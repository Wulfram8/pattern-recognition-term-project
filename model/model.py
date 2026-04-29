import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ResNet50FaceEmbedder(nn.Module):
    """
    This matches your training notebook:

    ResNet50 ImageNet backbone
    -> remove final classifier
    -> BatchNorm
    -> Dropout
    -> Linear to 512-d embedding
    -> BatchNorm
    -> L2 normalize

    In deployment we use weights=None because the trained weights are loaded
    from your checkpoint. Docker does not need to download ImageNet weights.
    """

    def __init__(self, emb_dim: int = 512, dropout: float = 0.20):
        super().__init__()
        base = models.resnet50(weights=None)
        in_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.backbone = base
        self.neck = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )

        # Same face-recognition trick used in training.
        nn.init.constant_(self.neck[-1].weight, 1.0)
        nn.init.constant_(self.neck[-1].bias, 0.0)
        self.neck[-1].bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        z = self.neck(z)
        return F.normalize(z, p=2, dim=1)


class MarginHead(nn.Module):
    """
    Training-compatible margin head.

    The checkpoint stores head.weight.
    For deployment, prediction uses class-center cosine logits:
        logits = s * cosine(embedding, class_weight)

    We do not subtract the CosFace margin during inference because there is
    no ground-truth class label at prediction time.
    """

    def __init__(
        self,
        emb_dim: int,
        num_classes: int,
        method: str = "cosface",
        s: float = 64.0,
        m: float = 0.35,
    ):
        super().__init__()
        self.method = method
        self.s = float(s)
        self.target_m = float(m)
        self.current_m = float(m)
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        if method == "arcface":
            self._update_arc_constants()

    def _update_arc_constants(self):
        m = self.current_m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_margin_scale(self, scale: float):
        self.current_m = self.target_m * float(scale)
        if self.method == "arcface":
            self._update_arc_constants()

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(
            -1.0 + 1e-7, 1.0 - 1e-7
        )

        if self.method == "softmax":
            return cosine * self.s

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        if self.method == "cosface":
            phi = cosine - self.current_m
        elif self.method == "arcface":
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0.0, 1.0))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.s

    @torch.no_grad()
    def inference_logits(self, emb: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1.0, 1.0)
        return cosine * self.s


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Supports checkpoints saved with or without nn.DataParallel."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_cfg(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


class CosFaceService:
    def __init__(
        self,
        checkpoint_path: str,
        gallery_root: Optional[str] = None,
        gallery_cache: Optional[str] = None,
        device: str = "auto",
        gallery_images_per_identity: int = 5,
        auto_build_gallery: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.gallery_root = gallery_root
        self.gallery_cache = gallery_cache
        self.gallery_images_per_identity = int(gallery_images_per_identity)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.cfg = {}
        self.method = "cosface"
        self.label_to_identity: Dict[int, str] = {}

        self.embedder: Optional[ResNet50FaceEmbedder] = None
        self.head: Optional[MarginHead] = None

        self.gallery_labels: List[str] = []
        self.gallery_prototypes: Optional[torch.Tensor] = None

        self.transform = None

        self.load_checkpoint()

        if gallery_cache and Path(gallery_cache).exists():
            self.load_gallery_cache(gallery_cache)
        elif auto_build_gallery and gallery_root and Path(gallery_root).exists():
            self.build_gallery(gallery_root, gallery_cache)

    def load_checkpoint(self):
        ckpt_path = Path(self.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. "
                "Copy your CosFace checkpoint to models/cosface_best.pt."
            )

        ckpt = safe_torch_load(str(ckpt_path), map_location="cpu")

        if "embedder" not in ckpt or "head" not in ckpt:
            raise KeyError(
                "This app expects the checkpoint from your notebook with keys: "
                "'embedder', 'head', 'cfg', and 'id_to_label'."
            )

        self.cfg = ckpt.get("cfg", {}) or {}
        self.method = str(ckpt.get("method", "cosface")).lower()

        embedder_state = strip_module_prefix(ckpt["embedder"])
        head_state = strip_module_prefix(ckpt["head"])

        num_classes, emb_dim_from_head = head_state["weight"].shape

        emb_dim = int(read_cfg(self.cfg, "emb_dim", int(emb_dim_from_head)))
        dropout = float(read_cfg(self.cfg, "dropout", 0.20))
        image_size = int(read_cfg(self.cfg, "image_size", 112))
        scale_s = float(read_cfg(self.cfg, "scale_s", 64.0))
        cosface_m = float(read_cfg(self.cfg, "cosface_m", 0.35))

        self.embedder = ResNet50FaceEmbedder(emb_dim=emb_dim, dropout=dropout)
        self.head = MarginHead(
            emb_dim=emb_dim,
            num_classes=int(num_classes),
            method=self.method,
            s=scale_s,
            m=cosface_m,
        )

        self.embedder.load_state_dict(embedder_state, strict=True)
        self.head.load_state_dict(head_state, strict=True)

        self.embedder.to(self.device).eval()
        self.head.to(self.device).eval()

        # Your notebook saves id_to_label as identity -> integer label.
        raw_id_to_label = ckpt.get("id_to_label", {}) or {}
        if raw_id_to_label:
            self.label_to_identity = {int(v): str(k) for k, v in raw_id_to_label.items()}
        else:
            self.label_to_identity = {i: f"class_{i}" for i in range(int(num_classes))}

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        x = self.preprocess(image)
        emb = self.embedder(x)
        return F.normalize(emb, p=2, dim=1)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        top_k = max(1, min(int(top_k), 20))
        emb = self.embed_image(image)

        logits = self.head.inference_logits(emb)
        probs = torch.softmax(logits, dim=1)
        values, indices = torch.topk(probs, k=min(top_k, probs.shape[1]), dim=1)
        cosine_scores = (logits / self.head.s)[0, indices[0]]

        head_topk = []
        for rank, (p, idx, cos) in enumerate(
            zip(
                values[0].detach().cpu().tolist(),
                indices[0].detach().cpu().tolist(),
                cosine_scores.detach().cpu().tolist(),
            ),
            start=1,
        ):
            head_topk.append(
                {
                    "rank": rank,
                    "label_index": int(idx),
                    "identity": self.label_to_identity.get(int(idx), f"class_{idx}"),
                    "probability": float(p),
                    "cosine_similarity_to_class_weight": float(cos),
                }
            )

        gallery_topk = []
        if self.gallery_prototypes is not None and len(self.gallery_labels) > 0:
            prototypes = self.gallery_prototypes.to(self.device)
            sims = (emb @ prototypes.T).squeeze(0)
            # Temperature scaling produces a probability-like confidence over gallery identities.
            gallery_probs = torch.softmax(sims * 30.0, dim=0)
            g_values, g_indices = torch.topk(
                gallery_probs, k=min(top_k, gallery_probs.shape[0]), dim=0
            )

            for rank, (p, idx) in enumerate(
                zip(g_values.detach().cpu().tolist(), g_indices.detach().cpu().tolist()),
                start=1,
            ):
                idx = int(idx)
                gallery_topk.append(
                    {
                        "rank": rank,
                        "identity": self.gallery_labels[idx],
                        "probability": float(p),
                        "cosine_similarity": float(sims[idx].detach().cpu()),
                    }
                )

        if gallery_topk:
            prediction = {
                "source": "gallery_prototype_search",
                "identity": gallery_topk[0]["identity"],
                "confidence": gallery_topk[0]["probability"],
                "cosine_similarity": gallery_topk[0]["cosine_similarity"],
            }
        else:
            prediction = {
                "source": "cosface_head",
                "identity": head_topk[0]["identity"],
                "confidence": head_topk[0]["probability"],
                "cosine_similarity": head_topk[0]["cosine_similarity_to_class_weight"],
            }

        return {
            "prediction": prediction,
            "head_topk": head_topk,
            "gallery_topk": gallery_topk,
            "model": self.info(),
        }

    def build_gallery(self, gallery_root: str, cache_path: Optional[str] = None) -> Dict[str, Any]:
        root = Path(gallery_root)
        if not root.exists():
            return {"gallery_size": 0, "message": f"Gallery root not found: {root}"}

        labels = []
        prototypes = []

        identity_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)

        for identity_dir in identity_dirs:
            image_paths = sorted(
                [
                    p
                    for p in identity_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ]
            )[: self.gallery_images_per_identity]

            if not image_paths:
                continue

            embeddings = []
            for path in image_paths:
                try:
                    with Image.open(path) as img:
                        embeddings.append(self.embed_image(img).detach().cpu())
                except Exception:
                    continue

            if not embeddings:
                continue

            proto = torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True)
            proto = F.normalize(proto, p=2, dim=1).squeeze(0)

            labels.append(identity_dir.name)
            prototypes.append(proto)

        if not prototypes:
            self.gallery_labels = []
            self.gallery_prototypes = None
            return {"gallery_size": 0, "message": "No valid gallery images found."}

        self.gallery_labels = labels
        self.gallery_prototypes = torch.stack(prototypes, dim=0).float()

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "labels": self.gallery_labels,
                    "prototypes": self.gallery_prototypes,
                    "gallery_root": str(root),
                },
                cache_path,
            )

        return {
            "gallery_size": len(self.gallery_labels),
            "gallery_root": str(root),
            "cache_path": cache_path,
        }

    def load_gallery_cache(self, cache_path: str):
        payload = safe_torch_load(cache_path, map_location="cpu")
        self.gallery_labels = [str(x) for x in payload.get("labels", [])]
        self.gallery_prototypes = payload.get("prototypes", None)
        if self.gallery_prototypes is not None:
            self.gallery_prototypes = F.normalize(self.gallery_prototypes.float(), p=2, dim=1)

    def info(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "method": self.method,
            "backbone": "resnet50_imagenet_pretrained_structure",
            "image_size": int(read_cfg(self.cfg, "image_size", 112)),
            "embedding_dim": int(read_cfg(self.cfg, "emb_dim", 512)),
            "scale_s": float(read_cfg(self.cfg, "scale_s", 64.0)),
            "cosface_m": float(read_cfg(self.cfg, "cosface_m", 0.35)),
            "num_classes": len(self.label_to_identity),
            "gallery_loaded": self.gallery_prototypes is not None,
            "gallery_size": len(self.gallery_labels),
        }
