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


def make_resnet_backbone(backbone="resnet34", pretrained=False):

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return base


class ResNetEmbedding(nn.Module):

    def __init__(self, backbone="resnet34", emb_dim=512, pretrained=False, dropout=0.10):
        super().__init__()
        base = make_resnet_backbone(backbone=backbone, pretrained=pretrained)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.neck = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        nn.init.xavier_uniform_(self.neck[1].weight)

    def forward(self, x):
        feat = self.backbone(x)
        raw_emb = self.neck(feat)
        emb = F.normalize(raw_emb, p=2, dim=1)
        return {"embeddings": emb, "raw_embeddings": raw_emb, "norms": raw_emb.norm(p=2, dim=1)}


class CosFaceHead(nn.Module):

    def __init__(self, emb_dim, num_classes, s=48.0, m=0.25):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.weight = nn.Parameter(torch.empty(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        if labels is None:
            return self.s * cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = cosine - one_hot * self.m
        return self.s * logits


class CosFaceModel(nn.Module):

    def __init__(self, num_classes, backbone="resnet34", emb_dim=512,
                 pretrained=False, dropout=0.10, cosface_s=48.0, cosface_m=0.25):
        super().__init__()
        self.encoder = ResNetEmbedding(backbone, emb_dim, pretrained, dropout)
        self.head = CosFaceHead(emb_dim, num_classes, s=cosface_s, m=cosface_m)

    def forward(self, x, labels=None):
        out = self.encoder(x)
        out["logits"] = self.head(out["embeddings"], labels)
        return out


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_cfg(cfg, key: str, default):
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


class FaceIdentificationService:
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
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.cfg: Dict[str, Any] = {}
        self.num_classes: int = 0
        self.model_name: str = "cosface"

        self.model: Optional[CosFaceModel] = None

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
                "Copy your CosFace checkpoint to model/checkpoints/cosface_best.pt."
            )

        ckpt = safe_torch_load(str(ckpt_path), map_location="cpu")

        # The notebook saves: model_name, epoch, best_val_top1,
        # model_state_dict, optimizer_state_dict, scheduler_state_dict,
        # cfg (dataclass as dict), num_classes.
        if "model_state_dict" not in ckpt:
            raise KeyError(
                "This app expects the checkpoint from the training notebook with key "
                "'model_state_dict'. Found keys: " + str(list(ckpt.keys()))
            )

        self.cfg = ckpt.get("cfg", {}) or {}
        self.model_name = str(ckpt.get("model_name", "cosface")).lower()
        self.num_classes = int(ckpt.get("num_classes", 0))

        # Read hyper-parameters from the saved config
        backbone = str(read_cfg(self.cfg, "backbone", "resnet34"))
        emb_dim = int(read_cfg(self.cfg, "embedding_dim", 512))
        dropout = float(read_cfg(self.cfg, "dropout", 0.10))
        img_size = int(read_cfg(self.cfg, "img_size", 112))
        cosface_s = float(read_cfg(self.cfg, "cosface_s", 48.0))
        cosface_m = float(read_cfg(self.cfg, "cosface_m", 0.25))

        # If num_classes wasn't saved, try to infer from head weight shape
        if self.num_classes == 0:
            state = strip_module_prefix(ckpt["model_state_dict"])
            for key in ("head.weight",):
                if key in state:
                    self.num_classes = state[key].shape[0]
                    break

        self.model = CosFaceModel(
            num_classes=self.num_classes,
            backbone=backbone,
            emb_dim=emb_dim,
            pretrained=False,
            dropout=dropout,
            cosface_s=cosface_s,
            cosface_m=cosface_m,
        )

        model_state = strip_module_prefix(ckpt["model_state_dict"])
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device).eval()

        self.transform = T.Compose([
            T.Resize((img_size, img_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> torch.Tensor:
        self.model.eval()
        x = self.preprocess(image)
        out = self.model.encoder(x)
        return out["embeddings"]  # already L2-normalized

    # ------------------------------------------------------------------
    # Prediction / identification
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """Identify a face using Prototype-Centroid Retrieval.

        The CosFace backbone is used purely as an embedding extractor.
        Identification is performed by computing cosine similarity between
        the probe embedding and each gallery identity centroid (prototype).
        """
        top_k = max(1, min(int(top_k), 20))
        emb = self.embed_image(image)

        if self.gallery_prototypes is None or len(self.gallery_labels) == 0:
            return {
                "prediction": {
                    "source": "prototype_centroid_retrieval",
                    "identity": "unknown",
                    "confidence": 0.0,
                    "cosine_similarity": 0.0,
                },
                "topk": [],
                "model": self.info(),
                "error": "Gallery not loaded. No identities to match against.",
            }

        # Cosine similarity between probe and every gallery centroid
        prototypes = self.gallery_prototypes.to(self.device)
        sims = (emb @ prototypes.T).squeeze(0)

        # Temperature-scaled softmax gives probability-like confidence scores
        gallery_probs = torch.softmax(sims * 30.0, dim=0)

        g_values, g_indices = torch.topk(
            gallery_probs, k=min(top_k, gallery_probs.shape[0]), dim=0
        )

        topk = []
        for rank, (p, idx) in enumerate(
            zip(g_values.detach().cpu().tolist(),
                g_indices.detach().cpu().tolist()),
            start=1,
        ):
            idx = int(idx)
            topk.append({
                "rank": rank,
                "identity": self.gallery_labels[idx],
                "probability": float(p),
                "cosine_similarity": float(sims[idx].detach().cpu()),
            })

        prediction = {
            "source": "prototype_centroid_retrieval",
            "identity": topk[0]["identity"],
            "confidence": topk[0]["probability"],
            "cosine_similarity": topk[0]["cosine_similarity"],
        }

        return {
            "prediction": prediction,
            "topk": topk,
            "model": self.info(),
        }

    def build_gallery(self, gallery_root: str, cache_path: Optional[str] = None) -> Dict[str, Any]:
        root = Path(gallery_root)
        if not root.exists():
            return {"gallery_size": 0, "message": f"Gallery root not found: {root}"}

        labels: List[str] = []
        prototypes: List[torch.Tensor] = []

        identity_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

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
            self.gallery_prototypes = F.normalize(
                self.gallery_prototypes.float(), p=2, dim=1)

    def add_to_gallery(self, label: str, image: Image.Image) -> Dict[str, Any]:
        emb = self.embed_image(image).detach().cpu()
        proto = F.normalize(emb, p=2, dim=1).squeeze(0)

        if label in self.gallery_labels:
            idx = self.gallery_labels.index(label)
            old_proto = self.gallery_prototypes[idx]
            new_proto = F.normalize((old_proto + proto).unsqueeze(0), p=2, dim=1).squeeze(0)
            self.gallery_prototypes[idx] = new_proto
            return {"action": "updated", "label": label, "gallery_size": len(self.gallery_labels)}

        self.gallery_labels.append(label)
        if self.gallery_prototypes is None:
            self.gallery_prototypes = proto.unsqueeze(0)
        else:
            self.gallery_prototypes = torch.cat(
                [self.gallery_prototypes, proto.unsqueeze(0)], dim=0
            )
        return {"action": "added", "label": label, "gallery_size": len(self.gallery_labels)}

    def info(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "model_name": self.model_name,
            "backbone": str(read_cfg(self.cfg, "backbone", "resnet34")),
            "image_size": int(read_cfg(self.cfg, "img_size", 112)),
            "embedding_dim": int(read_cfg(self.cfg, "embedding_dim", 512)),
            "cosface_s": float(read_cfg(self.cfg, "cosface_s", 48.0)),
            "cosface_m": float(read_cfg(self.cfg, "cosface_m", 0.25)),
            "num_classes": self.num_classes,
            "gallery_loaded": self.gallery_prototypes is not None,
            "gallery_size": len(self.gallery_labels),
        }
