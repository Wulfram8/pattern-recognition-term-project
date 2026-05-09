import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.model import FaceIdentificationService


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="Face Identification Deployment",
    description="Face identification using CosFace",
    version="2.0.0",
)

service: Optional[FaceIdentificationService] = None


@app.on_event("startup")
def startup():
    global service

    service = FaceIdentificationService(
        checkpoint_path=os.getenv("CHECKPOINT_PATH", "/models/cosface_best.pt"),
        gallery_root=os.getenv("GALLERY_ROOT", "/data/gallery"),
        gallery_cache=os.getenv("GALLERY_CACHE", "/models/gallery_cache.pt"),
        device=os.getenv("DEVICE", "auto"),
        gallery_images_per_identity=int(os.getenv("GALLERY_IMAGES_PER_IDENTITY", "5")),
        auto_build_gallery=os.getenv("AUTO_BUILD_GALLERY", "true").lower()
        in {"1", "true", "yes", "y"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    return {"status": "ok", "model": service.info()}


@app.get("/api/model-info")
def model_info():
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    return service.info()


@app.post("/api/build-gallery")
def build_gallery():
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")
    gallery_root = os.getenv("GALLERY_ROOT", "/data/gallery")
    gallery_cache = os.getenv("GALLERY_CACHE", "/models/gallery_cache.pt")
    return service.build_gallery(gallery_root, gallery_cache)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), top_k: int = Form(5)):
    if service is None:
        raise HTTPException(status_code=503, detail="Model service is not ready.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    try:
        return service.predict(image, top_k=top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
