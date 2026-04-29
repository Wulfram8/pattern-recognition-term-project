import sys
from pathlib import Path
from threading import Lock

from django.conf import settings

sys.path.insert(0, str(Path(settings.BASE_DIR)))

from model.model import CosFaceService

_service = None
_lock = Lock()


def get_service():
    global _service
    if _service is not None:
        return _service
    with _lock:
        if _service is not None:
            return _service
        _service = CosFaceService(
            checkpoint_path=settings.CHECKPOINT_PATH,
            gallery_root=settings.GALLERY_ROOT,
            device="cpu",
            gallery_images_per_identity=5,
            auto_build_gallery=True,
        )
        return _service
