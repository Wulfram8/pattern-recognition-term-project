import base64
import json
import os
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.core.files.base import ContentFile
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from PIL import Image

from .models import Identity
from .services import get_service


def index(request):
    return render(request, "recognition/index.html")


@csrf_exempt
@require_POST
def api_predict(request):
    image = None

    if request.FILES.get("image"):
        try:
            image = Image.open(request.FILES["image"])
        except Exception:
            return JsonResponse({"error": "Could not read uploaded image."}, status=400)
    else:
        try:
            body = json.loads(request.body)
            data_url = body.get("image_base64", "")
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(data_url)
            image = Image.open(BytesIO(image_bytes))
        except Exception:
            return JsonResponse({"error": "No valid image provided."}, status=400)

    if image is None:
        return JsonResponse({"error": "No image provided."}, status=400)

    service = get_service()
    result = service.predict(image, top_k=5)

    predicted_class_id = result["prediction"]["identity"]
    confidence = result["prediction"]["confidence"]

    identity = Identity.objects.filter(class_id=predicted_class_id).first()

    topk_list = []
    for item in result.get("topk", []):
        db_identity = Identity.objects.filter(class_id=item["identity"]).first()
        entry = {
            "rank": item["rank"],
            "class_id": item["identity"],
            "confidence": round(item["probability"] * 100, 2),
            "cosine_similarity": round(item.get("cosine_similarity", 0), 4),
        }
        if db_identity:
            entry["title"] = db_identity.title
            entry["avatar_url"] = db_identity.avatar.url if db_identity.avatar else None
        topk_list.append(entry)

    response_data = {
        "prediction": {
            "class_id": predicted_class_id,
            "confidence": round(confidence * 100, 2),
            "cosine_similarity": result["prediction"].get("cosine_similarity", 0),
        },
        "topk": topk_list,
        "identity_found": identity is not None,
    }

    if identity:
        response_data["identity"] = {
            "id": identity.id,
            "title": identity.title,
            "class_id": identity.class_id,
            "avatar_url": identity.avatar.url if identity.avatar else None,
        }

    return JsonResponse(response_data)


@csrf_exempt
@require_POST
def api_add_identity(request):
    title = request.POST.get("title", "").strip()
    class_id = request.POST.get("class_id", "").strip()
    avatar_file = request.FILES.get("avatar")
    avatar_base64 = request.POST.get("avatar_base64", "")

    if not title or not class_id:
        return JsonResponse({"error": "Title and class_id are required."}, status=400)

    if Identity.objects.filter(class_id=class_id).exists():
        return JsonResponse({"error": "Identity with this class_id already exists."}, status=400)

    image_bytes = None
    identity = Identity(title=title, class_id=class_id)

    if avatar_file:
        raw = avatar_file.read()
        image_bytes = raw
        avatar_file.seek(0)
        identity.avatar.save(f"{class_id}.jpg", avatar_file, save=False)
    elif avatar_base64:
        try:
            b64 = avatar_base64
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            image_bytes = base64.b64decode(b64)
            identity.avatar.save(f"{class_id}.jpg", ContentFile(image_bytes), save=False)
        except Exception:
            return JsonResponse({"error": "Invalid avatar image data."}, status=400)

    identity.save()

    if image_bytes:
        try:
            gallery_dir = Path(settings.GALLERY_ROOT) / class_id
            gallery_dir.mkdir(parents=True, exist_ok=True)
            gallery_path = gallery_dir / f"{class_id}_001.jpg"
            with open(gallery_path, "wb") as f:
                f.write(image_bytes)

            service = get_service()
            pil_image = Image.open(BytesIO(image_bytes))
            service.add_to_gallery(class_id, pil_image)
        except Exception:
            pass

    return JsonResponse({
        "success": True,
        "identity": {
            "id": identity.id,
            "title": identity.title,
            "class_id": identity.class_id,
            "avatar_url": identity.avatar.url if identity.avatar else None,
        },
    })
