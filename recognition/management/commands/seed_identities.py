from pathlib import Path

from PIL import Image

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand

from recognition.models import Identity
from recognition.services import get_service

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class Command(BaseCommand):

    def handle(self, *args, **options):
        # Gallery images live under data/train (folder-per-identity)
        gallery_root = Path(settings.GALLERY_ROOT)
        if not gallery_root.exists():
            self.stdout.write(self.style.ERROR(f"Gallery root not found: {gallery_root}"))
            return

        media_avatars = Path(settings.MEDIA_ROOT) / "avatars"
        media_avatars.mkdir(parents=True, exist_ok=True)

        service = get_service()

        identity_dirs = sorted(
            [p for p in gallery_root.iterdir() if p.is_dir()],
            key=lambda p: p.name,
        )

        created = 0
        for identity_dir in identity_dirs:
            images = sorted(
                [f for f in identity_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            )

            if not images:
                self.stdout.write(f"Skipping {identity_dir.name} (no images)")
                continue

            first_image = images[0]

            img = Image.open(first_image)
            result = service.predict(img, top_k=1)
            class_id = result["prediction"]["identity"]

            if Identity.objects.filter(class_id=class_id).exists():
                self.stdout.write(f"Skipping {identity_dir.name} -> {class_id} (already exists)")
                continue

            title = f"Identity {identity_dir.name}"

            identity = Identity(title=title, class_id=class_id)
            avatar_name = f"{class_id}{first_image.suffix.lower()}"
            with open(first_image, "rb") as f:
                identity.avatar.save(avatar_name, File(f), save=False)
            identity.save()

            created += 1
            self.stdout.write(f"Created {title} (folder={identity_dir.name}, class_id={class_id})")

        self.stdout.write(self.style.SUCCESS(f"Seeded {created} identities"))
