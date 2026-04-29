from django.db import models


class Identity(models.Model):
    title = models.CharField(max_length=255)
    class_id = models.CharField(max_length=255, unique=True)
    avatar = models.ImageField(upload_to="avatars/")

    class Meta:
        verbose_name_plural = "identities"
        ordering = ["class_id"]

    def __str__(self):
        return f"{self.title} ({self.class_id})"
