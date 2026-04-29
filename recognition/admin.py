from django.contrib import admin
from .models import Identity


@admin.register(Identity)
class IdentityAdmin(admin.ModelAdmin):
    list_display = ("title", "class_id", "avatar")
    search_fields = ("title", "class_id")
    list_filter = ("class_id",)
