from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("api/predict/", views.api_predict, name="api_predict"),
    path("api/add-identity/", views.api_add_identity, name="api_add_identity"),
]
