"""Root URL configuration for Hire Sense AI."""

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("coach.urls")),
]
