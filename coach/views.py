"""Views for the coach app."""

from django.shortcuts import render


def dashboard(request):
    """Render the main interview coach dashboard."""
    return render(request, "coach/dashboard.html")
