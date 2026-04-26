"""
ASGI config for Hire Sense AI.

Routes both HTTP and WebSocket traffic through Django Channels.
Daphne uses this as its entry point.
"""

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hiresense.settings")

# Initialize Django ASGI application early to ensure the AppRegistry is
# populated before importing consumers that depend on Django models/settings.
django_asgi_app = get_asgi_application()

# Import WebSocket routing AFTER Django is initialized
from coach.routing import websocket_urlpatterns  # noqa: E402

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        ),
    }
)
