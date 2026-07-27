"""Shared rate limiter instance.

Kept in its own module so both the app factory (which registers it) and the
routers (which decorate endpoints with it) import the same object.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
