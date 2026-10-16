"""Shared rate limiter instance.

Kept in its own module so both the app factory (which registers it) and the
routers (which decorate endpoints with it) import the same object.
"""

import ipaddress

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request


def client_ip(request: Request) -> str:
    """Rate-limit key: the rightmost public address in X-Forwarded-For.

    Render's proxy appends the connecting client's address to the header and
    keeps whatever the client sent, so entries to the left are spoofable;
    trusting the leftmost (as uvicorn --forwarded-allow-ips="*" does) lets
    anyone mint a fresh rate-limit bucket per request. Private internal hops
    are skipped.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    for part in reversed([p.strip() for p in forwarded.split(",") if p.strip()]):
        try:
            address = ipaddress.ip_address(part)
        except ValueError:
            continue
        if address.is_global:
            return str(address)
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)
