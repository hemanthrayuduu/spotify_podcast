"""Rate-limit key: must not be spoofable through X-Forwarded-For."""

from starlette.requests import Request

from app.core.limiter import client_ip


def _request(xff=None, peer="10.0.0.7"):
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_uses_the_address_the_proxy_appended():
    assert client_ip(_request("81.2.69.142")) == "81.2.69.142"


def test_ignores_client_supplied_entries_on_the_left():
    # A client sends "X-Forwarded-For: 6.6.6.6"; the proxy appends the real IP.
    assert client_ip(_request("6.6.6.6, 81.2.69.142")) == "81.2.69.142"


def test_skips_private_internal_hops():
    assert client_ip(_request("81.2.69.142, 10.1.2.3")) == "81.2.69.142"


def test_ignores_garbage_entries():
    assert client_ip(_request("81.2.69.142, not-an-ip")) == "81.2.69.142"


def test_falls_back_to_peer_without_header():
    assert client_ip(_request()) == "10.0.0.7"
