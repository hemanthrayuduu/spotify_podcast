"""Shared test fixtures."""

import httpx
import pytest_asyncio

from app.main import app


@pytest_asyncio.fixture
async def client():
    """An HTTP client bound to the app, with the lifespan (model load) run."""
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
