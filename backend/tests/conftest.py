"""Shared test fixtures."""

import os

# Keep tests hermetic: a developer's backend/.env may hold a real GROQ_API_KEY,
# which would make the app call Groq at startup. Environment variables take
# precedence over .env in pydantic-settings, so blank it before app import.
os.environ["GROQ_API_KEY"] = ""

import httpx  # noqa: E402
import pytest_asyncio  # noqa: E402

from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client():
    """An HTTP client bound to the app, with the lifespan (model load) run."""
    # Tests make far more requests than the per-IP limit allows.
    app.state.limiter.enabled = False
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
