"""Integration tests for POST /recommend."""

import json

import pytest_asyncio

from app.main import app

VALID_BODY = {
    "age": "25-34",
    "music_genre": ["Pop", "Rock"],
    "podcast_frequency": "Several times a week",
    "podcast_duration": "Medium (30-60 min)",
    "podcast_format": "Interview",
    "podcast_content": ["Technology", "Educational"],
    "content_language": "English",
    "region": "Global",
    "listening_mood": "Curious",
}


# --- Fake async Groq client (OpenAI-compatible chat.completions) ------------

_FAKE_RECS = [
    {
        "name": f"Test Podcast {i}",
        "creator": f"Creator {i}",
        "description": "A description.",
        "format": "Interview",
        "duration": "Medium (30-60 min)",
        "language": "English",
        "region": "Global",
        "reason": "Matches your interests.",
    }
    for i in range(1, 6)
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, recs):
        self._recs = recs

    async def create(self, **kwargs):
        return _FakeResponse(json.dumps({"recommendations": self._recs}))


class _FakeChat:
    def __init__(self, recs):
        self.completions = _FakeCompletions(recs)


class _FakeClient:
    def __init__(self, recs):
        self.chat = _FakeChat(recs)


@pytest_asyncio.fixture
async def mocked_llm(client):
    """Swap app.state.llm_client for a fake that returns 5 recs."""
    original = app.state.llm_client
    app.state.llm_client = _FakeClient(_FAKE_RECS)
    try:
        yield client
    finally:
        app.state.llm_client = original


# --- Tests ------------------------------------------------------------------


async def test_recommend_with_mocked_llm(mocked_llm):
    response = await mocked_llm.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
    names = [r["name"] for r in data["recommendations"]]
    assert names == [f"Test Podcast {i}" for i in range(1, 6)]
    # The server fills in a link for every recommendation.
    assert all(r["link"].startswith("https://") for r in data["recommendations"])


async def test_recommend_fallback_without_client(client):
    # The lifespan sets llm_client to None when no API key is present.
    assert app.state.llm_client is None
    response = await client.post("/recommend", json=VALID_BODY)
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
    assert data["recommendations"][0]["name"] == "The Daily"


async def test_recommend_empty_list_returns_422(client):
    response = await client.post("/recommend", json={**VALID_BODY, "music_genre": []})
    assert response.status_code == 422


async def test_recommend_missing_field_returns_422(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "region"}
    response = await client.post("/recommend", json=body)
    assert response.status_code == 422
