"""Unit tests for the Groq client helpers in app.services.llm."""

from app.core.config import Settings
from app.services.llm import check_model, generate_podcast_recommendations

PREFS = {
    "age": "25-34",
    "music_genre": ["Pop"],
    "podcast_frequency": "Daily",
    "podcast_duration": "Medium (30-60 min)",
    "podcast_format": "Interview",
    "podcast_content": ["Technology"],
    "content_language": "English",
    "region": "Global",
    "listening_mood": "Curious",
}


class _Models:
    def __init__(self, error=None):
        self._error = error

    async def retrieve(self, model):
        if self._error:
            raise self._error
        return {"id": model}


class _Completions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        raise RuntimeError("stop after capturing kwargs")


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _Client:
    def __init__(self, models_error=None):
        self.models = _Models(models_error)
        self.chat = _Chat()


def test_default_model_is_not_retired():
    # llama-3.3-70b-versatile was shut down by Groq on 2026-08-16.
    assert Settings(_env_file=None).groq_model == "openai/gpt-oss-120b"


async def test_check_model_ok():
    status = await check_model(_Client(), "openai/gpt-oss-120b")
    assert status == {"configured": True, "model": "openai/gpt-oss-120b", "ok": True, "error": None}


async def test_check_model_reports_unavailable_model():
    status = await check_model(_Client(RuntimeError("model_not_found")), "retired-model")
    assert status["ok"] is False
    assert "model_not_found" in status["error"]


async def test_check_model_without_client():
    status = await check_model(None, "openai/gpt-oss-120b")
    assert status == {
        "configured": False,
        "model": "openai/gpt-oss-120b",
        "ok": False,
        "error": "GROQ_API_KEY not set",
    }


async def test_gpt_oss_request_budgets_for_reasoning():
    client = _Client()
    await generate_podcast_recommendations(client, PREFS, {}, "openai/gpt-oss-120b")
    kwargs = client.chat.completions.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] >= 4096


async def test_non_reasoning_model_omits_reasoning_effort():
    client = _Client()
    await generate_podcast_recommendations(client, PREFS, {}, "some-other-model")
    assert "reasoning_effort" not in client.chat.completions.kwargs
