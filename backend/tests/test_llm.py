"""Unit tests for the Groq candidate generator in app.services.llm."""

import json

import pytest

from app.core.config import Settings
from app.services.llm import LLMError, check_model, suggest_podcasts
from tests.llm_fakes import FakeGroq, json_validate_failed

LABELS = {
    "age": "20–34",
    "music_genre": ["Jazz"],
    "podcast_frequency": "Daily",
    "podcast_duration": "Extra long (60+ min)",
    "podcast_format": "Storytelling / Narrative",
    "podcast_content": ["True Crime & Mystery", "History"],
    "content_language": "English",
    "region": "Europe",
    "listening_mood": "Deep thinking",
    "podcasts_enjoyed": "Hardcore History, Serial",
}
PERSONA = {"name": "Long-form fans · Twenty-somethings", "description": "38% of surveyed listeners."}

VALID = json.dumps(
    {
        "candidates": [
            {"title": "Casefile True Crime", "host": "Casefile Presents",
             "summary": "Anonymous host narrates true crime cases.", "reason": "Long narrative true crime."},
            {"title": "  ", "host": "x", "summary": "", "reason": ""},
            {"title": "The Rest Is History", "host": "Goalhanger",
             "summary": "Two historians talk history.", "reason": "History, from Europe."},
        ]
    }
)


# --- Model health ------------------------------------------------------------


def test_default_model_is_not_retired():
    # llama-3.3-70b-versatile was shut down by Groq on 2026-08-16.
    assert Settings(_env_file=None).groq_model == "openai/gpt-oss-120b"


async def test_check_model_ok():
    status = await check_model(FakeGroq([VALID]), "openai/gpt-oss-120b")
    assert status == {"configured": True, "model": "openai/gpt-oss-120b", "ok": True, "error": None}


async def test_check_model_reports_unavailable_model():
    status = await check_model(FakeGroq([VALID], models_error=RuntimeError("model_not_found")), "retired")
    assert status["ok"] is False
    assert "model_not_found" in status["error"]


async def test_check_model_without_client():
    status = await check_model(None, "openai/gpt-oss-120b")
    assert status["configured"] is False and status["ok"] is False


# --- Request shape -------------------------------------------------------------


async def test_gpt_oss_uses_strict_schema_and_reasoning_budget():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    kwargs = client.calls[0]
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] >= 4096


async def test_other_models_use_json_mode_without_reasoning_effort():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, LABELS, PERSONA, "some-other-model")
    kwargs = client.calls[0]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in kwargs


async def test_prompt_carries_profile_persona_and_exclusions():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b", n=8)
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "True Crime & Mystery, History" in prompt
    assert "Extra long (60+ min)" in prompt
    assert PERSONA["name"] in prompt
    assert "Hardcore History, Serial" in prompt and "do NOT recommend" in prompt
    assert "8 podcasts" in prompt


# --- Parsing -------------------------------------------------------------------


async def test_parses_candidates_and_drops_blank_titles():
    suggestions = await suggest_podcasts(FakeGroq([VALID]), LABELS, PERSONA, "openai/gpt-oss-120b")
    assert [s.title for s in suggestions] == ["Casefile True Crime", "The Rest Is History"]
    assert suggestions[0].host == "Casefile Presents"
    assert suggestions[0].reason


async def test_retries_once_on_invalid_output():
    client = FakeGroq(["not json", VALID])
    suggestions = await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    assert len(suggestions) == 2
    assert len(client.calls) == 2


async def test_raises_after_repeated_invalid_output():
    client = FakeGroq(["not json", json.dumps({"candidates": []})])
    with pytest.raises(LLMError):
        await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")


async def test_api_error_raises_llm_error():
    client = FakeGroq([RuntimeError("503 from Groq")])
    with pytest.raises(LLMError, match="503"):
        await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")


async def test_never_returns_more_than_requested():
    many = json.dumps({"candidates": [
        {"title": f"Show {i}", "host": "H", "summary": "s", "reason": "r"} for i in range(15)
    ]})
    suggestions = await suggest_podcasts(FakeGroq([many]), LABELS, PERSONA, "openai/gpt-oss-120b", n=8)
    assert len(suggestions) == 8


async def test_retries_once_when_the_model_refuses_in_prose():
    # Live failure: gpt-oss answered "I'm sorry, but I can't find enough Hindi-language
    # interview podcasts..." and Groq's strict JSON mode rejected it with a 400.
    client = FakeGroq([json_validate_failed(), VALID])
    suggestions = await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    assert len(suggestions) == 2
    assert len(client.calls) == 2


async def test_gives_up_after_repeated_refusals():
    client = FakeGroq([json_validate_failed(), json_validate_failed()])
    with pytest.raises(LLMError):
        await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    assert len(client.calls) == 2


async def test_prompt_asks_for_fewer_shows_rather_than_a_refusal():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "return fewer" in prompt.lower()


async def test_prompt_makes_episode_length_a_rule():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, LABELS, PERSONA, "openai/gpt-oss-120b")
    rules = client.calls[0]["messages"][-1]["content"].split("Rules:")[1]
    assert "Extra long (60+ min)" in rules


async def test_no_length_rule_without_a_preference():
    client = FakeGroq([VALID])
    await suggest_podcasts(client, {**LABELS, "podcast_duration": "No preference"}, PERSONA, "openai/gpt-oss-120b")
    rules = client.calls[0]["messages"][-1]["content"].split("Rules:")[1]
    assert "No preference" not in rules
