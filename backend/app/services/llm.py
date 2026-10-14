"""Groq-backed podcast candidate generation.

The LLM proposes candidate shows (title + host + why); it is never the source
of truth for what a show *is*. Every candidate is then verified against the
Apple catalog (app.services.catalog), which supplies the real metadata, and
unverifiable candidates are dropped.

gpt-oss models get strict JSON-schema output (constrained decoding), so the
response always parses; other models fall back to JSON mode plus validation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# gpt-oss models reason before answering; reasoning tokens share the completion
# budget, so a small max_tokens can cut the JSON off mid-object.
_MAX_COMPLETION_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are a meticulous podcast curator. You recommend only real podcasts that you "
    "are confident exist on Apple Podcasts and Spotify, and you respond with JSON only."
)


class LLMError(Exception):
    """The LLM failed or kept returning unusable output."""


class Suggestion(BaseModel):
    title: str
    host: str
    summary: str
    reason: str


class _Suggestions(BaseModel):
    candidates: List[Suggestion]


_SUGGESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "host": {"type": "string"},
                    "summary": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["title", "host", "summary", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


async def check_model(client: Optional[AsyncGroq], model: str) -> Dict[str, Any]:
    """Report whether the configured Groq model is reachable.

    Run at startup so a retired or mistyped model shows up in /health instead
    of every request silently degrading.
    """
    if client is None:
        return {"configured": False, "model": model, "ok": False, "error": "GROQ_API_KEY not set"}
    try:
        await client.models.retrieve(model)
    except Exception as exc:  # noqa: BLE001 - any failure means the model is unusable
        logger.error(f"Groq model {model!r} unavailable: {exc}")
        return {"configured": True, "model": model, "ok": False, "error": str(exc)}
    return {"configured": True, "model": model, "ok": True, "error": None}


def _join(value: Any) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value)


def _build_prompt(labels: Dict[str, Any], persona: Dict[str, Any], n: int) -> str:
    lines = [
        f"Recommend {n} podcasts for this listener.",
        "",
        "Listener profile:",
        f"- Age group: {labels['age']}",
        f"- Topics they want: {_join(labels['podcast_content'])}",
        f"- Preferred format: {labels['podcast_format']}",
        f"- Preferred episode length: {labels['podcast_duration']}",
        f"- How often they listen: {labels['podcast_frequency']}",
        f"- Music they enjoy: {_join(labels['music_genre'])}",
        f"- Language: {labels['content_language']}",
        f"- Region of interest: {labels['region']}",
        f"- Mood right now: {labels['listening_mood']}",
    ]
    if labels.get("podcasts_enjoyed"):
        lines.append(
            f"- Podcasts they already enjoy (do NOT recommend these; use them to infer taste): "
            f"{labels['podcasts_enjoyed']}"
        )
    if persona.get("name"):
        lines.append(f"- Survey persona: {persona['name']} — {persona.get('description', '')}")
    lines += [
        "",
        "Rules:",
        "- Only recommend real podcasts that exist on Apple Podcasts and Spotify today. Use the "
        "exact official show title and the host or publisher as listed there. Never invent a show.",
        "- Prefer shows that are still publishing new episodes.",
        f"- Every show must be in {labels['content_language']}.",
        "- Mix well-known and lesser-known shows; at most one show per publisher.",
        '- "summary": one factual sentence about what the show is.',
        '- "reason": one sentence on why it fits this listener, referring to their answers.',
        "",
        'Respond with JSON: {"candidates": [{"title": ..., "host": ..., "summary": ..., '
        '"reason": ...}]}',
    ]
    return "\n".join(lines)


def _request_kwargs(model: str) -> Dict[str, Any]:
    if _is_reasoning_model(model):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "podcast_candidates", "strict": True, "schema": _SUGGESTION_SCHEMA},
            },
            "reasoning_effort": "low",
            "include_reasoning": False,
        }
    return {"response_format": {"type": "json_object"}}


async def suggest_podcasts(
    client: AsyncGroq,
    labels: Dict[str, Any],
    persona: Dict[str, Any],
    model: str,
    n: int = 8,
) -> List[Suggestion]:
    """Ask the LLM for ``n`` candidate shows. Raises LLMError on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(labels, persona, n)},
    ]
    problem: object = None
    for _ in range(2):  # one retry on unusable output
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=_MAX_COMPLETION_TOKENS,
                temperature=0.4,
                **_request_kwargs(model),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as LLMError; caller degrades
            raise LLMError(f"Groq request failed (model={model}): {exc}") from exc
        try:
            parsed = _Suggestions.model_validate(json.loads(response.choices[0].message.content))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            problem = exc
            continue
        suggestions = [s for s in parsed.candidates if s.title.strip()]
        if suggestions:
            return suggestions
        problem = "no candidates"
    raise LLMError(f"unusable output from {model}: {problem}")
