"""Groq-backed podcast curation.

The LLM is a curator, not the source of truth for what a show *is*. It gets a
list of real candidate shows (from Apple's genre charts, with real episode
lengths) and picks the best fits by id, so those picks can't be invented. It
may also add a couple of shows of its own by title and host; those are
verified against the Apple catalog (app.services.catalog) like any other
claim, and dropped if they can't be found.

gpt-oss models get strict JSON-schema output (constrained decoding), so the
response always parses; other models fall back to JSON mode plus validation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from groq import AsyncGroq, BadRequestError
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


@dataclass(frozen=True)
class PoolEntry:
    """A real candidate show as the curator sees it."""

    id: int
    title: str
    publisher: str
    genres: List[str]
    minutes: Optional[int]  # median of recent episodes, from the catalog
    days_since_last: Optional[int]


class Pick(BaseModel):
    """A candidate from the list, chosen by id."""

    id: int
    summary: str
    reason: str


class Suggestion(BaseModel):
    """A show the curator proposes from outside the list; must be verified."""

    title: str
    host: str
    summary: str
    reason: str


class _Choice(BaseModel):
    picks: List[Pick] = []
    extras: List[Suggestion] = []


@dataclass
class Choice:
    picks: List[Pick] = field(default_factory=list)
    extras: List[Suggestion] = field(default_factory=list)


_TEXT = {"type": "string"}
_CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "summary": _TEXT, "reason": _TEXT},
                "required": ["id", "summary", "reason"],
                "additionalProperties": False,
            },
        },
        "extras": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": _TEXT, "host": _TEXT, "summary": _TEXT, "reason": _TEXT},
                "required": ["title", "host", "summary", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["picks", "extras"],
    "additionalProperties": False,
}


def _is_json_generation_failure(exc: Exception) -> bool:
    """Groq's 400 when strict JSON decoding fails, e.g. the model refused in prose."""
    if not isinstance(exc, BadRequestError) or not isinstance(exc.body, dict):
        return False
    error = exc.body.get("error")
    return isinstance(error, dict) and error.get("code") == "json_validate_failed"


def _is_reasoning_model(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


async def check_model(client: Optional[AsyncGroq], model: str) -> Dict[str, Any]:
    """Report whether the configured Groq model is reachable.

    Run at startup so a retired or mistyped model shows up in /health instead
    of every request silently degrading.
    """
    if client is None:
        return {"configured": False, "model": model, "ok": False, "error": "GROQ_API_KEY not set"}
    # List rather than models.retrieve(): groq>=1.x percent-encodes the id into
    # the URL path ("openai%2Fgpt-oss-120b"), which the API answers with a 404.
    try:
        available = {m.id for m in (await client.models.list()).data}
    except Exception as exc:  # noqa: BLE001 - any failure means the model is unusable
        logger.error(f"Groq model list unavailable: {exc}")
        return {"configured": True, "model": model, "ok": False, "error": str(exc)}
    if model not in available:
        error = f"model {model!r} is not available to this API key"
        logger.error(f"Groq {error}")
        return {"configured": True, "model": model, "ok": False, "error": error}
    return {"configured": True, "model": model, "ok": True, "error": None}


def _join(value: Any) -> str:
    return ", ".join(value) if isinstance(value, list) else str(value)


def _ago(days: Optional[int]) -> str:
    if days is None:
        return "date unknown"
    if days <= 0:
        return "today"
    return "1 day ago" if days == 1 else f"{days} days ago"


def _row(e: PoolEntry) -> str:
    genres = ", ".join(e.genres[:2]) or "genre unknown"
    length = f"~{e.minutes} min" if e.minutes else "length unknown"
    return f"{e.id} | {e.title} | {e.publisher} | {genres} | {length} | {_ago(e.days_since_last)}"


def _build_prompt(
    labels: Dict[str, Any],
    persona: Dict[str, Any],
    pool: Sequence[PoolEntry],
    n_picks: int,
    n_extras: int,
) -> str:
    lines = [
        "Pick podcasts for this listener from the candidate list below."
        if pool
        else f"Recommend {n_picks} podcasts for this listener.",
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
    if pool:
        lines += [
            "",
            "Candidate shows (real shows from Apple Podcasts charts, with real episode lengths):",
            "id | title | publisher | genres | typical episode | last new episode",
            *(_row(e) for e in pool),
        ]
    lines += ["", "Rules:"]
    if pool:
        lines += [
            f'- "picks": up to {n_picks} candidates that best fit this listener, best first. '
            "Use only ids from the list.",
            f'- "extras": optionally up to {n_extras} real podcasts that are not in the list and '
            "would fit clearly better than the remaining candidates. Use the exact official title "
            "and host or publisher as listed on Apple Podcasts. Never invent a show; leave "
            '"extras" empty if unsure.',
        ]
    else:
        lines += [
            '- Leave "picks" empty. Put your recommendations in "extras": real podcasts that '
            "exist on Apple Podcasts and Spotify today, with the exact official show title and "
            "the host or publisher as listed there. Never invent a show.",
            "- Prefer shows that are still publishing new episodes.",
        ]
    lines.append(f"- Every show must be in {labels['content_language']}.")
    if labels["podcast_duration"] != "No preference":
        lines.append(
            f"- Typical episodes should be {labels['podcast_duration']}; skip shows whose "
            "episodes usually run much shorter or longer."
        )
    lines += [
        "- At most one show per publisher.",
        "- If few shows fit, return fewer (even none) rather than explaining; never answer in "
        "prose.",
        '- "summary": one factual sentence about what the show is.',
        '- "reason": one sentence, shown to the listener, on why it fits; address the listener '
        'as "you" and refer to their answers. Don\'t state or describe episode length '
        '("long", "30-minute"): the app shows the real length from the catalog.',
        "",
        'Respond with JSON: {"picks": [{"id": ..., "summary": ..., "reason": ...}], '
        '"extras": [{"title": ..., "host": ..., "summary": ..., "reason": ...}]}',
    ]
    return "\n".join(lines)


def _request_kwargs(model: str) -> Dict[str, Any]:
    if _is_reasoning_model(model):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "podcast_choice", "strict": True, "schema": _CHOICE_SCHEMA},
            },
            "reasoning_effort": "low",
            "include_reasoning": False,
        }
    return {"response_format": {"type": "json_object"}}


def _usable(parsed: _Choice, pool_ids: set, n_picks: int, n_extras: int) -> Choice:
    picks: List[Pick] = []
    for pick in parsed.picks:
        if pick.id in pool_ids and pick.id not in {p.id for p in picks}:
            picks.append(pick)
    # Each extra costs an iTunes search, so never take more than allowed.
    extras = [e for e in parsed.extras if e.title.strip()][:n_extras]
    return Choice(picks=picks[:n_picks], extras=extras)


async def choose_podcasts(
    client: AsyncGroq,
    labels: Dict[str, Any],
    persona: Dict[str, Any],
    pool: Sequence[PoolEntry],
    model: str,
    n_picks: int = 8,
    n_extras: int = 2,
) -> Choice:
    """Have the curator pick from ``pool`` (and add up to ``n_extras`` of its own).

    With an empty pool (charts unavailable) it suggests up to ``n_picks`` shows
    freely, all of which the caller must verify. Raises LLMError on failure.
    """
    if not pool:
        n_extras = n_picks
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_prompt(labels, persona, pool, n_picks, n_extras)},
    ]
    pool_ids = {e.id for e in pool}
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
            if _is_json_generation_failure(exc):
                problem = exc  # sampling again usually yields valid JSON
                continue
            raise LLMError(f"Groq request failed (model={model}): {exc}") from exc
        try:
            parsed = _Choice.model_validate(json.loads(response.choices[0].message.content))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            problem = exc
            continue
        choice = _usable(parsed, pool_ids, n_picks, n_extras)
        if choice.picks or choice.extras:
            return choice
        problem = "no usable picks"
    raise LLMError(f"unusable output from {model}: {problem}")
