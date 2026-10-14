"""Request/response schemas for the recommendation endpoint."""

import json
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.vocabulary import allowed_values


def _check_allowed(field: str, value: str) -> str:
    allowed = allowed_values(field)
    if value not in allowed:
        raise ValueError(f"{value!r} is not a valid option; must be one of: {', '.join(allowed)}")
    return value


class UserPreferences(BaseModel):
    """Request body for /recommend. Values are option values from GET /options.

    Fields split by how they are used downstream:
      * Segment matching (persona): age, music_genre, podcast_frequency,
        podcast_duration, podcast_format, podcast_content.
      * AI personalization only (LLM prompt): content_language, region,
        listening_mood, podcasts_enjoyed.
    """

    # --- Used for segment matching ---
    age: str
    music_genre: List[str] = Field(..., description="At least one favorite music genre")
    podcast_frequency: str
    podcast_duration: str
    podcast_format: str
    podcast_content: List[str] = Field(..., description="At least one podcast content interest")

    # --- Used for AI personalization only ---
    content_language: str
    region: str
    listening_mood: str
    podcasts_enjoyed: str = Field("", max_length=500)

    @field_validator("music_genre", "podcast_content", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> Any:
        """Accept a JSON string or comma-separated string as a list."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            if "," in v:
                return [item.strip() for item in v.split(",") if item.strip()]
            return [v] if v else []
        return v

    @field_validator("music_genre", "podcast_content")
    @classmethod
    def valid_multi(cls, v: List[str], info) -> List[str]:
        if not v:
            raise ValueError("must contain at least one item")
        # Dedupe, keeping the user's order.
        return [_check_allowed(info.field_name, item) for item in dict.fromkeys(v)]

    @field_validator(
        "age", "podcast_frequency", "podcast_duration", "podcast_format",
        "content_language", "region", "listening_mood",
    )
    @classmethod
    def valid_single(cls, v: str, info) -> str:
        return _check_allowed(info.field_name, v)


class PersonaTrait(BaseModel):
    feature: str
    label: str
    share: float
    lift: float


class Persona(BaseModel):
    id: int
    name: str
    description: str
    traits: List[PersonaTrait]
    share: float


class MatchBreakdown(BaseModel):
    """Each component is in [0, 1]; see app.services.ranking.WEIGHTS."""

    topic: float
    duration: float
    endorsement: float
    freshness: float


class Match(BaseModel):
    score: int = Field(..., ge=0, le=100)
    breakdown: MatchBreakdown


class Recommendation(BaseModel):
    """A show verified in the Apple Podcasts catalog.

    Every field comes from the catalog except ``summary`` and ``reason``,
    which are AI-written when ``source == "ai"``.
    """

    id: int = Field(..., description="Apple Podcasts collection id")
    title: str
    publisher: str
    artwork_url: Optional[str]
    genres: List[str]
    apple_url: str
    spotify_url: str
    feed_url: Optional[str]
    episode_count: Optional[int]
    typical_episode_minutes: Optional[int] = Field(..., description="Median of recent episodes")
    last_episode_date: Optional[str]
    explicit: bool
    language: Optional[str] = Field(..., description="ISO 639-1 code declared by the RSS feed")
    summary: Optional[str]
    reason: str
    source: Literal["ai", "catalog"]
    match: Match


class RecommendMeta(BaseModel):
    mode: Literal["ai", "catalog_only"]
    llm_model: Optional[str]
    storefront: str
    llm_candidates: int = Field(..., description="Shows the LLM proposed")
    candidates_considered: int
    dropped_unverified: int = Field(..., description="LLM candidates not found in the catalog")
    dropped_language: int = Field(..., description="Shows whose feed is in another language")
    segment_model_version: str
    cached: bool
    latency_ms: int


class RecommendationResponse(BaseModel):
    persona: Persona
    recommendations: List[Recommendation]
    meta: RecommendMeta
