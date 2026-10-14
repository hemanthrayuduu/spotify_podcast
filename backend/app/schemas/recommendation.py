"""Request/response schemas for the recommendation endpoint."""

import json
from typing import Any, List

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


class Recommendation(BaseModel):
    name: str
    creator: str
    description: str
    format: str
    duration: str
    language: str
    region: str
    reason: str
    link: str


class RecommendationResponse(BaseModel):
    persona: Persona
    recommendations: List[Recommendation]
