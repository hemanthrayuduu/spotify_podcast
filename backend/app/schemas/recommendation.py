"""Request/response schemas for the recommendation endpoint."""

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class UserPreferences(BaseModel):
    """Request body for /recommend.

    Fields split by how they are used downstream:
      * Segment matching (ML feature vector): age, music_genre,
        podcast_frequency, podcast_duration, podcast_format, podcast_content.
      * AI personalization only (Claude prompt): content_language, region,
        listening_mood, podcasts_enjoyed.
    """

    # --- Used for KMeans segment matching ---
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
    podcasts_enjoyed: str = ""

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
    def not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("must contain at least one item")
        return v


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
    segment_profile: Dict[str, Any]
    recommendations: List[Recommendation]
