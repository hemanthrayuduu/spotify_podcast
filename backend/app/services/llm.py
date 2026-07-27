"""Groq-backed podcast recommendation generation.

Single source of truth for turning a user's preferences + their KMeans
segment profile into 5 podcast recommendations. Uses Groq (an open-source
Llama model) in JSON mode so the model returns a structured payload instead
of free-text we have to scrape.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from groq import AsyncGroq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a podcast recommendation expert. You always respond with a single "
    "valid JSON object and nothing else."
)

# The exact fields every recommendation must contain (mirrors the API's
# Recommendation schema, minus `link` which the server fills in).
_REQUIRED_FIELDS = (
    "name",
    "creator",
    "description",
    "format",
    "duration",
    "language",
    "region",
    "reason",
)


def _build_prompt(user_prefs: Dict[str, Any], segment_profile: Dict[str, Any]) -> str:
    music_genres = ", ".join(user_prefs.get("music_genre") or ["Various"])
    pod_content_topics = ", ".join(user_prefs.get("podcast_content") or ["Various"])

    top_music_genre = next(iter(segment_profile.get("fav_music_genre", {})), "Various")
    top_pod_genre = next(iter(segment_profile.get("fav_pod_genre", {})), "Various")
    segment_age = segment_profile.get("age_numeric", {}).get("mean", 30)

    prompt = f"""Based on the user profile and preferences below, suggest 5 real, high-quality podcasts they would enjoy. Be specific and personalized, not generic.

USER INFORMATION:
- Age group: {user_prefs.get("age", "25-34")}
- Favorite music genres: {music_genres}
- Podcast listening frequency: {user_prefs.get("podcast_frequency", "Weekly")}
- Preferred podcast duration: {user_prefs.get("podcast_duration", "Medium (30-60 min)")}
- Preferred podcast format: {user_prefs.get("podcast_format", "Interview")}
- Podcast content interests: {pod_content_topics}
- Preferred language: {user_prefs.get("content_language", "English")}
- Region of interest: {user_prefs.get("region", "Global")}
- Current listening mood: {user_prefs.get("listening_mood", "")}
"""
    if user_prefs.get("podcasts_enjoyed"):
        prompt += f"- Podcasts already enjoyed: {user_prefs['podcasts_enjoyed']}\n"

    prompt += """
Respond with a JSON object of this exact shape:
{
  "recommendations": [
    {
      "name": "Podcast name",
      "creator": "Creator or host",
      "description": "Brief, engaging 1-2 sentence description",
      "format": "Format type (Interview, Narrative, Educational, etc.)",
      "duration": "Typical episode length",
      "language": "Main language",
      "region": "Content region focus",
      "reason": "One-sentence personalized reason this matches the user"
    }
  ]
}
The "recommendations" array must contain exactly 5 items.
"""
    prompt += f"\n(Segment context — top music genre: {top_music_genre}, top podcast genre: {top_pod_genre}, average age: {segment_age}.)\n"
    return prompt


def _normalize(recs: List[Dict[str, Any]], user_prefs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ensure every recommendation has all required fields + a link.

    The model can omit or mistype fields; we backfill from the user's
    preferences so the response always validates against the API schema.
    """
    defaults = {
        "name": "Unknown Podcast",
        "creator": "Unknown Creator",
        "description": "",
        "format": user_prefs.get("podcast_format", "Interview"),
        "duration": user_prefs.get("podcast_duration", "Medium (30-60 min)"),
        "language": user_prefs.get("content_language", "English"),
        "region": user_prefs.get("region", "Global"),
        "reason": "Matches your listening preferences.",
    }
    normalized = []
    for rec in recs[:5]:
        if not isinstance(rec, dict):
            continue
        item = {f: str(rec.get(f) or defaults[f]) for f in _REQUIRED_FIELDS}
        name = item["name"].replace(" ", "+")
        creator = item["creator"].replace(" ", "+")
        item["link"] = f"https://www.google.com/search?q={name}+{creator}+podcast"
        normalized.append(item)
    return normalized


async def generate_podcast_recommendations(
    client: Optional[AsyncGroq],
    user_preferences: Dict[str, Any],
    segment_profile: Dict[str, Any],
    model: str,
) -> List[Dict[str, Any]]:
    """Return 5 podcast recommendations, falling back to a static list on failure."""
    if client is None:
        logger.warning("Groq client unavailable; returning fallback recommendations")
        return get_fallback_recommendations(user_preferences)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_prompt(user_preferences, segment_profile)},
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        recommendations = (json.loads(content) or {}).get("recommendations", [])
        if not recommendations:
            raise ValueError("no recommendations in model response")
        return _normalize(recommendations, user_preferences)
    except Exception as exc:  # noqa: BLE001 - any failure degrades to the static fallback
        logger.error(f"Groq recommendation error: {exc}")
        return get_fallback_recommendations(user_preferences)


def get_fallback_recommendations(user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Static recommendations used when the LLM is unavailable or errors."""
    pod_format = user_preferences.get("podcast_format", "Interview")
    pod_duration = user_preferences.get("podcast_duration", "Medium (30-60 min)")
    pod_content_topics = ", ".join(user_preferences.get("podcast_content") or ["Various"])
    language = user_preferences.get("content_language", "English")
    region = user_preferences.get("region", "Global")
    age = user_preferences.get("age", "25-34")

    return _normalize(
        [
            {
                "name": "The Daily",
                "creator": "The New York Times",
                "description": "The biggest stories of our time, told by the best journalists in the world.",
                "format": pod_format,
                "duration": pod_duration,
                "language": language,
                "region": region,
                "reason": f"Popular podcast that matches your interest in {pod_content_topics}.",
            },
            {
                "name": "TED Talks Daily",
                "creator": "TED",
                "description": "Thought-provoking ideas on every subject imaginable, every weekday.",
                "format": "Educational",
                "duration": "Short (< 30 min)",
                "language": language,
                "region": "Global",
                "reason": "Educational content that aligns with your listening preferences.",
            },
            {
                "name": "Freakonomics Radio",
                "creator": "Stephen J. Dubner",
                "description": "Discover the hidden side of everything with Stephen J. Dubner.",
                "format": "Interview",
                "duration": "Medium (30-60 min)",
                "language": language,
                "region": "Global",
                "reason": "Engaging economic and social topics presented in an accessible way.",
            },
            {
                "name": "SmartLess",
                "creator": "Jason Bateman, Sean Hayes, Will Arnett",
                "description": "Thoughtful dialogue and organic hilarity connecting people from all walks of life.",
                "format": "Interview",
                "duration": "Medium (30-60 min)",
                "language": language,
                "region": "Global",
                "reason": f"Popular {pod_format}-style podcast that many {age} listeners enjoy.",
            },
            {
                "name": "Stuff You Should Know",
                "creator": "iHeartRadio",
                "description": "Josh and Chuck make dense subjects easy to digest across a huge variety of topics.",
                "format": "Educational",
                "duration": pod_duration,
                "language": language,
                "region": "Global",
                "reason": f"Informative content about {pod_content_topics} in your preferred format.",
            },
        ],
        user_preferences,
    )
