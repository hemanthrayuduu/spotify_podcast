"""Claude-backed podcast recommendation generation.

Single source of truth for turning a user's preferences + their KMeans
segment profile into 5 podcast recommendations. Uses the async Anthropic
client and forced tool use so the model returns a guaranteed-shape payload
instead of free-text JSON we have to scrape with a regex.
"""

import logging
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

# Tool the model is forced to call. Its input schema *is* the response
# contract, which removes the old regex/JSON-parsing fallback entirely.
SUBMIT_RECOMMENDATIONS_TOOL = {
    "name": "submit_recommendations",
    "description": "Submit exactly 5 podcast recommendations tailored to the user.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "description": "Exactly 5 real, high-quality podcasts.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Podcast name"},
                        "creator": {"type": "string", "description": "Creator or host"},
                        "description": {
                            "type": "string",
                            "description": "Brief, engaging description (1-2 sentences)",
                        },
                        "format": {
                            "type": "string",
                            "description": "Format type (Interview, Narrative, Educational, etc.)",
                        },
                        "duration": {"type": "string", "description": "Typical episode length"},
                        "language": {"type": "string", "description": "Main language"},
                        "region": {"type": "string", "description": "Content region focus"},
                        "reason": {
                            "type": "string",
                            "description": "One-sentence personalized reason this matches the user",
                        },
                    },
                    "required": [
                        "name",
                        "creator",
                        "description",
                        "format",
                        "duration",
                        "language",
                        "region",
                        "reason",
                    ],
                },
            }
        },
        "required": ["recommendations"],
    },
}


def _build_prompt(user_prefs: Dict[str, Any], segment_profile: Dict[str, Any]) -> str:
    """Render the user profile + segment into the model prompt."""
    music_genres = ", ".join(user_prefs.get("music_genre") or ["Various"])
    pod_content_topics = ", ".join(user_prefs.get("podcast_content") or ["Various"])

    top_music_genre = next(iter(segment_profile.get("fav_music_genre", {})), "Various")
    top_pod_genre = next(iter(segment_profile.get("fav_pod_genre", {})), "Various")
    segment_age = segment_profile.get("age_numeric", {}).get("mean", 30)

    prompt = f"""You are a podcast recommendation expert. Based on the user profile and preferences, suggest 5 podcasts they would enjoy.

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

    prompt += f"""
LISTENER SEGMENT PROFILE:
- Top music genre in segment: {top_music_genre}
- Top podcast genre in segment: {top_pod_genre}
- Age demographics: {segment_age} (average)

Focus on real, high-quality podcasts that genuinely match the user's interests. Be specific, not generic. Call the submit_recommendations tool with exactly 5 recommendations.
"""
    return prompt


def _add_links(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach a Google search link to each recommendation."""
    for rec in recommendations:
        name = rec.get("name", "").replace(" ", "+")
        creator = rec.get("creator", "").replace(" ", "+")
        rec["link"] = f"https://www.google.com/search?q={name}+{creator}+podcast"
    return recommendations


async def generate_podcast_recommendations(
    client: Optional[anthropic.AsyncAnthropic],
    user_preferences: Dict[str, Any],
    segment_profile: Dict[str, Any],
    model: str,
) -> List[Dict[str, Any]]:
    """Return 5 podcast recommendations, falling back to a static list on failure."""
    if client is None:
        logger.warning("Anthropic client unavailable; returning fallback recommendations")
        return get_fallback_recommendations(user_preferences)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1500,
            tools=[SUBMIT_RECOMMENDATIONS_TOOL],
            tool_choice={"type": "tool", "name": "submit_recommendations"},
            messages=[{"role": "user", "content": _build_prompt(user_preferences, segment_profile)}],
        )
    except anthropic.APIError as exc:
        logger.error(f"Anthropic API error generating recommendations: {exc}")
        return get_fallback_recommendations(user_preferences)

    for block in response.content:
        if block.type == "tool_use":
            recommendations = block.input.get("recommendations", [])
            if recommendations:
                return _add_links(recommendations)

    logger.error("Claude response contained no tool_use recommendations; using fallback")
    return get_fallback_recommendations(user_preferences)


def get_fallback_recommendations(user_preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Static recommendations used when Claude is unavailable or errors."""
    pod_format = user_preferences.get("podcast_format", "Interview")
    pod_duration = user_preferences.get("podcast_duration", "Medium (30-60 min)")
    pod_content_topics = ", ".join(user_preferences.get("podcast_content") or ["Various"])
    language = user_preferences.get("content_language", "English")
    region = user_preferences.get("region", "Global")
    age = user_preferences.get("age", "25-34")

    return _add_links(
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
        ]
    )
