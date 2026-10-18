"""Single source of truth for every form field and option.

Each option has a stable ``value`` (what the client sends) and a ``label``
(what the user sees). Options that correspond to a survey answer carry the
cleaned survey ``code`` they map to; that code is what the segmentation model
was trained on, so training and serving share one vocabulary. Options with
``code=None`` are still used by the LLM and the catalog, but not for
segmentation.

The frontend renders its form from ``options_payload()`` (GET /options), so
the options cannot drift from what the backend understands.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Option:
    value: str
    label: str
    # Cleaned survey category this option maps to; None = not a segment feature.
    code: Optional[str] = None
    # podcast_content: Apple Podcasts chart genre id and the genre names that
    # count as a topic match (ids verified against live iTunes lookups).
    chart_genre_id: Optional[int] = None
    apple_genres: Tuple[str, ...] = ()
    # podcast_duration: preferred episode length in minutes (upper None = open).
    minutes: Optional[Tuple[int, Optional[int]]] = None
    # content_language / region: Apple storefront (ISO country) to search in.
    storefront: Optional[str] = None
    # content_language: ISO 639-1 code, checked against each feed's <language>.
    iso_language: Optional[str] = None


@dataclass(frozen=True)
class Field:
    name: str
    label: str
    options: Tuple[Option, ...]
    multi: bool = False
    # Ordered survey categories; each becomes one feature column.
    segment_codes: Tuple[str, ...] = ()
    # Encode the code's position as one scaled number instead of one-hot.
    ordinal: bool = False


FIELDS: Dict[str, Field] = {
    f.name: f
    for f in (
        Field(
            name="age",
            label="Age group",
            ordinal=True,
            segment_codes=("6-12", "12-20", "20-35", "35-60", "60+"),
            options=(
                Option("under_12", "Under 12", code="6-12"),
                Option("12_19", "12–19", code="12-20"),
                Option("20_34", "20–34", code="20-35"),
                Option("35_59", "35–59", code="35-60"),
                Option("60_plus", "60+", code="60+"),
            ),
        ),
        Field(
            name="music_genre",
            label="Music you enjoy",
            multi=True,
            segment_codes=(
                "pop", "rock", "rap", "melody", "classical", "jazz", "electronic", "kpop", "oldies",
            ),
            options=(
                Option("pop", "Pop", code="pop"),
                Option("rock", "Rock", code="rock"),
                Option("hip_hop", "Hip-Hop / Rap", code="rap"),
                Option("melodic", "Melodic / Soft", code="melody"),
                Option("classical", "Classical", code="classical"),
                Option("jazz", "Jazz", code="jazz"),
                Option("electronic", "Electronic / Dance", code="electronic"),
                Option("kpop", "K-pop", code="kpop"),
                Option("oldies", "Oldies / Retro", code="oldies"),
                Option("rnb", "R&B / Soul"),
                Option("country", "Country"),
                Option("metal", "Metal"),
                Option("folk", "Folk / Acoustic"),
                Option("indie", "Indie"),
                Option("latin", "Latin"),
            ),
        ),
        Field(
            name="podcast_content",
            label="Topics that interest you",
            multi=True,
            segment_codes=(
                "comedy", "true_crime", "sports", "technology", "lifestyle", "health_fitness",
                "food", "business", "news_politics", "self_help", "spirituality", "education",
                "fiction",
            ),
            options=(
                Option("comedy", "Comedy", code="comedy",
                       chart_genre_id=1303, apple_genres=("Comedy", "Comedy Interviews", "Improv", "Stand-Up")),
                Option("true_crime", "True Crime & Mystery", code="true_crime",
                       chart_genre_id=1488, apple_genres=("True Crime",)),
                Option("sports", "Sports", code="sports",
                       chart_genre_id=1545, apple_genres=("Sports",)),
                Option("technology", "Technology", code="technology",
                       chart_genre_id=1318, apple_genres=("Technology",)),
                Option("lifestyle", "Lifestyle & Wellness", code="lifestyle",
                       chart_genre_id=1517,
                       apple_genres=("Mental Health", "Alternative Health", "Leisure", "Relationships")),
                Option("health_fitness", "Health & Fitness", code="health_fitness",
                       chart_genre_id=1512, apple_genres=("Health & Fitness", "Fitness", "Nutrition")),
                Option("food", "Food & Cooking", code="food",
                       chart_genre_id=1306, apple_genres=("Food",)),
                Option("business", "Business & Finance", code="business",
                       chart_genre_id=1321, apple_genres=("Business", "Investing", "Entrepreneurship")),
                Option("news_politics", "News & Politics", code="news_politics",
                       chart_genre_id=1489, apple_genres=("News", "Politics", "Daily News")),
                Option("self_help", "Self-Improvement", code="self_help",
                       chart_genre_id=1500, apple_genres=("Self-Improvement",)),
                Option("spirituality", "Religion & Spirituality", code="spirituality",
                       chart_genre_id=1314, apple_genres=("Religion & Spirituality", "Spirituality")),
                Option("education", "Education & Learning", code="education",
                       chart_genre_id=1304, apple_genres=("Education", "Courses", "How To")),
                Option("fiction", "Fiction & Stories", code="fiction",
                       chart_genre_id=1483, apple_genres=("Fiction", "Drama", "Comedy Fiction")),
                Option("history", "History",
                       chart_genre_id=1487, apple_genres=("History",)),
                Option("science", "Science",
                       chart_genre_id=1533, apple_genres=("Science",)),
                Option("arts_culture", "Arts & Culture",
                       chart_genre_id=1301, apple_genres=("Arts", "Society & Culture", "Books")),
                Option("tv_film", "TV & Film",
                       chart_genre_id=1309, apple_genres=("TV & Film", "Film Reviews", "TV Reviews")),
                Option("music_talk", "Music",
                       chart_genre_id=1310, apple_genres=("Music", "Music Commentary", "Music History")),
                Option("kids_family", "Kids & Family",
                       chart_genre_id=1305, apple_genres=("Kids & Family",)),
            ),
        ),
        Field(
            name="podcast_frequency",
            label="How often do you listen?",
            segment_codes=("daily", "several_week", "weekly", "rarely", "never"),
            options=(
                Option("daily", "Daily", code="daily"),
                Option("several_week", "Several times a week", code="several_week"),
                Option("weekly", "Once a week", code="weekly"),
                Option("rarely", "Rarely", code="rarely"),
                Option("never", "I'm new to podcasts", code="never"),
            ),
        ),
        Field(
            name="podcast_format",
            label="Preferred format",
            segment_codes=("storytelling", "interview", "discussion", "conversational", "educational"),
            options=(
                Option("storytelling", "Storytelling / Narrative", code="storytelling"),
                Option("interview", "Interview", code="interview"),
                Option("discussion", "Panel / Discussion", code="discussion"),
                Option("conversational", "Conversational / Banter", code="conversational"),
                Option("educational", "Educational / Explainer", code="educational"),
                Option("news", "News / Daily briefing"),
                Option("any", "No preference"),
            ),
        ),
        Field(
            name="podcast_duration",
            label="Preferred episode length",
            segment_codes=("short", "medium", "long"),
            options=(
                Option("short", "Short (under 15 min)", code="short", minutes=(0, 15)),
                Option("medium", "Medium (15–30 min)", code="medium", minutes=(15, 30)),
                Option("long", "Long (30–60 min)", code="long", minutes=(30, 60)),
                Option("extra_long", "Extra long (60+ min)", code="long", minutes=(60, None)),
                Option("any", "No preference", minutes=(0, None)),
            ),
        ),
        Field(
            name="content_language",
            label="Preferred language",
            options=(
                # English searches the storefront chosen by region instead.
                Option("english", "English", iso_language="en"),
                Option("spanish", "Spanish", storefront="es", iso_language="es"),
                Option("french", "French", storefront="fr", iso_language="fr"),
                Option("german", "German", storefront="de", iso_language="de"),
                Option("portuguese", "Portuguese", storefront="br", iso_language="pt"),
                Option("italian", "Italian", storefront="it", iso_language="it"),
                Option("hindi", "Hindi", storefront="in", iso_language="hi"),
                Option("telugu", "Telugu", storefront="in", iso_language="te"),
                Option("tamil", "Tamil", storefront="in", iso_language="ta"),
                Option("japanese", "Japanese", storefront="jp", iso_language="ja"),
                Option("korean", "Korean", storefront="kr", iso_language="ko"),
                Option("mandarin", "Mandarin", storefront="tw", iso_language="zh"),
                Option("arabic", "Arabic", storefront="sa", iso_language="ar"),
            ),
        ),
        Field(
            name="region",
            label="Content region",
            options=(
                Option("global", "Global", storefront="us"),
                Option("north_america", "North America", storefront="us"),
                Option("europe", "Europe", storefront="gb"),
                Option("asia", "Asia", storefront="in"),
                # Region storefronts are only used for English; Brazil's charts are Portuguese.
                Option("south_america", "South America", storefront="us"),
                Option("africa", "Africa", storefront="za"),
                Option("oceania", "Australia / Oceania", storefront="au"),
            ),
        ),
        Field(
            name="listening_mood",
            label="What are you in the mood for?",
            options=(
                Option("learn", "Learn something new"),
                Option("entertained", "Be entertained"),
                Option("informed", "Stay informed"),
                Option("relax", "Relax & unwind"),
                Option("motivated", "Get motivated"),
                Option("deep_thinking", "Deep thinking"),
                Option("laugh", "Laugh"),
            ),
        ),
    )
}

SEGMENT_FIELDS: Tuple[str, ...] = tuple(n for n, f in FIELDS.items() if f.segment_codes)

_OPTIONS: Dict[str, Dict[str, Option]] = {
    name: {o.value: o for o in field.options} for name, field in FIELDS.items()
}


def option(field: str, value: str) -> Option:
    """Look up an option; raises KeyError for an unknown field or value."""
    return _OPTIONS[field][value]


def allowed_values(field: str) -> List[str]:
    return list(_OPTIONS[field])


def to_labels(prefs: Dict[str, Any]) -> Dict[str, Any]:
    """Replace option values with their human labels (for prompts and display)."""
    labeled: Dict[str, Any] = {}
    for name, value in prefs.items():
        if name not in _OPTIONS:
            labeled[name] = value
        elif isinstance(value, list):
            labeled[name] = [option(name, v).label for v in value]
        else:
            labeled[name] = option(name, value).label
    return labeled


def options_payload() -> Dict[str, Any]:
    return {
        "fields": [
            {
                "name": f.name,
                "label": f.label,
                "multi": f.multi,
                "options": [{"value": o.value, "label": o.label} for o in f.options],
            }
            for f in FIELDS.values()
        ]
    }
