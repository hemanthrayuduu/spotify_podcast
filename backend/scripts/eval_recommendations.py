"""Measure recommendation accuracy against the live Groq + iTunes services.

Run from backend/ (uses GROQ_API_KEY from backend/.env if set):

    python -m scripts.eval_recommendations

For a set of diverse listener profiles it reports, per profile and overall:

  * LLM verified   share of the LLM's proposals that are real shows. Picks from
                   the chart list are real by construction; extras (its own
                   suggestions) are verified in Apple's catalog. (Extras whose
                   search failed count as unverified; see catalog errors.)
  * pool / search  candidate shows the LLM chose from; how many came from
                   language search (thin non-English chart lists)
  * extras         how many of its own
                   suggestions it added and got verified
  * ai_top5        share of the returned shows the AI picked (vs. chart order)
                   — whether the AI step actually shapes the results
  * in length band share of recommendations whose real median episode
                   length falls inside the requested band
  * language ok    share whose feed-declared language matches (unknown feeds
                   count as not ok)
  * topic match    share whose Apple genres match a requested topic
  * publishers     distinct publishers among the recommendations
  * catalog errors iTunes calls that failed (rate limiting); rows with errors
                   understate verification
  * latency        end-to-end pipeline time (includes waiting on the call budget)

Without GROQ_API_KEY it evaluates catalog-only mode.
"""

import asyncio
import statistics
import time
from typing import Any, Dict, List

import httpx

from app.core.config import settings
from app.core.vocabulary import option
from app.main import _init_llm_client
from app.ml.segmenter import Segmenter
from app.schemas.recommendation import UserPreferences
from app.services.catalog import ITunesCatalog, RateBudget
from app.services.recommender import Recommender


def _p(**kw: Any) -> Dict[str, Any]:
    base = {
        "age": "20_34", "music_genre": ["pop"], "podcast_frequency": "weekly",
        "podcast_duration": "medium", "podcast_format": "interview", "podcast_content": ["comedy"],
        "content_language": "english", "region": "global", "listening_mood": "entertained",
        "podcasts_enjoyed": "",
    }
    return UserPreferences(**{**base, **kw}).model_dump()


PROFILES: Dict[str, Dict[str, Any]] = {
    "true crime, long, UK": _p(podcast_content=["true_crime"], podcast_duration="extra_long",
                               podcast_format="storytelling", region="europe",
                               podcasts_enjoyed="Serial"),
    "history nerd": _p(podcast_content=["history"], podcast_duration="extra_long",
                       listening_mood="deep_thinking", podcasts_enjoyed="Hardcore History"),
    "teen science, short": _p(age="12_19", podcast_content=["science", "technology"],
                              podcast_duration="short", podcast_format="educational",
                              listening_mood="learn"),
    "kids & family": _p(age="under_12", podcast_content=["kids_family", "fiction"],
                        podcast_duration="short", podcast_format="storytelling"),
    "business, Spanish": _p(age="35_59", podcast_content=["business"], content_language="spanish",
                            region="south_america", listening_mood="informed"),
    "comedy, Korean": _p(podcast_content=["comedy", "tv_film"], content_language="korean",
                         region="asia", podcast_duration="any", podcast_format="conversational"),
    "self-help, Hindi": _p(podcast_content=["self_help", "health_fitness"],
                           content_language="hindi", region="asia", podcast_duration="long"),
    "daily news": _p(age="35_59", podcast_content=["news_politics"], podcast_frequency="daily",
                     podcast_duration="short", podcast_format="news", listening_mood="informed"),
    "tech deep dives": _p(podcast_content=["technology", "business"], podcast_duration="extra_long",
                          podcasts_enjoyed="Acquired, Lex Fridman Podcast"),
    "wellness, 60+": _p(age="60_plus", podcast_content=["lifestyle", "spirituality"],
                        podcast_duration="medium", listening_mood="relax"),
    "sports banter": _p(podcast_content=["sports", "comedy"], podcast_format="conversational",
                        podcast_duration="long", podcasts_enjoyed="Pardon My Take"),
    "French culture": _p(podcast_content=["arts_culture", "history"], content_language="french",
                         region="europe", podcast_duration="long"),
    "English, South America": _p(podcast_content=["sports", "news_politics"], region="south_america",
                                 podcast_duration="medium"),
    "Telugu movies": _p(podcast_content=["tv_film", "comedy"], content_language="telugu",
                        region="asia", podcast_duration="any", podcast_format="conversational"),
}


def _share(flags: List[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _metrics(prefs: Dict[str, Any], response: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    recs, meta = response["recommendations"], response["meta"]
    lo, hi = option("podcast_duration", prefs["podcast_duration"]).minutes
    want_lang = option("content_language", prefs["content_language"]).iso_language
    in_band = [
        r["typical_episode_minutes"] is not None
        and lo <= r["typical_episode_minutes"] <= (hi if hi is not None else 10**6)
        for r in recs
    ]
    proposed = meta["llm_candidates"]
    picks = proposed - meta["llm_extras"]
    return {
        "mode": meta["mode"],
        "recs": len(recs),
        "pool": meta["pool_size"],
        "search": meta["search_candidates"],
        "extras": f"{meta['llm_verified'] - picks}/{meta['llm_extras']}",
        "llm_verified": meta["llm_verified"] / proposed if proposed else None,
        "ai_top5": _share([r["source"] == "ai" for r in recs]) if meta["mode"] == "ai" else None,
        "in_band": _share(in_band),
        "language_ok": _share([r["language"] == want_lang for r in recs]),
        "topic_match": _share([r["match"]["breakdown"]["topic"] == 1.0 for r in recs]),
        "publishers": len({r["publisher"] for r in recs}),
        "catalog_errors": meta["catalog_errors"],
        "seconds": seconds,
    }


def _fmt(v: Any, col: str = "") -> str:
    if v is None:
        return "—"
    if col == "seconds":
        return f"{v:.1f}s"
    if isinstance(v, float):
        return f"{v:.0%}" if col not in ("publishers", "catalog_errors") else f"{v:.1f}"
    return str(v)


async def main() -> None:
    llm = _init_llm_client()
    async with httpx.AsyncClient(timeout=settings.itunes_timeout_s) as http:
        # Same iTunes call budget as production, but wait for tokens instead of
        # failing fast, so a dozen back-to-back profiles don't get rate limited.
        catalog = ITunesCatalog(
            http, budget=RateBudget(settings.itunes_calls_per_minute, max_wait=120)
        )
        recommender = Recommender(
            Segmenter.load(settings.model_dir), catalog, llm, settings.groq_model, cache_ttl=0
        )
        rows = []
        for name, prefs in PROFILES.items():
            started = time.perf_counter()
            response = await recommender.recommend(prefs)
            rows.append((name, _metrics(prefs, response, time.perf_counter() - started)))
            titles = ", ".join(r["title"] for r in response["recommendations"])
            print(f"- {name}: {titles}", flush=True)

    cols = [
        "mode", "recs", "pool", "search", "extras", "llm_verified", "ai_top5", "in_band", "language_ok",
        "topic_match", "publishers", "catalog_errors", "seconds",
    ]
    print("\n| profile | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, m in rows:
        print(f"| {name} | " + " | ".join(_fmt(m[c], c) for c in cols) + " |")

    def mean(col: str) -> Any:
        values = [m[col] for _, m in rows if isinstance(m[col], (int, float))]
        return statistics.mean(values) if values else None

    numeric = cols[5:]  # skip mode, recs, pool, search, extras
    print("| **mean** | | | | | | " + " | ".join(_fmt(mean(c), c) for c in numeric) + " |")
    latencies = sorted(m["seconds"] for _, m in rows)
    print(f"\nlatency p50 {statistics.median(latencies):.1f}s, max {latencies[-1]:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
