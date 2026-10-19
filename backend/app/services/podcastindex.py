"""Podcast Index client: trending shows filtered by language and category.

Apple's charts are per storefront, not per language, so a Telugu listener in
the Indian store sees mostly English and Hindi shows. Podcast Index filters
trending shows by the language their feed declares, which fills that gap.
Only shows that are also on Apple Podcasts are used (callers look them up by
``itunes_id``), so every recommendation still carries Apple metadata.

API docs: https://podcastindex-org.github.io/docs-api/ (free key + secret).
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from app.services.catalog import normalize_language

logger = logging.getLogger(__name__)

BASE_URL = "https://api.podcastindex.org/api/1.0"
USER_AGENT = "spotify-podcast-recommender/2.0"
_MAX_CACHE_ENTRIES = 512


class PodcastIndexError(Exception):
    """Podcast Index refused the request or returned something unusable."""


@dataclass(frozen=True)
class TrendingShow:
    feed_id: int
    itunes_id: Optional[int]
    title: str
    language: Optional[str]


class PodcastIndex:
    def __init__(
        self,
        http: httpx.AsyncClient,
        key: str,
        secret: str,
        cache_ttl: float = 6 * 3600,
        clock: Callable[[], float] = time.time,
    ):
        self._http = http
        self._key = key
        self._secret = secret
        self._cache_ttl = cache_ttl
        self._clock = clock
        self._cache: Dict[Tuple, Tuple[float, List[TrendingShow]]] = {}

    def _headers(self) -> Dict[str, str]:
        """Auth per the API: Authorization = sha1(key + secret + unix time)."""
        now = str(int(self._clock()))
        digest = hashlib.sha1(f"{self._key}{self._secret}{now}".encode()).hexdigest()
        return {
            "X-Auth-Key": self._key,
            "X-Auth-Date": now,
            "Authorization": digest,
            "User-Agent": USER_AGENT,
        }

    async def trending(
        self, language: str, categories: Sequence[int], max_results: int = 15
    ) -> List[TrendingShow]:
        """Trending shows whose feed declares ``language`` (ISO 639-1), in any of ``categories``."""
        params: Dict[str, Any] = {
            "lang": language,
            "cat": ",".join(str(c) for c in categories),
            "max": max_results,
        }
        key = tuple(sorted(params.items()))
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        try:
            response = await self._http.get(
                f"{BASE_URL}/podcasts/trending", params=params, headers=self._headers(), timeout=8.0
            )
        except httpx.HTTPError as exc:
            raise PodcastIndexError(f"trending request failed: {exc}") from exc
        if response.status_code != 200:
            raise PodcastIndexError(f"trending returned HTTP {response.status_code}")
        try:
            feeds = response.json().get("feeds") or []
        except (ValueError, AttributeError):
            raise PodcastIndexError("trending returned unexpected content") from None

        shows = []
        for f in feeds if isinstance(feeds, list) else []:
            if not isinstance(f, dict) or not f.get("id"):
                continue
            itunes_id = f.get("itunesId")
            shows.append(
                TrendingShow(
                    feed_id=int(f["id"]),
                    itunes_id=int(itunes_id) if isinstance(itunes_id, int) and itunes_id > 0 else None,
                    title=str(f.get("title") or ""),
                    language=normalize_language(str(f.get("language") or "")) if f.get("language") else None,
                )
            )
        if len(self._cache) >= _MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (time.monotonic() + self._cache_ttl, shows)
        return shows
