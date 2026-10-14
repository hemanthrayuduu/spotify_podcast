"""Apple iTunes Search API client and the candidate → real-show resolver.

Every recommendation is grounded here: a show only reaches the user if it
exists in Apple's podcast catalog, and its artwork, links, genres, episode
lengths and last-episode date come from the catalog rather than the LLM.

iTunes allows roughly 20 calls per minute per IP, so responses are cached,
concurrency is capped, and shows + episodes are fetched in batched lookups.
Every call asks for English genre names (``lang=en_us``) so topic matching
works in every storefront.
"""

import asyncio
import re
import statistics
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from rapidfuzz import fuzz

BASE_URL = "https://itunes.apple.com"
_RETRYABLE = {403, 429, 500, 502, 503, 504}  # iTunes answers 403 when rate limiting
_MAX_CACHE_ENTRIES = 2048
_LOOKUP_CHUNK = 6  # ids per lookup; big batches are slow (~4s for 10 shows)
_FEED_BYTES = 32 * 1024  # <language> sits in the channel header, before any items
# Some hosts (anchor.fm, audioboom) wrap the value: <language><![CDATA[es]]></language>
_LANGUAGE_TAG = re.compile(rb"<language>\s*(?:<!\[CDATA\[)?\s*([^<\]\s]+)", re.IGNORECASE)
_ISO_639_2 = {
    "eng": "en", "spa": "es", "fra": "fr", "fre": "fr", "deu": "de", "ger": "de", "por": "pt",
    "ita": "it", "hin": "hi", "tel": "te", "tam": "ta", "jpn": "ja", "kor": "ko", "zho": "zh",
    "chi": "zh", "ara": "ar",
}
_LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "español": "es", "french": "fr", "français": "fr",
    "german": "de", "deutsch": "de", "portuguese": "pt", "português": "pt", "italian": "it",
    "italiano": "it", "hindi": "hi", "japanese": "ja", "korean": "ko", "chinese": "zh", "arabic": "ar",
}


def normalize_language(raw: str) -> Optional[str]:
    """``en-US`` / ``es_ES`` / ``spa`` / ``English`` → ISO 639-1, or None."""
    base = re.split(r"[-_]", raw.strip().casefold())[0]
    if len(base) == 2 and base.isalpha():
        return base
    return _ISO_639_2.get(base) or _LANGUAGE_NAMES.get(base)


class CatalogError(Exception):
    """The catalog could not be reached (after a retry)."""


@dataclass(frozen=True)
class Show:
    id: int
    title: str
    publisher: str
    artwork_url: Optional[str]
    genres: List[str]
    apple_url: str
    feed_url: Optional[str]
    episode_count: Optional[int]
    explicit: bool


@dataclass(frozen=True)
class EpisodeStats:
    typical_minutes: Optional[int]
    last_episode_date: Optional[date]


def _parse_show(r: Dict[str, Any]) -> Show:
    return Show(
        id=int(r["collectionId"]),
        title=r.get("collectionName") or r.get("trackName") or "",
        publisher=r.get("artistName") or "",
        artwork_url=r.get("artworkUrl600") or r.get("artworkUrl100"),
        genres=[g for g in r.get("genres", []) if isinstance(g, str) and g != "Podcasts"],
        apple_url=(r.get("collectionViewUrl") or r.get("trackViewUrl") or "").split("?")[0],
        feed_url=r.get("feedUrl"),
        episode_count=r.get("trackCount"),
        explicit=r.get("collectionExplicitness") == "explicit",
    )


def _date(value: Optional[str]) -> Optional[date]:
    try:
        return date.fromisoformat(value[:10]) if value else None
    except ValueError:
        return None


class ITunesCatalog:
    def __init__(
        self,
        http: httpx.AsyncClient,
        max_concurrency: int = 4,
        cache_ttl: float = 6 * 3600,
        retry_delay: float = 1.0,
    ):
        self._http = http
        self._sem = asyncio.Semaphore(max_concurrency)
        self._cache: Dict[Tuple, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl = cache_ttl
        self._retry_delay = retry_delay
        # Feeds live on publishers' hosts, not Apple's, so they get their own cap.
        self._feed_sem = asyncio.Semaphore(12)
        self._feed_languages: Dict[str, Optional[str]] = {}

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        key = (path, tuple(sorted(params.items())))
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            return hit[1]

        error: object = None
        async with self._sem:
            for attempt in range(2):
                try:
                    response = await self._http.get(BASE_URL + path, params=params)
                except httpx.HTTPError as exc:
                    error = exc
                else:
                    if response.status_code == 200:
                        body = response.json()
                        if len(self._cache) >= _MAX_CACHE_ENTRIES:
                            self._cache.pop(next(iter(self._cache)))
                        self._cache[key] = (time.monotonic() + self._cache_ttl, body)
                        return body
                    error = f"HTTP {response.status_code}"
                    if response.status_code not in _RETRYABLE:
                        break
                if attempt == 0:
                    await asyncio.sleep(self._retry_delay)
        raise CatalogError(f"iTunes {path} failed: {error}")

    async def search(
        self, term: str, country: str, limit: int = 10, explicit: bool = True
    ) -> List[Show]:
        params: Dict[str, Any] = {
            "term": term,
            "media": "podcast",
            "entity": "podcast",
            "country": country,
            "limit": limit,
            "lang": "en_us",
        }
        if not explicit:
            params["explicit"] = "No"
        results = (await self._get("/search", params)).get("results", [])
        return [_parse_show(r) for r in results if r.get("kind") == "podcast" and r.get("collectionId")]

    async def lookup(
        self, ids: Sequence[int], country: str, per_show: int = 5
    ) -> Tuple[Dict[int, Show], Dict[int, EpisodeStats]]:
        """Shows plus typical episode length / last-episode date, in batched calls."""
        ids = list(dict.fromkeys(ids))
        chunks = [ids[i : i + _LOOKUP_CHUNK] for i in range(0, len(ids), _LOOKUP_CHUNK)]
        bodies = await asyncio.gather(
            *(
                self._get(
                    "/lookup",
                    {
                        "id": ",".join(str(i) for i in chunk),
                        "entity": "podcastEpisode",
                        "limit": per_show,
                        "country": country,
                        "lang": "en_us",
                    },
                )
                for chunk in chunks
            )
        )
        shows: Dict[int, Show] = {}
        minutes: Dict[int, List[float]] = {}
        dates: Dict[int, List[date]] = {}
        for r in (r for body in bodies for r in body.get("results", [])):
            if r.get("kind") == "podcast" and r.get("collectionId"):
                show = _parse_show(r)
                shows[show.id] = show
            elif r.get("wrapperType") == "podcastEpisode":
                cid = int(r["collectionId"])
                if r.get("trackTimeMillis"):
                    minutes.setdefault(cid, []).append(r["trackTimeMillis"] / 60000)
                if d := _date(r.get("releaseDate")):
                    dates.setdefault(cid, []).append(d)
        stats = {
            cid: EpisodeStats(
                typical_minutes=round(statistics.median(minutes[cid])) if minutes.get(cid) else None,
                last_episode_date=max(dates[cid]) if dates.get(cid) else None,
            )
            for cid in set(minutes) | set(dates)
        }
        return shows, stats

    async def top_chart(self, genre_id: int, country: str, limit: int = 25) -> List[int]:
        """Collection ids of a storefront's top shows in an Apple genre, in chart order."""
        body = await self._get(f"/{country}/rss/toppodcasts/limit={limit}/genre={genre_id}/json", {})
        entries = body.get("feed", {}).get("entry", [])
        if isinstance(entries, dict):  # the legacy feed collapses a single entry
            entries = [entries]
        return [int(e["id"]["attributes"]["im:id"]) for e in entries]


    async def feed_language(self, feed_url: Optional[str]) -> Optional[str]:
        """The language a show's RSS feed declares (ISO 639-1), or None if unknown.

        Reads only the first few KB of the feed. iTunes has no language field,
        so this is the only real signal for "shows in my language".
        """
        if not feed_url or not feed_url.startswith(("https://", "http://")):
            return None
        if feed_url in self._feed_languages:
            return self._feed_languages[feed_url]
        language = None
        try:
            async with self._feed_sem:
                async with self._http.stream("GET", feed_url, timeout=4.0, follow_redirects=True) as r:
                    if r.status_code == 200:
                        head = b""
                        async for chunk in r.aiter_bytes():
                            head += chunk
                            if match := _LANGUAGE_TAG.search(head):
                                language = normalize_language(match.group(1).decode("utf-8", "ignore"))
                                break
                            if len(head) >= _FEED_BYTES:
                                break
        except httpx.HTTPError:
            language = None
        if len(self._feed_languages) >= _MAX_CACHE_ENTRIES:
            self._feed_languages.pop(next(iter(self._feed_languages)))
        self._feed_languages[feed_url] = language
        return language


# --- Resolver -----------------------------------------------------------------

_TITLE_STOPWORDS = {"the", "a", "podcast"}


def _norm_title(s: str) -> str:
    s = s.casefold().replace("&", " and ").replace("'", "").replace("’", "")
    tokens = re.sub(r"[^\w\s]", " ", s).split()
    return " ".join(t for t in tokens if t not in _TITLE_STOPWORDS)


def _norm_name(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.casefold()).split())


def best_match(title: str, host: str, shows: Sequence[Show]) -> Optional[Show]:
    """Pick the catalog show an LLM candidate refers to, or None.

    Accept a show when the titles match closely, or when the candidate title
    is contained in the show title and something else corroborates it (the
    host matches, it's a long prefix, or it has 3+ words). Among accepted
    shows, prefer a matching host, then iTunes' own ranking — this matters
    because copycat shows reuse popular titles under other publishers.
    """
    want = _norm_title(title)
    if not want:
        return None
    want_host = _norm_name(host)
    best, best_score = None, float("-inf")
    for rank, show in enumerate(shows):
        have = _norm_title(show.title)
        full = fuzz.ratio(want, have)
        subset = fuzz.token_set_ratio(want, have)
        host_score = fuzz.token_sort_ratio(want_host, _norm_name(show.publisher)) if want_host else 0
        corroborated = (
            host_score >= 80
            or len(want.split()) >= 3
            or (have.startswith(want) and len(want) >= 8)
        )
        if not (full >= 88 or (subset >= 95 and corroborated)):
            continue
        score = max(full, subset) + 0.5 * host_score - rank
        if score > best_score:
            best, best_score = show, score
    return best
