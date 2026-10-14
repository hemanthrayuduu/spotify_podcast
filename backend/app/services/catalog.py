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
    """The catalog could not be reached, refused us, or returned garbage."""


class RateBudget:
    """Process-wide token bucket for iTunes calls.

    Apple allows roughly 20 calls per minute per IP. The semaphore caps
    concurrency, not rate; this caps the rate, failing fast once a short wait
    wouldn't help, so a traffic burst degrades results instead of getting the
    server's IP blocked.
    """

    def __init__(self, calls_per_minute: int = 18, max_wait: float = 2.0):
        self._capacity = float(calls_per_minute)
        self._tokens = float(calls_per_minute)
        self._per_second = calls_per_minute / 60
        self._max_wait = max_wait
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._per_second)
        self._updated = now

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._per_second
                if wait > self._max_wait:
                    raise CatalogError("iTunes call budget exhausted; try again shortly")
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1


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


def _shows(results: Any) -> List[Show]:
    """Parse show records, skipping malformed ones rather than failing the request."""
    shows = []
    for r in results if isinstance(results, list) else []:
        if not isinstance(r, dict) or r.get("kind") != "podcast" or not r.get("collectionId"):
            continue
        try:
            shows.append(_parse_show(r))
        except (KeyError, TypeError, ValueError):
            continue
    return shows


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
        budget: Optional[RateBudget] = None,
    ):
        self._http = http
        self._budget = budget
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
                if self._budget:
                    await self._budget.acquire()
                try:
                    response = await self._http.get(BASE_URL + path, params=params)
                except httpx.HTTPError as exc:
                    error = exc
                else:
                    if response.status_code == 200:
                        try:
                            body = response.json()
                        except ValueError:
                            raise CatalogError(f"iTunes {path} returned non-JSON") from None
                        if not isinstance(body, dict):
                            raise CatalogError(f"iTunes {path} returned unexpected JSON")
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
        return _shows((await self._get("/search", params)).get("results"))

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
        results = [r for body in bodies for r in (body.get("results") or []) if isinstance(r, dict)]
        shows = {show.id: show for show in _shows(results)}
        minutes: Dict[int, List[float]] = {}
        dates: Dict[int, List[date]] = {}
        for r in results:
            if r.get("wrapperType") == "podcastEpisode" and isinstance(r.get("collectionId"), int):
                cid = r["collectionId"]
                if isinstance(r.get("trackTimeMillis"), (int, float)) and r["trackTimeMillis"] > 0:
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
        try:
            entries = body.get("feed", {}).get("entry", [])
            if isinstance(entries, dict):  # the legacy feed collapses a single entry
                entries = [entries]
            return [int(e["id"]["attributes"]["im:id"]) for e in entries]
        except (AttributeError, KeyError, TypeError, ValueError):
            raise CatalogError(f"chart {genre_id} ({country}) has an unexpected format") from None


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
        except Exception:  # noqa: BLE001 - best effort: bad URLs, TLS, decoding, ...
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


def _tokens(s: str) -> List[str]:
    return s.split()


def best_match(title: str, host: str, shows: Sequence[Show]) -> Optional[Show]:
    """Pick the catalog show an LLM candidate refers to, or None.

    Copycat feeds reuse popular titles ("Serials", "Stuff You Should Know
    About Money", "Joe Rogan Experience Review"), and when the real show is
    missing from the results they are the closest thing left. So:

      * an exact title is accepted unless the host clearly points elsewhere
        and it isn't iTunes' top result (networks often publish the shows
        hosts front, so a host/publisher mismatch alone proves little);
      * a near-identical title needs a publisher related to the host or title;
      * a title contained in a longer one ("Hardcore History" in "Dan
        Carlin's Hardcore History") needs a matching host, or a related
        publisher and a short prefix extension ("Freakonomics" → "…Radio").

    Among accepted shows, prefer a matching host, then iTunes' ranking.
    """
    want = _norm_title(title)
    if not want:
        return None
    want_host = _norm_name(host)
    best, best_score = None, float("-inf")
    for rank, show in enumerate(shows):
        have = _norm_title(show.title)
        publisher = _norm_name(show.publisher)
        full = fuzz.ratio(want, have)
        host_score = fuzz.token_sort_ratio(want_host, publisher) if want_host else 0
        related = (
            None
            if not want_host
            else host_score >= 60
            or bool(set(_tokens(publisher)) & (set(_tokens(want_host)) | set(_tokens(want))))
        )
        extra = len(set(_tokens(have)) - set(_tokens(want)))
        subset = set(_tokens(want)) <= set(_tokens(have))

        if want == have:
            accepted = related is not False or rank == 0
        elif full >= 92:
            accepted = related is True
        elif subset and extra <= 2:
            accepted = host_score >= 80 or (
                related is not False and have.startswith(want) and len(want) >= 8 and extra <= 1
            )
        else:
            accepted = False
        if not accepted:
            continue
        score = full + 0.5 * host_score - rank
        if score > best_score:
            best, best_score = show, score
    return best


def match_enjoyed(name: str, shows: Sequence[Show]) -> Optional[Show]:
    """Resolve a show name the user typed (no host given; shorthand is common).

    Lenient on purpose: a wrong match only excludes a show and nudges topic
    fit, while a miss lets an already-enjoyed show be recommended back.
    """
    want = _norm_title(name)
    if not want:
        return None
    for show in shows:  # iTunes ranks the popular, real show first
        have = _norm_title(show.title)
        extra = len(set(_tokens(have)) - set(_tokens(want)))
        if want == have or fuzz.ratio(want, have) >= 85 or (
            set(_tokens(want)) <= set(_tokens(have)) and extra <= 3
        ):
            return show
    return None


def same_title(name: str, show: Show) -> bool:
    """Strict check used to exclude enjoyed shows by name alone."""
    want, have = _norm_title(name), _norm_title(show.title)
    return bool(want) and (want == have or fuzz.ratio(want, have) >= 92)
