"""The recommendation pipeline.

    persona   ← survey segment for the user's answers
    seeds     ← podcasts they already enjoy, resolved in the catalog (excluded,
                and their genres count toward topic fit)
    pool      ← Apple's genre charts for the user's topics, with real episode
                lengths; ineligible shows filtered out, best fits kept
    choose    ← the LLM picks from the pool by id and may add up to 2 shows of
                its own (extras)
    verify    ← each extra resolved to a real Apple Podcasts show; the rest dropped
    rank      ← transparent score over pool + extras, language check, one show
                per publisher; without an LLM the pool is ranked directly

Nothing the user sees about a show comes from the LLM except the clearly
labelled AI-written ``reason`` and ``summary``.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

from app.core.vocabulary import Option, option, to_labels
from app.ml.features import codes_from_prefs, encode
from app.ml.segmenter import Segmenter, personalize
from app.services.catalog import (
    CatalogError,
    ITunesCatalog,
    Show,
    best_match,
    match_enjoyed,
    same_title,
)
from app.services.llm import Choice, LLMError, PoolEntry, Suggestion, choose_podcasts
from app.services.ranking import Candidate, Ranked, rank

logger = logging.getLogger(__name__)

MINOR_AGES = {"under_12", "12_19"}
_MAX_SEEDS = 3
_SEARCH_LIMIT = 8  # same limit for seed and verify searches so they share cache entries
_MAX_CHART_TOPICS = 3
_CHART_DEPTH = 15  # top shows taken from each genre chart
_POOL_SIZE = 30  # candidates shown to the LLM (~1k prompt tokens)
_POOL_LOOKUP_CHUNK = 10  # pool lookups: fewer, bigger calls from the iTunes budget
_POOL_EPISODES = 3  # recent episodes per show: enough for a median length
_MAX_LOOKUP_IDS = 45
_LANGUAGE_BATCH = 12  # feeds checked per parallel batch
_MAX_CACHE_ENTRIES = 256


class Recommender:
    def __init__(
        self,
        segmenter: Segmenter,
        catalog: ITunesCatalog,
        llm_client: Any,
        model: str,
        *,
        top_n: int = 5,
        n_picks: int = 8,
        n_extras: int = 2,
        cache_ttl: float = 600,
        today: Callable[[], date] = date.today,
    ):
        self.segmenter = segmenter
        self.catalog = catalog
        self.llm_client = llm_client
        self.model = model
        self.top_n = top_n
        self.n_picks = n_picks
        self.n_extras = n_extras
        self.cache_ttl = cache_ttl
        self.today = today
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    async def recommend(self, prefs: Dict[str, Any]) -> Dict[str, Any]:
        key = json.dumps(prefs, sort_keys=True)
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            return {**hit[1], "meta": {**hit[1]["meta"], "cached": True}}

        started = time.perf_counter()
        run = _Run(self, prefs)
        response = await run.execute()
        response["meta"]["latency_ms"] = round((time.perf_counter() - started) * 1000)

        # Only cache complete answers: a result degraded by a catalog or LLM
        # failure should be retried by the next request, not served for minutes.
        if not run.degraded:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (time.monotonic() + self.cache_ttl, response)
        return response


class _Run:
    """State for one recommendation request."""

    def __init__(self, rec: Recommender, prefs: Dict[str, Any]):
        self.rec = rec
        self.prefs = prefs
        self.labels = to_labels(prefs)
        codes = codes_from_prefs(prefs)
        self.persona = personalize(rec.segmenter.predict(encode(codes)), codes)
        names = re.split(r"[,;\n]", prefs.get("podcasts_enjoyed") or "")
        self.enjoyed_names = [n.strip() for n in names if n.strip()][:_MAX_SEEDS]
        language = option("content_language", prefs["content_language"])
        self.country = language.storefront or option("region", prefs["region"]).storefront
        self.allow_explicit = prefs["age"] not in MINOR_AGES
        self.topics: List[Option] = [option("podcast_content", v) for v in prefs["podcast_content"]]
        self.target_genres = {g for t in self.topics for g in t.apple_genres}
        self.band = option("podcast_duration", prefs["podcast_duration"]).minutes
        self.language = language.iso_language
        self.feed_languages: Dict[int, Optional[str]] = {}
        self.dropped_language = 0
        self.catalog_ok = 0
        self.catalog_failed = 0
        self.llm_failed = False

    async def _search(self, term: str, limit: int) -> Optional[List[Show]]:
        try:
            shows = await self.rec.catalog.search(term, self.country, limit, self.allow_explicit)
        except CatalogError as exc:
            logger.warning(f"Catalog search {term!r} failed: {exc}")
            self.catalog_failed += 1
            return None
        self.catalog_ok += 1
        return shows

    @property
    def degraded(self) -> bool:
        return self.catalog_failed > 0 or self.llm_failed

    async def _seeds(self) -> List[Show]:
        names = self.enjoyed_names
        results = await asyncio.gather(*(self._search(n, _SEARCH_LIMIT) for n in names))
        return [m for n, shows in zip(names, results) if shows and (m := match_enjoyed(n, shows))]

    async def _choose(self, pool: List[Candidate]) -> Optional[Choice]:
        if self.rec.llm_client is None:
            return None
        today = self.rec.today()
        entries = [
            PoolEntry(
                id=c.show.id,
                title=c.show.title,
                publisher=c.show.publisher,
                genres=c.show.genres,
                minutes=c.stats.typical_minutes if c.stats else None,
                days_since_last=(today - c.stats.last_episode_date).days
                if c.stats and c.stats.last_episode_date
                else None,
            )
            for c in pool
        ]
        try:
            return await choose_podcasts(
                self.rec.llm_client, self.labels, self.persona, entries, self.rec.model,
                n_picks=self.rec.n_picks, n_extras=self.rec.n_extras,
            )
        except LLMError as exc:
            logger.warning(f"LLM unavailable, using catalog only: {exc}")
            self.llm_failed = True
            return None

    async def _verify(
        self, suggestions: Sequence[Suggestion], first_rank: int = 0
    ) -> Tuple[List[Candidate], int]:
        """Resolve the LLM's own suggestions to real shows; count the ones that aren't."""
        results = await asyncio.gather(*(self._search(s.title, _SEARCH_LIMIT) for s in suggestions))
        verified, dropped = [], 0
        for i, (s, shows) in enumerate(zip(suggestions, results), start=first_rank):
            if shows is None:  # the search failed; that's a catalog error, not a fake show
                continue
            match = best_match(s.title, s.host, shows)
            if match is None:
                dropped += 1
                continue
            verified.append(
                Candidate(show=match, stats=None, llm_rank=i, reason=s.reason, summary=s.summary)
            )
        return verified, dropped

    async def _chart(self, topic: Option) -> List[int]:
        try:
            ids = await self.rec.catalog.top_chart(topic.chart_genre_id, self.country)
        except CatalogError as exc:
            logger.warning(f"Chart {topic.chart_genre_id} failed: {exc}")
            self.catalog_failed += 1
            return []
        self.catalog_ok += 1
        return ids[:_CHART_DEPTH]

    async def _lookup(self, ids: List[int], **kw: Any) -> Tuple[Dict[int, Show], Dict[int, Any]]:
        if not ids:
            return {}, {}
        try:
            result = await self.rec.catalog.lookup(ids[:_MAX_LOOKUP_IDS], self.country, **kw)
        except CatalogError as exc:
            logger.warning(f"Lookup failed: {exc}")
            self.catalog_failed += 1
            return {}, {}
        self.catalog_ok += 1
        return result

    async def _charts(self) -> List[List[int]]:
        """Chart ids per topic (cached for hours)."""
        return list(await asyncio.gather(*(self._chart(t) for t in self.topics[:_MAX_CHART_TOPICS])))

    async def _chart_candidates(self) -> List[Candidate]:
        """Top shows from the Apple genre chart of each chosen topic, with real stats."""
        topics = self.topics[:_MAX_CHART_TOPICS]
        charts = await self._charts()
        ids = list(dict.fromkeys(i for chart in charts for i in chart))
        shows, stats = await self._lookup(ids, per_show=_POOL_EPISODES, chunk=_POOL_LOOKUP_CHUNK)
        candidates = []
        for topic, ids in zip(topics, charts):
            for position, show_id in enumerate(ids):
                if show_id in shows:
                    reason = (
                        f"#{position + 1} in Apple Podcasts' {topic.label} chart "
                        f"({self.country.upper()})."
                    )
                    candidates.append(
                        Candidate(show=shows[show_id], stats=stats.get(show_id),
                                  catalog_rank=position, reason=reason)
                    )
        return candidates

    async def _enrich(self, candidates: List[Candidate]) -> None:
        """Attach real episode stats to shows that lack them (the LLM's verified extras)."""
        _, stats = await self._lookup([c.show.id for c in candidates if c.stats is None])
        for c in candidates:
            c.stats = c.stats or stats.get(c.show.id)

    async def _languages(self, shows: Sequence[Show]) -> None:
        unique = {s.id: s for s in shows}  # Show holds a list, so dedupe by id
        unknown = [s for i, s in unique.items() if i not in self.feed_languages]
        found = await asyncio.gather(*(self.rec.catalog.feed_language(s.feed_url) for s in unknown))
        self.feed_languages.update({s.id: lang for s, lang in zip(unknown, found)})

    async def _in_language(self, candidates: List[Candidate], seeds: List[Show]) -> List[Ranked]:
        """Best ``top_n`` shows whose feed is in the requested language.

        Feeds are checked in parallel batches, best-ranked first. Shows whose
        feed confirms the language come first; feeds that don't declare one
        are kept but only fill remaining slots: we can't prove them wrong, and
        dropping them would punish older feeds rather than wrong ones.
        """
        ordered = self._rank(candidates, seeds, top_n=None)
        confirmed: List[Ranked] = []
        undeclared: List[Ranked] = []
        rejected = set()
        for start in range(0, len(ordered), _LANGUAGE_BATCH):
            batch = ordered[start : start + _LANGUAGE_BATCH]
            await self._languages([r.candidate.show for r in batch])
            for r in batch:
                lang = self.feed_languages.get(r.candidate.show.id)
                if lang is None:
                    undeclared.append(r)
                elif lang == self.language:
                    confirmed.append(r)
                else:
                    rejected.add(r.candidate.show.id)
            if len(confirmed) >= self.rec.top_n:
                break
        self.dropped_language = len(rejected)
        return (confirmed + undeclared)[: self.rec.top_n]

    def _rank(
        self, candidates: List[Candidate], seeds: List[Show], top_n: Optional[int]
    ) -> List[Ranked]:
        # Exclude enjoyed shows by name too, so it doesn't hinge on their search succeeding.
        enjoyed = {s.id for s in seeds} | {
            c.show.id for c in candidates if any(same_title(n, c.show) for n in self.enjoyed_names)
        }
        return rank(
            candidates,
            target_genres=self.target_genres,
            seed_genres={g for s in seeds for g in s.genres},
            seed_ids=enjoyed,
            minutes=self.band,
            allow_explicit=self.allow_explicit,
            today=self.rec.today(),
            top_n=top_n,
        )

    def _pool(self, charted: List[Candidate], seeds: List[Show]) -> List[Candidate]:
        """The chart shows worth showing the LLM: eligible, one per publisher, best first."""
        ordered = self._rank(charted, seeds, top_n=None)
        return [r.candidate for r in ordered][:_POOL_SIZE]

    async def execute(self) -> Dict[str, Any]:
        seeds, charted = await asyncio.gather(self._seeds(), self._chart_candidates())
        pool = self._pool(charted, seeds)
        choice = await self._choose(pool)

        candidates: List[Candidate] = list(pool)
        dropped = 0
        if choice:
            by_id = {c.show.id: c for c in pool}
            for i, pick in enumerate(choice.picks):
                picked = by_id[pick.id]
                picked.llm_rank, picked.reason, picked.summary = i, pick.reason, pick.summary
            extras, dropped = await self._verify(choice.extras, first_rank=len(choice.picks))
            candidates += extras
        # Extras still need episode stats (Apple) and every show its language
        # (publishers' feeds): overlap them.
        await asyncio.gather(
            self._enrich([c for c in candidates if c.stats is None]),
            self._languages([c.show for c in candidates if c.llm_rank is not None]),
        )
        ranked = await self._in_language(candidates, seeds)

        if not ranked and self.catalog_failed and not self.catalog_ok:
            raise CatalogError("every catalog request failed")

        return {
            "persona": self.persona,
            "recommendations": [self._format(r, self.feed_languages) for r in ranked],
            "meta": {
                "mode": "ai" if choice else "catalog_only",
                "llm_model": self.rec.model if choice else None,
                "storefront": self.country,
                "pool_size": len(pool),
                "llm_candidates": len(choice.picks) + len(choice.extras) if choice else 0,
                "llm_extras": len(choice.extras) if choice else 0,
                "llm_verified": sum(1 for c in candidates if c.llm_rank is not None),
                "candidates_considered": len({c.show.id for c in candidates}),
                "dropped_unverified": dropped,
                "dropped_language": self.dropped_language,
                "catalog_errors": self.catalog_failed,
                "segment_model_version": self.rec.segmenter.model_version,
                "cached": False,
            },
        }

    @staticmethod
    def _format(r: Ranked, languages: Dict[int, Optional[str]]) -> Dict[str, Any]:
        c, show = r.candidate, r.candidate.show
        last = c.stats.last_episode_date if c.stats else None
        return {
            "id": show.id,
            "title": show.title,
            "publisher": show.publisher,
            "artwork_url": show.artwork_url,
            "genres": show.genres,
            "apple_url": show.apple_url,
            "spotify_url": "https://open.spotify.com/search/" + quote(show.title, safe=""),
            "feed_url": show.feed_url,
            "episode_count": show.episode_count,
            "typical_episode_minutes": c.stats.typical_minutes if c.stats else None,
            "last_episode_date": last.isoformat() if last else None,
            "explicit": show.explicit,
            "language": languages.get(show.id),
            "summary": c.summary,
            "reason": c.reason or "",
            "source": "ai" if c.llm_rank is not None else "catalog",
            "match": {"score": r.score, "breakdown": asdict(r.breakdown)},
        }

