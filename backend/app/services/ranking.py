"""Transparent scoring of verified shows against what the user asked for.

Every component is in [0, 1] and is returned to the client, so the match
score can be explained ("why is this 82%?") rather than invented:

  * topic       Apple genres overlap the user's topics (or shows they enjoy)
  * duration    real median episode length vs. the requested length band
  * endorsement how strongly the LLM put it forward (or its genre-chart rank)
  * freshness   how recently a new episode came out
"""

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.services.catalog import EpisodeStats, Show

WEIGHTS: Dict[str, float] = {"topic": 0.35, "duration": 0.30, "endorsement": 0.20, "freshness": 0.15}

_INF = float("inf")
MIN_EPISODES = 3  # trailer-only and brand-new feeds aren't worth recommending yet


@dataclass
class Candidate:
    show: Show
    stats: Optional[EpisodeStats]
    llm_rank: Optional[int] = None  # 0 = the model's top pick; None = found by catalog search
    catalog_rank: int = 0  # position in the Apple genre chart
    reason: Optional[str] = None  # AI-written
    summary: Optional[str] = None  # AI-written


@dataclass
class Breakdown:
    topic: float
    duration: float
    endorsement: float
    freshness: float


@dataclass
class Ranked:
    candidate: Candidate
    score: int  # 0-100
    breakdown: Breakdown


def duration_fit(minutes: Optional[int], band: Tuple[int, Optional[int]]) -> float:
    if minutes is None:
        return 0.5
    lo, hi = band
    # Squared ratio: twice the wanted length scores 0.25, not 0.5.
    if minutes < lo:
        return (minutes / lo) ** 2
    if hi is not None and minutes > hi:
        return (hi / minutes) ** 2
    return 1.0


def freshness(last_episode: Optional[date], today: date) -> float:
    if last_episode is None:
        return 0.3
    days = (today - last_episode).days
    for limit, score in ((30, 1.0), (90, 0.8), (365, 0.5), (730, 0.25)):
        if days <= limit:
            return score
    return 0.1


def topic_fit(genres: Iterable[str], target: Set[str], seed: Set[str]) -> float:
    genres = set(genres)
    if genres & target:
        return 1.0
    if genres & seed:
        return 0.6
    return 0.0


def endorsement(c: Candidate) -> float:
    if c.llm_rank is not None:
        return max(0.5, 1.0 - 0.07 * c.llm_rank)
    return max(0.2, 0.6 - 0.02 * c.catalog_rank)


def _merge(candidates: Sequence[Candidate]) -> List[Candidate]:
    """One entry per show, keeping the strongest LLM endorsement."""
    by_id: Dict[int, Candidate] = {}
    for c in candidates:
        kept = by_id.get(c.show.id)
        if kept is None or (c.llm_rank if c.llm_rank is not None else _INF) < (
            kept.llm_rank if kept.llm_rank is not None else _INF
        ):
            by_id[c.show.id] = c
    return list(by_id.values())


def rank(
    candidates: Sequence[Candidate],
    *,
    target_genres: Set[str],
    seed_genres: Set[str],
    seed_ids: Set[int],
    minutes: Tuple[int, Optional[int]],
    allow_explicit: bool,
    today: date,
    top_n: Optional[int] = 5,
) -> List[Ranked]:
    """Score, filter and diversify; ``top_n=None`` returns the whole ordering."""
    scored: List[Ranked] = []
    for c in _merge(candidates):
        if c.show.id in seed_ids or (c.show.explicit and not allow_explicit):
            continue
        if c.show.episode_count is not None and c.show.episode_count < MIN_EPISODES:
            continue
        b = Breakdown(
            topic=topic_fit(c.show.genres, target_genres, seed_genres),
            duration=round(duration_fit(c.stats.typical_minutes if c.stats else None, minutes), 2),
            endorsement=round(endorsement(c), 2),
            freshness=freshness(c.stats.last_episode_date if c.stats else None, today),
        )
        score = round(100 * sum(WEIGHTS[k] * getattr(b, k) for k in WEIGHTS))
        scored.append(Ranked(candidate=c, score=score, breakdown=b))

    scored.sort(
        key=lambda r: (
            -r.score,
            r.candidate.llm_rank if r.candidate.llm_rank is not None else _INF,
            r.candidate.catalog_rank,
        )
    )
    picked, publishers = [], set()
    for r in scored:
        publisher = r.candidate.show.publisher.casefold()
        if publisher in publishers:
            continue
        publishers.add(publisher)
        picked.append(r)
        if len(picked) == top_n:
            break
    return picked
