"""Tests for transparent scoring, filtering and diversification."""

from datetime import date

import pytest

from app.services.catalog import EpisodeStats, Show
from app.services.ranking import (
    WEIGHTS,
    Candidate,
    duration_fit,
    freshness,
    rank,
    topic_fit,
)

TODAY = date(2026, 10, 14)


def _show(id, publisher="P", genres=("Comedy",), explicit=False, episodes=50):
    return Show(id=id, title=f"Show {id}", publisher=f"{publisher}{id}" if publisher == "P" else publisher,
                artwork_url=None, genres=list(genres), apple_url="", feed_url=None,
                episode_count=episodes, explicit=explicit)


def _cand(id, llm_rank=None, minutes=45, last=date(2026, 10, 1), **show_kw):
    return Candidate(
        show=_show(id, **show_kw),
        stats=EpisodeStats(typical_minutes=minutes, last_episode_date=last),
        llm_rank=llm_rank,
        catalog_rank=0,
    )


# --- Component scores ----------------------------------------------------------


def test_duration_inside_band_scores_full():
    assert duration_fit(45, (30, 60)) == 1.0
    assert duration_fit(90, (60, None)) == 1.0


def test_duration_outside_band_decays():
    near, far = duration_fit(70, (30, 60)), duration_fit(200, (30, 60))
    assert 0 < far < near < 1
    assert duration_fit(5, (30, 60)) < 1


def test_duration_misses_are_penalized_quadratically():
    # A 30-minute show for someone who wants under 15 minutes is a real miss.
    assert duration_fit(30, (0, 15)) == pytest.approx(0.25)
    assert duration_fit(15, (30, 60)) == pytest.approx(0.25)


def test_length_outweighs_llm_endorsement():
    assert WEIGHTS["duration"] > WEIGHTS["endorsement"]


def test_unknown_duration_is_neutral():
    assert duration_fit(None, (30, 60)) == 0.5


def test_freshness_prefers_recent_shows():
    assert freshness(date(2026, 10, 1), TODAY) == 1.0
    assert freshness(date(2026, 6, 1), TODAY) < 1.0
    assert freshness(date(2020, 1, 1), TODAY) < freshness(date(2026, 6, 1), TODAY)
    assert freshness(None, TODAY) < 0.5


def test_topic_fit_matches_apple_genres():
    assert topic_fit(["True Crime"], {"True Crime"}, set()) == 1.0
    assert topic_fit(["History"], {"True Crime"}, {"History"}) == pytest.approx(0.6)
    assert topic_fit(["Sports"], {"True Crime"}, set()) == 0.0


# --- Ranking -------------------------------------------------------------------


def _rank(cands, **kw):
    defaults = dict(target_genres={"Comedy"}, seed_genres=set(), seed_ids=set(),
                    minutes=(30, 60), allow_explicit=True, today=TODAY, top_n=5)
    return rank(cands, **{**defaults, **kw})


def test_score_is_weighted_sum_of_breakdown():
    (rec,) = _rank([_cand(1, llm_rank=0)])
    b = rec.breakdown
    expected = sum(WEIGHTS[k] * getattr(b, k) for k in WEIGHTS)
    assert rec.score == round(100 * expected)


def test_returns_top_n_by_score():
    cands = [_cand(i, llm_rank=i) for i in range(8)]
    ranked = _rank(cands)
    assert len(ranked) == 5
    assert [r.candidate.show.id for r in ranked] == [0, 1, 2, 3, 4]


def test_better_duration_fit_ranks_higher():
    ranked = _rank([_cand(1, llm_rank=0, minutes=180), _cand(2, llm_rank=1, minutes=45)])
    assert ranked[0].candidate.show.id == 2


def test_one_show_per_publisher():
    ranked = _rank([_cand(1, llm_rank=0, publisher="NPR"), _cand(2, llm_rank=1, publisher="NPR"),
                    _cand(3, llm_rank=2)])
    assert [r.candidate.show.id for r in ranked] == [1, 3]


def test_explicit_shows_dropped_for_minors():
    ranked = _rank([_cand(1, llm_rank=0, explicit=True), _cand(2, llm_rank=1)], allow_explicit=False)
    assert [r.candidate.show.id for r in ranked] == [2]


def test_already_enjoyed_shows_are_excluded():
    ranked = _rank([_cand(1, llm_rank=0), _cand(2, llm_rank=1)], seed_ids={1})
    assert [r.candidate.show.id for r in ranked] == [2]


def test_duplicate_shows_are_merged_keeping_best_llm_rank():
    ranked = _rank([_cand(1, llm_rank=3), _cand(1, llm_rank=0)])
    assert len(ranked) == 1
    assert ranked[0].candidate.llm_rank == 0


def test_shows_with_too_few_episodes_are_dropped():
    # Trailer-only and brand-new feeds aren't worth recommending yet.
    ranked = _rank([_cand(1, llm_rank=0, episodes=1), _cand(2, llm_rank=1)])
    assert [r.candidate.show.id for r in ranked] == [2]
