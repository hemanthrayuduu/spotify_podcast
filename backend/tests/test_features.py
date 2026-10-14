"""Unit tests for the shared feature encoding (used by training and serving)."""

import numpy as np
import pytest

from app.ml.features import FEATURE_NAMES, codes_from_prefs, encode

BASE_PREFS = {
    "age": "20_34",
    "music_genre": ["pop"],
    "podcast_content": ["technology"],
    "podcast_frequency": "daily",
    "podcast_duration": "medium",
    "podcast_format": "interview",
}


def _vec(prefs):
    return dict(zip(FEATURE_NAMES, encode(codes_from_prefs(prefs))))


def test_feature_names_start_with_age_then_field_codes():
    assert FEATURE_NAMES[0] == "age"
    assert "podcast_content:comedy" in FEATURE_NAMES
    assert "podcast_duration:short" in FEATURE_NAMES


def test_vector_length_matches_feature_names():
    assert encode(codes_from_prefs(BASE_PREFS)).shape == (len(FEATURE_NAMES),)


def test_single_select_is_one_hot():
    v = _vec(BASE_PREFS)
    assert v["podcast_frequency:daily"] == 1
    assert v["podcast_frequency:rarely"] == 0
    assert v["podcast_format:interview"] == 1


def test_multi_select_is_normalized_to_sum_one():
    v = _vec({**BASE_PREFS, "music_genre": ["pop", "rock"]})
    assert v["music_genre:pop"] == pytest.approx(0.5)
    assert v["music_genre:rock"] == pytest.approx(0.5)


def test_llm_only_options_do_not_affect_vector():
    # "country" has no survey category, so the music block stays empty.
    v = _vec({**BASE_PREFS, "music_genre": ["country"]})
    assert all(v[n] == 0 for n in FEATURE_NAMES if n.startswith("music_genre:"))


def test_hip_hop_maps_to_survey_rap():
    assert _vec({**BASE_PREFS, "music_genre": ["hip_hop"]})["music_genre:rap"] == 1


def test_no_preference_is_all_zero_like_unanswered_survey():
    v = _vec({**BASE_PREFS, "podcast_duration": "any"})
    assert all(v[n] == 0 for n in FEATURE_NAMES if n.startswith("podcast_duration:"))


def test_age_is_ordinal_scaled_to_unit_interval():
    assert _vec({**BASE_PREFS, "age": "under_12"})["age"] == 0.0
    assert _vec({**BASE_PREFS, "age": "60_plus"})["age"] == 1.0
    assert 0 < _vec(BASE_PREFS)["age"] < 1


def test_long_and_extra_long_share_the_survey_long_category():
    long_v = encode(codes_from_prefs({**BASE_PREFS, "podcast_duration": "long"}))
    xl_v = encode(codes_from_prefs({**BASE_PREFS, "podcast_duration": "extra_long"}))
    assert np.array_equal(long_v, xl_v)
