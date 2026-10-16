"""Tests for the runtime segmenter (numpy nearest-centroid over segments.json)."""

import copy

import numpy as np
import pytest

from app.core.config import settings
from app.ml.features import FEATURE_NAMES, codes_from_prefs, encode
from app.ml.segmenter import SCHEMA_VERSION, Segmenter, personalize


def _artifact():
    n = len(FEATURE_NAMES)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": "test",
        "feature_names": list(FEATURE_NAMES),
        "feature_scale": [1.0] * n,
        "centroids": [[0.0] * n, [1.0] * n],
        "personas": [
            {"id": 0, "name": "Zero", "description": "", "traits": [], "share": 0.5},
            {"id": 1, "name": "One", "description": "", "traits": [], "share": 0.5},
        ],
        "metrics": {},
        "training": {},
    }


def test_rejects_feature_mismatch():
    bad = _artifact()
    bad["feature_names"] = bad["feature_names"][:-1]
    with pytest.raises(RuntimeError, match="feature"):
        Segmenter.from_artifact(bad)


def test_rejects_schema_version():
    bad = copy.deepcopy(_artifact())
    bad["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(RuntimeError, match="schema"):
        Segmenter.from_artifact(bad)


def test_predict_returns_nearest_persona():
    seg = Segmenter.from_artifact(_artifact())
    assert seg.predict(np.zeros(len(FEATURE_NAMES)))["name"] == "Zero"
    assert seg.predict(np.ones(len(FEATURE_NAMES)))["name"] == "One"


def test_predict_applies_feature_scale():
    art = _artifact()
    n = len(FEATURE_NAMES)
    art["feature_scale"] = [2.0] * n
    # 0.4 is nearer centroid 0 unscaled, but scaled to 0.8 it is nearer centroid 1.
    assert Segmenter.from_artifact(art).predict(np.full(n, 0.4))["name"] == "One"


def test_committed_artifact_loads_and_predicts():
    seg = Segmenter.load(settings.model_dir)
    prefs = {
        "age": "20_34",
        "music_genre": ["jazz"],
        "podcast_content": ["true_crime"],
        "podcast_frequency": "daily",
        "podcast_duration": "long",
        "podcast_format": "storytelling",
    }
    persona = seg.predict(encode(codes_from_prefs(prefs)))
    assert persona["name"] and persona["traits"]
    assert seg.model_version


def test_committed_artifact_has_at_least_three_personas():
    # Silhouette favours k=2 on this weakly-structured survey, which yields a
    # bare age split; training searches k >= 3 so personas are informative.
    assert len(Segmenter.load(settings.model_dir).personas) >= 3


PERSONA = {
    "id": 0, "name": "Mid-career listeners · Long-form fans", "share": 0.24, "description": "x",
    "traits": [
        {"feature": "age:35-60", "label": "Mid-career listeners", "share": 0.85, "lift": 3.3},
        {"feature": "podcast_duration:long", "label": "Long-form fans", "share": 0.28, "lift": 1.3},
    ],
}


def test_personalize_keeps_only_traits_the_user_shares():
    codes = codes_from_prefs({"age": "35_59", "podcast_duration": "short", "music_genre": [],
                              "podcast_content": [], "podcast_frequency": "", "podcast_format": ""})
    persona = personalize(PERSONA, codes)
    assert [t["feature"] for t in persona["traits"]] == ["age:35-60"]
    assert persona["name"] == "Mid-career listeners"
    assert "Long-form" not in persona["description"]


def test_personalize_without_shared_traits_says_so():
    codes = codes_from_prefs({"age": "60_plus", "podcast_duration": "short", "music_genre": [],
                              "podcast_content": [], "podcast_frequency": "", "podcast_format": ""})
    persona = personalize(PERSONA, codes)
    assert persona["traits"] == []
    assert persona["name"] == "Eclectic listener"


@pytest.mark.parametrize("age", ["under_12", "12_19", "20_34", "35_59", "60_plus"])
@pytest.mark.parametrize("duration", ["short", "medium", "long", "extra_long", "any"])
def test_committed_personas_never_contradict_the_user(age, duration):
    seg = Segmenter.load(settings.model_dir)
    prefs = {"age": age, "podcast_duration": duration, "music_genre": ["pop"],
             "podcast_content": ["comedy"], "podcast_frequency": "weekly", "podcast_format": "interview"}
    codes = codes_from_prefs(prefs)
    persona = personalize(seg.predict(encode(codes)), codes)
    for trait in persona["traits"]:
        field, code = trait["feature"].split(":")
        assert code in codes[field], (age, duration, trait)
