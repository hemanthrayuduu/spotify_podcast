"""Tests for the survey cleaning + segmentation training pipeline.

These need the training extras (pandas, scikit-learn) and the survey CSV.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from app.ml import training  # noqa: E402
from app.ml.features import FEATURE_NAMES, encode  # noqa: E402
from app.ml.segmenter import Segmenter  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "Spotify_user_research.csv")


@pytest.fixture(scope="module")
def survey():
    return training.load_survey(DATA_PATH)


@pytest.fixture(scope="module")
def survey_codes(survey):
    return training.survey_codes(survey)


def test_every_raw_survey_value_is_mapped(survey):
    # survey_codes raises on an unmapped value; check the message is useful.
    assert training.unmapped_values(survey) == {}


@pytest.mark.parametrize(
    "field, raw, expected",
    [
        ("music_genre", "Classical & melody, dance", {"classical", "melody", "electronic"}),
        ("music_genre", "classical", {"classical"}),
        ("music_genre", "All", set()),
        ("podcast_duration", "Shorter", {"short"}),
        ("podcast_duration", "Both", set()),
        ("podcast_content", "Murder Mystery ", {"true_crime"}),
        ("podcast_format", "Story telling", {"storytelling"}),
        ("age", "20~35", {"20-35"}),
    ],
)
def test_raw_values_clean_to_codes(field, raw, expected):
    assert training.clean_value(field, raw) == expected


def test_missing_answers_clean_to_no_codes():
    assert training.clean_value("podcast_format", float("nan")) == set()


def test_unknown_raw_value_raises():
    with pytest.raises(ValueError, match="unmapped"):
        training.clean_value("music_genre", "Polka")


@pytest.fixture(scope="module")
def X(survey_codes):
    return np.vstack([encode(c) for c in survey_codes])


def test_block_scale_equalizes_field_variance(X):
    scaled = X * training.block_scale(X)
    for field in ("age", "podcast_frequency", "podcast_content"):
        idx = [i for i, n in enumerate(FEATURE_NAMES) if n == field or n.startswith(field + ":")]
        assert scaled[:, idx].var(axis=0).sum() == pytest.approx(1.0)


def _artifact(X, k=3):
    scale = training.block_scale(X)
    km = training.fit_kmeans(X * scale, k=k)
    return km, training.build_artifact(X, km, scale, data_sha256="test", silhouette_by_k={k: 0.0})


def test_nearest_centroid_matches_sklearn(X):
    km, artifact = _artifact(X)
    # Segmenter takes unscaled vectors (what features.encode produces).
    assert np.array_equal(Segmenter.from_artifact(artifact).nearest(X), km.labels_)


def test_personas_have_names_and_lifted_traits(X):
    _, artifact = _artifact(X)
    personas = artifact["personas"]
    assert len(personas) == 3
    assert len({p["name"] for p in personas}) == 3, "persona names must be distinct"
    for p in personas:
        assert p["traits"], p
        assert all(t["lift"] > 1 for t in p["traits"])
    assert sum(p["share"] for p in personas) == pytest.approx(1.0)


def test_training_is_reproducible(tmp_path):
    a = training.train(DATA_PATH, k_range=range(2, 5))
    b = training.train(DATA_PATH, k_range=range(2, 5))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["metrics"]["k"] in (2, 3, 4)
