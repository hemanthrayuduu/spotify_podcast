"""Unit tests for the feature-vector mapping.

Uses a fake ModelBundle with an identity scaler so we can assert the exact
one-hot vector prepare_features builds, independent of the trained model.
"""

from app.ml.features import prepare_features
from app.ml.loader import ModelBundle


class _IdentityScaler:
    def transform(self, X):
        return X


VALID_FEATURES = [
    "age_numeric",
    "fav_music_genre_Pop",
    "fav_music_genre_Rock",
    "fav_pod_genre_Technology",
    "pod_lis_frequency_Daily",
    "preffered_pod_duration_Medium (30-60 min)",
    "preffered_pod_format_Interview",
]


def _bundle():
    return ModelBundle(
        kmeans_model=None,
        scaler=_IdentityScaler(),
        valid_features=VALID_FEATURES,
        segment_profiles={},
    )


BASE_PREFS = {
    "age": "25-34",
    "music_genre": ["Pop"],
    "podcast_content": ["Technology"],
    "podcast_frequency": "Daily",
    "podcast_duration": "Medium (30-60 min)",
    "podcast_format": "Interview",
}


def test_shape_matches_valid_features():
    vec = prepare_features(_bundle(), BASE_PREFS)
    assert vec.shape == (1, len(VALID_FEATURES))


def test_known_values_are_one_hot_encoded():
    vec = prepare_features(_bundle(), BASE_PREFS)[0]
    idx = {name: i for i, name in enumerate(VALID_FEATURES)}
    assert vec[idx["age_numeric"]] == 30  # midpoint of 25-34
    assert vec[idx["fav_music_genre_Pop"]] == 1
    assert vec[idx["fav_pod_genre_Technology"]] == 1
    assert vec[idx["pod_lis_frequency_Daily"]] == 1
    # A genre not selected stays 0.
    assert vec[idx["fav_music_genre_Rock"]] == 0


def test_multiple_genres_all_set():
    prefs = {**BASE_PREFS, "music_genre": ["Pop", "Rock"]}
    vec = prepare_features(_bundle(), prefs)[0]
    idx = {name: i for i, name in enumerate(VALID_FEATURES)}
    assert vec[idx["fav_music_genre_Pop"]] == 1
    assert vec[idx["fav_music_genre_Rock"]] == 1


def test_unknown_age_defaults_to_30():
    prefs = {**BASE_PREFS, "age": "does-not-exist"}
    vec = prepare_features(_bundle(), prefs)[0]
    idx = {name: i for i, name in enumerate(VALID_FEATURES)}
    assert vec[idx["age_numeric"]] == 30
