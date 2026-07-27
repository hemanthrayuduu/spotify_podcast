"""Map user preferences to the scaled one-hot feature vector."""

from typing import Any, Dict, List

import numpy as np

from app.ml.loader import ModelBundle

AGE_MAP = {"18-24": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55+": 60}


def prepare_features(bundle: ModelBundle, preferences: Dict[str, Any]) -> np.ndarray:
    """Build the scaled feature vector for KMeans prediction."""
    valid_features = bundle.valid_features
    features: Dict[str, int] = {"age_numeric": AGE_MAP.get(preferences["age"], 30)}

    def set_multi(base_name: str, values: List[str]) -> None:
        for value in values:
            if value and f"{base_name}_{value}" in valid_features:
                features[f"{base_name}_{value}"] = 1

    def set_single(base_name: str, value: str) -> None:
        if value and f"{base_name}_{value}" in valid_features:
            features[f"{base_name}_{value}"] = 1

    set_multi("fav_music_genre", preferences["music_genre"])
    set_multi("fav_pod_genre", preferences["podcast_content"])
    set_single("pod_lis_frequency", preferences["podcast_frequency"])
    set_single("preffered_pod_duration", preferences["podcast_duration"])
    set_single("preffered_pod_format", preferences["podcast_format"])

    feature_vector = np.array(
        [[features.get(name, 0) for name in valid_features]], dtype=float
    )
    return bundle.scaler.transform(feature_vector)
