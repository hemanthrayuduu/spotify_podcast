"""Load ML model artifacts, failing loudly if any are missing or corrupt."""

import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    kmeans_model: Any
    scaler: Any
    valid_features: Any
    segment_profiles: Dict[str, Any]


def load_model_bundle(model_dir: str) -> ModelBundle:
    """Load all artifacts from ``model_dir``.

    We fail loudly at startup rather than silently degrading to fallback
    recommendations, so a broken deploy is visible immediately instead of
    quietly serving low-quality results forever.
    """
    pickles = {
        "kmeans_model": "kmeans_model.pkl",
        "scaler": "scaler.pkl",
        "valid_features": "valid_features.pkl",
    }
    loaded: Dict[str, Any] = {}
    for key, filename in pickles.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            raise RuntimeError(f"Required model artifact missing: {path}")
        with open(path, "rb") as f:
            loaded[key] = pickle.load(f)

    segment_path = os.path.join(model_dir, "segment_profiles.json")
    if not os.path.exists(segment_path):
        raise RuntimeError(f"Required model artifact missing: {segment_path}")
    with open(segment_path, "r") as f:
        segment_profiles = json.load(f)

    logger.info("Models and segment profiles loaded successfully.")
    return ModelBundle(
        kmeans_model=loaded["kmeans_model"],
        scaler=loaded["scaler"],
        valid_features=loaded["valid_features"],
        segment_profiles=segment_profiles,
    )
