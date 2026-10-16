"""Runtime listener segmentation: nearest KMeans centroid, numpy only.

The trained model is a plain JSON artifact (models/segments.json) written by
``backend/train.py``: centroids, the feature names they were trained on, and a
persona per segment. Loading JSON instead of pickles means no code execution
on load and no scikit-learn version skew at serving time.
"""

import json
import logging
import os
from typing import Any, Dict

import numpy as np

from app.ml.features import FEATURE_NAMES, SegmentCodes

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
ARTIFACT_NAME = "segments.json"


class Segmenter:
    def __init__(
        self, centroids: np.ndarray, scale: np.ndarray, personas: list, model_version: str
    ):
        self.centroids = centroids
        self.scale = scale
        self.personas = personas
        self.model_version = model_version

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "Segmenter":
        """Validate and wrap an artifact; fails loudly on any mismatch."""
        if artifact.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"segments.json schema {artifact.get('schema_version')} != {SCHEMA_VERSION}; "
                "re-run train.py"
            )
        if artifact.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError(
                "segments.json feature names don't match app.ml.features; re-run train.py"
            )
        centroids = np.asarray(artifact["centroids"], dtype=float)
        scale = np.asarray(artifact["feature_scale"], dtype=float)
        personas = sorted(artifact["personas"], key=lambda p: p["id"])
        if centroids.shape != (len(personas), len(FEATURE_NAMES)) or scale.shape != (
            len(FEATURE_NAMES),
        ):
            raise RuntimeError("segments.json centroids/feature_scale have unexpected shapes")
        return cls(centroids, scale, personas, artifact["model_version"])

    @classmethod
    def load(cls, model_dir: str) -> "Segmenter":
        path = os.path.join(model_dir, ARTIFACT_NAME)
        if not os.path.exists(path):
            raise RuntimeError(f"Required model artifact missing: {path}")
        with open(path) as f:
            segmenter = cls.from_artifact(json.load(f))
        logger.info(
            f"Segment model {segmenter.model_version} loaded ({len(segmenter.personas)} personas)."
        )
        return segmenter

    def nearest(self, X: np.ndarray) -> np.ndarray:
        """Index of the nearest centroid for each row (same rule as KMeans.predict).

        Takes unscaled vectors from features.encode; centroids live in the
        block-scaled space the model was trained in.
        """
        X = np.atleast_2d(X) * self.scale
        distances = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def predict(self, vector: np.ndarray) -> Dict[str, Any]:
        return self.personas[int(self.nearest(vector)[0])]


def personalize(persona: Dict[str, Any], codes: SegmentCodes) -> Dict[str, Any]:
    """Keep only the persona traits this user actually has.

    The nearest segment can still differ from the user on its defining
    traits (a 60+ listener landing in "Mid-career listeners"), and showing
    those, or prompting the LLM with them, would contradict their answers.
    """

    def has(feature: str) -> bool:
        field, code = feature.split(":", 1)
        return code in codes.get(field, set())

    shared = [t for t in persona["traits"] if has(t["feature"])]
    if shared:
        name = " · ".join(t["label"] for t in shared[:2])
        description = (
            f"You share the defining traits of a group that is {persona['share']:.0%} of surveyed "
            "listeners: "
            + ", ".join(f"{t['label'].lower()} ({t['lift']:.1f}× more common than average)" for t in shared)
            + "."
        )
    else:
        name = "Eclectic listener"
        description = (
            "Your answers don't line up with the defining traits of any one survey group, "
            "so your recommendations lean on your answers alone."
        )
    return {**persona, "name": name, "description": description, "traits": shared}
