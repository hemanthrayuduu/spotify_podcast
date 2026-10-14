"""Encode segment answers as the feature vector the KMeans model was trained on.

Training (app.ml.training) and serving both go through ``encode``, so the two
can't disagree about feature names or values.

Layout, in ``FEATURE_NAMES`` order:
  * ``age``: survey age bucket position, scaled to [0, 1]
  * multi-select fields: multi-hot, normalized to sum to 1, so someone who
    ticks many genres doesn't outweigh someone who ticks one
  * single-select fields: one-hot
An unanswered field (or a "No preference"/LLM-only option) encodes as all
zeros, the same as an unanswered survey question.
"""

from typing import Any, Dict, List, Set

import numpy as np

from app.core.vocabulary import FIELDS, SEGMENT_FIELDS, option

SegmentCodes = Dict[str, Set[str]]


def _feature_names() -> List[str]:
    names: List[str] = []
    for name in SEGMENT_FIELDS:
        field = FIELDS[name]
        if field.ordinal:
            names.append(name)
        else:
            names.extend(f"{name}:{code}" for code in field.segment_codes)
    return names


FEATURE_NAMES: List[str] = _feature_names()
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def codes_from_prefs(prefs: Dict[str, Any]) -> SegmentCodes:
    """Map request option values to survey codes, dropping LLM-only options."""
    codes: SegmentCodes = {}
    for name in SEGMENT_FIELDS:
        raw = prefs.get(name)
        values = raw if isinstance(raw, list) else [raw] if raw else []
        codes[name] = {c for c in (option(name, v).code for v in values) if c}
    return codes


def encode(codes: SegmentCodes) -> np.ndarray:
    vec = np.zeros(len(FEATURE_NAMES), dtype=float)
    for name in SEGMENT_FIELDS:
        field = FIELDS[name]
        selected = codes.get(name) or set()
        if not selected:
            continue
        if field.ordinal:
            (code,) = selected
            vec[_INDEX[name]] = field.segment_codes.index(code) / (len(field.segment_codes) - 1)
            continue
        weight = 1.0 / len(selected) if field.multi else 1.0
        for code in selected:
            vec[_INDEX[f"{name}:{code}"]] = weight
    return vec
