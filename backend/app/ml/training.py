"""Train the listener segmentation model from the survey CSV.

Pipeline: clean every raw survey answer into a vocabulary code → encode with
the same ``app.ml.features.encode`` used at serving time → KMeans for each k
in a range → keep the k with the best silhouette score (subject to a minimum
segment size) → describe each segment by its most *over-represented* answers
(lift vs the whole survey), not its most common ones, which are the same in
every segment.

Training-only dependencies (pandas, scikit-learn) live in
requirements-train.txt; the API never imports this module.
"""

import hashlib
import math
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.vocabulary import FIELDS, SEGMENT_FIELDS
from app.ml.features import FEATURE_NAMES, SegmentCodes, encode
from app.ml.segmenter import SCHEMA_VERSION

RANDOM_STATE = 42
MIN_SEGMENT_SHARE = 0.05
SILHOUETTE_SAMPLE = 5000
# A trait must describe at least this share of the segment, and be this much
# more common than in the whole survey, to count as distinguishing.
MIN_TRAIT_SUPPORT = 0.2
MIN_TRAIT_LIFT = 1.15
MAX_TRAITS = 3

SURVEY_COLUMNS = {
    "age": "Age",
    "music_genre": "fav_music_genre",
    "podcast_content": "fav_pod_genre",
    "podcast_frequency": "pod_lis_frequency",
    "podcast_format": "preffered_pod_format",
    "podcast_duration": "preffered_pod_duration",
}

# Every distinct raw answer (stripped, casefolded) → vocabulary codes. Junk or
# ambiguous answers map to no codes explicitly, so an answer we've never seen
# fails training instead of being silently dropped.
RAW_TO_CODES: Dict[str, Dict[str, Set[str]]] = {
    "age": {
        "6~12": {"6-12"},
        "12~20": {"12-20"},
        "20~35": {"20-35"},
        "35~60": {"35-60"},
        "60+": {"60+"},
    },
    "music_genre": {
        "melody": {"melody"},
        "rap": {"rap"},
        "pop": {"pop"},
        "rock": {"rock"},
        "classical": {"classical"},
        "jazz": {"jazz"},
        "electronic/dance": {"electronic"},
        "kpop": {"kpop"},
        "old songs": {"oldies"},
        "classical & melody, dance": {"classical", "melody", "electronic"},
        "trending songs random": set(),
        "all": set(),
    },
    "podcast_content": {
        "lifestyle and health": {"lifestyle"},
        "comedy": {"comedy"},
        "sports": {"sports"},
        "technology": {"technology"},
        "health and fitness": {"health_fitness"},
        "food and cooking": {"food"},
        "novels": {"fiction"},
        "stories": {"fiction"},
        "self help": {"self_help"},
        "murder mystery": {"true_crime"},
        "business": {"business"},
        "finance related and current affairs": {"business", "news_politics"},
        "political, informative, topics that interests me": {"news_politics"},
        "spiritual and devotional": {"spirituality"},
        "general knowledge": {"education"},
        "informative stuff": {"education"},
        "educational": {"education"},
        "dance and relevant cases": set(),
        "everything": set(),
        "none": set(),
    },
    "podcast_frequency": {
        "daily": {"daily"},
        "several times a week": {"several_week"},
        "once a week": {"weekly"},
        "rarely": {"rarely"},
        "never": {"never"},
    },
    "podcast_format": {
        "story telling": {"storytelling"},
        "interview": {"interview"},
        "discussion": {"discussion"},
        "conversational": {"conversational"},
        "educational": {"educational"},
        "none": set(),
    },
    "podcast_duration": {
        "short (less than 15 minutes)": {"short"},
        "shorter": {"short"},
        "medium (15-30 minutes)": {"medium"},
        "long (more than 30 minutes)": {"long"},
        "longer": {"long"},
        "both": set(),
        "none": set(),
    },
}

# How each code reads as a persona trait.
TRAIT_PHRASES: Dict[str, Dict[str, str]] = {
    "age": {
        "6-12": "Under-12s", "12-20": "Teens", "20-35": "Twenty-somethings",
        "35-60": "Mid-career listeners", "60+": "Listeners 60+",
    },
    "music_genre": {
        "pop": "Pop fans", "rock": "Rock fans", "rap": "Hip-hop heads", "melody": "Melody lovers",
        "classical": "Classical ears", "jazz": "Jazz fans", "electronic": "Dance-music fans",
        "kpop": "K-pop fans", "oldies": "Retro-music fans",
    },
    "podcast_content": {
        "comedy": "Comedy seekers", "true_crime": "True-crime sleuths", "sports": "Sports fans",
        "technology": "Tech enthusiasts", "lifestyle": "Wellness seekers",
        "health_fitness": "Fitness buffs", "food": "Foodies", "business": "Business minds",
        "news_politics": "News followers", "self_help": "Self-improvers",
        "spirituality": "Spiritual seekers", "education": "Lifelong learners",
        "fiction": "Story lovers",
    },
    "podcast_frequency": {
        "daily": "Daily listeners", "several_week": "Frequent listeners",
        "weekly": "Weekly listeners", "rarely": "Occasional listeners",
        "never": "Podcast newcomers",
    },
    "podcast_format": {
        "storytelling": "Narrative fans", "interview": "Interview fans",
        "discussion": "Panel-show fans", "conversational": "Banter fans",
        "educational": "Explainer fans",
    },
    "podcast_duration": {
        "short": "Quick-listen fans", "medium": "Mid-length fans", "long": "Long-form fans",
    },
}


def load_survey(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()
    return df[list(SURVEY_COLUMNS.values())]


def _key(raw: Any) -> str:
    return str(raw).strip().casefold()


def _is_missing(raw: Any) -> bool:
    return raw is None or (isinstance(raw, float) and math.isnan(raw))


def clean_value(field: str, raw: Any) -> Set[str]:
    """Raw survey answer → set of vocabulary codes (empty = no answer)."""
    if _is_missing(raw):
        return set()
    try:
        return set(RAW_TO_CODES[field][_key(raw)])
    except KeyError:
        raise ValueError(f"unmapped {field} survey value {raw!r}; add it to RAW_TO_CODES") from None


def unmapped_values(df: pd.DataFrame) -> Dict[str, List[str]]:
    missing: Dict[str, List[str]] = {}
    for field, column in SURVEY_COLUMNS.items():
        values = {v for v in df[column].dropna().unique() if _key(v) not in RAW_TO_CODES[field]}
        if values:
            missing[field] = sorted(values)
    return missing


def survey_codes(df: pd.DataFrame) -> List[SegmentCodes]:
    rows = df[[SURVEY_COLUMNS[f] for f in SEGMENT_FIELDS]].itertuples(index=False, name=None)
    return [{f: clean_value(f, raw) for f, raw in zip(SEGMENT_FIELDS, row)} for row in rows]


def block_scale(X: np.ndarray) -> np.ndarray:
    """Per-feature multipliers giving every survey field equal total variance.

    Without this the never-missing single-select fields (listening frequency)
    dominate the distance and KMeans just re-discovers that one answer.
    """
    scale = np.ones(X.shape[1])
    for field in SEGMENT_FIELDS:
        idx = [i for i, n in enumerate(FEATURE_NAMES) if n == field or n.startswith(field + ":")]
        variance = X[:, idx].var(axis=0).sum()
        if variance > 0:
            scale[idx] = 1.0 / np.sqrt(variance)
    return scale


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)


def _indicators(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Boolean column per trait feature (age is split back into its buckets)."""
    out: Dict[str, np.ndarray] = {}
    age_codes = FIELDS["age"].segment_codes
    age_idx = np.rint(X[:, FEATURE_NAMES.index("age")] * (len(age_codes) - 1)).astype(int)
    for i, code in enumerate(age_codes):
        out[f"age:{code}"] = age_idx == i
    for j, name in enumerate(FEATURE_NAMES):
        if name != "age":
            out[name] = X[:, j] > 0
    return out


def _phrase(feature: str) -> str:
    field, code = feature.split(":", 1)
    return TRAIT_PHRASES[field][code]


def _traits(mask: np.ndarray, indicators: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    candidates = []
    for feature, col in indicators.items():
        overall = col.mean()
        if overall == 0:
            continue
        share = col[mask].mean()
        lift = share / overall
        candidates.append((lift, share, feature))
    candidates.sort(reverse=True)
    strong = [c for c in candidates if c[1] >= MIN_TRAIT_SUPPORT and c[0] >= MIN_TRAIT_LIFT]
    # One trait per field reads better than three flavours of the same answer.
    chosen, fields_seen = [], set()
    for lift, share, feature in strong or [c for c in candidates if c[0] > 1][:1]:
        field = feature.split(":", 1)[0]
        if field in fields_seen:
            continue
        fields_seen.add(field)
        chosen.append(
            {"feature": feature, "label": _phrase(feature), "share": round(float(share), 3),
             "lift": round(float(lift), 2)}
        )
        if len(chosen) == MAX_TRAITS:
            break
    return chosen


def _personas(X: np.ndarray, labels: np.ndarray, k: int) -> List[Dict[str, Any]]:
    indicators = _indicators(X)
    personas, names = [], set()
    for segment in range(k):
        mask = labels == segment
        traits = _traits(mask, indicators)
        name = " · ".join(t["label"] for t in traits[:2])
        if name in names:
            name = " · ".join(t["label"] for t in traits)
        names.add(name)
        share = float(mask.mean())
        description = (
            f"{share:.0%} of surveyed listeners. Compared with the average respondent, "
            "this group over-indexes on: "
            + ", ".join(f"{t['label'].lower()} ({t['lift']:.1f}×)" for t in traits)
            + "."
        )
        personas.append(
            {"id": segment, "name": name, "description": description, "traits": traits,
             "share": round(share, 4)}
        )
    return personas


def build_artifact(
    X: np.ndarray,
    km: KMeans,
    scale: np.ndarray,
    data_sha256: str,
    silhouette_by_k: Dict[int, float],
) -> Dict[str, Any]:
    """``km`` was fit on ``X * scale``; ``X`` is unscaled (for persona stats)."""
    k = int(km.n_clusters)
    shares = np.bincount(km.labels_, minlength=k) / len(km.labels_)
    feature_hash = hashlib.sha256("|".join(FEATURE_NAMES).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": f"{data_sha256[:8]}-k{k}-{feature_hash[:6]}",
        "feature_names": list(FEATURE_NAMES),
        "feature_scale": scale.tolist(),
        # Rounded: KMeans' parallel reductions differ run-to-run in the last ulp,
        # which would make the artifact non-reproducible byte-for-byte.
        "centroids": np.round(km.cluster_centers_, 10).tolist(),
        "personas": _personas(X, km.labels_, k),
        "metrics": {
            "k": k,
            "silhouette_by_k": {str(kk): round(float(s), 4) for kk, s in silhouette_by_k.items()},
            "min_segment_share": round(float(shares.min()), 4),
        },
        "training": {
            "data_sha256": data_sha256,
            "rows": int(len(X)),
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
        },
    }


def train(data_path: str, k_range: Iterable[int] = range(3, 9)) -> Dict[str, Any]:
    with open(data_path, "rb") as f:
        data_sha256 = hashlib.sha256(f.read()).hexdigest()
    df = load_survey(data_path)
    X = np.vstack([encode(codes) for codes in survey_codes(df)])
    scale = block_scale(X)
    Xs = X * scale

    silhouette_by_k: Dict[int, float] = {}
    best = None
    for k in k_range:
        km = fit_kmeans(Xs, k)
        score = silhouette_score(
            Xs, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(X)), random_state=RANDOM_STATE
        )
        silhouette_by_k[k] = float(score)
        min_share = np.bincount(km.labels_, minlength=k).min() / len(X)
        if min_share >= MIN_SEGMENT_SHARE and (best is None or score > silhouette_by_k[best.n_clusters]):
            best = km
    if best is None:
        raise RuntimeError("no k produced segments above the minimum size")
    return build_artifact(X, best, scale, data_sha256, silhouette_by_k)
