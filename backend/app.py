"""Spotify Podcast Recommender API.

Assigns a user to a KMeans listening segment, then asks Claude for 5
personalized podcast recommendations based on that segment and the user's
stated preferences.
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import anthropic
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from llm_integration import generate_podcast_recommendations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- Configuration -----------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# Comma-separated list of allowed browser origins. Defaults to local dev; set
# ALLOWED_ORIGINS in the environment for production (the deployed frontend URL).
_default_origins = "http://localhost:3000,http://127.0.0.1:3000"
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

AGE_MAP = {"18-24": 21, "25-34": 30, "35-44": 40, "45-54": 50, "55+": 60}


# --- Model loading (fail loud) ----------------------------------------------


def _load_models() -> Dict[str, Any]:
    """Load all ML artifacts, raising if any are missing or corrupt.

    We fail loudly at startup rather than silently degrading to fallback
    recommendations, so a broken deploy is visible immediately instead of
    quietly serving low-quality results forever.
    """
    required_pickles = {
        "kmeans_model": "kmeans_model.pkl",
        "scaler": "scaler.pkl",
        "valid_features": "valid_features.pkl",
    }
    artifacts: Dict[str, Any] = {}
    for key, filename in required_pickles.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            raise RuntimeError(f"Required model artifact missing: {path}")
        with open(path, "rb") as f:
            artifacts[key] = pickle.load(f)

    segment_path = os.path.join(MODEL_DIR, "segment_profiles.json")
    if not os.path.exists(segment_path):
        raise RuntimeError(f"Required model artifact missing: {segment_path}")
    with open(segment_path, "r") as f:
        artifacts["segment_profiles"] = json.load(f)

    logger.info("Models and segment profiles loaded successfully.")
    return artifacts


_models = _load_models()
kmeans_model = _models["kmeans_model"]
scaler = _models["scaler"]
valid_features = _models["valid_features"]
segment_profiles = _models["segment_profiles"]


# --- Anthropic client --------------------------------------------------------


def _init_anthropic_client() -> Optional[anthropic.AsyncAnthropic]:
    """Create the async Anthropic client, or None if no API key is set."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set; recommendations will use fallback mode.")
        return None
    logger.info(f"Anthropic client initialized (model={ANTHROPIC_MODEL}).")
    return anthropic.AsyncAnthropic(api_key=api_key, timeout=30.0, max_retries=2)


anthropic_client = _init_anthropic_client()


# --- Schemas -----------------------------------------------------------------


class UserPreferences(BaseModel):
    """Request body for /recommend.

    Fields split by how they are used downstream:
      * Segment matching (ML feature vector): age, music_genre,
        podcast_frequency, podcast_duration, podcast_format, podcast_content.
      * AI personalization only (Claude prompt): content_language, region,
        listening_mood, podcasts_enjoyed.
    """

    # --- Used for KMeans segment matching ---
    age: str
    music_genre: List[str] = Field(..., description="At least one favorite music genre")
    podcast_frequency: str
    podcast_duration: str
    podcast_format: str
    podcast_content: List[str] = Field(..., description="At least one podcast content interest")

    # --- Used for AI personalization only ---
    content_language: str
    region: str
    listening_mood: str
    podcasts_enjoyed: str = ""

    @field_validator("music_genre", "podcast_content", mode="before")
    @classmethod
    def coerce_to_list(cls, v: Any) -> Any:
        """Accept a JSON string or comma-separated string as a list."""
        if v is None:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            if "," in v:
                return [item.strip() for item in v.split(",") if item.strip()]
            return [v] if v else []
        return v

    @field_validator("music_genre", "podcast_content")
    @classmethod
    def not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("must contain at least one item")
        return v


class Recommendation(BaseModel):
    name: str
    creator: str
    description: str
    format: str
    duration: str
    language: str
    region: str
    reason: str
    link: str


class RecommendationResponse(BaseModel):
    segment_profile: Dict[str, Any]
    recommendations: List[Recommendation]


# --- Feature preparation -----------------------------------------------------


def prepare_features(preferences: Dict[str, Any]) -> np.ndarray:
    """Map user preferences to the scaled one-hot feature vector."""
    features: Dict[str, int] = {}
    features["age_numeric"] = AGE_MAP.get(preferences["age"], 30)

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
    return scaler.transform(feature_vector)


# --- App ---------------------------------------------------------------------

app = FastAPI(
    title="Spotify Podcast Recommender API",
    description="Recommends podcasts based on user preferences and a KMeans listening segment.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to the Spotify Podcast Recommender API"}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_podcasts(preferences: UserPreferences) -> RecommendationResponse:
    prefs = preferences.model_dump()
    logger.info(f"Received recommendation request for age={prefs['age']}")

    try:
        user_features = prepare_features(prefs)
        segment_id = (await run_in_threadpool(kmeans_model.predict, user_features))[0]
        user_segment = segment_profiles.get(f"Segment_{segment_id}", {})

        recommendations = await generate_podcast_recommendations(
            anthropic_client, prefs, user_segment, ANTHROPIC_MODEL
        )
        return RecommendationResponse(
            segment_profile=user_segment, recommendations=recommendations
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail="Error generating recommendations") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
