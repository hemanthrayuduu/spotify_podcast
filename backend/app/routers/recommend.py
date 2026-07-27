"""Podcast recommendation endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.limiter import limiter
from app.ml.features import prepare_features
from app.schemas.recommendation import RecommendationResponse, UserPreferences
from app.services.llm import generate_podcast_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
@limiter.limit(settings.rate_limit)
async def recommend_podcasts(
    preferences: UserPreferences, request: Request
) -> RecommendationResponse:
    bundle = request.app.state.bundle
    client = request.app.state.anthropic_client
    prefs = preferences.model_dump()
    logger.info(f"Received recommendation request for age={prefs['age']}")

    try:
        user_features = prepare_features(bundle, prefs)
        segment_id = (await run_in_threadpool(bundle.kmeans_model.predict, user_features))[0]
        user_segment = bundle.segment_profiles.get(f"Segment_{segment_id}", {})

        recommendations = await generate_podcast_recommendations(
            client, prefs, user_segment, settings.anthropic_model
        )
        return RecommendationResponse(
            segment_profile=user_segment, recommendations=recommendations
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail="Error generating recommendations") from exc
