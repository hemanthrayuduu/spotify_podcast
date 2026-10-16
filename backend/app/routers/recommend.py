"""Podcast recommendation endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.limiter import limiter
from app.core.vocabulary import to_labels
from app.ml.features import codes_from_prefs, encode
from app.schemas.recommendation import RecommendationResponse, UserPreferences
from app.services.llm import generate_podcast_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
@limiter.limit(settings.rate_limit)
async def recommend_podcasts(
    preferences: UserPreferences, request: Request
) -> RecommendationResponse:
    segmenter = request.app.state.segmenter
    client = request.app.state.llm_client
    prefs = preferences.model_dump()
    logger.info(f"Received recommendation request for age={prefs['age']}")

    try:
        persona = segmenter.predict(encode(codes_from_prefs(prefs)))
        recommendations = await generate_podcast_recommendations(
            client, to_labels(prefs), persona, settings.groq_model
        )
        return RecommendationResponse(persona=persona, recommendations=recommendations)
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail="Error generating recommendations") from exc
