"""Podcast recommendation endpoint."""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.core.limiter import limiter
from app.schemas.recommendation import RecommendationResponse, UserPreferences
from app.services.catalog import CatalogError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
@limiter.limit(settings.rate_limit)
async def recommend_podcasts(preferences: UserPreferences, request: Request):
    prefs = preferences.model_dump()
    logger.info(f"Recommendation request: topics={prefs['podcast_content']} age={prefs['age']}")
    try:
        return await request.app.state.recommender.recommend(prefs)
    except CatalogError as exc:
        logger.error(f"Catalog unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="The podcast catalog is unavailable right now. Please try again in a minute.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        logger.exception("Error generating recommendations")
        raise HTTPException(status_code=500, detail="Error generating recommendations") from exc
