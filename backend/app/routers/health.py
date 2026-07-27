"""Root and health-check endpoints."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Welcome to the Spotify Podcast Recommender API"}


@router.get("/health")
async def health(request: Request):
    """Report readiness. Suitable for the Render health check."""
    models_loaded = getattr(request.app.state, "bundle", None) is not None
    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
    }
