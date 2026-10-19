"""Root and health-check endpoints."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Welcome to the Spotify Podcast Recommender API"}


@router.get("/health")
async def health(request: Request):
    """Report readiness. Suitable for the Render health check."""
    segmenter = getattr(request.app.state, "segmenter", None)
    return {
        "status": "ok" if segmenter is not None else "degraded",
        "models_loaded": segmenter is not None,
        "segment_model_version": segmenter.model_version if segmenter else None,
        "llm": getattr(request.app.state, "llm_status", None),
        "podcast_index": {
            "configured": getattr(getattr(request.app.state, "recommender", None), "podcast_index", None)
            is not None
        },
    }
