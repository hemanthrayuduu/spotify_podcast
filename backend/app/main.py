"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.ml.loader import load_model_bundle
from app.routers import health, recommend

logger = logging.getLogger(__name__)


def _init_anthropic_client() -> Optional[anthropic.AsyncAnthropic]:
    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set; recommendations will use fallback mode.")
        return None
    logger.info(f"Anthropic client initialized (model={settings.anthropic_model}).")
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key, timeout=30.0, max_retries=2
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loud here: if artifacts are missing/corrupt, startup raises.
    app.state.bundle = load_model_bundle(settings.model_dir)
    app.state.anthropic_client = _init_anthropic_client()
    yield


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Spotify Podcast Recommender API",
        description="Recommends podcasts based on user preferences and a KMeans listening segment.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(recommend.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
