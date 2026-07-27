"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from groq import AsyncGroq
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.limiter import limiter
from app.core.logging import configure_logging
from app.ml.loader import load_model_bundle
from app.routers import health, recommend

logger = logging.getLogger(__name__)


def _init_llm_client() -> Optional[AsyncGroq]:
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY not set; recommendations will use fallback mode.")
        return None
    logger.info(f"Groq client initialized (model={settings.groq_model}).")
    return AsyncGroq(api_key=settings.groq_api_key, timeout=30.0, max_retries=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail loud here: if artifacts are missing/corrupt, startup raises.
    app.state.bundle = load_model_bundle(settings.model_dir)
    app.state.llm_client = _init_llm_client()
    yield


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Spotify Podcast Recommender API",
        description="Recommends podcasts based on user preferences and a KMeans listening segment.",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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
