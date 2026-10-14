"""Application configuration via pydantic-settings."""

import os
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    # protected_namespaces=() avoids a false-positive warning on `model_dir`
    # (pydantic reserves the `model_` prefix by default).
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    groq_api_key: str = ""
    # llama-3.3-70b-versatile was retired by Groq on 2026-08-16; gpt-oss-120b is
    # Groq's recommended open-weight replacement.
    groq_model: str = "openai/gpt-oss-120b"

    # Comma-separated in the environment; parsed into a list below.
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_dir: str = os.path.join(_BACKEND_DIR, "models")

    rate_limit: str = "10/minute"

    # Apple iTunes Search API (~20 calls/min/IP): cap concurrency, cache hard.
    itunes_timeout_s: float = 8.0
    itunes_max_concurrency: int = 4
    catalog_cache_ttl_s: int = 6 * 3600
    response_cache_ttl_s: int = 600

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
