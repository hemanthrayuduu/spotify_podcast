"""Application configuration via pydantic-settings."""

import os
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    # Comma-separated in the environment; parsed into a list below.
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_dir: str = os.path.join(_BACKEND_DIR, "models")

    rate_limit: str = "10/minute"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()
