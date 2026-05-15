"""
Application Configuration

Environment-based configuration using pydantic-settings.
Loads from .env file or environment variables.
"""

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required:
        HF_MODEL_ID: HuggingFace model repository ID
        HF_TOKEN: HuggingFace API token (for private repos)

    Optional:
        MODEL_CACHE_DIR: Local directory for caching downloaded models
        DEBUG: Enable debug logging
        ALLOWED_ORIGINS: CORS allowed origins (comma-separated)
    """

    HF_MODEL_ID: str = 'DanielTobi0/climate-rnn-model'
    HF_TOKEN: str  # Required for private repository

    MODEL_CACHE_DIR: Path = Path('./model_cache')

    DEBUG: bool = False
    ALLOWED_ORIGINS: str = '*'  # Comma-separated list or "*" for all

    APP_NAME: str = 'Climate Forecasting API'
    APP_VERSION: str = '1.0.0'

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse ALLOWED_ORIGINS into a list."""
        if self.ALLOWED_ORIGINS == '*':
            return ['*']
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',') if origin.strip()]


# Global settings instance
settings = Settings()
