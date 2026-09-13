from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from the local environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Evidence-First Contract & Document Risk Assistant"
    app_env: str = "development"
    app_version: str = "0.1.0"
    debug: bool = False
    max_upload_size_bytes: int = 25 * 1024 * 1024
    max_pdf_page_count: int = 250
    max_pdf_chunk_count: int = 5_000

    database_url: str

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()