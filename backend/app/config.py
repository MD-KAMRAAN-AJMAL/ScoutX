"""Application settings loaded from environment variables or a local .env file."""

from functools import lru_cache

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ScoutX API"
    environment: str = "development"
    database_url: PostgresDsn = Field(default="postgresql+psycopg://scoutx@localhost:5432/scoutx")
    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "file_content_chunks"
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    embedding_device: str = Field(
        default="auto", pattern="^(auto|cpu|cuda|mps)$", validation_alias="SCOUTX_EMBEDDING_DEVICE"
    )
    embedding_cpu_threads: int = Field(
        default=4, ge=1, le=256, validation_alias="SCOUTX_EMBEDDING_CPU_THREADS"
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=1024, validation_alias="SCOUTX_EMBEDDING_BATCH_SIZE"
    )
    excluded_paths: list[str] = Field(default_factory=list)
    sql_echo: bool = False
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object; clear the cache in tests after env changes."""
    return Settings()
