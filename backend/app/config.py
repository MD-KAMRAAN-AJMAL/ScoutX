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
    ollama_base_url: AnyHttpUrl = Field(
        default="http://localhost:11434", validation_alias="SCOUTX_OLLAMA_BASE_URL"
    )
    ollama_model: str = Field(default="qwen3:8b", validation_alias="SCOUTX_OLLAMA_MODEL")
    ollama_timeout_seconds: float = Field(
        default=20.0, gt=0, le=300, validation_alias="SCOUTX_OLLAMA_TIMEOUT_SECONDS"
    )
    embedding_device: str = Field(
        default="auto", pattern="^(auto|cpu|cuda|mps)$", validation_alias="SCOUTX_EMBEDDING_DEVICE"
    )
    embedding_cpu_threads: int = Field(
        default=4, ge=1, le=256, validation_alias="SCOUTX_EMBEDDING_CPU_THREADS"
    )
    embedding_batch_size: int = Field(
        default=32, ge=1, le=1024, validation_alias="SCOUTX_EMBEDDING_BATCH_SIZE"
    )
    reranker_enabled: bool = Field(default=True, validation_alias="SCOUTX_RERANKER_ENABLED")
    reranker_model: str = Field(
        default="Qwen/Qwen3-Reranker-0.6B", validation_alias="SCOUTX_RERANKER_MODEL"
    )
    reranker_local_files_only: bool = Field(
        default=False, validation_alias="SCOUTX_RERANKER_LOCAL_FILES_ONLY"
    )
    reranker_candidate_limit: int = Field(
        default=50, ge=1, le=500, validation_alias="SCOUTX_RERANKER_CANDIDATE_LIMIT"
    )
    reranker_top_n: int = Field(default=20, ge=1, le=100, validation_alias="SCOUTX_RERANKER_TOP_N")
    reranker_batch_size: int = Field(
        default=8, ge=1, le=128, validation_alias="SCOUTX_RERANKER_BATCH_SIZE"
    )
    reranker_max_length: int = Field(
        default=2048, ge=128, le=8192, validation_alias="SCOUTX_RERANKER_MAX_LENGTH"
    )
    reranker_fallback_enabled: bool = Field(
        default=True, validation_alias="SCOUTX_RERANKER_FALLBACK_ENABLED"
    )
    rag_context_token_budget: int = Field(
        default=6000, ge=256, le=32000, validation_alias="SCOUTX_RAG_CONTEXT_TOKEN_BUDGET"
    )
    rag_min_confidence: float = Field(
        default=0.08, ge=0.0, le=1.0, validation_alias="SCOUTX_RAG_MIN_CONFIDENCE"
    )
    excluded_paths: list[str] = Field(default_factory=list)
    sql_echo: bool = False
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object; clear the cache in tests after env changes."""
    return Settings()
