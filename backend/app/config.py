"""Application settings loaded from environment variables or a local .env file."""

from functools import lru_cache

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, model_validator
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
    qdrant_timeout_seconds: float = Field(default=5.0, gt=0, le=120)
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
    api_local_only: bool = Field(default=True, validation_alias="SCOUTX_API_LOCAL_ONLY")
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="SCOUTX_CORS_ORIGINS",
    )
    max_request_body_bytes: int = Field(
        default=1_048_576,
        ge=16_384,
        le=16_777_216,
        validation_alias="SCOUTX_MAX_REQUEST_BODY_BYTES",
    )
    parser_max_file_bytes: int = Field(
        default=536_870_912,
        ge=1_048_576,
        le=4_294_967_296,
        validation_alias="SCOUTX_PARSER_MAX_FILE_BYTES",
    )
    parser_subprocess_timeout_seconds: int = Field(
        default=300, ge=1, le=3600, validation_alias="SCOUTX_PARSER_TIMEOUT_SECONDS"
    )
    worker_concurrency: int = Field(
        default=2, ge=1, le=32, validation_alias="SCOUTX_WORKER_CONCURRENCY"
    )
    worker_task_soft_limit_seconds: int = Field(
        default=3300, ge=60, le=7200, validation_alias="SCOUTX_WORKER_TASK_SOFT_LIMIT_SECONDS"
    )
    worker_task_hard_limit_seconds: int = Field(
        default=3600, ge=60, le=7200, validation_alias="SCOUTX_WORKER_TASK_HARD_LIMIT_SECONDS"
    )
    otel_enabled: bool = Field(default=False, validation_alias="SCOUTX_OTEL_ENABLED")
    otel_exporter_endpoint: AnyHttpUrl | None = Field(
        default=None, validation_alias="SCOUTX_OTEL_EXPORTER_ENDPOINT"
    )
    sql_echo: bool = False
    db_pool_size: int = Field(default=5, ge=1, le=100)
    db_max_overflow: int = Field(default=10, ge=0, le=100)

    @model_validator(mode="after")
    def validate_worker_timeouts(self) -> "Settings":
        if self.worker_task_hard_limit_seconds <= self.worker_task_soft_limit_seconds:
            raise ValueError("worker hard time limit must exceed its soft time limit")
        if not self.api_local_only:
            raise ValueError(
                "SCOUTX_API_LOCAL_ONLY cannot be disabled until a network authentication provider exists"
            )
        if self.otel_enabled and self.otel_exporter_endpoint is None:
            raise ValueError(
                "SCOUTX_OTEL_EXPORTER_ENDPOINT is required when SCOUTX_OTEL_ENABLED is true"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object; clear the cache in tests after env changes."""
    return Settings()
