"""FastAPI application entry point for the ScoutX API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis import Redis
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api import feedback, files, indexing, personalization, recommendations, roots, search
from app.config import get_settings
from app.database.engine import engine
from app.observability import (
    TelemetryRuntime,
    configure_structured_logging,
    configure_telemetry,
    install_correlation_log_context,
)
from app.security.api_middleware import APISafetyMiddleware
from app.vectorstore.qdrant_client import QdrantStore

logger = logging.getLogger(__name__)
API_PREFIX = "/api/v1"
_telemetry_runtime: TelemetryRuntime | None = None
install_correlation_log_context()
configure_structured_logging()


def _allowed_origins() -> list[str]:
    configured = get_settings().cors_origins
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    for origin in origins:
        try:
            parsed = urlsplit(origin)
            host = parsed.hostname
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("CORS origins must be valid loopback origins") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or host not in {"localhost", "127.0.0.1", "::1"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("CORS origins must use localhost or a loopback IP address")
    return origins


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Reserve application resources for initialization and cleanup.

    Database and worker resources are intentionally not initialized here until
    their shared contracts and lifecycle are supplied by the infrastructure phase.
    """
    logger.info("Starting ScoutX API")
    try:
        yield
    finally:
        if _telemetry_runtime is not None:
            _telemetry_runtime.tracer_provider.shutdown()
            _telemetry_runtime.meter_provider.shutdown()
        logger.info("Stopping ScoutX API")


app = FastAPI(
    title="ScoutX API",
    description="Local-first intelligent file recommendation and retrieval API.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(APISafetyMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail and "message" in detail:
        error = detail
    else:
        error = {"code": "http_error", "message": str(detail), "details": None}
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request did not match the expected schema.",
                "details": {
                    "fields": [
                        {
                            "location": [str(part) for part in error.get("loc", ())],
                            "type": str(error.get("type", "invalid")),
                        }
                        for error in exc.errors()
                    ]
                },
            }
        },
    )


@app.exception_handler(SQLAlchemyError)
async def database_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    logger.error(
        "Database request failed",
        extra={
            "component": "api",
            "operation": "database",
            "error_type": type(exc).__name__,
            "request_id": getattr(request.state, "correlation_id", None),
        },
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "database_unavailable",
                "message": "The database is temporarily unavailable. Please retry.",
                "details": None,
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled API exception",
        extra={
            "component": "api",
            "operation": "request",
            "error_type": type(exc).__name__,
            "request_id": getattr(request.state, "correlation_id", None),
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected server error occurred.",
                "details": None,
            }
        },
    )


class HealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, str]


def _check_postgres() -> str:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return "available"


def _check_qdrant() -> str:
    store = QdrantStore()
    try:
        store.client.get_collections()
        return "available"
    finally:
        close = getattr(store.client, "close", None)
        if close:
            close()


def _check_redis() -> str:
    client = Redis.from_url(
        str(get_settings().redis_url), socket_connect_timeout=2, socket_timeout=2
    )
    try:
        client.ping()
        return "available"
    finally:
        client.close()


def _check_ollama() -> str:
    import httpx

    response = httpx.get(f"{str(get_settings().ollama_base_url).rstrip('/')}/api/tags", timeout=2)
    response.raise_for_status()
    return "available"


def _probe(check) -> str:
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 - a health probe must degrade, not fail the request
        logger.info(
            "A health dependency check failed",
            extra={
                "component": "health",
                "dependency": check.__name__,
                "error_type": type(exc).__name__,
            },
        )
        return "unavailable"


@app.get(f"{API_PREFIX}/health", tags=["health"], response_model=HealthResponse)
async def health() -> HealthResponse:
    checks = await asyncio.gather(
        *(
            asyncio.to_thread(_probe, check)
            for check in (_check_postgres, _check_qdrant, _check_redis, _check_ollama)
        )
    )
    dependencies = dict(zip(("postgres", "qdrant", "redis", "ollama"), checks, strict=True))
    return HealthResponse(
        status="ok" if all(value == "available" for value in dependencies.values()) else "degraded",
        service="scoutx-api",
        dependencies=dependencies,
    )


@app.get(f"{API_PREFIX}/live", tags=["health"], response_model=dict[str, str])
async def liveness() -> dict[str, str]:
    """Cheap process liveness check that does not contact dependencies."""
    return {"status": "ok", "service": "scoutx-api"}


@app.get(f"{API_PREFIX}/ready", tags=["health"], response_model=HealthResponse)
async def readiness() -> HealthResponse:
    """Readiness check; PostgreSQL and Qdrant are required, other services optional."""
    result = await health()
    if any(result.dependencies.get(service) != "available" for service in ("postgres", "qdrant")):
        raise HTTPException(
            status_code=503,
            detail={
                "code": "dependencies_not_ready",
                "message": "Core ScoutX search dependencies are temporarily unavailable.",
                "details": result.dependencies,
            },
        )
    return result


app.include_router(search.router, prefix=API_PREFIX)
app.include_router(files.router, prefix=API_PREFIX)
app.include_router(roots.router, prefix=API_PREFIX)
app.include_router(indexing.router, prefix=API_PREFIX)
app.add_api_route(
    f"{API_PREFIX}/indexing/status",
    indexing.indexing_status,
    methods=["GET"],
    response_model=indexing.IndexStatusResponse,
    tags=["indexing"],
)
app.include_router(feedback.router, prefix=API_PREFIX)
app.include_router(personalization.router, prefix=API_PREFIX)
app.include_router(recommendations.router, prefix=API_PREFIX)
_telemetry_runtime = configure_telemetry(app)
