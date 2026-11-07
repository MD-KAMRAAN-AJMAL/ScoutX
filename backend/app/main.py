"""FastAPI application entry point for the ScoutX API."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import feedback, files, indexing, roots, search

logger = logging.getLogger(__name__)
API_PREFIX = "/api/v1"


def _allowed_origins() -> list[str]:
    configured = os.getenv(
        "SCOUTX_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


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
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


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
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request did not match the expected schema.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API exception")
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


@app.get(f"{API_PREFIX}/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "scoutx-api"}


app.include_router(search.router, prefix=API_PREFIX)
app.include_router(files.router, prefix=API_PREFIX)
app.include_router(roots.router, prefix=API_PREFIX)
app.include_router(indexing.router, prefix=API_PREFIX)
app.include_router(feedback.router, prefix=API_PREFIX)
