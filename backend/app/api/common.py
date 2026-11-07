"""Shared API response schemas and explicit unavailable response helper."""

from typing import Any, NoReturn

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorBody(APIModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(APIModel):
    error: ErrorBody


def not_implemented(capability: str) -> NoReturn:
    raise HTTPException(
        status_code=501,
        detail={
            "code": "not_implemented",
            "message": f"{capability} is not available in this application phase.",
            "details": None,
        },
    )
