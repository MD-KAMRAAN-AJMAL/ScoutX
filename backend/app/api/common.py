"""Shared API response schemas and explicit unavailable response helper."""

import os
import uuid
from typing import Any, NoReturn

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.engine import get_db
from app.database.models import User
from app.database.repositories import UserRepository


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ErrorBody(APIModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(APIModel):
    error: ErrorBody


LOCAL_USER_ID = uuid.UUID(
    os.getenv("SCOUTX_LOCAL_USER_ID", "00000000-0000-4000-8000-000000000001")
)


def get_local_user_id(
    request: Request, session: Session = Depends(get_db)
) -> uuid.UUID:
    """Use middleware identity, with a development-only local workspace fallback."""
    authenticated_id = getattr(request.state, "user_id", None)
    if authenticated_id is not None:
        try:
            return uuid.UUID(str(authenticated_id))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=401,
                detail={"code": "invalid_user_context", "message": "The authenticated user context is invalid."},
            ) from exc

    if get_settings().environment.lower() != "development":
        raise HTTPException(
            status_code=401,
            detail={"code": "authentication_required", "message": "An authenticated user context is required."},
        )

    repository = UserRepository(session)
    if repository.get(LOCAL_USER_ID) is None:
        session.add(User(id=LOCAL_USER_ID, display_name="Local user"))
        session.commit()
    return LOCAL_USER_ID


def not_implemented(capability: str) -> NoReturn:
    raise HTTPException(
        status_code=501,
        detail={
            "code": "not_implemented",
            "message": f"{capability} is not available in this application phase.",
            "details": None,
        },
    )
