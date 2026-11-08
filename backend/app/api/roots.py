"""Authorized filesystem root registration endpoints."""

from __future__ import annotations

import os
import stat
import uuid
from datetime import datetime
from pathlib import Path as FileSystemPath
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.models import IndexRoot
from app.database.repositories import IndexRootRepository
from app.database.schemas import IndexRootCreate

router = APIRouter(prefix="/roots", tags=["roots"])


class RootCreateRequest(APIModel):
    path: str = Field(min_length=1, max_length=4096)
    display_name: str | None = Field(default=None, max_length=255)


class RootRecord(APIModel):
    id: uuid.UUID
    path: str
    display_name: str | None = None
    status: Literal["enabled", "disabled"]
    last_reconciled_at: datetime | None = None


class RootListResponse(APIModel):
    items: list[RootRecord]


def _root_response(root: IndexRoot) -> RootRecord:
    return RootRecord(
        id=root.id,
        path=root.display_name or FileSystemPath(root.canonical_path).name,
        display_name=root.display_name,
        status="enabled" if root.is_enabled else "disabled",
        last_reconciled_at=root.last_reconciled_at,
    )


def _validate_directory(raw_path: str) -> str:
    requested = FileSystemPath(raw_path).expanduser()
    if not requested.is_absolute():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_root_path",
                "message": "Enter an absolute directory path.",
                "details": None,
            },
        )
    try:
        path = requested.resolve(strict=True)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "root_inaccessible",
                "message": "ScoutX does not have permission to access this directory.",
                "details": None,
            },
        ) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_root_path",
                "message": "The directory does not exist or cannot be resolved.",
                "details": None,
            },
        ) from exc

    try:
        is_directory = stat.S_ISDIR(path.stat().st_mode)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "root_inaccessible",
                "message": "ScoutX does not have permission to access this directory.",
                "details": None,
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_root_path",
                "message": "The directory could not be inspected.",
                "details": None,
            },
        ) from exc

    if not is_directory:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_root_path",
                "message": "The selected path is not a directory.",
                "details": None,
            },
        )

    try:
        # Opening the directory catches inaccessible roots before they are saved.
        with os.scandir(path):
            pass
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "root_inaccessible",
                "message": "ScoutX does not have permission to read this directory.",
                "details": None,
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_root_path",
                "message": "The directory could not be opened.",
                "details": None,
            },
        ) from exc
    return str(path)


@router.get(
    "",
    response_model=RootListResponse,
    responses={401: {"model": ErrorResponse}},
)
def list_roots(
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> RootListResponse:
    roots = IndexRootRepository(session).list_for_user(user_id)
    return RootListResponse(items=[_root_response(root) for root in roots])


@router.post(
    "",
    response_model=RootRecord,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
def register_root(
    body: RootCreateRequest,
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> RootRecord:
    canonical_path = _validate_directory(body.path)
    repository = IndexRootRepository(session)
    existing = repository.list_for_user(user_id)
    matching_root = next((root for root in existing if root.canonical_path == canonical_path), None)
    if matching_root is not None:
        if matching_root.is_enabled:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "root_already_registered",
                    "message": "This directory is already registered.",
                    "details": None,
                },
            )
        matching_root.is_enabled = True
        if body.display_name:
            matching_root.display_name = body.display_name
        session.commit()
        session.refresh(matching_root)
        return _root_response(matching_root)

    try:
        root = repository.create(
            user_id,
            IndexRootCreate(
                canonical_path=canonical_path,
                display_name=body.display_name,
            ),
        )
        session.commit()
        session.refresh(root)
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "root_already_registered",
                "message": "This directory is already registered.",
                "details": None,
            },
        ) from exc
    return _root_response(root)


@router.delete(
    "/{root_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def unregister_root(
    root_id: uuid.UUID = Path(),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> None:
    root = IndexRootRepository(session).get(user_id, root_id)
    if root is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "root_not_found",
                "message": "The registered directory was not found.",
                "details": None,
            },
        )

    # Keep discovered file metadata for history; disabling prevents future scans.
    root.is_enabled = False
    session.commit()
