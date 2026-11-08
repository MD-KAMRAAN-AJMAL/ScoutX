"""User-scoped file listing over scanner-owned PostgreSQL records."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.models import File
from app.database.repositories import FileRepository, IndexRootRepository

router = APIRouter(prefix="/files", tags=["files"])

FileStatus = Literal[
    "pending",
    "processing",
    "indexed",
    "partial",
    "failed",
    "unsupported",
    "deleted",
]


class FileRecord(APIModel):
    id: uuid.UUID
    root_id: uuid.UUID
    name: str
    path: str
    mime_type: str | None = None
    size_bytes: int = Field(ge=0)
    modified_at: datetime | None = None
    indexing_status: FileStatus
    extraction_error: str | None = None


class FileListResponse(APIModel):
    items: list[FileRecord]
    limit: int
    offset: int
    total: int


class FileStatusCounts(APIModel):
    pending: int = 0
    processing: int = 0
    indexed: int = 0
    partial: int = 0
    failed: int = 0
    unsupported: int = 0
    deleted: int = 0


def _normalized_status(db_status: str, extra_metadata: dict[str, Any]) -> FileStatus:
    extraction_state = extra_metadata.get("extraction_status")
    if db_status == "excluded":
        return "unsupported"
    if db_status == "queued":
        return "pending"
    if db_status == "indexed" and (
        extraction_state == "partial"
        or extra_metadata.get("partial") is True
        or extra_metadata.get("extraction_error")
    ):
        return "partial"
    if db_status in {"pending", "processing", "indexed", "failed", "deleted"}:
        return cast(FileStatus, db_status)
    # Unknown legacy values should remain visible while being treated as queued.
    return "pending"


def _extraction_error(extra_metadata: dict[str, Any]) -> str | None:
    value = extra_metadata.get("extraction_error") or extra_metadata.get("error")
    return value if isinstance(value, str) else None


def _file_response(file: File) -> FileRecord:
    metadata = file.extra_metadata if isinstance(file.extra_metadata, dict) else {}
    return FileRecord(
        id=file.id,
        root_id=file.root_id,
        name=file.filename,
        path=file.canonical_path,
        mime_type=file.mime_type,
        size_bytes=file.size_bytes,
        modified_at=file.modified_at,
        indexing_status=_normalized_status(file.indexing_status, metadata),
        extraction_error=_extraction_error(metadata),
    )


def _file_conditions(user_id: uuid.UUID, root_id: uuid.UUID | None) -> list[ColumnElement[bool]]:
    conditions = [File.user_id == user_id]
    if root_id is not None:
        conditions.append(File.root_id == root_id)
    return conditions


@router.get(
    "",
    response_model=FileListResponse,
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def list_files(
    root_id: uuid.UUID | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> FileListResponse:
    if root_id is not None and IndexRootRepository(session).get(user_id, root_id) is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "root_not_found",
                "message": "The registered directory was not found.",
                "details": None,
            },
        )

    conditions = _file_conditions(user_id, root_id)
    total = session.scalar(select(func.count(File.id)).where(*conditions)) or 0
    if root_id is not None:
        records = FileRepository(session).list_for_root(
            user_id, root_id, limit=limit, offset=offset
        )
    else:
        records = session.scalars(
            select(File)
            .where(*conditions)
            .order_by(File.canonical_path, File.id)
            .offset(offset)
            .limit(limit)
        ).all()
    return FileListResponse(
        items=[_file_response(file) for file in records],
        limit=limit,
        offset=offset,
        total=total,
    )
