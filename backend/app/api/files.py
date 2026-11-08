"""User-scoped file listing over scanner-owned PostgreSQL records."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import Path as APIPath
from pydantic import Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql.elements import ColumnElement

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.database.engine import get_db
from app.database.models import DocumentChunk, File, IndexRoot
from app.database.repositories import FileRepository, IndexRootRepository
from app.security.permissions import FilesystemPermissionError, resolve_file_under_root

router = APIRouter(prefix="/files", tags=["files"])
_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?<![\w:/])/(?!/)[^\s,;:'\"<>]+|(?<![\w])[A-Za-z]:\\[^\s,;:'\"<>]+"
)

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
    parser_status: str | None = None
    extraction_warnings: list[Any] = Field(default_factory=list)
    extraction_error: str | None = None
    extension: str | None = None
    version: int = 1
    last_indexed_at: datetime | None = None
    created_at: datetime | None = None


class FileSnippet(APIModel):
    chunk_id: uuid.UUID
    snippet: str
    modality: str
    content_version: int
    section_title: str | None = None
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    start_time_ms: int | None = None
    end_time_ms: int | None = None


class FileDetailResponse(APIModel):
    file: FileRecord
    snippets: list[FileSnippet] = Field(default_factory=list)


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


def normalize_file_status(db_status: str, extra_metadata: dict[str, Any]) -> FileStatus:
    parser_status = extra_metadata.get("parser_status")
    if (
        db_status == "indexed"
        and isinstance(parser_status, str)
        and parser_status in {"partial", "failed", "unsupported"}
    ):
        return cast(FileStatus, parser_status)
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


def _extraction_warnings(extra_metadata: dict[str, Any]) -> list[Any]:
    warnings = extra_metadata.get("extraction_warnings", [])
    return warnings if isinstance(warnings, list) else []


def _sanitize_metadata_paths(value: Any, file: File, root: IndexRoot | None) -> Any:
    if isinstance(value, str):
        safe_file_path = _display_path(file, root)
        result = value.replace(file.canonical_path, safe_file_path)
        root_path = getattr(root or getattr(file, "root", None), "canonical_path", None)
        if isinstance(root_path, str):
            root_label = getattr(root, "display_name", None) or Path(root_path).name
            result = result.replace(root_path, root_label)
        return _ABSOLUTE_PATH_PATTERN.sub("[local path]", result)
    if isinstance(value, list):
        return [_sanitize_metadata_paths(item, file, root) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_metadata_paths(item, file, root) for key, item in value.items()}
    return value


def _display_path(file: File, root: IndexRoot | None = None) -> str:
    canonical = Path(file.canonical_path)
    root_path = getattr(root or getattr(file, "root", None), "canonical_path", None)
    if isinstance(root_path, str):
        try:
            relative = canonical.relative_to(Path(root_path))
            if not relative.is_absolute() and ".." not in relative.parts:
                return relative.as_posix()
        except ValueError:
            pass
    return canonical.name


def _file_response(file: File, root: IndexRoot | None = None) -> FileRecord:
    metadata = file.extra_metadata if isinstance(file.extra_metadata, dict) else {}
    return FileRecord(
        id=file.id,
        root_id=file.root_id,
        name=file.filename,
        path=_display_path(file, root),
        mime_type=file.mime_type,
        size_bytes=file.size_bytes,
        modified_at=file.modified_at,
        indexing_status=normalize_file_status(file.indexing_status, metadata),
        parser_status=metadata.get("parser_status")
        if isinstance(metadata.get("parser_status"), str)
        else None,
        extraction_warnings=_sanitize_metadata_paths(_extraction_warnings(metadata), file, root),
        extraction_error=_sanitize_metadata_paths(_extraction_error(metadata), file, root),
        extension=getattr(file, "extension", None),
        version=getattr(file, "version", 1),
        last_indexed_at=getattr(file, "last_indexed_at", None),
        created_at=getattr(file, "created_at", None),
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
    filename_contains: str | None = Query(default=None, min_length=1, max_length=255),
    extension: list[str] | None = Query(default=None, max_length=32),
    indexing_status: FileStatus | None = None,
    modified_after: datetime | None = None,
    modified_before: datetime | None = None,
    sort_by: Literal["name", "modified_at", "size_bytes", "last_indexed_at"] = "name",
    sort_order: Literal["asc", "desc"] = "asc",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> FileListResponse:
    if modified_after and modified_before and modified_after > modified_before:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_date_range",
                "message": "modified_after must not be later than modified_before.",
            },
        )
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
    conditions.append(
        File.root_id.in_(
            select(IndexRoot.id).where(IndexRoot.user_id == user_id, IndexRoot.is_enabled.is_(True))
        )
    )
    if filename_contains:
        conditions.append(File.filename.ilike(f"%{filename_contains.strip()}%"))
    if modified_after:
        conditions.append(File.modified_at >= modified_after)
    if modified_before:
        conditions.append(File.modified_at <= modified_before)
    normalized_extensions = [item.strip().lower() for item in extension or []]
    normalized_extensions = [
        item if item.startswith(".") else f".{item}" for item in normalized_extensions
    ]
    if normalized_extensions:
        conditions.append(func.lower(File.extension).in_(normalized_extensions))
    if indexing_status:
        if indexing_status == "pending":
            conditions.append(File.indexing_status.in_(("pending", "queued")))
        elif indexing_status == "unsupported":
            conditions.append(File.indexing_status == "excluded")
        elif indexing_status == "partial":
            conditions.append(
                (File.indexing_status == "indexed")
                & or_(
                    File.extra_metadata["parser_status"].astext == "partial",
                    File.extra_metadata["extraction_status"].astext == "partial",
                    File.extra_metadata["partial"].astext == "true",
                )
            )
        else:
            conditions.append(File.indexing_status == indexing_status)
    total = session.scalar(select(func.count(File.id)).where(*conditions)) or 0
    sort_column = {
        "name": File.filename,
        "modified_at": File.modified_at,
        "size_bytes": File.size_bytes,
        "last_indexed_at": File.last_indexed_at,
    }[sort_by]
    order = (
        sort_column.desc().nullslast() if sort_order == "desc" else sort_column.asc().nullslast()
    )
    records = session.scalars(
        select(File)
        .options(joinedload(File.root))
        .where(*conditions)
        .order_by(order, File.id)
        .offset(offset)
        .limit(limit)
    ).all()
    readable_records = []
    for file in records:
        root = getattr(file, "root", None) or IndexRootRepository(session).get(
            user_id, file.root_id
        )
        if root is None or not root.is_enabled:
            continue
        try:
            resolve_file_under_root(file.canonical_path, root.canonical_path)
        except OSError:
            continue
        readable_records.append(file)
    return FileListResponse(
        items=[_file_response(file) for file in readable_records],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get(
    "/{file_id}",
    response_model=FileDetailResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
    },
)
def file_detail(
    file_id: uuid.UUID = APIPath(),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> FileDetailResponse:
    file = FileRepository(session).get(user_id, file_id)
    if file is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "file_not_found", "message": "The file was not found."},
        )
    root = IndexRootRepository(session).get(user_id, file.root_id)
    if root is None or not root.is_enabled:
        raise HTTPException(
            status_code=404,
            detail={"code": "file_not_found", "message": "The file was not found."},
        )
    try:
        resolve_file_under_root(file.canonical_path, root.canonical_path)
    except FilesystemPermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "file_inaccessible",
                "message": "ScoutX cannot currently read this file.",
            },
        ) from exc

    chunks = session.scalars(
        select(DocumentChunk)
        .join(File, File.id == DocumentChunk.file_id)
        .where(
            DocumentChunk.file_id == file.id,
            DocumentChunk.content_version == File.version,
            File.user_id == user_id,
            File.indexing_status == "indexed",
        )
        .order_by(DocumentChunk.chunk_index)
        .limit(30)
    ).all()
    return FileDetailResponse(
        file=_file_response(file, root),
        snippets=[
            FileSnippet(
                chunk_id=chunk.id,
                snippet=chunk.content[:1200],
                modality=chunk.modality,
                content_version=chunk.content_version,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                start_time_ms=chunk.start_time_ms,
                end_time_ms=chunk.end_time_ms,
            )
            for chunk in chunks
        ],
    )
