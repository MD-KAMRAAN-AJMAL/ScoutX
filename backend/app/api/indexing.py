"""Indexing status and reconciliation task submission endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.common import APIModel, ErrorResponse, get_local_user_id
from app.api.files import FileStatusCounts, normalize_file_status
from app.database.engine import get_db
from app.database.models import File, IndexJob
from app.database.repositories import IndexJobRepository, IndexRootRepository

router = APIRouter(prefix="/index", tags=["indexing"])


class IndexJobProgress(APIModel):
    id: uuid.UUID
    job_type: str
    status: str
    root_id: uuid.UUID | None = None
    file_id: uuid.UUID | None = None
    attempts: int = Field(ge=0)
    progress_percent: int | None = Field(default=None, ge=0, le=100)
    stage: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class IndexStatusResponse(APIModel):
    state: str
    files_discovered: int
    files_indexed: int
    files_failed: int
    status_counts: FileStatusCounts
    roots_registered: int
    last_reconciled_at: datetime | None = None
    recent_jobs: list[IndexJobProgress] = Field(default_factory=list)
    recent_job_counts: dict[str, int] = Field(default_factory=dict)


class ReconcileRequest(APIModel):
    root_id: uuid.UUID | None = None


class ReconcileResponse(APIModel):
    accepted: bool
    root_ids: list[uuid.UUID]
    task_ids: list[str | None]
    submitted_at: datetime


class ReconciliationDispatcher(Protocol):
    """Application integration point for the scanner's background task queue."""

    def __call__(self, user_id: uuid.UUID, root_id: uuid.UUID) -> str | None: ...


STATUS_KEYS = (
    "pending",
    "processing",
    "indexed",
    "partial",
    "failed",
    "unsupported",
    "deleted",
)


def _api_status(db_status: str, metadata: dict) -> str:
    return normalize_file_status(db_status, metadata)


def _job_progress(job: IndexJob) -> IndexJobProgress:
    payload = job.payload
    payload = payload if isinstance(payload, dict) else {}
    raw_percent = payload.get("progress_percent")
    progress_percent = (
        raw_percent
        if isinstance(raw_percent, int)
        and not isinstance(raw_percent, bool)
        and 0 <= raw_percent <= 100
        else None
    )
    if job.status == "succeeded":
        progress_percent = 100
    raw_stage = payload.get("stage")
    return IndexJobProgress(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        root_id=job.root_id,
        file_id=job.file_id,
        attempts=job.attempts,
        progress_percent=progress_percent,
        stage=raw_stage if isinstance(raw_stage, str) else None,
        error_message=job.error_message[:1000] if job.error_message else None,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _dispatch_celery_reconciliation(user_id: uuid.UUID, root_id: uuid.UUID) -> str | None:
    """Publish through Person A's existing reconcile task and Redis broker."""
    from app.workers.tasks import reconcile_root

    return reconcile_root.delay(str(user_id), str(root_id)).id


@router.get(
    "/status",
    response_model=IndexStatusResponse,
    responses={401: {"model": ErrorResponse}},
)
def indexing_status(
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> IndexStatusResponse:
    roots = IndexRootRepository(session).list_for_user(user_id)
    counts = dict.fromkeys(STATUS_KEYS, 0)
    file_rows = session.execute(
        select(File.indexing_status, File.extra_metadata).where(File.user_id == user_id)
    ).all()
    for db_status, metadata in file_rows:
        details = metadata if isinstance(metadata, dict) else {}
        counts[_api_status(db_status, details)] += 1

    active = counts["pending"] + counts["processing"]
    state = (
        "processing"
        if counts["processing"]
        else "pending"
        if active
        else "failed"
        if counts["failed"]
        else "partial"
        if counts["partial"]
        else "idle"
    )
    last_reconciled = max(
        (root.last_reconciled_at for root in roots if root.last_reconciled_at is not None),
        default=None,
    )
    jobs = IndexJobRepository(session).list_for_user(user_id, limit=100)
    job_counts = {status: 0 for status in ("queued", "running", "succeeded", "failed", "cancelled")}
    for job in jobs:
        if job.status in job_counts:
            job_counts[job.status] += 1
    return IndexStatusResponse(
        state=state,
        files_discovered=sum(counts.values()) - counts["deleted"],
        files_indexed=counts["indexed"],
        files_failed=counts["failed"],
        status_counts=FileStatusCounts(**counts),
        roots_registered=sum(root.is_enabled for root in roots),
        last_reconciled_at=last_reconciled,
        recent_jobs=[_job_progress(job) for job in jobs[:10]],
        recent_job_counts=job_counts,
    )


@router.post(
    "/reconcile",
    response_model=ReconcileResponse,
    status_code=202,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def reconcile(
    request: Request,
    body: ReconcileRequest = Body(default_factory=ReconcileRequest),
    session: Session = Depends(get_db),
    user_id: uuid.UUID = Depends(get_local_user_id),
) -> ReconcileResponse:
    repository = IndexRootRepository(session)
    if body.root_id is not None:
        root = repository.get(user_id, body.root_id)
        if root is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "root_not_found",
                    "message": "The registered directory was not found.",
                },
            )
        if not root.is_enabled:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "root_disabled",
                    "message": "Re-enable this directory before reconciling it.",
                },
            )
        roots = [root]
    else:
        roots = repository.list_for_user(user_id, enabled_only=True)

    if not roots:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "no_registered_roots",
                "message": "Register a directory before reconciling.",
            },
        )

    dispatcher: ReconciliationDispatcher | None = getattr(
        request.app.state, "reconciliation_dispatcher", _dispatch_celery_reconciliation
    )
    if dispatcher is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "indexing_service_unavailable",
                "message": "The filesystem indexing worker is not connected yet.",
            },
        )

    try:
        task_ids = [dispatcher(user_id, root.id) for root in roots]
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "reconciliation_enqueue_failed",
                "message": "The reconciliation job could not be queued. Please retry.",
            },
        ) from exc

    return ReconcileResponse(
        accepted=True,
        root_ids=[root.id for root in roots],
        task_ids=task_ids,
        submitted_at=datetime.now(UTC),
    )
