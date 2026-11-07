"""Idempotent filesystem inventory tasks.

Watchdog only publishes these jobs. It never hashes large files, mutates the
database, extracts document content, or creates embeddings itself.
"""

from __future__ import annotations

import logging
import os
import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from celery import Task
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.engine import SessionLocal
from app.database.models import DocumentChunk, File, IndexJob, IndexRoot
from app.database.repositories import IndexRootRepository
from app.ingestion.fingerprints import FileChangedDuringHashing
from app.ingestion.scanner import FileScanner
from app.security.permissions import RootUnavailableError, is_within, require_readable_directory
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="scoutx.discover_file",
    max_retries=3,
    autoretry_for=(SQLAlchemyError, FileChangedDuringHashing),
    retry_backoff=True,
    retry_jitter=True,
)
def discover_file(
    self: Task,
    user_id: str,
    root_id: str,
    path: str,
    *,
    old_path: str | None = None,
) -> dict[str, Any]:
    """Hash and upsert one created/modified/moved file, then queue processing."""
    owner_id = uuid.UUID(user_id)
    registered_root_id = uuid.UUID(root_id)
    with SessionLocal() as session:
        scanner = FileScanner(session)
        file, changed = scanner.discover_file(
            owner_id,
            registered_root_id,
            path,
            old_path=old_path,
            force_hash=True,
        )
        if file is None:
            session.rollback()
            return {"status": "ignored", "path": path}
        file_id, version = file.id, file.version
        if changed:
            file.indexing_status = "queued"
            _ensure_process_job(session, file, user_id=owner_id)
        session.commit()
    if changed:
        process_file.apply_async(
            args=[str(owner_id), str(file_id), version],
            task_id=f"process-{file_id}-{version}",
        )
    return {
        "status": "queued" if changed else "unchanged",
        "file_id": str(file_id),
        "version": version,
    }


@celery_app.task(
    bind=True,
    name="scoutx.process_file",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
)
def process_file(self: Task, user_id: str, file_id: str, version: int) -> dict[str, Any]:
    """Validate the durable processing request for downstream extraction workers.

    Content extraction and embedding are owned by later phases. This task leaves
    the PostgreSQL job queued for that processor instead of claiming the file was
    indexed when no extractor has run.
    """
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    with SessionLocal() as session:
        file = session.scalar(select(File).where(File.id == record_id, File.user_id == owner_id))
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.version != version:
            return {"status": "superseded", "file_id": file_id, "current_version": file.version}
        job = _find_process_job(session, file, owner_id)
        if job is None:
            return {"status": "missing_job", "file_id": file_id}
        return {
            "status": "queued_for_content_processor",
            "file_id": file_id,
            "version": version,
            "index_job_id": str(job.id),
        }


@celery_app.task(
    bind=True,
    name="scoutx.reindex_file",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
)
def reindex_file(self: Task, user_id: str, file_id: str) -> dict[str, Any]:
    """Queue an explicit reindex request for the current indexed file version."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    with SessionLocal() as session:
        file = session.scalar(select(File).where(File.id == record_id, File.user_id == owner_id))
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.indexing_status in {"queued", "processing"}:
            return {"status": "already_queued", "file_id": file_id, "version": file.version}
        file.indexing_status = "queued"
        job = _ensure_process_job(session, file, user_id=owner_id, force=True)
        version = file.version
        session.commit()
    process_file.apply_async(
        args=[str(owner_id), str(record_id), version],
        task_id=f"reindex-{record_id}-{version}-{job.id}",
    )
    return {"status": "queued", "file_id": file_id, "version": version}


@celery_app.task(
    bind=True,
    name="scoutx.delete_file",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
)
def delete_file(self: Task, user_id: str, root_id: str, path: str) -> dict[str, Any]:
    """Idempotently mark one removed path deleted and remove its stored chunks."""
    owner_id, registered_root_id = uuid.UUID(user_id), uuid.UUID(root_id)
    canonical_path = os.path.abspath(path)
    with SessionLocal() as session:
        root = IndexRootRepository(session).get(owner_id, registered_root_id)
        if root is None or not is_within(canonical_path, root.canonical_path):
            return {"status": "ignored", "path": canonical_path}
        try:
            mounted_root = require_readable_directory(root.canonical_path)
        except RootUnavailableError:
            return {"status": "root_unavailable", "path": canonical_path}
        expected_device = (root.settings or {}).get("filesystem_device_id")
        current_device = getattr(mounted_root.stat(), "st_dev", None)
        if expected_device is not None and current_device != expected_device:
            return {"status": "root_unavailable", "path": canonical_path}
        file = session.scalar(
            select(File).where(
                File.user_id == owner_id,
                File.root_id == registered_root_id,
                File.canonical_path == canonical_path,
            )
        )
        if file is None or file.indexing_status == "deleted":
            return {"status": "already_deleted", "path": canonical_path}
        file_id = file.id
        file.indexing_status = "deleted"
        session.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file_id))
        _cancel_file_jobs(session, owner_id, file_id, "File was deleted")
        session.commit()
    return {"status": "deleted", "file_id": str(file_id)}


@celery_app.task(
    bind=True,
    name="scoutx.reconcile_root",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
)
def reconcile_root(self: Task, user_id: str, root_id: str) -> dict[str, Any]:
    """Reconcile one registered root; an unavailable drive never implies deletion."""
    owner_id, registered_root_id = uuid.UUID(user_id), uuid.UUID(root_id)
    with SessionLocal() as session:
        scanner = FileScanner(session)
        try:
            report = scanner.reconcile_root(owner_id, registered_root_id)
        except RootUnavailableError as exc:
            session.rollback()
            logger.info("Skipping unavailable root %s: %s", registered_root_id, exc)
            return {"status": "root_unavailable", "root_id": str(registered_root_id)}
        process_requests: list[tuple[uuid.UUID, int]] = []
        for file_id in report.changed_file_ids:
            file = session.get(File, file_id)
            if file is None:
                continue
            file.indexing_status = "queued"
            _ensure_process_job(session, file, user_id=owner_id)
            process_requests.append((file.id, file.version))
        for path in report.missing_paths:
            file = session.scalar(
                select(File).where(
                    File.user_id == owner_id,
                    File.root_id == registered_root_id,
                    File.canonical_path == path,
                )
            )
            if file is not None and file.indexing_status != "deleted":
                file.indexing_status = "deleted"
                session.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file.id))
                _cancel_file_jobs(session, owner_id, file.id, "File was absent during reconciliation")
        queued_jobs = session.scalars(
            select(IndexJob).where(
                IndexJob.user_id == owner_id,
                IndexJob.root_id == registered_root_id,
                IndexJob.job_type == "process_file",
                IndexJob.status == "queued",
            )
        ).all()
        queued_versions = set(process_requests)
        for job in queued_jobs:
            file = session.get(File, job.file_id) if job.file_id else None
            version = job.payload.get("file_version")
            if (
                file is not None
                and file.indexing_status != "deleted"
                and version == file.version
                and (file.id, version) not in queued_versions
            ):
                process_requests.append((file.id, version))
                queued_versions.add((file.id, version))
        session.commit()
    for file_id, version in process_requests:
        process_file.apply_async(
            args=[str(owner_id), str(file_id), version],
            task_id=f"process-{file_id}-{version}",
        )
    return {
        "status": "reconciled",
        "root_id": str(registered_root_id),
        "discovered": report.discovered,
        "created": report.created,
        "changed": report.changed,
        "unchanged": report.unchanged,
        "deleted": len(report.missing_paths),
        "issues": len(report.issues),
    }


@celery_app.task(
    bind=True,
    name="scoutx.reconcile_enabled_roots",
    max_retries=3,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
)
def reconcile_enabled_roots(self: Task) -> dict[str, int]:
    """Fan out the periodic Celery Beat reconciliation over enabled root records."""
    with SessionLocal() as session:
        roots = session.execute(
            select(IndexRoot.user_id, IndexRoot.id).where(IndexRoot.is_enabled.is_(True))
        ).all()
    for owner_id, root_id in roots:
        reconcile_root.delay(str(owner_id), str(root_id))
    return {"scheduled_roots": len(roots)}


def _ensure_process_job(
    session: Session,
    file: File,
    *,
    user_id: uuid.UUID,
    force: bool = False,
) -> IndexJob:
    """Return an active same-version job or create one durable, repeatable request."""
    lock_material = f"{file.id}:{file.version}".encode("ascii")
    lock_key = int.from_bytes(hashlib.sha256(lock_material).digest()[:8], "big", signed=True)
    # Serialize the read-create pair per file version so duplicate Watchdog,
    # reconciliation, and retry deliveries cannot create parallel work records.
    session.execute(select(func.pg_advisory_xact_lock(lock_key)))
    jobs = session.scalars(
        select(IndexJob).where(
            IndexJob.user_id == user_id,
            IndexJob.file_id == file.id,
            IndexJob.job_type == "process_file",
        )
    ).all()
    for existing in jobs:
        if existing.payload.get("file_version") == file.version and existing.status in {"queued", "running"}:
            return existing
        if not force and existing.payload.get("file_version") == file.version and existing.status == "succeeded":
            return existing
    job = IndexJob(
        user_id=user_id,
        root_id=file.root_id,
        file_id=file.id,
        job_type="process_file",
        status="queued",
        payload={
            "file_version": file.version,
            "canonical_path": file.canonical_path,
            "requested_at": datetime.now(UTC).isoformat(),
            "force": force,
        },
    )
    session.add(job)
    session.flush()
    return job


def _find_process_job(session: Session, file: File, user_id: uuid.UUID) -> IndexJob | None:
    jobs = session.scalars(
        select(IndexJob)
        .where(
            IndexJob.user_id == user_id,
            IndexJob.file_id == file.id,
            IndexJob.job_type == "process_file",
        )
        .order_by(IndexJob.created_at.desc())
    ).all()
    return next((job for job in jobs if job.payload.get("file_version") == file.version), None)


def _cancel_file_jobs(
    session: Session, user_id: uuid.UUID, file_id: uuid.UUID, reason: str
) -> None:
    session.execute(
        update(IndexJob)
        .where(
            IndexJob.user_id == user_id,
            IndexJob.file_id == file_id,
            IndexJob.status.in_(("queued", "running")),
        )
        .values(status="cancelled", finished_at=datetime.now(UTC), error_message=reason)
    )

