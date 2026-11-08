"""Idempotent filesystem inventory tasks.

Watchdog only publishes these jobs. It never hashes large files, mutates the
database, extracts document content, or creates embeddings itself.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
from celery import Task
from kombu.exceptions import OperationalError as BrokerOperationalError
from opentelemetry import trace
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.engine import SessionLocal
from app.database.models import DocumentChunk, File, IndexJob, IndexRoot
from app.database.repositories import IndexRootRepository
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import DocumentChunker
from app.ingestion.fingerprints import FileChangedDuringHashing
from app.ingestion.normalized import ExtractionStatus
from app.ingestion.parser_registry import get_default_registry
from app.ingestion.scanner import FileScanner
from app.security.permissions import (
    FilesystemPermissionError,
    RootUnavailableError,
    is_within,
    require_readable_directory,
    resolve_file_under_root,
)
from app.vectorstore.indexing import prepare_point, verify_point_batch
from app.vectorstore.qdrant_client import get_qdrant_store
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)
TRANSIENT_STORAGE_ERRORS = (
    SQLAlchemyError,
    ApiException,
    ResponseHandlingException,
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
)


class IndexingTask(Task):
    """Mark an indexing request failed only after Celery exhausts retries."""

    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        try:
            user_id, file_id, version = args[:3]
            _record_pipeline_error(
                uuid.UUID(str(user_id)),
                uuid.UUID(str(file_id)),
                int(version),
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        except (ValueError, TypeError, IndexError):
            logger.exception("Could not record terminal indexing failure for task %s", task_id)
        return super().on_failure(exc, task_id, args, kwargs, einfo)


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
    base=IndexingTask,
    name="scoutx.process_file",
    max_retries=3,
    autoretry_for=(SQLAlchemyError, BrokerOperationalError),
    retry_backoff=True,
    retry_jitter=True,
)
def process_file(self: Task, user_id: str, file_id: str, version: int) -> dict[str, Any]:
    """Start the durable extraction, chunking and vector-indexing pipeline."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    with SessionLocal() as session:
        file = session.scalar(
            select(File).where(File.id == record_id, File.user_id == owner_id).with_for_update()
        )
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.version != version:
            return {"status": "superseded", "file_id": file_id, "current_version": file.version}
        job = _find_process_job(session, file, owner_id)
        if job is None:
            return {"status": "missing_job", "file_id": file_id}
        settings = file.extra_metadata if isinstance(file.extra_metadata, dict) else {}
        if (
            job.status == "succeeded"
            and settings.get("active_content_version") == version
            and settings.get("active_embedding_model") == EmbeddingService().fingerprint
        ):
            return {"status": "already_indexed", "file_id": file_id, "version": version}
        file.indexing_status = "processing"
        job.status = "running"
        job.started_at = job.started_at or datetime.now(UTC)
        job.finished_at = None
        session.commit()
    create_chunks.apply_async(
        args=[str(owner_id), str(record_id), version],
        task_id=f"chunks-{record_id}-{version}",
    )
    return {"status": "processing", "file_id": file_id, "version": version}


@celery_app.task(
    bind=True,
    base=IndexingTask,
    name="scoutx.create_chunks",
    max_retries=3,
    autoretry_for=(SQLAlchemyError, OSError),
    retry_backoff=True,
    retry_jitter=True,
)
def create_chunks(self: Task, user_id: str, file_id: str, version: int) -> dict[str, Any]:
    """Extract and persist deterministic PostgreSQL chunks for one file version."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    with SessionLocal() as session:
        file = session.scalar(select(File).where(File.id == record_id, File.user_id == owner_id))
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.version != version:
            return {"status": "superseded", "file_id": file_id, "current_version": file.version}
        existing_count = (
            session.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(
                    DocumentChunk.file_id == record_id,
                    DocumentChunk.content_version == version,
                )
            )
            or 0
        )
        root = session.scalar(
            select(IndexRoot).where(
                IndexRoot.id == file.root_id,
                IndexRoot.user_id == owner_id,
                IndexRoot.is_enabled.is_(True),
            )
        )
        if root is None:
            job = _find_process_job(session, file, owner_id)
            if job is not None:
                job.status = "cancelled"
                job.finished_at = datetime.now(UTC)
                job.error_message = "Authorized folder was disabled before extraction"
            session.commit()
            return {"status": "root_disabled", "file_id": file_id}
        try:
            path = str(resolve_file_under_root(file.canonical_path, root.canonical_path))
        except RootUnavailableError:
            file.indexing_status = "queued"
            job = _find_process_job(session, file, owner_id)
            if job is not None:
                job.status = "queued"
                job.started_at = None
                job.finished_at = None
                job.error_message = None
            session.commit()
            return {"status": "root_unavailable", "file_id": file_id}
        except FilesystemPermissionError:
            logger.warning(
                "Refusing to parse a file that is no longer safely accessible",
                extra={"file_id": file_id, "root_id": str(root.id)},
            )
            file.indexing_status = "failed"
            file.extra_metadata = {
                **(file.extra_metadata or {}),
                "filesystem_access_error": "File is no longer safely accessible under its authorized root",
            }
            job = _find_process_job(session, file, owner_id)
            if job is not None:
                job.status = "failed"
                job.finished_at = datetime.now(UTC)
                job.error_message = "File is no longer safely accessible under its authorized root"
            session.commit()
            return {"status": "inaccessible", "file_id": file_id}
        mime_type = file.mime_type
    if existing_count:
        generate_embeddings.delay(str(owner_id), str(record_id), version)
        return {"status": "chunks_reused", "file_id": file_id, "chunks": existing_count}

    tracer = trace.get_tracer("scoutx.ingestion")
    with tracer.start_as_current_span("scoutx.ingestion.extract") as span:
        extracted = get_default_registry().parse(path, record_id, version, mime_type)
        span.set_attribute("scoutx.ingestion.extraction_status", extracted.status.value)
        span.set_attribute("scoutx.ingestion.block_count", len(extracted.blocks))
    if extracted.status in {ExtractionStatus.FAILED, ExtractionStatus.UNSUPPORTED}:
        _record_pipeline_error(
            owner_id,
            record_id,
            version,
            extracted.status.value,
            extracted.error or "; ".join(extracted.warnings),
        )
        return {"status": extracted.status.value, "file_id": file_id, "error": extracted.error}
    with tracer.start_as_current_span("scoutx.ingestion.chunk") as span:
        chunks = DocumentChunker().chunk(extracted)
        span.set_attribute("scoutx.ingestion.chunk_count", len(chunks))
    if not chunks:
        _record_pipeline_error(
            owner_id, record_id, version, "failed", "Extraction returned no chunkable content"
        )
        return {
            "status": "failed",
            "file_id": file_id,
            "error": "Extraction returned no chunkable content",
        }

    with SessionLocal() as session:
        file = session.scalar(
            select(File).where(File.id == record_id, File.user_id == owner_id).with_for_update()
        )
        if file is None or file.indexing_status == "deleted" or file.version != version:
            session.rollback()
            return {"status": "superseded", "file_id": file_id}
        # A second delivery may have parsed concurrently. The file-row lock makes
        # replacement for this exact version deterministic and never touches an
        # older active version.
        session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.file_id == record_id,
                DocumentChunk.content_version == version,
            )
        )
        session.add_all(
            DocumentChunk(
                id=chunk.id,
                file_id=record_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                modality=chunk.modality,
                language=chunk.language,
                page_number=chunk.page_number,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                start_time_ms=chunk.start_time_ms,
                end_time_ms=chunk.end_time_ms,
                section_title=chunk.section_title,
                content_version=version,
                chunk_metadata=chunk.metadata,
            )
            for chunk in chunks
        )
        metadata = dict(file.extra_metadata or {})
        metadata.update(
            {
                "extraction_status": extracted.status.value,
                "extraction_parser": extracted.parser_name,
                "extraction_warnings": extracted.warnings,
                "extraction_error": extracted.error,
                "chunk_count": len(chunks),
            }
        )
        file.extra_metadata = metadata
        file.indexing_status = "processing"
        session.commit()
    generate_embeddings.delay(str(owner_id), str(record_id), version)
    return {
        "status": "chunks_created",
        "file_id": file_id,
        "version": version,
        "chunks": len(chunks),
    }


@celery_app.task(
    bind=True,
    base=IndexingTask,
    name="scoutx.generate_embeddings",
    max_retries=3,
    autoretry_for=TRANSIENT_STORAGE_ERRORS,
    retry_backoff=True,
    retry_jitter=True,
)
def generate_embeddings(self: Task, user_id: str, file_id: str, version: int) -> dict[str, Any]:
    """Embed missing chunks in batches and dispatch verified Qdrant upserts."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    with SessionLocal() as session:
        file = session.scalar(select(File).where(File.id == record_id, File.user_id == owner_id))
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.version != version:
            return {"status": "superseded", "file_id": file_id, "current_version": file.version}
        chunks = session.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.file_id == record_id,
                DocumentChunk.content_version == version,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
        if not chunks:
            raise RuntimeError("No PostgreSQL chunks exist for embedding")
        file_snapshot = file

    service = EmbeddingService()
    store = get_qdrant_store()
    prepared = [(chunk, *prepare_point(chunk, file_snapshot, service)) for chunk in chunks]
    stored = store.existing_fingerprints(owner_id, [point_id for _, point_id, _, _ in prepared])
    pending = []
    expected_existing: dict[UUID, str] = {}
    for chunk, point_id, fingerprint, payload in prepared:
        if stored.get(point_id) == fingerprint:
            expected_existing[point_id] = fingerprint
            store.update_point_payload(owner_id, point_id, payload)
        else:
            pending.append((chunk, point_id, fingerprint))
    if expected_existing:
        if not verify_point_batch(stored, expected_existing):
            raise RuntimeError("Existing Qdrant vectors did not match their content fingerprints")
        pointer_rows = [
            (chunk.id, point_id, fingerprint)
            for chunk, point_id, fingerprint, _payload in prepared
            if point_id in expected_existing
        ]
        _save_chunk_embedding_pointers(
            owner_id, record_id, version, pointer_rows, service.fingerprint
        )

    cursor = 0
    for _texts, vectors in service.embed_documents(
        chunk.content for chunk, _point_id, _fingerprint in pending
    ):
        batch = pending[cursor : cursor + len(vectors)]
        cursor += len(vectors)
        payloads = [
            {
                "chunk_id": str(chunk.id),
                "point_id": str(point_id),
                "embedding_fingerprint": fingerprint,
                "embedding_model_fingerprint": service.fingerprint,
                "vector": vector,
            }
            for (chunk, point_id, fingerprint), vector in zip(batch, vectors, strict=True)
        ]
        upsert_result = upsert_vectors.apply(
            args=[str(owner_id), str(record_id), version, payloads]
        )
        upsert_result.get(propagate=True)
    cleanup_old_version.delay(str(owner_id), str(record_id), version)
    return {
        "status": "vectors_ready",
        "file_id": file_id,
        "version": version,
        "embedded": len(pending),
        "reused": len(expected_existing),
    }


@celery_app.task(
    bind=True,
    base=IndexingTask,
    name="scoutx.upsert_vectors",
    max_retries=3,
    autoretry_for=TRANSIENT_STORAGE_ERRORS,
    retry_backoff=True,
    retry_jitter=True,
)
def upsert_vectors(
    self: Task,
    user_id: str,
    file_id: str,
    version: int,
    vectors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Idempotently upsert one bounded vector batch and persist its chunk links."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    if not vectors:
        return {"status": "empty", "upserted": 0}
    chunk_ids = [uuid.UUID(item["chunk_id"]) for item in vectors]
    with SessionLocal() as session:
        file = session.scalar(
            select(File).where(File.id == record_id, File.user_id == owner_id).with_for_update()
        )
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "upserted": 0}
        if file.version != version:
            return {"status": "superseded", "upserted": 0}
        chunks = session.scalars(
            select(DocumentChunk).where(
                DocumentChunk.id.in_(chunk_ids),
                DocumentChunk.file_id == record_id,
                DocumentChunk.content_version == version,
            )
        ).all()
        by_id = {chunk.id: chunk for chunk in chunks}
        service = EmbeddingService()
        point_payloads = []
        point_records = []
        for item in vectors:
            chunk_id = uuid.UUID(item["chunk_id"])
            chunk = by_id.get(chunk_id)
            if chunk is None:
                raise LookupError(f"Chunk {chunk_id} is not owned by this file/version")
            point_id, fingerprint, payload = prepare_point(chunk, file, service)
            if (
                point_id != uuid.UUID(item["point_id"])
                or fingerprint != item["embedding_fingerprint"]
            ):
                raise ValueError(
                    "Vector payload does not match current chunk content/model fingerprint"
                )
            if item["embedding_model_fingerprint"] != service.fingerprint:
                raise ValueError("Vector payload was generated by a different embedding model")
            point_payloads.append({"id": str(point_id), "payload": payload})
            point_records.append((chunk, point_id, fingerprint))
        vector_values = [[float(value) for value in item["vector"]] for item in vectors]
        store = get_qdrant_store()
        point_ids = [point_id for _chunk, point_id, _fingerprint in point_records]
        active_flags = store.existing_active_flags(owner_id, point_ids)
        store.upsert_chunks(owner_id, point_payloads, vector_values, active_flags=active_flags)
        persisted = store.existing_fingerprints(owner_id, point_ids)
        expected = {point_id: fingerprint for _, point_id, fingerprint in point_records}
        if not verify_point_batch(persisted, expected):
            raise RuntimeError("Qdrant did not return the expected vector batch after upsert")
        for chunk, point_id, fingerprint in point_records:
            stored_chunk = session.get(DocumentChunk, chunk.id)
            if stored_chunk is None or stored_chunk.content_version != version:
                continue
            stored_chunk.qdrant_point_id = point_id
            metadata = dict(stored_chunk.chunk_metadata or {})
            metadata["embedding_fingerprint"] = fingerprint
            metadata["embedding_model_fingerprint"] = service.fingerprint
            stored_chunk.chunk_metadata = metadata
        session.commit()
    return {"status": "upserted", "upserted": len(point_records)}


@celery_app.task(
    bind=True,
    base=IndexingTask,
    name="scoutx.cleanup_old_version",
    max_retries=5,
    autoretry_for=TRANSIENT_STORAGE_ERRORS,
    retry_backoff=True,
    retry_jitter=True,
)
def cleanup_old_version(self: Task, user_id: str, file_id: str, version: int) -> dict[str, Any]:
    """Verify a full replacement, activate it, then remove stale PostgreSQL/Qdrant data."""
    owner_id, record_id = uuid.UUID(user_id), uuid.UUID(file_id)
    service = EmbeddingService()
    store = get_qdrant_store()
    with SessionLocal() as session:
        file = session.scalar(
            select(File).where(File.id == record_id, File.user_id == owner_id).with_for_update()
        )
        if file is None or file.indexing_status == "deleted":
            return {"status": "obsolete", "file_id": file_id}
        if file.version != version:
            return {"status": "superseded", "file_id": file_id, "current_version": file.version}
        chunks = session.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.file_id == record_id,
                DocumentChunk.content_version == version,
            )
            .order_by(DocumentChunk.chunk_index)
        ).all()
        if not chunks or any(chunk.qdrant_point_id is None for chunk in chunks):
            raise RuntimeError(
                "Cannot activate a version until every PostgreSQL chunk has a Qdrant point"
            )
        expected = {
            chunk.qdrant_point_id: (chunk.chunk_metadata or {}).get("embedding_fingerprint")
            for chunk in chunks
        }
        if any(not value for value in expected.values()):
            raise RuntimeError("Embedding fingerprints are missing from PostgreSQL chunk records")
        stored = store.existing_fingerprints(owner_id, list(expected))
        if not verify_point_batch(stored, expected):
            raise RuntimeError("Replacement vector set is incomplete; retaining prior active index")

        # Keep the file row lock through Qdrant activation and the canonical PG
        # pointer update. Old points are not removed until this replacement is
        # complete and verified.
        store.activate_version(owner_id, record_id, version, list(expected))
        metadata = dict(file.extra_metadata or {})
        metadata["active_content_version"] = version
        metadata["active_embedding_model"] = service.fingerprint
        metadata["indexed_chunk_count"] = len(chunks)
        metadata.pop("indexing_retry_error", None)
        metadata.pop("extraction_error", None)
        file.extra_metadata = metadata
        file.indexing_status = "indexed"
        file.last_indexed_at = datetime.now(UTC)
        job = _find_process_job(session, file, owner_id)
        if job is not None and job.payload.get("file_version") == version:
            job.status = "succeeded"
            job.finished_at = datetime.now(UTC)
            job.error_message = None
        session.commit()

    # Deleting is deliberately after replacement verification and activation.
    store.delete_obsolete_file_points(owner_id, record_id, list(expected))
    with SessionLocal() as session:
        session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.file_id == record_id,
                DocumentChunk.content_version < version,
            )
        )
        session.commit()
    return {"status": "indexed", "file_id": file_id, "version": version, "chunks": len(chunks)}


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
    autoretry_for=TRANSIENT_STORAGE_ERRORS,
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
            select(File)
            .where(
                File.user_id == owner_id,
                File.root_id == registered_root_id,
                File.canonical_path == canonical_path,
            )
            .with_for_update()
        )
        if file is None or file.indexing_status == "deleted":
            return {"status": "already_deleted", "path": canonical_path}
        file_id = file.id
        get_qdrant_store().delete_file_points(owner_id, file_id)
        file.indexing_status = "deleted"
        session.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file_id))
        _cancel_file_jobs(session, owner_id, file_id, "File was deleted")
        session.commit()
    return {"status": "deleted", "file_id": str(file_id)}


@celery_app.task(
    bind=True,
    name="scoutx.reconcile_root",
    max_retries=3,
    autoretry_for=TRANSIENT_STORAGE_ERRORS,
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
        removed_file_ids: set[uuid.UUID] = set()
        for file_id in report.changed_file_ids:
            file = session.get(File, file_id)
            if file is None:
                continue
            file.indexing_status = "queued"
            _ensure_process_job(session, file, user_id=owner_id)
            process_requests.append((file.id, file.version))
        for path in report.missing_paths:
            file = session.scalar(
                select(File)
                .where(
                    File.user_id == owner_id,
                    File.root_id == registered_root_id,
                    File.canonical_path == path,
                )
                .with_for_update()
            )
            if file is not None and file.indexing_status != "deleted":
                file.indexing_status = "deleted"
                removed_file_ids.add(file.id)
                session.execute(delete(DocumentChunk).where(DocumentChunk.file_id == file.id))
                _cancel_file_jobs(
                    session, owner_id, file.id, "File was absent during reconciliation"
                )
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
        for removed_file_id in removed_file_ids:
            get_qdrant_store().delete_file_points(owner_id, removed_file_id)
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
        if existing.payload.get("file_version") == file.version and existing.status in {
            "queued",
            "running",
        }:
            return existing
        if (
            not force
            and existing.payload.get("file_version") == file.version
            and existing.status == "succeeded"
        ):
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


def _record_pipeline_error(
    user_id: uuid.UUID,
    file_id: uuid.UUID,
    version: int,
    status: str,
    error: str | None,
) -> None:
    """Record terminal extraction/index errors without invalidating an active index."""
    with SessionLocal() as session:
        file = session.scalar(
            select(File).where(File.id == file_id, File.user_id == user_id).with_for_update()
        )
        if file is None or file.version != version:
            return
        metadata = dict(file.extra_metadata or {})
        active_same_version = metadata.get("active_content_version") == version
        if error:
            metadata["indexing_retry_error" if active_same_version else "extraction_error"] = error[
                :4000
            ]
        if not active_same_version:
            metadata["extraction_status"] = status
        file.extra_metadata = metadata
        if not active_same_version:
            file.indexing_status = (
                "excluded" if status == ExtractionStatus.UNSUPPORTED.value else "failed"
            )
        job = _find_process_job(session, file, user_id)
        if job is not None and job.payload.get("file_version") == version:
            job.status = "succeeded" if status == ExtractionStatus.UNSUPPORTED.value else "failed"
            job.finished_at = datetime.now(UTC)
            job.error_message = error[:4000] if error else None
        session.commit()


def _save_chunk_embedding_pointers(
    user_id: uuid.UUID,
    file_id: uuid.UUID,
    version: int,
    pointers: list[tuple[uuid.UUID, uuid.UUID, str]],
    model_fingerprint: str,
) -> None:
    """Persist deterministic Qdrant pointers after confirming their DB owner."""
    if not pointers:
        return
    chunk_ids = [chunk_id for chunk_id, _point_id, _fingerprint in pointers]
    fingerprints = {
        chunk_id: (point_id, fingerprint) for chunk_id, point_id, fingerprint in pointers
    }
    with SessionLocal() as session:
        chunks = session.scalars(
            select(DocumentChunk)
            .join(File, File.id == DocumentChunk.file_id)
            .where(
                File.user_id == user_id,
                File.id == file_id,
                File.version == version,
                DocumentChunk.id.in_(chunk_ids),
                DocumentChunk.content_version == version,
            )
        ).all()
        for chunk in chunks:
            point_id, content_fingerprint = fingerprints[chunk.id]
            chunk.qdrant_point_id = point_id
            metadata = dict(chunk.chunk_metadata or {})
            metadata["embedding_fingerprint"] = content_fingerprint
            metadata["embedding_model_fingerprint"] = model_fingerprint
            chunk.chunk_metadata = metadata
        session.commit()
