"""Authorized recursive filesystem discovery and PostgreSQL reconciliation."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import File, IndexRoot
from app.database.repositories import IndexRootRepository
from app.database.schemas import IndexRootCreate
from app.ingestion.file_detector import FileDetector
from app.ingestion.fingerprints import (
    FileChangedDuringHashing,
    FileFingerprint,
    FileStat,
    content_version,
    fingerprint_file,
    stat_file,
)
from app.security.permissions import (
    FilesystemPermissionError,
    RootUnavailableError,
    canonicalize_authorized_directory,
    is_excluded_path,
    is_within,
    require_readable_directory,
    resolve_directory_under_root,
    resolve_file_under_root,
)


@dataclass(frozen=True, slots=True)
class ScanIssue:
    path: str
    message: str


@dataclass(slots=True)
class ReconcileReport:
    root_id: uuid.UUID
    discovered: int = 0
    created: int = 0
    changed: int = 0
    unchanged: int = 0
    missing_paths: list[str] = field(default_factory=list)
    changed_file_ids: list[uuid.UUID] = field(default_factory=list)
    issues: list[ScanIssue] = field(default_factory=list)
    root_available: bool = True


def register_directory(
    session: Session,
    user_id: uuid.UUID,
    path: str | os.PathLike[str],
    *,
    display_name: str | None = None,
    excluded_paths: Iterable[str | os.PathLike[str]] = (),
) -> IndexRoot:
    """Register a directory selected by the authenticated user after OS checks."""
    canonical = canonicalize_authorized_directory(path)
    configured = (*get_settings().excluded_paths, *tuple(excluded_paths))
    if is_excluded_path(canonical, configured_paths=configured):
        raise FilesystemPermissionError(f"Selected directory is excluded by policy: {canonical}")
    info = canonical.stat()
    settings = {
        "filesystem_device_id": getattr(info, "st_dev", None),
        "filesystem_inode": getattr(info, "st_ino", None),
    }
    roots = IndexRootRepository(session)
    existing = next(
        (root for root in roots.list_for_user(user_id) if root.canonical_path == str(canonical)),
        None,
    )
    if existing is not None:
        existing.is_enabled = True
        if display_name is not None:
            existing.display_name = display_name
        existing.settings = {**(existing.settings or {}), **settings}
        session.flush()
        return existing
    return roots.create(
        user_id,
        IndexRootCreate(
            canonical_path=str(canonical),
            display_name=display_name or canonical.name or str(canonical),
            settings=settings,
        ),
    )


class FileScanner:
    """Scan user-authorized roots and update inventory rows incrementally."""

    def __init__(
        self,
        session: Session,
        *,
        detector: FileDetector | None = None,
        excluded_paths: Iterable[str | os.PathLike[str]] = (),
        hash_chunk_size: int = 1024 * 1024,
    ) -> None:
        self.session = session
        self.detector = detector or FileDetector()
        self.excluded_paths = tuple(excluded_paths)
        self.hash_chunk_size = hash_chunk_size

    def scan_directory(self, path: str | os.PathLike[str]) -> Iterator[FileFingerprint]:
        """Yield content-hashed snapshots; inaccessible descendants are skipped."""
        root = require_readable_directory(path).resolve(strict=True)
        configured = self._configured_exclusions()
        stack = [root]
        while stack:
            try:
                directory = resolve_directory_under_root(stack.pop(), root)
            except (OSError, FilesystemPermissionError):
                continue
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        try:
                            if entry.is_symlink():
                                continue
                            if is_excluded_path(entry_path, configured_paths=configured):
                                continue
                            info = entry.stat(follow_symlinks=False)
                            if stat.S_ISDIR(info.st_mode):
                                if os.access(entry_path, os.R_OK | os.X_OK):
                                    stack.append(resolve_directory_under_root(entry_path, root))
                                continue
                            if not stat.S_ISREG(info.st_mode) or not os.access(entry_path, os.R_OK):
                                continue
                            yield self._snapshot(resolve_file_under_root(entry_path, root))
                        except (OSError, FilesystemPermissionError):
                            continue
            except OSError:
                # Descendant errors do not fail the whole scan; reconciliation below
                # separately protects all rows under inaccessible directory prefixes.
                continue

    def discover_file(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        path: str | os.PathLike[str],
        *,
        old_path: str | os.PathLike[str] | None = None,
        force_hash: bool = True,
    ) -> tuple[File | None, bool]:
        """Upsert one Watchdog event, returning (row, content_or_identity_changed)."""
        root = self._get_root(user_id, root_id)
        root_path = require_readable_directory(root.canonical_path).resolve(strict=True)
        self._check_registered_device(root, root_path)
        requested_target = Path(os.path.abspath(path))
        if not is_within(requested_target, root_path) or is_excluded_path(
            requested_target, configured_paths=self._configured_exclusions(root)
        ):
            return None, False
        try:
            target = resolve_file_under_root(requested_target, root_path)
        except (OSError, FilesystemPermissionError):
            return None, False
        prior = self._find_existing(user_id, root_id, target, old_path, root_path)
        stat_now = stat_file(target)
        if prior is not None and not force_hash and _metadata_matches(prior, stat_now):
            return prior, False
        try:
            fingerprint = self._snapshot(target)
        except FileNotFoundError:
            return None, False
        except FileChangedDuringHashing:
            raise
        file, changed = self._upsert(user_id, root_id, prior, fingerprint)
        return file, changed

    def reconcile_root(self, user_id: uuid.UUID, root_id: uuid.UUID) -> ReconcileReport:
        """Reconcile one registered root; never infer mass deletion from an unavailable root."""
        root = self._get_root(user_id, root_id)
        root_path = require_readable_directory(root.canonical_path).resolve(strict=True)
        self._check_registered_device(root, root_path)
        configured = self._configured_exclusions(root)
        report = ReconcileReport(root_id=root_id)
        existing = list(
            self.session.scalars(
                select(File).where(File.user_id == user_id, File.root_id == root_id)
            ).all()
        )
        existing_by_path = {file.canonical_path: file for file in existing}
        seen: set[str] = set()
        protected_prefixes: list[str] = []
        stack = [root_path]

        while stack:
            directory = stack.pop()
            try:
                entries = os.scandir(directory)
            except OSError as exc:
                report.issues.append(ScanIssue(str(directory), str(exc)))
                protected_prefixes.append(str(directory))
                continue
            try:
                with entries:
                    for entry in entries:
                        candidate = Path(entry.path)
                        candidate_text = os.path.abspath(candidate)
                        try:
                            if entry.is_symlink():
                                continue
                            if is_excluded_path(candidate, configured_paths=configured):
                                if entry.is_dir(follow_symlinks=False):
                                    protected_prefixes.append(candidate_text)
                                continue
                            info = entry.stat(follow_symlinks=False)
                            if stat.S_ISDIR(info.st_mode):
                                if not os.access(candidate, os.R_OK | os.X_OK):
                                    report.issues.append(
                                        ScanIssue(candidate_text, "directory is not readable")
                                    )
                                    protected_prefixes.append(candidate_text)
                                else:
                                    stack.append(resolve_directory_under_root(candidate, root_path))
                                continue
                            if not stat.S_ISREG(info.st_mode):
                                continue
                            seen.add(candidate_text)
                            report.discovered += 1
                            safe_candidate = resolve_file_under_root(candidate, root_path)
                            previous = existing_by_path.get(candidate_text)
                            current_stat = stat_file(safe_candidate)
                            if previous is not None and _metadata_matches(previous, current_stat):
                                report.unchanged += 1
                                continue
                            try:
                                snapshot = self._snapshot(safe_candidate)
                            except OSError as exc:
                                report.issues.append(ScanIssue(candidate_text, str(exc)))
                                protected_prefixes.append(candidate_text)
                                continue
                            row, changed = self._upsert(user_id, root_id, previous, snapshot)
                            if previous is None:
                                report.created += 1
                            elif changed:
                                report.changed += 1
                            else:
                                report.unchanged += 1
                            if changed:
                                report.changed_file_ids.append(row.id)
                        except (OSError, FilesystemPermissionError) as exc:
                            report.issues.append(ScanIssue(candidate_text, str(exc)))
                            protected_prefixes.append(candidate_text)
            except OSError as exc:
                report.issues.append(ScanIssue(str(directory), str(exc)))
                protected_prefixes.append(str(directory))

        for file in existing:
            if file.indexing_status == "deleted" or file.canonical_path in seen:
                continue
            if is_excluded_path(file.canonical_path, configured_paths=configured):
                continue
            if any(is_within(file.canonical_path, prefix) for prefix in protected_prefixes):
                continue
            report.missing_paths.append(file.canonical_path)
        root.last_reconciled_at = datetime.now(UTC)
        self.session.flush()
        return report

    def _snapshot(self, path: Path) -> FileFingerprint:
        fingerprint = fingerprint_file(path, chunk_size=self.hash_chunk_size)
        # Use content bytes (libmagic), never the suffix, as the MIME authority.
        mime_type = self.detector.detect(path)
        return _with_mime(fingerprint, mime_type)

    def _upsert(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        previous: File | None,
        snapshot: FileFingerprint,
    ) -> tuple[File, bool]:
        if previous is None:
            row = File(
                user_id=user_id,
                root_id=root_id,
                canonical_path=snapshot.canonical_path,
                filename=snapshot.filename,
                extension=snapshot.extension,
                mime_type=snapshot.mime_type or "application/octet-stream",
                size_bytes=snapshot.size_bytes,
                modified_at=snapshot.modified_at,
                device_id=snapshot.device_id,
                inode=snapshot.inode,
                content_hash=snapshot.content_hash,
                version=1,
                indexing_status="pending",
            )
            self.session.add(row)
            self.session.flush()
            return row, True
        content_changed = previous.content_hash != snapshot.content_hash
        path_changed = previous.canonical_path != snapshot.canonical_path
        was_deleted = previous.indexing_status == "deleted"
        prior_hash = previous.content_hash
        previous.canonical_path = snapshot.canonical_path
        previous.filename = snapshot.filename
        previous.extension = snapshot.extension
        previous.mime_type = snapshot.mime_type or "application/octet-stream"
        previous.size_bytes = snapshot.size_bytes
        previous.modified_at = snapshot.modified_at
        previous.device_id = snapshot.device_id
        previous.inode = snapshot.inode
        previous.content_hash = snapshot.content_hash
        previous.version = content_version(prior_hash, snapshot.content_hash, previous.version)
        if content_changed or path_changed or was_deleted:
            previous.indexing_status = "pending"
        self.session.flush()
        return previous, content_changed or path_changed or was_deleted

    def _find_existing(
        self,
        user_id: uuid.UUID,
        root_id: uuid.UUID,
        path: Path,
        old_path: str | os.PathLike[str] | None,
        root_path: Path,
    ) -> File | None:
        direct = self.session.scalar(
            select(File).where(
                File.user_id == user_id,
                File.root_id == root_id,
                File.canonical_path == str(path),
            )
        )
        if direct is not None or old_path is None:
            return direct
        old = Path(os.path.abspath(old_path))
        if not is_within(old, root_path):
            return None
        return self.session.scalar(
            select(File).where(
                File.user_id == user_id,
                File.root_id == root_id,
                File.canonical_path == str(old),
            )
        )

    def _get_root(self, user_id: uuid.UUID, root_id: uuid.UUID) -> IndexRoot:
        root = IndexRootRepository(self.session).get(user_id, root_id)
        if root is None or not root.is_enabled:
            raise LookupError("Enabled index root does not exist for this user")
        return root

    def _check_registered_device(self, root: IndexRoot, current_path: Path) -> None:
        info = current_path.stat()
        current_device = getattr(info, "st_dev", None)
        recorded_device = (root.settings or {}).get("filesystem_device_id")
        if recorded_device is not None and current_device != recorded_device:
            raise RootUnavailableError(
                f"Registered root is on a different filesystem device; refusing reconciliation: {current_path}"
            )
        if recorded_device is None:
            root.settings = {
                **(root.settings or {}),
                "filesystem_device_id": current_device,
                "filesystem_inode": getattr(info, "st_ino", None),
            }

    def _configured_exclusions(
        self, root: IndexRoot | None = None
    ) -> tuple[str | os.PathLike[str], ...]:
        settings_paths = get_settings().excluded_paths
        root_paths = (root.settings or {}).get("excluded_paths", []) if root is not None else []
        return (*settings_paths, *root_paths, *self.excluded_paths)


def _metadata_matches(file: File, current: FileStat) -> bool:
    return (
        file.indexing_status != "deleted"
        and file.size_bytes == current.size_bytes
        and file.modified_at == current.modified_at
        and file.device_id == current.device_id
        and file.inode == current.inode
        and file.content_hash is not None
    )


def _with_mime(fingerprint: FileFingerprint, mime_type: str) -> FileFingerprint:
    # Keep fingerprint immutable while attaching the separately detected MIME result.
    return replace(fingerprint, mime_type=mime_type)
