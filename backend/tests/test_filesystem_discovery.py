from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy.sql import operators

from app.database.models import File, IndexRoot
from app.ingestion.fingerprints import fingerprint_file, hash_file_streaming
from app.ingestion.file_detector import FileDetector
from app.ingestion.scanner import FileScanner, register_directory
from app.ingestion.watcher import DebouncedFilesystemEventHandler
from app.security import permissions
from app.security.permissions import FilesystemPermissionError, RootUnavailableError


class FakeDetector:
    def detect(self, path: str | os.PathLike[str]) -> str:
        with open(path, "rb") as source:
            header = source.read(8)
        return "application/pdf" if header.startswith(b"%PDF") else "text/plain"


class FakeResult:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows

    def all(self) -> list[Any]:
        return list(self.rows)


class FakeSession:
    """Small SQLAlchemy-session stand-in for scanner and event-handler tests."""

    def __init__(self, roots: list[IndexRoot] | None = None, files: list[File] | None = None) -> None:
        self.roots = roots or []
        self.files = files or []
        self.added: list[Any] = []
        self.committed = False
        self.rolled_back = False

    def scalar(self, statement: Any) -> Any | None:
        entity = statement.column_descriptions[0]["entity"]
        rows = self.roots if entity is IndexRoot else self.files
        return next((row for row in rows if _matches(row, statement._where_criteria)), None)

    def scalars(self, statement: Any) -> FakeResult:
        entity = statement.column_descriptions[0]["entity"]
        rows = self.roots if entity is IndexRoot else self.files
        return FakeResult([row for row in rows if _matches(row, statement._where_criteria)])

    def add(self, row: Any) -> None:
        self.added.append(row)
        if isinstance(row, File):
            self.files.append(row)
        elif isinstance(row, IndexRoot):
            self.roots.append(row)

    def flush(self) -> None:
        for row in self.added:
            if getattr(row, "id", None) is None:
                row.id = uuid.uuid4()

    def execute(self, *_: Any, **__: Any) -> None:
        return None

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def _matches(row: Any, criteria: tuple[Any, ...]) -> bool:
    for criterion in criteria:
        if getattr(criterion, "operator", None) is not operators.eq:
            continue
        key = getattr(criterion.left, "key", None)
        value = getattr(criterion.right, "value", object())
        if key is not None and getattr(row, key, None) != value:
            return False
    return True


def _make_root(path: Path, *, device_id: int | None = None) -> tuple[uuid.UUID, IndexRoot]:
    user_id = uuid.uuid4()
    info = path.stat()
    root = IndexRoot(
        id=uuid.uuid4(),
        user_id=user_id,
        canonical_path=str(path.resolve()),
        display_name=path.name,
        is_enabled=True,
        settings={
            "filesystem_device_id": info.st_dev if device_id is None else device_id,
            "filesystem_inode": info.st_ino,
        },
    )
    return user_id, root


def test_streaming_sha256_fingerprint_and_hash_verification(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    data = (b"ScoutX content\0" * 50_000) + b"end"
    path.write_bytes(data)

    assert hash_file_streaming(path, chunk_size=97) == hashlib.sha256(data).hexdigest()
    fingerprint = fingerprint_file(path, chunk_size=31)
    assert fingerprint.content_hash == hashlib.sha256(data).hexdigest()
    assert fingerprint.size_bytes == len(data)
    assert fingerprint.device_id is not None
    assert fingerprint.inode is not None


def test_mime_uses_libmagic_header_not_extension(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_bytes(b"%PDF-1.7\nbody")
    try:
        import magic  # noqa: F401
    except ImportError:
        pytest.skip("native libmagic is not installed on this host")
    try:
        assert FileDetector().detect(path) == "application/pdf"
    except OSError as exc:
        pytest.fail(f"libmagic failed to inspect a readable fixture: {exc}")


def test_new_modified_duplicate_and_moved_file_upserts(tmp_path: Path) -> None:
    root_path = tmp_path / "authorized"
    root_path.mkdir()
    first_path = root_path / "notes.txt"
    first_path.write_text("first content")
    user_id, root = _make_root(root_path)
    session = FakeSession(roots=[root])
    scanner = FileScanner(session, detector=FakeDetector())

    record, changed = scanner.discover_file(user_id, root.id, first_path)
    assert changed is True
    assert record is not None and record.version == 1
    original_id = record.id

    original_hash = record.content_hash
    old_mtime_ns = first_path.stat().st_mtime_ns
    os.utime(first_path, ns=(old_mtime_ns, old_mtime_ns + 2_000_000))
    record, changed = scanner.discover_file(user_id, root.id, first_path)
    assert changed is False
    assert record.content_hash == original_hash
    assert record.version == 1

    first_path.write_text("second content is longer")
    record, changed = scanner.discover_file(user_id, root.id, first_path)
    assert changed is True
    assert record.version == 2
    assert record.content_hash == hashlib.sha256(b"second content is longer").hexdigest()

    record, changed = scanner.discover_file(user_id, root.id, first_path)
    assert changed is False
    assert record.version == 2

    moved_path = root_path / "renamed.txt"
    first_path.rename(moved_path)
    record, changed = scanner.discover_file(
        user_id,
        root.id,
        moved_path,
        old_path=first_path,
    )
    assert changed is True
    assert record.id == original_id
    assert record.canonical_path == str(moved_path)
    assert record.filename == "renamed.txt"
    assert record.version == 2


def test_reconciliation_finds_new_and_missing_files(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    new_file = root_path / "new.txt"
    new_file.write_text("available")
    missing_file = File(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        root_id=uuid.uuid4(),
        canonical_path=str(root_path / "removed.txt"),
        filename="removed.txt",
        size_bytes=5,
        version=1,
        indexing_status="indexed",
        content_hash="old-hash",
    )
    user_id, root = _make_root(root_path)
    missing_file.user_id = user_id
    missing_file.root_id = root.id
    session = FakeSession(roots=[root], files=[missing_file])

    report = FileScanner(session, detector=FakeDetector()).reconcile_root(user_id, root.id)
    assert report.created == 1
    assert report.missing_paths == [str(root_path / "removed.txt")]
    assert report.changed_file_ids
    assert root.last_reconciled_at is not None


def test_reconciliation_does_not_delete_rows_below_inaccessible_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "root"
    protected = root_path / "locked"
    protected.mkdir(parents=True)
    user_id, root = _make_root(root_path)
    old_file = File(
        id=uuid.uuid4(),
        user_id=user_id,
        root_id=root.id,
        canonical_path=str(protected / "still-indexed.txt"),
        filename="still-indexed.txt",
        size_bytes=20,
        version=1,
        indexing_status="indexed",
        content_hash="known-hash",
    )
    session = FakeSession(roots=[root], files=[old_file])
    real_scandir = os.scandir

    def inaccessible_scandir(path: str | os.PathLike[str]) -> Any:
        if Path(path) == protected:
            raise PermissionError("simulated inaccessible directory")
        return real_scandir(path)

    monkeypatch.setattr("app.ingestion.scanner.os.scandir", inaccessible_scandir)
    report = FileScanner(session, detector=FakeDetector()).reconcile_root(user_id, root.id)
    assert report.missing_paths == []
    assert any(issue.path == str(protected) for issue in report.issues)


def test_discovery_reactivates_a_restored_deleted_file_without_bumping_content_version(
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    path = root_path / "restored.txt"
    path.write_text("same bytes")
    user_id, root = _make_root(root_path)
    row = File(
        id=uuid.uuid4(),
        user_id=user_id,
        root_id=root.id,
        canonical_path=str(path),
        filename=path.name,
        extension=".txt",
        mime_type="text/plain",
        size_bytes=10,
        modified_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
        device_id=path.stat().st_dev,
        inode=path.stat().st_ino,
        content_hash=hashlib.sha256(b"same bytes").hexdigest(),
        version=4,
        indexing_status="deleted",
    )
    session = FakeSession(roots=[root], files=[row])

    restored, changed = FileScanner(session, detector=FakeDetector()).discover_file(
        user_id, root.id, path
    )
    assert changed is True
    assert restored.id == row.id
    assert restored.indexing_status == "pending"
    assert restored.version == 4


def test_register_directory_persists_only_authorized_canonical_path(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    user_id = uuid.uuid4()
    session = FakeSession()

    root = register_directory(session, user_id, selected, display_name="My files")
    assert root.canonical_path == str(selected.resolve())
    assert root.display_name == "My files"
    assert root.settings["filesystem_device_id"] == selected.stat().st_dev
    assert session.roots == [root]


def test_reconciliation_protects_files_when_root_device_changes(tmp_path: Path) -> None:
    root_path = tmp_path / "remounted"
    root_path.mkdir()
    user_id, root = _make_root(root_path, device_id=-1)
    session = FakeSession(roots=[root])
    scanner = FileScanner(session, detector=FakeDetector())

    with pytest.raises(RootUnavailableError):
        scanner.reconcile_root(user_id, root.id)


def test_scan_does_not_follow_symlinks_or_index_excluded_paths(tmp_path: Path) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    (root_path / "visible.txt").write_text("visible")
    virtualenv = root_path / ".venv"
    virtualenv.mkdir()
    (virtualenv / "secret.txt").write_text("excluded")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    try:
        (root_path / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("filesystem does not permit symlink creation")

    scanner = FileScanner(FakeSession(), detector=FakeDetector())
    snapshots = list(scanner.scan_directory(root_path))
    assert [snapshot.filename for snapshot in snapshots] == ["visible.txt"]


def test_permission_failure_prevents_root_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.security import permissions as permission_module

    monkeypatch.setattr(permission_module, "_access", lambda *_: False)
    with pytest.raises(FilesystemPermissionError):
        permission_module.canonicalize_authorized_directory(tmp_path)


def test_watchdog_debounces_duplicate_events_and_reports_move_delete(tmp_path: Path) -> None:
    root_path = tmp_path / "watched"
    root_path.mkdir()
    source = root_path / "before.txt"
    source.write_text("stable")
    destination = root_path / "after.txt"
    calls: list[tuple[str, str, str | None]] = []
    ready = threading.Event()

    class Publisher:
        def discover_file(self, user_id: uuid.UUID, root_id: uuid.UUID, path: str, *, old_path: str | None = None) -> None:
            calls.append(("discover", path, old_path))
            ready.set()

        def delete_file(self, user_id: uuid.UUID, root_id: uuid.UUID, path: str) -> None:
            calls.append(("delete", path, None))
            ready.set()

    handler = DebouncedFilesystemEventHandler(
        uuid.uuid4(),
        uuid.uuid4(),
        root_path,
        publisher=Publisher(),
        debounce_seconds=0.03,
        stability_timeout=0.2,
        stability_interval=0,
        stable_samples=2,
    )
    event = SimpleNamespace(src_path=str(source), is_directory=False)
    handler.on_created(event)
    handler.on_modified(event)
    assert ready.wait(1)
    time.sleep(0.06)
    assert [call[0] for call in calls] == ["discover"]

    source.rename(destination)
    ready.clear()
    handler.on_moved(
        SimpleNamespace(src_path=str(source), dest_path=str(destination), is_directory=False)
    )
    assert ready.wait(1)
    assert calls[-1] == ("discover", str(destination), str(source))

    destination.unlink()
    ready.clear()
    handler.on_deleted(SimpleNamespace(src_path=str(destination), is_directory=False))
    assert ready.wait(1)
    assert calls[-1] == ("delete", str(destination), None)
    handler.close()


def test_watchdog_suppresses_unmount_delete_burst(tmp_path: Path) -> None:
    root_path = tmp_path / "drive"
    root_path.mkdir()
    file_path = root_path / "item.txt"
    file_path.write_text("data")
    calls: list[str] = []

    class Publisher:
        def discover_file(self, *_: Any, **__: Any) -> None:
            return None

        def delete_file(self, user_id: uuid.UUID, root_id: uuid.UUID, path: str) -> None:
            calls.append(path)

    handler = DebouncedFilesystemEventHandler(
        uuid.uuid4(),
        uuid.uuid4(),
        root_path,
        publisher=Publisher(),
        expected_device_id=-1,
        debounce_seconds=0.01,
    )
    file_path.unlink()
    handler.on_deleted(SimpleNamespace(src_path=str(file_path), is_directory=False))
    time.sleep(0.05)
    assert calls == []
    handler.close()


def test_root_registration_rejects_filesystem_root(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Unix filesystem-root policy test")
    with pytest.raises(FilesystemPermissionError):
        permissions.canonicalize_authorized_directory(Path("/"))


def test_delete_task_marks_file_deleted_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "root"
    root_path.mkdir()
    path = root_path / "gone.txt"
    user_id, root = _make_root(root_path)
    file = File(
        id=uuid.uuid4(),
        user_id=user_id,
        root_id=root.id,
        canonical_path=str(path),
        filename=path.name,
        size_bytes=4,
        version=1,
        indexing_status="indexed",
        content_hash="hash",
    )
    session = FakeSession(roots=[root], files=[file])
    monkeypatch.setattr("app.workers.tasks.SessionLocal", lambda: session)
    from app.workers.tasks import delete_file

    first = delete_file.run(str(user_id), str(root.id), str(path))
    second = delete_file.run(str(user_id), str(root.id), str(path))
    assert first["status"] == "deleted"
    assert second["status"] == "already_deleted"
    assert file.indexing_status == "deleted"
    assert session.committed is True


def test_delete_task_preserves_rows_when_external_root_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_path = tmp_path / "offline-drive"
    root_path.mkdir()
    path = root_path / "not-really-gone.txt"
    user_id, root = _make_root(root_path, device_id=-1)
    file = File(
        id=uuid.uuid4(),
        user_id=user_id,
        root_id=root.id,
        canonical_path=str(path),
        filename=path.name,
        size_bytes=1,
        version=1,
        indexing_status="indexed",
        content_hash="hash",
    )
    session = FakeSession(roots=[root], files=[file])
    monkeypatch.setattr("app.workers.tasks.SessionLocal", lambda: session)
    from app.workers.tasks import delete_file

    result = delete_file.run(str(user_id), str(root.id), str(path))
    assert result["status"] == "root_unavailable"
    assert file.indexing_status == "indexed"
