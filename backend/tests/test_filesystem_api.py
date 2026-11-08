from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest

from app.api import feedback as feedback_api
from app.api import files as files_api
from app.api import indexing as indexing_api
from app.api import roots as roots_api
from app.api.common import get_local_user_id
from app.config import get_settings
from app.database.engine import get_db
from app.main import app

USER_ID = uuid.UUID("68e6be19-b06c-4b30-a4fa-48ce1bc649d8")
ROOT_ID = uuid.UUID("1069579a-0320-4e94-a223-a419b7dfad9d")
FILE_ID = uuid.UUID("746dc34e-3946-4d7f-91ce-99304f95103a")


class FakeResult:
    def __init__(self, values: list[Any]) -> None:
        self.values = values

    def all(self) -> list[Any]:
        return self.values


class FakeSession:
    def __init__(self) -> None:
        self.count = 1
        self.file_rows: list[Any] = []
        self.status_rows: list[Any] = []

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def refresh(self, _: Any) -> None:
        pass

    def scalar(self, _: Any) -> int:
        return self.count

    def scalars(self, _: Any) -> FakeResult:
        return FakeResult(self.file_rows)

    def execute(self, _: Any) -> FakeResult:
        return FakeResult(self.status_rows)


class FakeRootRepository:
    roots: ClassVar[list[Any]] = []
    requested: ClassVar[list[Any]] = []

    def __init__(self, _: FakeSession) -> None:
        pass

    def create(self, user_id: uuid.UUID, data: Any) -> Any:
        root = SimpleNamespace(
            id=ROOT_ID,
            user_id=user_id,
            canonical_path=data.canonical_path,
            display_name=data.display_name,
            is_enabled=True,
            last_reconciled_at=None,
        )
        self.roots.append(root)
        self.requested.append(data)
        return root

    def get(self, user_id: uuid.UUID, root_id: uuid.UUID) -> Any | None:
        return next(
            (root for root in self.roots if root.id == root_id and root.user_id == user_id), None
        )

    def list_for_user(self, user_id: uuid.UUID, *, enabled_only: bool = False) -> list[Any]:
        items = [root for root in self.roots if root.user_id == user_id]
        if enabled_only:
            items = [root for root in items if root.is_enabled]
        return items


class FakeIndexJobRepository:
    jobs: ClassVar[list[Any]] = []

    def __init__(self, _: FakeSession) -> None:
        pass

    def list_for_user(self, _: uuid.UUID, *, limit: int = 100) -> list[Any]:
        return self.jobs[:limit]


@pytest.fixture(autouse=True)
def api_dependencies(monkeypatch: pytest.MonkeyPatch) -> Any:
    session = FakeSession()

    def override_session():
        yield session

    app.dependency_overrides[get_db] = override_session
    app.dependency_overrides[get_local_user_id] = lambda: USER_ID
    monkeypatch.setattr(roots_api, "IndexRootRepository", FakeRootRepository)
    monkeypatch.setattr(files_api, "IndexRootRepository", FakeRootRepository)
    monkeypatch.setattr(files_api, "resolve_file_under_root", lambda path, _root: Path(path))
    monkeypatch.setattr(indexing_api, "IndexRootRepository", FakeRootRepository)
    monkeypatch.setattr(indexing_api, "IndexJobRepository", FakeIndexJobRepository)
    FakeRootRepository.roots = []
    FakeRootRepository.requested = []
    FakeIndexJobRepository.jobs = []
    app.state.reconciliation_dispatcher = None
    yield session
    app.dependency_overrides.clear()
    if hasattr(app.state, "reconciliation_dispatcher"):
        del app.state.reconciliation_dispatcher


async def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        return await client.request(method, path, **kwargs)


def call(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(request(method, path, **kwargs))


def test_register_root_validates_and_canonicalizes_directory(tmp_path: Path) -> None:
    response = call("POST", "/api/v1/roots", json={"path": str(tmp_path), "display_name": "Work"})

    assert response.status_code == 201
    assert response.json()["path"] == "Work"
    assert str(tmp_path) not in response.text
    assert response.json()["status"] == "enabled"
    assert FakeRootRepository.requested[0].canonical_path == str(tmp_path.resolve())


def test_register_root_rejects_invalid_path() -> None:
    response = call("POST", "/api/v1/roots", json={"path": "/path/that/does/not/exist"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_root_path"


def test_register_root_requires_absolute_path() -> None:
    response = call("POST", "/api/v1/roots", json={"path": "relative/folder"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_root_path"


def test_api_requires_authentication_outside_development(
    api_dependencies: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    app.dependency_overrides.pop(get_local_user_id)
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_settings.cache_clear()

    response = call("GET", "/api/v1/roots")

    get_settings.cache_clear()
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"


def test_register_root_handles_inaccessible_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny_scandir(_: Any) -> Any:
        raise PermissionError("access denied")

    monkeypatch.setattr(roots_api.os, "scandir", deny_scandir)
    response = call("POST", "/api/v1/roots", json={"path": str(tmp_path)})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "root_inaccessible"


def test_register_root_rejects_duplicate_directory(tmp_path: Path) -> None:
    payload = {"path": str(tmp_path)}
    assert call("POST", "/api/v1/roots", json=payload).status_code == 201

    duplicate = call("POST", "/api/v1/roots", json=payload)

    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "root_already_registered"


def test_delete_root_disables_registration_and_listing_keeps_it(tmp_path: Path) -> None:
    FakeRootRepository.roots = [
        SimpleNamespace(
            id=ROOT_ID,
            user_id=USER_ID,
            canonical_path=str(tmp_path),
            display_name=None,
            is_enabled=True,
            last_reconciled_at=None,
        )
    ]

    deleted = call("DELETE", f"/api/v1/roots/{ROOT_ID}")
    listed = call("GET", "/api/v1/roots")

    assert deleted.status_code == 204
    assert listed.status_code == 200
    assert listed.json()["items"][0]["status"] == "disabled"


def test_file_listing_includes_mime_size_modified_status_and_extraction_error(
    api_dependencies: FakeSession,
) -> None:
    root = SimpleNamespace(
        id=ROOT_ID,
        user_id=USER_ID,
        is_enabled=True,
        canonical_path="/authorized",
    )
    FakeRootRepository.roots = [root]
    api_dependencies.file_rows = [
        SimpleNamespace(
            id=FILE_ID,
            root_id=ROOT_ID,
            filename="notes.pdf",
            canonical_path="/authorized/notes.pdf",
            mime_type="application/pdf",
            size_bytes=4096,
            modified_at=datetime(2026, 1, 2, tzinfo=UTC),
            indexing_status="indexed",
            extra_metadata={
                "parser_status": "partial",
                "extraction_warnings": ["Page 2 contains no extractable text"],
                "extraction_error": "Page 2 unreadable",
            },
        )
    ]

    response = call("GET", "/api/v1/files?root_id=" + str(ROOT_ID))

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["mime_type"] == "application/pdf"
    assert item["size_bytes"] == 4096
    assert item["modified_at"] == "2026-01-02T00:00:00Z"
    assert item["indexing_status"] == "partial"
    assert item["parser_status"] == "partial"
    assert item["extraction_warnings"] == ["Page 2 contains no extractable text"]
    assert item["extraction_error"] == "Page 2 unreadable"
    assert item["path"] == "notes.pdf"
    assert not Path(item["path"]).is_absolute()


def test_file_listing_supports_server_side_filters_and_sort_validation(
    api_dependencies: FakeSession,
) -> None:
    FakeRootRepository.roots = [
        SimpleNamespace(
            id=ROOT_ID,
            user_id=USER_ID,
            is_enabled=True,
            canonical_path="/authorized",
        )
    ]
    api_dependencies.file_rows = [
        SimpleNamespace(
            id=FILE_ID,
            root_id=ROOT_ID,
            filename="report.pdf",
            canonical_path="/authorized/report.pdf",
            mime_type="application/pdf",
            extension=".pdf",
            size_bytes=512,
            modified_at=None,
            indexing_status="indexed",
            version=1,
            last_indexed_at=None,
            created_at=None,
            extra_metadata={},
        )
    ]
    response = call(
        "GET",
        "/api/v1/files?filename_contains=report&extension=pdf&sort_by=size_bytes&sort_order=desc",
    )
    assert response.status_code == 200
    assert response.json()["items"][0]["extension"] == ".pdf"
    assert response.json()["items"][0]["path"] == "report.pdf"

    invalid = call(
        "GET",
        "/api/v1/files?modified_after=2026-11-08T00:00:00Z&modified_before=2026-11-07T00:00:00Z",
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_date_range"


def test_index_status_maps_database_statuses(api_dependencies: FakeSession) -> None:
    now = datetime(2026, 11, 8, 9, tzinfo=UTC)
    FakeRootRepository.roots = [
        SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True, last_reconciled_at=None)
    ]
    FakeIndexJobRepository.jobs = [
        SimpleNamespace(
            id=FILE_ID,
            user_id=USER_ID,
            root_id=ROOT_ID,
            file_id=FILE_ID,
            job_type="process_file",
            status="running",
            attempts=1,
            payload={"stage": "embedding", "progress_percent": 42},
            error_message=None,
            created_at=now,
            started_at=now,
            finished_at=None,
        ),
        SimpleNamespace(
            id=uuid.UUID("80eaa197-6bc8-401f-90fa-7b292596fb1a"),
            user_id=USER_ID,
            root_id=ROOT_ID,
            file_id=uuid.UUID("80eaa197-6bc8-401f-90fa-7b292596fb1a"),
            job_type="process_file",
            status="failed",
            attempts=3,
            payload={"stage": "extract"},
            error_message="Parser rejected this file",
            created_at=now,
            started_at=now,
            finished_at=now,
        ),
    ]
    api_dependencies.status_rows = [
        ("queued", {}),
        ("processing", {}),
        ("indexed", {"parser_status": "partial"}),
        ("failed", {}),
        ("excluded", {}),
        ("deleted", {}),
    ]

    response = call("GET", "/api/v1/index/status")

    assert response.status_code == 200
    assert response.json()["status_counts"] == {
        "pending": 1,
        "processing": 1,
        "indexed": 0,
        "partial": 1,
        "failed": 1,
        "unsupported": 1,
        "deleted": 1,
    }
    assert response.json()["files_indexed"] == 0
    assert response.json()["state"] == "processing"
    assert response.json()["recent_job_counts"]["running"] == 1
    assert response.json()["recent_job_counts"]["failed"] == 1
    assert response.json()["recent_jobs"][0]["stage"] == "embedding"
    assert response.json()["recent_jobs"][0]["progress_percent"] == 42
    assert response.json()["recent_jobs"][1]["error_message"] == "Parser rejected this file"

    alias = call("GET", "/api/v1/indexing/status")
    assert alias.status_code == 200
    assert alias.json()["files_discovered"] == response.json()["files_discovered"]


def test_reconcile_requires_dispatcher_and_queues_when_configured() -> None:
    FakeRootRepository.roots = [
        SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True, last_reconciled_at=None)
    ]
    unavailable = call("POST", "/api/v1/index/reconcile", json={"root_id": str(ROOT_ID)})
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "indexing_service_unavailable"

    app.state.reconciliation_dispatcher = lambda user_id, root_id: f"task-{root_id}"
    queued = call("POST", "/api/v1/index/reconcile", json={"root_id": str(ROOT_ID)})
    assert queued.status_code == 202
    assert queued.json()["task_ids"] == [f"task-{ROOT_ID}"]

    queued_all = call("POST", "/api/v1/index/reconcile")
    assert queued_all.status_code == 202
    assert queued_all.json()["root_ids"] == [str(ROOT_ID)]


def test_reconcile_uses_existing_celery_task_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeRootRepository.roots = [
        SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True, last_reconciled_at=None)
    ]
    if hasattr(app.state, "reconciliation_dispatcher"):
        del app.state.reconciliation_dispatcher
    submitted: list[tuple[uuid.UUID, uuid.UUID]] = []

    def enqueue(user_id: uuid.UUID, root_id: uuid.UUID) -> str:
        submitted.append((user_id, root_id))
        return "celery-task-123"

    monkeypatch.setattr(indexing_api, "_dispatch_celery_reconciliation", enqueue)
    response = call("POST", "/api/v1/index/reconcile", json={"root_id": str(ROOT_ID)})

    assert response.status_code == 202
    assert submitted == [(USER_ID, ROOT_ID)]
    assert response.json()["task_ids"] == ["celery-task-123"]


def test_file_detail_is_user_scoped_and_returns_versioned_snippets(
    api_dependencies: FakeSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_path = tmp_path / "authorized"
    root_path.mkdir()
    actual_file = root_path / "notes.txt"
    actual_file.write_text("notes", encoding="utf-8")
    root = SimpleNamespace(canonical_path=str(root_path))
    FakeRootRepository.roots = [
        SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True, canonical_path=str(root_path))
    ]
    file = SimpleNamespace(
        id=FILE_ID,
        user_id=USER_ID,
        root_id=ROOT_ID,
        root=root,
        filename="notes.txt",
        canonical_path=str(actual_file),
        mime_type="text/plain",
        extension=".txt",
        size_bytes=5,
        modified_at=None,
        indexing_status="indexed",
        version=2,
        last_indexed_at=datetime(2026, 11, 8, tzinfo=UTC),
        created_at=datetime(2026, 11, 7, tzinfo=UTC),
        extra_metadata={"parser_status": "success"},
    )
    chunk = SimpleNamespace(
        id=uuid.uuid4(),
        content="A searchable note.",
        modality="text",
        content_version=2,
        section_title="Summary",
        page_number=1,
        line_start=None,
        line_end=None,
        start_time_ms=None,
        end_time_ms=None,
        chunk_index=0,
    )
    monkeypatch.setattr(files_api.FileRepository, "get", lambda *_args: file)
    monkeypatch.setattr(files_api, "resolve_file_under_root", lambda *_args: actual_file)
    api_dependencies.file_rows = [chunk]

    response = call("GET", f"/api/v1/files/{FILE_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["file"]["path"] == "notes.txt"
    assert body["file"]["version"] == 2
    assert body["snippets"][0]["snippet"] == "A searchable note."
    assert body["snippets"][0]["section_title"] == "Summary"
    assert str(tmp_path) not in response.text


def test_file_detail_returns_safe_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    file = SimpleNamespace(id=FILE_ID, root_id=ROOT_ID, canonical_path="/private/blocked.txt")
    FakeRootRepository.roots = [
        SimpleNamespace(
            id=ROOT_ID,
            user_id=USER_ID,
            is_enabled=True,
            canonical_path="/private",
        )
    ]
    monkeypatch.setattr(files_api.FileRepository, "get", lambda *_args: file)
    monkeypatch.setattr(
        files_api,
        "resolve_file_under_root",
        lambda *_args: (_ for _ in ()).throw(files_api.FilesystemPermissionError("private path")),
    )

    response = call("GET", f"/api/v1/files/{FILE_ID}")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "file_inaccessible"
    assert "/private" not in response.text


def test_feedback_is_persisted_in_user_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: list[Any] = []
    activities: list[Any] = []

    class FakeFeedbackRepository:
        def __init__(self, _session: FakeSession) -> None:
            pass

        def create(self, user_id: uuid.UUID, data: Any) -> Any:
            saved.append((user_id, data))
            return data

    class FakeActivityRepository:
        def __init__(self, _session: FakeSession) -> None:
            pass

        def record(self, user_id: uuid.UUID, event_type: str, **kwargs: Any) -> None:
            activities.append((user_id, event_type, kwargs))

    monkeypatch.setattr(feedback_api, "RecommendationFeedbackRepository", FakeFeedbackRepository)
    monkeypatch.setattr(feedback_api, "ActivityRepository", FakeActivityRepository)
    response = call(
        "POST",
        "/api/v1/feedback",
        json={"file_id": str(FILE_ID), "action": "opened", "query": "budget"},
    )

    assert response.status_code == 202
    assert response.json() == {"accepted": True}
    assert saved[0][0] == USER_ID
    assert saved[0][1].rating == 3
    assert activities[0][1] == "opened"


def test_health_reports_dependency_degradation_without_sensitive_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app import main as main_api

    monkeypatch.setattr(main_api, "_check_postgres", lambda: "available")
    monkeypatch.setattr(main_api, "_check_qdrant", lambda: "unavailable")
    monkeypatch.setattr(main_api, "_check_redis", lambda: "available")
    monkeypatch.setattr(main_api, "_check_ollama", lambda: "available")

    response = call("GET", "/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["qdrant"] == "unavailable"
    assert "password" not in response.text.lower()
