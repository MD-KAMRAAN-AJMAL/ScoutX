from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest

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
        return next((root for root in self.roots if root.id == root_id and root.user_id == user_id), None)

    def list_for_user(self, user_id: uuid.UUID, *, enabled_only: bool = False) -> list[Any]:
        items = [root for root in self.roots if root.user_id == user_id]
        if enabled_only:
            items = [root for root in items if root.is_enabled]
        return items


@pytest.fixture(autouse=True)
def api_dependencies(monkeypatch: pytest.MonkeyPatch) -> Any:
    session = FakeSession()

    def override_session():
        yield session

    app.dependency_overrides[get_db] = override_session
    app.dependency_overrides[get_local_user_id] = lambda: USER_ID
    monkeypatch.setattr(roots_api, "IndexRootRepository", FakeRootRepository)
    monkeypatch.setattr(files_api, "IndexRootRepository", FakeRootRepository)
    monkeypatch.setattr(indexing_api, "IndexRootRepository", FakeRootRepository)
    FakeRootRepository.roots = []
    FakeRootRepository.requested = []
    app.state.reconciliation_dispatcher = None
    yield session
    app.dependency_overrides.clear()
    if hasattr(app.state, "reconciliation_dispatcher"):
        del app.state.reconciliation_dispatcher


async def request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


def call(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(request(method, path, **kwargs))


def test_register_root_validates_and_canonicalizes_directory(tmp_path: Path) -> None:
    response = call("POST", "/api/v1/roots", json={"path": str(tmp_path), "display_name": "Work"})

    assert response.status_code == 201
    assert response.json()["path"] == str(tmp_path.resolve())
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
    root = SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True)
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


def test_index_status_maps_database_statuses(api_dependencies: FakeSession) -> None:
    FakeRootRepository.roots = [
        SimpleNamespace(id=ROOT_ID, user_id=USER_ID, is_enabled=True, last_reconciled_at=None)
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
