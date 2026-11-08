from __future__ import annotations

import asyncio
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import httpx
import pytest

from app.api.common import get_local_user_id
from app.api.files import _sanitize_metadata_paths
from app.config import get_settings
from app.database.engine import get_db
from app.main import app
from app.retrieval.common import canonical_filter
from app.security.parser_sandbox import ParserInputError, run_tool, validate_input
from app.security.permissions import FilesystemPermissionError, resolve_file_under_root


@pytest.fixture(autouse=True)
def override_api_dependencies():
    def fake_db():
        yield object()

    app.dependency_overrides[get_db] = fake_db
    app.dependency_overrides[get_local_user_id] = lambda: uuid.UUID(
        "68e6be19-b06c-4b30-a4fa-48ce1bc649d8"
    )
    yield
    app.dependency_overrides.clear()


def test_resolve_file_under_root_rejects_traversal_and_absolute_injection(tmp_path: Path) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")

    for path in (root / ".." / "outside.txt", outside):
        with pytest.raises(FilesystemPermissionError):
            resolve_file_under_root(path, root)


def test_resolve_file_under_root_rejects_internal_and_escaping_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "authorized"
    root.mkdir()
    inside = root / "real.txt"
    inside.write_text("inside")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    internal_link = root / "inside-link.txt"
    external_link = root / "outside-link.txt"
    try:
        internal_link.symlink_to(inside)
        external_link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    assert resolve_file_under_root(inside, root) == inside
    with pytest.raises(FilesystemPermissionError):
        resolve_file_under_root(internal_link, root)
    with pytest.raises(FilesystemPermissionError):
        resolve_file_under_root(external_link, root)


def test_active_search_query_excludes_disabled_authorized_roots() -> None:
    from sqlalchemy.dialects import postgresql

    statement = canonical_filter(None, uuid.UUID("68e6be19-b06c-4b30-a4fa-48ce1bc649d8"))
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "index_roots" in sql
    assert "is_enabled" in sql


def test_validate_input_enforces_zip_member_and_expansion_safety(tmp_path: Path) -> None:
    archive_path = tmp_path / "malformed.docx"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "unsafe")

    with pytest.raises(ParserInputError, match="unsafe member path"):
        validate_input(archive_path)


def test_validate_input_uses_configured_file_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCOUTX_PARSER_MAX_FILE_BYTES", str(1_048_576))
    get_settings.cache_clear()
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (1_048_576 + 1))
    try:
        with pytest.raises(ParserInputError, match="exceeds parser limit"):
            validate_input(path)
    finally:
        get_settings.cache_clear()


def test_parser_subprocess_timeout_is_enforced() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_tool(
            sys.executable,
            ["-c", "import time; time.sleep(2)"],
            timeout=0.05,
        )


async def _http_request(
    method: str,
    path: str,
    *,
    client: tuple[str, int] = ("127.0.0.1", 40000),
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=client),
        base_url="http://testserver",
    ) as http:
        return await http.request(method, path, headers=headers, content=content)


def test_local_only_api_rejects_non_loopback_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCOUTX_API_LOCAL_ONLY", "true")
    get_settings.cache_clear()
    try:
        response = asyncio.run(_http_request("GET", "/api/v1/live", client=("203.0.113.12", 5000)))
    finally:
        get_settings.cache_clear()
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "local_access_only"


def test_network_api_exposure_cannot_be_enabled_without_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("SCOUTX_API_LOCAL_ONLY", "false")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValidationError, match="network authentication provider"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_celery_worker_and_visibility_limits_follow_configuration() -> None:
    from app.workers.celery_app import celery_app

    settings = get_settings()
    assert celery_app.conf.worker_concurrency == settings.worker_concurrency
    assert celery_app.conf.task_soft_time_limit == settings.worker_task_soft_limit_seconds
    assert celery_app.conf.task_time_limit == settings.worker_task_hard_limit_seconds
    assert (
        celery_app.conf.broker_transport_options["visibility_timeout"]
        > celery_app.conf.task_time_limit
    )


def test_api_bounds_streamed_request_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCOUTX_MAX_REQUEST_BODY_BYTES", "16384")
    get_settings.cache_clear()
    try:
        response = asyncio.run(
            _http_request(
                "POST",
                "/api/v1/roots",
                headers={"Content-Type": "application/json"},
                content=b"{" + b" " * 20_000 + b"}",
            )
        )
    finally:
        get_settings.cache_clear()
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_cors_preflight_allows_preferences_put() -> None:
    response = asyncio.run(
        _http_request(
            "OPTIONS",
            "/api/v1/personalization/preferences",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    )
    assert response.status_code == 200
    assert "PUT" in response.headers["access-control-allow-methods"]


def test_cors_configuration_rejects_non_loopback_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import _allowed_origins

    monkeypatch.setenv("SCOUTX_CORS_ORIGINS", "https://example.com")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="loopback"):
            _allowed_origins()
    finally:
        get_settings.cache_clear()


def test_request_validation_does_not_reflect_untrusted_values() -> None:
    response = asyncio.run(
        _http_request(
            "POST",
            "/api/v1/roots",
            headers={"Content-Type": "application/json"},
            content=b'{"path":"SECRET_PATH","unexpected":"SENSITIVE_VALUE"}',
        )
    )
    assert response.status_code == 422
    assert "SECRET_PATH" not in response.text
    assert "SENSITIVE_VALUE" not in response.text


def test_api_sets_request_correlation_header() -> None:
    response = asyncio.run(_http_request("GET", "/api/v1/live"))
    assert response.status_code == 200
    assert response.headers.get("x-request-id")


def test_structured_logs_omit_interpolated_sensitive_values() -> None:
    import logging

    from app.observability import SafeJsonFormatter

    record = logging.LogRecord(
        "app.test", logging.ERROR, __file__, 1, "file operation failed %s", ("/private/path",), None
    )
    record.correlation_id = "request-123"
    rendered = SafeJsonFormatter().format(record)
    assert "/private/path" not in rendered
    assert "request-123" in rendered
    assert '"severity": "ERROR"' in rendered


def test_file_error_responses_redact_absolute_paths() -> None:
    file = type(
        "File", (), {"canonical_path": "/authorized/report.pdf", "filename": "report.pdf"}
    )()
    result = _sanitize_metadata_paths(
        "Parser failed while opening /Users/private/work/cache/doc.tmp", file, None
    )
    assert "/Users/private" not in result
    assert "[local path]" in result
