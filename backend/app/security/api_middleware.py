"""Local-access, bounded-body, and correlation controls for the single-user API."""

from __future__ import annotations

import ipaddress
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.responses import JSONResponse

from app.config import get_settings
from app.observability import record_api_request, reset_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)
ASGIApp = Callable[
    [dict[str, Any], Callable[..., Awaitable[Any]], Callable[..., Awaitable[Any]]], Awaitable[None]
]


class APISafetyMiddleware:
    """Keep the local workspace API local and bound request memory use."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        if settings.api_local_only and not _is_loopback_client(
            scope, trusted_proxy_ips=settings.trusted_proxy_ips
        ):
            await JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "code": "local_access_only",
                        "message": "ScoutX API is configured for local access only.",
                        "details": None,
                    }
                },
            )(scope, receive, send)
            return

        started = time.perf_counter()
        correlation_id = str(uuid.uuid4())
        correlation_token = set_correlation_id(correlation_id)
        scope.setdefault("state", {})["correlation_id"] = correlation_id
        response_status = 500

        async def tracked_send(message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                message["headers"].append((b"x-request-id", correlation_id.encode("ascii")))
            await send(message)

        async def reject_body() -> None:
            nonlocal response_status
            response_status = 413
            route = scope.get("route")
            route_template = getattr(route, "path", "unmatched")
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            record_api_request(
                method=scope.get("method", "unknown"),
                route=route_template,
                status=response_status,
                duration_ms=duration_ms,
            )
            logger.info(
                "API request rejected",
                extra={
                    "component": "api",
                    "operation": "request_size_check",
                    "request_id": correlation_id,
                    "duration_ms": duration_ms,
                    "status": response_status,
                    "route": route_template,
                },
            )
            try:
                await _too_large(scope, receive, tracked_send)
            finally:
                reset_correlation_id(correlation_token)

        if scope.get("method") in {"POST", "PUT", "PATCH", "DELETE"}:
            content_length = _header(scope, b"content-length")
            try:
                declared_size = int(content_length) if content_length else 0
            except ValueError:
                declared_size = settings.max_request_body_bytes + 1
            if declared_size > settings.max_request_body_bytes:
                await reject_body()
                return
            buffered, total = [], 0
            while True:
                event = await receive()
                buffered.append(event)
                if event["type"] == "http.request":
                    total += len(event.get("body", b""))
                    if total > settings.max_request_body_bytes:
                        await reject_body()
                        return
                    if not event.get("more_body", False):
                        break
                elif event["type"] == "http.disconnect":
                    break
            cursor = 0

            async def replay_receive():
                nonlocal cursor
                if cursor < len(buffered):
                    event = buffered[cursor]
                    cursor += 1
                    return event
                return {"type": "http.request", "body": b"", "more_body": False}

            request_receive = replay_receive
        else:
            request_receive = receive

        try:
            await self.app(scope, request_receive, tracked_send)
        finally:
            route = scope.get("route")
            route_template = getattr(route, "path", "unmatched")
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            record_api_request(
                method=scope.get("method", "unknown"),
                route=route_template,
                status=response_status,
                duration_ms=duration_ms,
            )
            logger.info(
                "API request completed",
                extra={
                    "component": "api",
                    "operation": scope.get("method", "unknown"),
                    "request_id": correlation_id,
                    "duration_ms": duration_ms,
                    "status": response_status,
                    "route": route_template,
                },
            )
            reset_correlation_id(correlation_token)


async def _too_large(scope, receive, send) -> None:
    response = JSONResponse(
        status_code=413,
        content={
            "error": {
                "code": "request_too_large",
                "message": "The request exceeds the configured size limit.",
                "details": None,
            }
        },
    )
    await response(scope, receive, send)


def _header(scope, key: bytes) -> str | None:
    return next(
        (value.decode("latin1") for name, value in scope.get("headers", []) if name.lower() == key),
        None,
    )


def _is_loopback_client(scope, *, trusted_proxy_ips: str = "") -> bool:
    client = scope.get("client")
    if not client:
        return False
    try:
        immediate_peer = ipaddress.ip_address(client[0])
    except (TypeError, ValueError):
        return False
    if immediate_peer.is_loopback:
        return True

    trusted = set()
    for value in trusted_proxy_ips.split(","):
        value = value.strip()
        if value:
            try:
                trusted.add(ipaddress.ip_address(value))
            except ValueError:
                continue
    if immediate_peer not in trusted:
        return False

    forwarded_for = next(
        (
            value.decode("latin1")
            for name, value in scope.get("headers", [])
            if name.lower() == b"x-forwarded-for"
        ),
        "",
    )
    first_address = forwarded_for.split(",", maxsplit=1)[0].strip()
    try:
        return ipaddress.ip_address(first_address).is_loopback
    except ValueError:
        return False
