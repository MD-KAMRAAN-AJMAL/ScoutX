"""Optional local OpenTelemetry export plus bounded application metrics."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)
_correlation_id: ContextVar[str | None] = ContextVar("scoutx_correlation_id", default=None)
_log_factory_installed = False


@dataclass
class TelemetryRuntime:
    tracer_provider: Any
    meter_provider: Any


_request_counter = None
_request_duration = None


class SafeJsonFormatter(logging.Formatter):
    """Emit allowlisted structured fields without exception text or call arguments."""

    _FIELDS = (
        "component",
        "operation",
        "correlation_id",
        "request_id",
        "duration_ms",
        "status",
        "route",
        "dependency",
        "error_type",
        "candidate_count",
        "retry_count",
        "category",
        "result_count",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "event": str(record.msg),
        }
        for field in self._FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info and record.exc_info[1] is not None:
            payload["error_type"] = type(record.exc_info[1]).__name__
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_structured_logging() -> None:
    """Configure application logs as safe JSON records while leaving Uvicorn intact."""
    application_logger = logging.getLogger("app")
    if any(getattr(handler, "_scoutx_json", False) for handler in application_logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeJsonFormatter())
    handler._scoutx_json = True  # type: ignore[attr-defined]
    application_logger.addHandler(handler)
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False


def install_correlation_log_context() -> None:
    """Attach the current request correlation ID to emitted Python log records."""
    global _log_factory_installed
    if _log_factory_installed:
        return
    previous_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous_factory(*args, **kwargs)
        record.correlation_id = _correlation_id.get()
        return record

    logging.setLogRecordFactory(factory)
    _log_factory_installed = True


def set_correlation_id(value: str):
    return _correlation_id.set(value)


def reset_correlation_id(token) -> None:
    _correlation_id.reset(token)


def configure_telemetry(app) -> TelemetryRuntime | None:
    """Enable tracing/metrics only when an explicit OTLP endpoint is configured."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.otel_enabled:
        return None

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    endpoint = str(settings.otel_exporter_endpoint).rstrip("/")
    endpoint = endpoint.removesuffix("/v1/traces")
    resource = Resource.create(
        {SERVICE_NAME: "scoutx-api", "deployment.environment.name": settings.environment}
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
    )
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"), export_interval_millis=30_000
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(meter_provider)
    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        excluded_urls=r"/api/v1/(health|live|ready)$",
    )
    from app.database.engine import engine

    SQLAlchemyInstrumentor().instrument(engine=engine, enable_commenter=False)
    HTTPXClientInstrumentor().instrument()
    global _request_counter, _request_duration
    meter = meter_provider.get_meter("scoutx.api")
    _request_counter = meter.create_counter("scoutx.api.requests", unit="{request}")
    _request_duration = meter.create_histogram("scoutx.api.duration", unit="ms")
    logger.info("OpenTelemetry tracing and metrics enabled")
    return TelemetryRuntime(tracer_provider, meter_provider)


def record_api_request(*, method: str, route: str, status: int, duration_ms: float) -> None:
    """Record low-cardinality API metrics without user text or file paths."""
    if _request_counter is None or _request_duration is None:
        return
    attributes = {
        "http.request.method": method,
        "http.route": route,
        "http.response.status_code": status,
    }
    _request_counter.add(1, attributes)
    _request_duration.record(duration_ms, attributes)
