"""Laya System-1 advisory adapter and contracts for ScoutX.

This module integrates the non-autoregressive Laya decision engine in strict shadow mode:
- Laya evaluates query complexity and suggests a search lane and escalation probability.
- Suggestions are recorded exclusively as shadow telemetry in RouterTraceStep.laya_advisory.
- ScoutX's ComplexityRoutingAgent remains the sole autonomous decision-maker.
- No third-party dependencies are required; HTTP communication uses urllib.request in an isolated helper process.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from scoutx.models import LayaAdvisorSuggestion, ParsedQuery, SearchLane

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PRINTABLE_ASCII_RE = re.compile(r"^[\x20-\x7E]+$")

DEFAULT_LAYA_TIMEOUT_SECONDS: float = 0.5  # 500 ms client-side request deadline
DEFAULT_ESCALATION_THRESHOLD: float = 0.5  # Shadow comparison convention (not calibrated)

LAYA_ROUTING_QUESTIONS: dict[str, Any] = {
    "lane": {
        "type": "choice",
        "instructions": "Classify the appropriate search lane for this file discovery query based on ambiguity and constraints.",
        "criteria": {
            "fast": "Specific filename hints, exact file paths, or unambiguous file targets with zero complexity.",
            "standard": "Clear intent with specific keyword, date, or extension constraints requiring normal multi-tool search.",
            "deep": "Vague, highly ambiguous, multi-intent, or difficult conceptual queries needing widened search and reasoning.",
        },
    },
    "requires_escalation": {
        "type": "noul",
        "instructions": "Does this query require escalating beyond the standard retrieval lane?",
    },
}


def _redact_secret(text: str, secret: str | None) -> str:
    if not text:
        return ""
    redacted = text
    if secret and len(secret) >= 4:
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _sanitize_string(value: Any, max_len: int = 64) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = "".join(ch for ch in value if 32 <= ord(ch) <= 126).strip()
    return cleaned[:max_len]


@dataclass(frozen=True)
class LayaConfig:
    """Configuration for Laya System-1 advisory integration in shadow mode."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8001"
    api_key_env: str | None = None
    timeout_seconds: float = DEFAULT_LAYA_TIMEOUT_SECONDS
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD
    max_query_chars: int = 500

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"enabled must be a bool, got {self.enabled!r}")

        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        parsed = urllib.parse.urlparse(self.base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"base_url must be a valid http(s) URL, got {self.base_url!r}")
        normalized_base = self.base_url.strip().rstrip("/")
        # If the caller supplied path ending with /v1/systemone, strip it so it normalizes cleanly
        if normalized_base.endswith("/v1/systemone"):
            normalized_base = normalized_base[:-len("/v1/systemone")]
        object.__setattr__(self, "base_url", normalized_base)

        if self.api_key_env is not None:
            if not isinstance(self.api_key_env, str) or not _ENV_VAR_NAME_RE.match(self.api_key_env):
                raise ValueError(
                    "api_key_env must be a valid environment variable name (e.g. 'LAYA_API_KEY'), never a raw secret"
                )

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or float(self.timeout_seconds) <= 0.0
        ):
            raise ValueError(
                f"timeout_seconds must be a positive finite number, got {self.timeout_seconds!r}"
            )
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

        if (
            isinstance(self.escalation_threshold, bool)
            or not isinstance(self.escalation_threshold, (int, float))
            or not math.isfinite(self.escalation_threshold)
            or not (0.0 <= float(self.escalation_threshold) <= 1.0)
        ):
            raise ValueError(
                f"escalation_threshold must be a finite number in [0.0, 1.0], got {self.escalation_threshold!r}"
            )
        object.__setattr__(self, "escalation_threshold", float(self.escalation_threshold))

        if (
            isinstance(self.max_query_chars, bool)
            or not isinstance(self.max_query_chars, int)
            or self.max_query_chars < 32
        ):
            raise ValueError(
                f"max_query_chars must be an integer >= 32, got {self.max_query_chars!r}"
            )

    def resolve_api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        return value if value else None


def build_laya_query_state(parsed_query: ParsedQuery, complexity_score: float, max_chars: int = 500) -> str:
    """Build a sanitized query description string for Laya evaluation.

    Strictly excludes host filesystem paths, full document contents, and profile data.
    """
    raw_query = (parsed_query.raw or "").strip()[:max_chars]
    intent = (parsed_query.intent or "").strip()
    keywords = " ".join(parsed_query.keywords)
    file_types = " ".join(parsed_query.file_types)
    hints = " ".join(parsed_query.filename_hints)
    parts = [f"query: {raw_query}"]
    if intent:
        parts.append(f"intent: {intent}")
    if keywords:
        parts.append(f"keywords: {keywords}")
    if file_types:
        parts.append(f"file_types: {file_types}")
    if hints:
        parts.append(f"filename_hints: {hints}")
    parts.append(f"complexity_score: {complexity_score:.2f}")
    return "\n".join(parts)[:max_chars]


def validate_laya_response(
    response_json: Any,
    *,
    escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD,
    elapsed_ms: float = 0.0,
) -> LayaAdvisorSuggestion:
    """Strictly validate and sanitize a Laya POST /v1/systemone response.

    Returns LayaAdvisorSuggestion with status='ok' on success, or status='invalid_response'
    with a sanitized error message on any structural or type violation.
    """
    if not isinstance(response_json, Mapping):
        return LayaAdvisorSuggestion(
            status="invalid_response",
            error_message=f"Laya response must be a JSON object, got {type(response_json).__name__}",
            latency_ms=elapsed_ms,
        )

    answers = response_json.get("answers")
    if not isinstance(answers, Mapping):
        return LayaAdvisorSuggestion(
            status="invalid_response",
            error_message="Laya response missing 'answers' object",
            latency_ms=elapsed_ms,
        )

    # 1. Validate lane choice
    lane_entry = answers.get("lane")
    if not isinstance(lane_entry, Mapping):
        return LayaAdvisorSuggestion(
            status="invalid_response",
            error_message="Laya response missing 'lane' object in answers",
            latency_ms=elapsed_ms,
        )

    raw_lane = lane_entry.get("choice")
    if not isinstance(raw_lane, str) or raw_lane not in {"fast", "standard", "deep"}:
        return LayaAdvisorSuggestion(
            status="invalid_response",
            error_message=f"Invalid lane choice in Laya response: {raw_lane!r}",
            latency_ms=elapsed_ms,
        )
    suggested_lane = {
        "fast": SearchLane.FAST,
        "standard": SearchLane.STANDARD,
        "deep": SearchLane.DEEP,
    }[raw_lane]

    # Model-reported confidence
    confidence_val = lane_entry.get("confidence")
    confidence: float | None = None
    if confidence_val is not None:
        if isinstance(confidence_val, bool) or not isinstance(confidence_val, (int, float)):
            return LayaAdvisorSuggestion(
                status="invalid_response",
                error_message=f"Confidence must be a numeric float, got {confidence_val!r}",
                latency_ms=elapsed_ms,
            )
        f_conf = float(confidence_val)
        if not math.isfinite(f_conf) or not (0.0 <= f_conf <= 1.0):
            return LayaAdvisorSuggestion(
                status="invalid_response",
                error_message=f"Confidence must be in [0.0, 1.0], got {confidence_val!r}",
                latency_ms=elapsed_ms,
            )
        confidence = f_conf

    # 2. Validate escalation decision
    esc_entry = answers.get("requires_escalation")
    escalation_prob: float | None = None
    suggested_escalate: bool | None = None

    if isinstance(esc_entry, Mapping):
        noul_val = esc_entry.get("noul")
        if noul_val is not None:
            if isinstance(noul_val, bool) or not isinstance(noul_val, (int, float)):
                return LayaAdvisorSuggestion(
                    status="invalid_response",
                    error_message=f"Escalation 'noul' must be a numeric float, got {noul_val!r}",
                    latency_ms=elapsed_ms,
                )
            f_noul = float(noul_val)
            if not math.isfinite(f_noul) or not (0.0 <= f_noul <= 1.0):
                return LayaAdvisorSuggestion(
                    status="invalid_response",
                    error_message=f"Escalation 'noul' must be in [0.0, 1.0], got {noul_val!r}",
                    latency_ms=elapsed_ms,
                )
            escalation_prob = f_noul
            suggested_escalate = escalation_prob >= escalation_threshold

    # 3. Model identifier & latency
    routing = response_json.get("routing")
    model_identifier = "laya"
    reported_latency = elapsed_ms
    if isinstance(routing, Mapping):
        raw_model = routing.get("model")
        if raw_model is not None:
            model_identifier = _sanitize_string(raw_model, max_len=64) or "laya"
        raw_lat = routing.get("latency_ms")
        if (
            raw_lat is not None
            and not isinstance(raw_lat, bool)
            and isinstance(raw_lat, (int, float))
            and math.isfinite(raw_lat)
            and float(raw_lat) >= 0.0
        ):
            reported_latency = float(raw_lat)

    return LayaAdvisorSuggestion(
        suggested_lane=suggested_lane,
        suggested_escalate=suggested_escalate,
        escalation_probability=escalation_prob,
        confidence=confidence,
        model_identifier=model_identifier,
        latency_ms=reported_latency,
        status="ok",
    )


@runtime_checkable
class LayaAdvisor(Protocol):
    """Protocol for Laya System-1 advisory evaluations."""

    def advise_search(
        self,
        parsed_query: ParsedQuery,
        complexity_score: float,
    ) -> LayaAdvisorSuggestion:
        """Evaluate query complexity and produce a single advisory suggestion per search."""
        ...


def _laya_request_worker(
    url: str,
    body_bytes: bytes,
    headers: dict[str, str],
    timeout_seconds: float,
    pipe_send: Any,
) -> None:
    """Isolated, terminable helper process target for HTTP execution.

    Runs in a dedicated helper process so that the parent can enforce a hard total
    wall-clock deadline and reliably terminate the child process if the deadline expires,
    avoiding leaked sockets or unkillable background threads.
    """
    try:
        req = urllib.request.Request(url, data=body_bytes, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            pipe_send.send(("ok", 200, raw_body, None))
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:200]
        except Exception:
            pass
        pipe_send.send(("http_error", exc.code, error_body, str(exc.reason)))
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            pipe_send.send(("timeout", None, None, str(reason)))
        else:
            pipe_send.send(("url_error", None, None, str(reason)))
    except (TimeoutError, socket.timeout) as exc:
        pipe_send.send(("timeout", None, None, str(exc)))
    except OSError as exc:
        pipe_send.send(("os_error", None, None, str(exc)))
    except Exception as exc:
        pipe_send.send(("error", None, None, str(exc)))
    finally:
        try:
            pipe_send.close()
        except Exception:
            pass


class LayaHTTPAdvisor:
    """HTTP client for Laya's POST /v1/systemone endpoint with strict deadline enforcement."""

    def __init__(self, config: LayaConfig):
        self.config = config

    def advise_search(
        self,
        parsed_query: ParsedQuery,
        complexity_score: float,
    ) -> LayaAdvisorSuggestion:
        if not self.config.enabled:
            return LayaAdvisorSuggestion(status="disabled")

        state_text = build_laya_query_state(
            parsed_query, complexity_score, max_chars=self.config.max_query_chars
        )
        url = f"{self.config.base_url}/v1/systemone"
        payload = {
            "state": state_text,
            "questions": LAYA_ROUTING_QUESTIONS,
        }
        api_key = self.config.resolve_api_key()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        start_time = time.monotonic()
        deadline = self.config.timeout_seconds

        recv_conn, send_conn = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(
            target=_laya_request_worker,
            args=(url, body_bytes, headers, deadline, send_conn),
        )

        try:
            p.start()
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            recv_conn.close()
            send_conn.close()
            msg = _redact_secret(f"Failed to launch Laya helper: {exc}", api_key)
            return LayaAdvisorSuggestion(
                status="unavailable",
                error_message=_sanitize_string(msg, max_len=200),
                latency_ms=elapsed_ms,
            )

        # Close send end in parent process so EOF can be detected
        send_conn.close()

        # Wait on pipe with remaining wall-clock deadline
        rem_seconds = max(0.0, deadline - (time.monotonic() - start_time))
        has_data = recv_conn.poll(rem_seconds)
        raw_result: tuple[str, Any, Any, Any] | None = None
        timed_out = not has_data

        if has_data:
            try:
                raw_result = recv_conn.recv()
            except (EOFError, BrokenPipeError):
                raw_result = None

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Terminate helper process if still alive or timed out
        if timed_out or p.is_alive():
            try:
                p.terminate()
                p.join(timeout=0.05)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=0.05)
            except Exception:
                pass
        else:
            try:
                p.join(timeout=0.05)
            except Exception:
                pass

        try:
            p.close()
        except Exception:
            pass
        try:
            recv_conn.close()
        except Exception:
            pass

        if timed_out or (raw_result is not None and raw_result[0] == "timeout"):
            return LayaAdvisorSuggestion(
                status="timeout",
                error_message=f"Laya request exceeded deadline ({self.config.timeout_seconds}s)",
                latency_ms=elapsed_ms,
            )

        if raw_result is None:
            # Child process exited without sending result (e.g. killed or crashed)
            return LayaAdvisorSuggestion(
                status="unavailable",
                error_message="Laya helper process exited unexpectedly",
                latency_ms=elapsed_ms,
            )

        result_tag, code_or_reason, body_or_err, extra = raw_result

        if result_tag == "http_error":
            msg = _redact_secret(
                f"HTTP {code_or_reason} from Laya: {body_or_err or extra}", api_key
            )
            return LayaAdvisorSuggestion(
                status="unavailable",
                error_message=_sanitize_string(msg, max_len=200),
                latency_ms=elapsed_ms,
            )

        if result_tag in {"url_error", "os_error", "error"}:
            msg = _redact_secret(f"Failed to reach Laya: {extra or body_or_err}", api_key)
            return LayaAdvisorSuggestion(
                status="unavailable",
                error_message=_sanitize_string(msg, max_len=200),
                latency_ms=elapsed_ms,
            )

        if result_tag != "ok":
            return LayaAdvisorSuggestion(
                status="unavailable",
                error_message=f"Unknown Laya helper result: {result_tag}",
                latency_ms=elapsed_ms,
            )

        raw_response = body_or_err
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return LayaAdvisorSuggestion(
                status="invalid_response",
                error_message=f"Laya returned non-JSON response: {exc}",
                latency_ms=elapsed_ms,
            )

        return validate_laya_response(
            parsed,
            escalation_threshold=self.config.escalation_threshold,
            elapsed_ms=elapsed_ms,
        )


class DeterministicFakeLayaAdvisor:
    """Completely offline, deterministic fake Laya advisor for tests."""

    def __init__(
        self,
        *,
        suggested_lane: SearchLane | None = SearchLane.STANDARD,
        escalation_probability: float | None = 0.20,
        confidence: float | None = 0.85,
        status: str = "ok",
        error_message: str | None = None,
        latency_ms: float = 12.0,
        model_identifier: str = "fake-laya-v1",
        escalation_threshold: float = DEFAULT_ESCALATION_THRESHOLD,
    ):
        self.suggested_lane = suggested_lane
        self.escalation_probability = escalation_probability
        self.confidence = confidence
        self.status = status
        self.error_message = error_message
        self.latency_ms = latency_ms
        self.model_identifier = model_identifier
        self.escalation_threshold = escalation_threshold
        self.call_count = 0

    def advise_search(
        self,
        parsed_query: ParsedQuery,
        complexity_score: float,
    ) -> LayaAdvisorSuggestion:
        self.call_count += 1
        suggested_escalate = (
            self.escalation_probability >= self.escalation_threshold
            if self.escalation_probability is not None
            else None
        )
        return LayaAdvisorSuggestion(
            suggested_lane=self.suggested_lane,
            suggested_escalate=suggested_escalate,
            escalation_probability=self.escalation_probability,
            confidence=self.confidence,
            model_identifier=self.model_identifier,
            latency_ms=self.latency_ms,
            status=self.status,
            error_message=self.error_message,
        )
