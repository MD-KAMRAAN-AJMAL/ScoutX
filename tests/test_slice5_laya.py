"""Deterministic unit and loopback integration tests for Slice 5:
Laya System-1 shadow advisory integration.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scoutx.cli import main as cli_main
from scoutx.document_loader import DocumentLoader
from scoutx.engine import ScoutXEngine
from scoutx.laya import (
    DEFAULT_LAYA_TIMEOUT_SECONDS,
    DeterministicFakeLayaAdvisor,
    LayaConfig,
    LayaHTTPAdvisor,
    build_laya_query_state,
    validate_laya_response,
)
from scoutx.models import LayaAdvisorSuggestion, SearchLane
from scoutx.query_understanding import QueryUnderstandingModule
from scoutx.retrieval import DocumentCorpus, RetrievalToolbox
from scoutx.router import ComplexityRoutingAgent


class Slice5LayaShadowTests(unittest.TestCase):
    def test_01_laya_config_validation_and_url_normalization(self) -> None:
        # Invalid configs
        with self.assertRaises(ValueError):
            LayaConfig(enabled="yes")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LayaConfig(base_url="not-a-url")
        with self.assertRaises(ValueError):
            LayaConfig(api_key_env="raw-key-value-1234")
        with self.assertRaises(ValueError):
            LayaConfig(timeout_seconds=-0.5)
        with self.assertRaises(ValueError):
            LayaConfig(timeout_seconds=float("nan"))
        with self.assertRaises(ValueError):
            LayaConfig(escalation_threshold=1.5)
        with self.assertRaises(ValueError):
            LayaConfig(max_query_chars=10)

        # URL normalization: strips trailing slashes and redundant endpoint suffixes
        cfg1 = LayaConfig(base_url="http://127.0.0.1:8001/")
        self.assertEqual(cfg1.base_url, "http://127.0.0.1:8001")
        cfg2 = LayaConfig(base_url="http://127.0.0.1:8001/v1/systemone")
        self.assertEqual(cfg2.base_url, "http://127.0.0.1:8001")

        # API key resolution
        cfg3 = LayaConfig(api_key_env="TEST_LAYA_KEY")
        with patch.dict(os.environ, {"TEST_LAYA_KEY": "secret-token-xyz"}):
            self.assertEqual(cfg3.resolve_api_key(), "secret-token-xyz")
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(cfg3.resolve_api_key())

    def test_02_query_state_builder_privacy_boundaries(self) -> None:
        parsed = QueryUnderstandingModule().parse("find the Q3 financial forecast spreadsheet")
        state = build_laya_query_state(parsed, complexity_score=0.42, max_chars=300)
        self.assertLessEqual(len(state), 300)
        self.assertIn("financial", state)
        self.assertIn("complexity_score: 0.42", state)
        # Ensure zero paths or system artifacts are present
        self.assertNotIn("/Users/", state)
        self.assertNotIn("path", state.lower().split("\n")[0])

    def test_03_response_validation_conformance_and_error_handling(self) -> None:
        # Valid standard response
        valid_payload = {
            "answers": {
                "lane": {"choice": "standard", "confidence": 0.88},
                "requires_escalation": {"noul": 0.15},
            },
            "routing": {"model": "laya-multilingual", "latency_ms": 18.5},
        }
        suggestion = validate_laya_response(valid_payload, escalation_threshold=0.5)
        self.assertEqual(suggestion.status, "ok")
        self.assertEqual(suggestion.suggested_lane, SearchLane.STANDARD)
        self.assertEqual(suggestion.escalation_probability, 0.15)
        self.assertFalse(suggestion.suggested_escalate)
        self.assertEqual(suggestion.confidence, 0.88)
        self.assertEqual(suggestion.model_identifier, "laya-multilingual")
        self.assertEqual(suggestion.latency_ms, 18.5)

        # Valid escalation response (noul >= 0.5)
        esc_payload = {
            "answers": {
                "lane": {"choice": "deep"},
                "requires_escalation": {"noul": 0.85},
            }
        }
        esc_sugg = validate_laya_response(esc_payload, escalation_threshold=0.5)
        self.assertEqual(esc_sugg.status, "ok")
        self.assertEqual(esc_sugg.suggested_lane, SearchLane.DEEP)
        self.assertTrue(esc_sugg.suggested_escalate)
        self.assertEqual(esc_sugg.escalation_probability, 0.85)
        self.assertIsNone(esc_sugg.confidence)

        # Invalid responses: non-dict top-level
        bad1 = validate_laya_response(["answers"])
        self.assertEqual(bad1.status, "invalid_response")

        # Invalid responses: missing answers
        bad2 = validate_laya_response({"routing": {}})
        self.assertEqual(bad2.status, "invalid_response")

        # Invalid responses: invalid lane choice
        bad3 = validate_laya_response({"answers": {"lane": {"choice": "super_deep"}}})
        self.assertEqual(bad3.status, "invalid_response")

        # Invalid responses: boolean confidence rejected
        bad4 = validate_laya_response(
            {"answers": {"lane": {"choice": "fast", "confidence": True}}}
        )
        self.assertEqual(bad4.status, "invalid_response")

        # Invalid responses: out-of-range confidence rejected
        bad5 = validate_laya_response(
            {"answers": {"lane": {"choice": "fast", "confidence": 1.5}}}
        )
        self.assertEqual(bad5.status, "invalid_response")

        # Invalid responses: boolean noul rejected
        bad6 = validate_laya_response(
            {
                "answers": {
                    "lane": {"choice": "fast"},
                    "requires_escalation": {"noul": False},
                }
            }
        )
        self.assertEqual(bad6.status, "invalid_response")

        # Invalid responses: out-of-range noul rejected
        bad7 = validate_laya_response(
            {
                "answers": {
                    "lane": {"choice": "fast"},
                    "requires_escalation": {"noul": -0.1},
                }
            }
        )
        self.assertEqual(bad7.status, "invalid_response")

    def test_04_shadow_mode_invariance_under_adversarial_advisor_disagreement(self) -> None:
        """Adversarial test: verify that even when Laya aggressively disagrees with ScoutX
        on lane selection and escalation, the authoritative search results, strategy, tools,
        reflection decisions, and candidate ordering remain 100% IDENTICAL.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            f1 = root / "Pricing_v3.pptx"
            f1.write_text("enterprise pricing forecast revenue slides", encoding="utf-8")
            f2 = root / "meeting_notes.md"
            f2.write_text("sync notes and action items", encoding="utf-8")

            corpus = DocumentCorpus.from_directory(root)
            toolbox = RetrievalToolbox(corpus)

            # 1. Run clear query without any advisor
            agent_no_advisor = ComplexityRoutingAgent(toolbox)
            parsed_clear = QueryUnderstandingModule().parse('"Pricing_v3.pptx"')
            res_baseline = agent_no_advisor.run(parsed_clear)

            # 2. Run clear query with adversarial fake advisor that advises DEEP and escalate=True
            adversarial_advisor = DeterministicFakeLayaAdvisor(
                suggested_lane=SearchLane.DEEP,
                escalation_probability=0.99,
                confidence=0.99,
            )
            agent_with_advisor = ComplexityRoutingAgent(toolbox, laya_advisor=adversarial_advisor)
            res_advised = agent_with_advisor.run(parsed_clear)

            # Verify identical outcome: candidates, lane, reflection
            self.assertEqual(
                tuple(c.document.id for c in res_baseline.candidates),
                tuple(c.document.id for c in res_advised.candidates),
            )
            self.assertEqual(
                tuple(c.score for c in res_baseline.candidates),
                tuple(c.score for c in res_advised.candidates),
            )
            self.assertEqual(res_baseline.trace[0].strategy.lane, SearchLane.FAST)
            self.assertEqual(res_advised.trace[0].strategy.lane, SearchLane.FAST)
            self.assertEqual(
                res_baseline.trace[0].reflection.confident,
                res_advised.trace[0].reflection.confident,
            )

            # Verify shadow telemetry was captured in iteration 0
            adv_trace = res_advised.trace[0].laya_advisory
            self.assertIsNotNone(adv_trace)
            assert adv_trace is not None
            self.assertEqual(adv_trace.suggested_lane, SearchLane.DEEP)
            self.assertEqual(adv_trace.escalation_probability, 0.99)
            self.assertTrue(adv_trace.suggested_escalate)
            self.assertEqual(adv_trace.confidence, 0.99)
            self.assertEqual(adversarial_advisor.call_count, 1)

            # 3. Test multi-pass escalation with an ambiguous query
            parsed_vague = QueryUnderstandingModule().parse("that thing about numbers from a while ago")
            res_vague_baseline = agent_no_advisor.run(parsed_vague)

            adversarial_advisor.call_count = 0
            # Adversary suggests FAST and no escalation
            adversarial_advisor.suggested_lane = SearchLane.FAST
            adversarial_advisor.escalation_probability = 0.05
            res_vague_advised = agent_with_advisor.run(parsed_vague)

            # Multi-pass trace must match in candidate sets, strategies, and reflections
            self.assertEqual(len(res_vague_baseline.trace), len(res_vague_advised.trace))
            for step_b, step_a in zip(res_vague_baseline.trace, res_vague_advised.trace):
                self.assertEqual(step_b.strategy.lane, step_a.strategy.lane)
                self.assertEqual(step_b.reflection.confident, step_a.reflection.confident)

            # Exactly one Laya advisory call per search, attached strictly to iteration 0
            self.assertEqual(adversarial_advisor.call_count, 1)
            self.assertIsNotNone(res_vague_advised.trace[0].laya_advisory)
            for later_step in res_vague_advised.trace[1:]:
                self.assertIsNone(later_step.laya_advisory)

    def test_05_http_bridge_contract_payload_and_bearer_token(self) -> None:
        captured_requests: list[tuple[str, dict[str, str], dict[str, Any]]] = []

        class _LayaLoopbackHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")
                parsed_body = json.loads(raw_body)
                headers_snap = {k.lower(): v for k, v in self.headers.items()}
                captured_requests.append((self.path, headers_snap, parsed_body))

                resp_obj = {
                    "answers": {
                        "lane": {"choice": "standard", "confidence": 0.82},
                        "requires_escalation": {"noul": 0.25},
                    },
                    "routing": {"model": "laya-systemone-0.1", "latency_ms": 22.0},
                }
                resp_bytes = json.dumps(resp_obj).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_bytes)))
                self.end_headers()
                self.wfile.write(resp_bytes)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), _LayaLoopbackHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"SECRET_LAYA_ENV": "BearerTokenSecret999"}):
                cfg = LayaConfig(
                    enabled=True,
                    base_url=f"http://127.0.0.1:{port}",
                    api_key_env="SECRET_LAYA_ENV",
                    timeout_seconds=1.0,
                )
                advisor = LayaHTTPAdvisor(cfg)
                parsed = QueryUnderstandingModule().parse("quarterly financial plan")
                sugg = advisor.advise_search(parsed, complexity_score=0.45)

                self.assertEqual(sugg.status, "ok")
                self.assertEqual(sugg.suggested_lane, SearchLane.STANDARD)
                self.assertEqual(sugg.escalation_probability, 0.25)
                self.assertFalse(sugg.suggested_escalate)
                self.assertEqual(sugg.confidence, 0.82)
                self.assertEqual(sugg.model_identifier, "laya-systemone-0.1")

                # Verify captured HTTP request
                self.assertEqual(len(captured_requests), 1)
                req_path, req_headers, req_body = captured_requests[0]
                self.assertEqual(req_path, "/v1/systemone")
                self.assertEqual(
                    req_headers.get("authorization"), "Bearer BearerTokenSecret999"
                )
                self.assertIn("financial", req_body["state"])
                self.assertIn("complexity_score: 0.45", req_body["state"])
                self.assertEqual(req_body["questions"]["lane"]["type"], "choice")
                self.assertIn("fast", req_body["questions"]["lane"]["criteria"])
                self.assertEqual(req_body["questions"]["requires_escalation"]["type"], "noul")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_06_http_bridge_timeout_deadline_enforcement_and_secret_redaction(self) -> None:
        mode = {"sleep": 0.0, "status": 200}

        class _SlowOrFailingHandler(BaseHTTPRequestHandler):
            def handle(self) -> None:
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                if mode["sleep"] > 0.0:
                    time.sleep(mode["sleep"])
                try:
                    if mode["status"] == 500:
                        err_body = b'{"error": "crash with secret key BearerTokenSecret999"}'
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_body)))
                        self.end_headers()
                        self.wfile.write(err_body)
                        return
                    resp_bytes = b'{"answers": {"lane": {"choice": "fast"}}}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp_bytes)))
                    self.end_headers()
                    self.wfile.write(resp_bytes)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), _SlowOrFailingHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"SECRET_LAYA_ENV": "BearerTokenSecret999"}):
                # 1. Timeout deadline: server sleeps 0.6s, deadline is 0.2s
                mode["sleep"] = 0.6
                cfg_timeout = LayaConfig(
                    enabled=True,
                    base_url=f"http://127.0.0.1:{port}",
                    api_key_env="SECRET_LAYA_ENV",
                    timeout_seconds=0.2,
                )
                advisor = LayaHTTPAdvisor(cfg_timeout)
                parsed = QueryUnderstandingModule().parse("find my notes")
                sugg_timeout = advisor.advise_search(parsed, complexity_score=0.2)
                self.assertEqual(sugg_timeout.status, "timeout")
                self.assertIn("deadline", (sugg_timeout.error_message or "").lower())

                # 2. HTTP 500 failure with secret redaction
                mode["sleep"] = 0.0
                mode["status"] = 500
                cfg_fail = LayaConfig(
                    enabled=True,
                    base_url=f"http://127.0.0.1:{port}",
                    api_key_env="SECRET_LAYA_ENV",
                    timeout_seconds=1.0,
                )
                sugg_500 = LayaHTTPAdvisor(cfg_fail).advise_search(parsed, complexity_score=0.2)
                self.assertEqual(sugg_500.status, "unavailable")
                self.assertNotIn("BearerTokenSecret999", sugg_500.error_message or "")
                self.assertIn("[REDACTED]", sugg_500.error_message or "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    def test_07_cli_laya_shadow_flag_and_trace_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            root.mkdir()
            doc_path = root / "pricing_summary.md"
            doc_path.write_text("annual enterprise pricing forecast sheet", encoding="utf-8")

            # Run search with --laya-shadow pointing to an unavailable port
            # CLI must not crash, search must succeed, and trace must print Laya status
            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                exit_code = cli_main(
                    [
                        "search",
                        "pricing forecast",
                        "--root",
                        str(root),
                        "--trace",
                        "--laya-shadow",
                        "--laya-base-url",
                        "http://127.0.0.1:1",  # Unavailable port
                        "--laya-timeout",
                        "0.2",
                    ]
                )
            self.assertEqual(exit_code, 0)
            output = out_buf.getvalue()
            self.assertIn("pricing_summary.md", output)
            self.assertIn("Router trace:", output)
            self.assertIn("laya[status=unavailable", output)

    def test_08_http_bridge_slow_streaming_response_exceeds_total_deadline(self) -> None:
        """Loopback test: server sends valid chunks slowly where each chunk arrives in <500ms,
        but total streaming duration exceeds the 500ms total deadline.

        Asserts that:
        - LayaHTTPAdvisor returns status='timeout' within bounded tolerance of deadline.
        - ScoutX still returns its normal deterministic search result.
        - No Laya suggestion changes strategy, tools, candidates, or reflection.
        - Telemetry is bounded and sanitized.
        - No retry is performed (server receives exactly expected requests).
        """
        request_counter = {"count": 0}

        class _SlowStreamingHandler(BaseHTTPRequestHandler):
            def handle(self) -> None:
                try:
                    super().handle()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_POST(self) -> None:
                request_counter["count"] += 1
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                # 4 chunks, 0.2s delay before each = 0.8s total (> 0.5s deadline)
                # Each chunk delay (0.2s) is < 0.5s
                chunks = [
                    b'{"answers": ',
                    b'{"lane": {"choice": "deep"}, "confidence": 0.99}, ',
                    b'"requires_escalation": {"noul": 0.99}}, ',
                    b'"routing": {"model": "slow-streamer-v1"}}',
                ]
                try:
                    for chunk in chunks:
                        time.sleep(0.2)
                        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args: object) -> None:
                return

        server = HTTPServer(("127.0.0.1", 0), _SlowStreamingHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            cfg = LayaConfig(
                enabled=True,
                base_url=f"http://127.0.0.1:{port}",
                timeout_seconds=0.5,
            )
            advisor = LayaHTTPAdvisor(cfg)
            parsed = QueryUnderstandingModule().parse("revenue projections spreadsheet")

            t0 = time.monotonic()
            sugg = advisor.advise_search(parsed, complexity_score=0.35)
            t1 = time.monotonic()
            elapsed = t1 - t0

            # 1. Advisor returns timeout within bounded tolerance of 0.5s (not 0.8s+)
            self.assertEqual(sugg.status, "timeout")
            self.assertGreaterEqual(elapsed, 0.45)
            self.assertLess(elapsed, 0.75)
            self.assertIn("deadline", (sugg.error_message or "").lower())
            self.assertEqual(request_counter["count"], 1)  # No retry performed

            # 2. ScoutX search returns normal deterministic result under timeout
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                f1 = root / "revenue_forecast.xlsx"
                f1.write_text("q3 financial projections and revenue forecast", encoding="utf-8")
                f2 = root / "other.txt"
                f2.write_text("random file", encoding="utf-8")

                corpus = DocumentCorpus.from_directory(root)
                toolbox = RetrievalToolbox(corpus)

                # Baseline without advisor
                agent_baseline = ComplexityRoutingAgent(toolbox)
                res_baseline = agent_baseline.run(parsed)

                # Run with slow streaming advisor that times out
                agent_with_advisor = ComplexityRoutingAgent(toolbox, laya_advisor=advisor)
                res_advised = agent_with_advisor.run(parsed)

                # Exactly identical candidates, scores, lane, reflection
                self.assertEqual(
                    tuple(c.document.id for c in res_baseline.candidates),
                    tuple(c.document.id for c in res_advised.candidates),
                )
                self.assertEqual(
                    tuple(c.score for c in res_baseline.candidates),
                    tuple(c.score for c in res_advised.candidates),
                )
                self.assertEqual(res_baseline.trace[0].strategy.lane, res_advised.trace[0].strategy.lane)
                self.assertEqual(
                    res_baseline.trace[0].reflection.confident,
                    res_advised.trace[0].reflection.confident,
                )

                # Telemetry in trace step 0 is bounded and sanitized
                adv_telemetry = res_advised.trace[0].laya_advisory
                self.assertIsNotNone(adv_telemetry)
                assert adv_telemetry is not None
                self.assertEqual(adv_telemetry.status, "timeout")
                self.assertIsNone(adv_telemetry.suggested_lane)
                self.assertIsNone(adv_telemetry.suggested_escalate)
                self.assertIn("deadline", (adv_telemetry.error_message or "").lower())

                # Exactly 2 requests total: 1 direct call + 1 search call. Zero retries.
                self.assertEqual(request_counter["count"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
