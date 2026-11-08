from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.compare import compare_reports
from evaluation.dataset import load_dataset
from evaluation.metrics import (
    average_precision,
    bootstrap_mean_interval,
    edit_error_rate,
    hit_rate_at_k,
    query_metrics,
    reciprocal_rank,
)
from evaluation.privacy import safe_configuration
from evaluation.report import new_report, write_reports
from evaluation.runner import (
    evaluate_agent,
    evaluate_citations,
    evaluate_observations,
    evaluate_rag,
    evaluate_rankings,
    evaluate_recommendations,
    evaluate_reranking,
    offline_agent_routing_run,
    offline_lexical_run,
    offline_personalization_runs,
)
from evaluation.schema import EvaluationDataset, RankingRun


def test_versioned_controlled_dataset_validates_and_has_stable_ids() -> None:
    dataset = load_dataset()
    assert dataset.name == "scoutx-small"
    assert dataset.version == "1.0.0"
    assert len({item.logical_file_id for item in dataset.documents}) == len(dataset.documents)
    assert len({item.query_id for item in dataset.queries}) == len(dataset.queries)
    assert {item.kind for item in dataset.queries} >= {
        "filename",
        "semantic",
        "hybrid",
        "code",
        "spreadsheet",
        "ocr",
        "audio",
        "video",
        "no_result",
    }


def test_dataset_rejects_unknown_judgment_and_duplicate_ids() -> None:
    payload = {
        "name": "broken",
        "version": "1.0.0",
        "description": "x",
        "judgment_policy": "x",
        "documents": [
            {
                "logical_file_id": "d",
                "filename": "d.txt",
                "extension": ".txt",
                "mime_type": "text/plain",
            }
        ],
        "queries": [
            {
                "query_id": "q",
                "text": "q",
                "kind": "x",
                "judgments": [{"logical_file_id": "absent", "grade": 1}],
            }
        ],
    }
    with pytest.raises(ValidationError, match="unknown judged"):
        EvaluationDataset.model_validate(payload)
    payload["queries"][0]["judgments"] = [
        {"logical_file_id": "d", "grade": 1},
        {"logical_file_id": "d", "grade": 2},
    ]
    with pytest.raises(ValidationError, match="one judgment"):
        EvaluationDataset.model_validate(payload)


@pytest.mark.parametrize("grade", [-1, 4])
def test_relevance_grade_range_is_validated(grade: int) -> None:
    payload = {
        "name": "x",
        "version": "1.0.0",
        "description": "x",
        "judgment_policy": "x",
        "documents": [
            {
                "logical_file_id": "d",
                "filename": "d.txt",
                "extension": ".txt",
                "mime_type": "text/plain",
            }
        ],
        "queries": [
            {
                "query_id": "q",
                "text": "q",
                "kind": "x",
                "judgments": [{"logical_file_id": "d", "grade": grade}],
            }
        ],
    }
    with pytest.raises(ValidationError):
        EvaluationDataset.model_validate(payload)


def test_ranking_metrics_match_hand_calculated_example() -> None:
    ranked = ["irrelevant", "a", "b", "a"]
    judgments = {"a": 3, "b": 1, "c": 2}
    metrics = query_metrics(ranked, judgments, [1, 2, 3])
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["recall@3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx(1 / 2)
    assert metrics["map"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert metrics["hit_rate@1"] == 0.0
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["duplicate_result_count"] == 1


def test_unjudged_policy_is_explicit_and_changes_ranks() -> None:
    relevance = {"relevant": 3}
    penalized = query_metrics(
        ["unjudged", "relevant"], relevance, [1], unjudged_policy="irrelevant"
    )
    excluded = query_metrics(["unjudged", "relevant"], relevance, [1], unjudged_policy="exclude")
    assert penalized["mrr"] == 0.5
    assert excluded["mrr"] == 1.0
    assert excluded["precision@1"] == 1.0


def test_empty_results_and_empty_relevance_are_defined() -> None:
    metrics = query_metrics([], {}, [1, 5])
    assert metrics["zero_result"] == 1
    assert metrics["mrr"] == 0
    assert metrics["map"] == 0
    assert metrics["recall@5"] == 0
    assert hit_rate_at_k([], {}, 3) == 0
    assert reciprocal_rank([], {}) == 0
    assert average_precision([], {}) == 0


def test_invalid_k_and_policy_fail_clearly() -> None:
    with pytest.raises(ValueError, match="positive"):
        query_metrics([], {}, [0])
    with pytest.raises(ValueError, match="unjudged_policy"):
        query_metrics([], {}, [1], unjudged_policy="unknown")  # type: ignore[arg-type]


def test_evaluation_ranking_produces_aggregate_per_query_latency_and_ci() -> None:
    dataset = load_dataset()
    run = RankingRun(
        name="captured",
        predictions=[
            {"query_id": "q-semantic-cache", "result_ids": ["aurora-design"], "latency_ms": 8.0}
        ],
    )
    result = evaluate_rankings(dataset, run, k_values=[1, 3])
    assert result["aggregate"]["query_count"] == len(dataset.queries)
    assert result["aggregate"]["prediction_count"] == 1
    assert result["per_query"][2]["metrics"]["hit_rate@1"] == 1.0
    assert result["per_query"][0]["missing_prediction"] is True
    assert result["aggregate"]["p95_latency_ms"] == 8.0
    assert result["confidence_intervals"]["mrr"]["n"] == len(dataset.queries)


def test_duplicate_prediction_query_ids_rejected_by_evaluator() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RankingRun(
            name="dup",
            predictions=[
                {"query_id": "q-no-match", "result_ids": []},
                {"query_id": "q-no-match", "result_ids": ["aurora-design"]},
            ],
        )


def test_offline_lexical_runner_is_repeatable_and_labeled_nonproduction() -> None:
    dataset = load_dataset()
    first = offline_lexical_run(dataset)
    second = offline_lexical_run(dataset)
    assert first.name == "offline_lexical_token_overlap"
    assert first.configuration["production_scoutx_pipeline"] is False
    assert [x.result_ids for x in first.predictions] == [x.result_ids for x in second.predictions]


def test_offline_personalization_uses_fixed_queries_and_disabled_profile_matches_baseline() -> None:
    dataset = load_dataset()
    baseline = offline_lexical_run(dataset)
    runs = offline_personalization_runs(dataset)
    cold_start = next(run for run in runs if run.name.endswith("cold_start_disabled"))
    assert len(runs) >= 6
    assert [item.result_ids for item in cold_start.predictions] == [
        item.result_ids for item in baseline.predictions
    ]
    assert all(len(run.predictions) == len(dataset.queries) for run in runs)
    assert all(run.configuration["database_activity_used"] is False for run in runs)


def test_reranking_comparison_uses_paired_query_set() -> None:
    dataset = load_dataset()
    result = evaluate_reranking(
        dataset,
        [
            {
                "query_id": "q-semantic-cache",
                "before_ids": ["vector-notes", "aurora-design"],
                "after_ids": ["aurora-design", "vector-notes"],
                "candidate_count": 2,
                "status": "qwen3",
                "latency_ms": 12.0,
            }
        ],
        k_values=[1, 3],
    )
    assert result["per_query_changes"][2]["changed"] is True
    assert result["per_query_changes"][2]["mrr_delta"] == pytest.approx(0.5)
    assert result["reranker_status_counts"] == {"qwen3": 1}


def test_rag_checks_answerability_evidence_and_annotation_not_llm_judging() -> None:
    dataset = load_dataset()
    result = evaluate_rag(
        dataset,
        [
            {
                "query_id": "q-semantic-cache",
                "answer": "Aurora entries expire after fifteen minutes.",
                "context_file_ids": ["aurora-design"],
                "answerable": True,
                "claims": [{"supported": True}],
            }
        ],
    )
    row = result["per_query"][0]
    assert row["answerable_correct"] is True
    assert row["expected_context_recall"] == 1.0
    assert row["annotated_claim_support_rate"] == 1.0
    assert row["answer_token_f1"] > 0


def test_citation_evaluator_separates_validity_support_and_provenance() -> None:
    dataset = load_dataset()
    result = evaluate_citations(
        dataset,
        [
            {
                "query_id": "q-hybrid-failover",
                "retrieved_file_ids": ["quasar-runbook"],
                "citations": [
                    {
                        "citation_id": "S1",
                        "logical_file_id": "quasar-runbook",
                        "section": "Failover",
                    }
                ],
                "claims": [{"text": "Promote east", "citation_ids": ["S1"]}],
            }
        ],
    )
    assert result["aggregate"]["citation_precision"] == 1
    assert result["aggregate"]["citation_validity_rate"] == 1
    assert result["aggregate"]["citation_recall"] == 1
    assert result["aggregate"]["provenance_accuracy"] == 1


def test_citations_count_duplicates_and_invalid_source_ids_separately() -> None:
    dataset = load_dataset()
    result = evaluate_citations(
        dataset,
        [
            {
                "query_id": "q-hybrid-failover",
                "retrieved_file_ids": ["quasar-runbook"],
                "citations": [
                    {
                        "citation_id": "S1",
                        "logical_file_id": "quasar-runbook",
                        "section": "Failover",
                    },
                    {
                        "citation_id": "S1",
                        "logical_file_id": "quasar-runbook",
                        "section": "Failover",
                    },
                    {"citation_id": "S2", "logical_file_id": "missing-file"},
                ],
            }
        ],
    )
    row = result["per_query"][0]
    assert row["citation_precision"] == 0.5
    assert row["citation_validity_rate"] == 0.5
    assert row["duplicate_citation_count"] == 1
    assert row["invalid_or_unretrieved"] == 1


def test_agent_planner_evaluation_respects_valid_tool_alternatives() -> None:
    dataset = load_dataset()
    result = offline_agent_routing_run(dataset)
    assert result["sample_count"] == len(dataset.queries)
    assert result["aggregate"]["authorized_plan_rate"] == 1
    assert result["aggregate"]["routing_accuracy"] == 1
    assert "No graph nodes" in result["scope"]
    # A bad route is observable as a false rather than an exception.
    hand = evaluate_agent(
        dataset, [{"query_id": "q-filename-aurora", "tools_invoked": ["semantic"]}]
    )
    assert hand["aggregate"]["routing_accuracy"] == 0
    assert hand["aggregate"]["tool_invocation_success_rate"] is None


def test_recommendation_scenarios_report_quality_and_empty_case() -> None:
    dataset = load_dataset()
    result = evaluate_recommendations(
        dataset,
        [
            {"scenario_id": "cold-start", "result_ids": ["aurora-design", "vector-notes"]},
            {"scenario_id": "no-available-files", "result_ids": []},
        ],
        k_values=[1, 2],
    )
    assert result["sample_count"] == 2
    assert result["aggregate"]["empty_case_accuracy"] == 1
    assert result["aggregate"]["recommendation_coverage"] > 0


def test_ingestion_observation_metrics_keep_unsupported_separate() -> None:
    result = evaluate_observations(
        [
            {
                "eligible": 4,
                "discovered": 4,
                "correctly_discovered": 4,
                "indexed": 3,
                "supported": 3,
                "extracted": 2,
                "mime_checked": 4,
                "mime_correct": 3,
                "unsupported_checked": 1,
                "unsupported_handled": 1,
            }
        ],
        "ingestion",
    )
    assert result["aggregate"]["discovery_recall"] == 1
    assert result["aggregate"]["extraction_success_rate"] == pytest.approx(2 / 3)
    assert result["aggregate"]["unsupported_handling_rate"] == 1


def test_performance_and_reliability_summary() -> None:
    perf = evaluate_observations(
        [
            {
                "phase": "cold",
                "latency_ms": 20,
                "items": 1,
                "duration_seconds": 1,
                "memory_bytes": 100,
            },
            {
                "phase": "warm",
                "latency_ms": 10,
                "items": 2,
                "duration_seconds": 1,
                "cpu_percent": 20,
                "bytes": 50,
                "chunks": 3,
                "stage_latency_ms": {"retrieval": 8},
            },
        ],
        "performance",
    )
    assert perf["aggregate"]["p95_latency_ms"] == pytest.approx(19.5)
    assert perf["aggregate"]["throughput_per_second"] == pytest.approx(1.5)
    assert perf["aggregate"]["bytes_per_second"] == pytest.approx(25)
    assert perf["aggregate"]["stage_mean_latency_ms"]["retrieval"] == 8
    rel = evaluate_observations(
        [
            {
                "scenario": "qdrant_offline",
                "expected_failure_detected": True,
                "recovered": True,
                "reported_success_on_failure": False,
            }
        ],
        "reliability",
    )
    assert rel["aggregate"]["failure_detection_rate"] == 1
    assert rel["aggregate"]["incorrect_success_count"] == 0


def test_extraction_expected_term_coverage() -> None:
    result = evaluate_observations(
        [
            {
                "status": "success",
                "expected_terms": ["CacheLease", "invalidate"],
                "extracted_content": "CacheLease invalidates the key",
            }
        ],
        "extraction",
    )
    assert result["aggregate"]["extraction_success_rate"] == 1
    assert result["aggregate"]["expected_term_coverage"] == 1


def test_character_and_word_error_rates_use_trusted_reference_text() -> None:
    assert edit_error_rate("hello world", "hello word", unit="word") == pytest.approx(0.5)
    assert edit_error_rate("cat", "cut", unit="character") == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="reference text"):
        edit_error_rate("  ", "anything", unit="word")


def test_extraction_cer_wer_requires_trusted_reference() -> None:
    result = evaluate_observations(
        [
            {
                "status": "success",
                "reference_trusted": True,
                "reference_text": "hello world",
                "extracted_content": "hello word",
            },
            {
                "status": "success",
                "reference_trusted": False,
                "reference_text": "do not use this",
                "extracted_content": "different text",
            },
        ],
        "extraction",
    )
    assert result["aggregate"]["trusted_transcription_samples"] == 1
    assert result["aggregate"]["wer"] == pytest.approx(0.5)


def test_bootstrap_interval_is_seeded_and_empty_is_explicit() -> None:
    left = bootstrap_mean_interval([0.2, 0.7, 0.9], seed=42)
    right = bootstrap_mean_interval([0.2, 0.7, 0.9], seed=42)
    assert left == right
    assert left["n"] == 3
    assert left["lower"] <= left["mean"] <= left["upper"]
    assert bootstrap_mean_interval([])["mean"] is None


def test_baseline_comparison_reports_paired_interval_and_only_configured_thresholds() -> None:
    dataset = load_dataset()
    base = new_report(
        dataset=dataset.name, dataset_version=dataset.version, mode="retrieval", seed=17
    )
    current = new_report(
        dataset=dataset.name, dataset_version=dataset.version, mode="retrieval", seed=17
    )
    base["results"] = {
        "quality": {
            "aggregate": {"mrr": 0.5},
            "per_query": [{"query_id": "q", "metrics": {"mrr": 0.5}}],
        }
    }
    current["results"] = {
        "quality": {
            "aggregate": {"mrr": 0.8},
            "per_query": [{"query_id": "q", "metrics": {"mrr": 0.8}}],
        }
    }
    result = compare_reports(current, base, thresholds={"mrr": -0.1})
    assert result["comparisons"]["quality"]["aggregate_deltas"]["mrr"] == pytest.approx(0.3)
    assert result["comparisons"]["quality"]["status"] == "paired_bootstrap_interval_reported"


def test_baseline_comparison_matches_named_retrieval_configurations() -> None:
    dataset = load_dataset()
    base = evaluate_rankings(
        dataset,
        RankingRun(
            name="hybrid",
            predictions=[
                {"query_id": "q-semantic-cache", "result_ids": ["vector-notes", "aurora-design"]}
            ],
        ),
        k_values=[1],
    )
    current = evaluate_rankings(
        dataset,
        RankingRun(
            name="hybrid",
            predictions=[{"query_id": "q-semantic-cache", "result_ids": ["aurora-design"]}],
        ),
        k_values=[1],
    )
    result = compare_reports(
        {"run_id": "c", "results": {"retrieval": {"runs": [current]}}},
        {"run_id": "b", "results": {"retrieval": {"runs": [base]}}},
        thresholds={"retrieval.hybrid.mrr": {"min_delta": 0.03}},
    )
    comparison = result["comparisons"]["retrieval"]["runs"]["hybrid"]
    assert comparison["aggregate_deltas"]["mrr"] == pytest.approx(0.5 / len(dataset.queries))
    assert comparison["configured_thresholds"]["mrr"]["threshold_crossed"] is False
    assert comparison["paired_query_count"] == len(dataset.queries)


def test_baseline_comparison_rejects_dataset_mismatch() -> None:
    with pytest.raises(ValueError, match="same dataset"):
        compare_reports(
            {"dataset": "d", "dataset_version": "1.0.0", "results": {}},
            {"dataset": "d", "dataset_version": "2.0.0", "results": {}},
        )


def test_reports_serialize_json_csv_markdown_without_raw_paths(tmp_path: Path) -> None:
    report = new_report(dataset="d", dataset_version="1.0.0", mode="retrieval", seed=17)
    report["results"] = {
        "retrieval": {
            "aggregate": {"mrr": 0.5},
            "per_query": [{"query_id": "q", "metrics": {"mrr": 0.5}}],
        }
    }
    paths = write_reports(report, tmp_path, {"json", "csv", "markdown"})
    assert {path.suffix for path in paths} == {".json", ".csv", ".md"}
    saved = json.loads(next(path for path in paths if path.suffix == ".json").read_text())
    assert saved["run_id"] == report["run_id"]


def test_report_configuration_redacts_credentials_and_paths() -> None:
    safe = safe_configuration(
        {
            "embedding_model": "Qwen3-Embedding-0.6B",
            "database_url": "postgresql://name:secret@localhost/db",
            "file_path": "/Users/private/sample.pdf",
            "api_key": "secret-value",
            "reranker_enabled": True,
        }
    )
    assert safe == {"embedding_model": "Qwen3-Embedding-0.6B", "reranker_enabled": True}
