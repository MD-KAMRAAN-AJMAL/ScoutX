"""Deterministic metric runners over the versioned corpus and observed artifacts."""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.agents.planner import deterministic_understanding, plan_retrieval
from app.agents.schemas import ToolName
from app.personalization.profile import UserProfile
from app.personalization.scoring import personalize_ranked_candidates
from evaluation.metrics import bootstrap_mean_interval, edit_error_rate, percentile, query_metrics
from evaluation.privacy import safe_configuration, safe_label
from evaluation.schema import EvaluationDataset, EvaluationQuery, RankingRun

_TOKEN = re.compile(r"[\w.+-]+", re.UNICODE)


def _rate(numerator: float, denominator: float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def _relevance(query: EvaluationQuery) -> dict[str, int]:
    return {item.logical_file_id: item.grade for item in query.judgments}


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_rankings(
    dataset: EvaluationDataset,
    run: RankingRun,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    unjudged_policy: str = "irrelevant",
    seed: int = 17,
) -> dict[str, Any]:
    predictions = {item.query_id: item for item in run.predictions}
    unknown_queries = set(predictions) - {item.query_id for item in dataset.queries}
    if unknown_queries:
        raise ValueError(f"predictions contain unknown query IDs: {sorted(unknown_queries)}")
    per_query = []
    metric_rows: dict[str, list[float]] = {}
    latencies: list[float] = []
    for query in dataset.queries:
        prediction = predictions.get(query.query_id)
        ids = prediction.result_ids if prediction else []
        metrics = query_metrics(
            ids,
            _relevance(query),
            k_values,
            unjudged_policy=unjudged_policy,  # type: ignore[arg-type]
        )
        if prediction and prediction.latency_ms is not None:
            latencies.append(prediction.latency_ms)
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and name not in {
                "zero_result",
                "duplicate_result_count",
                "unjudged_result_count",
            }:
                metric_rows.setdefault(name, []).append(float(value))
        per_query.append(
            {
                "query_id": query.query_id,
                "kind": query.kind,
                "result_count": len(ids),
                "result_ids": ids,
                "judged_relevant_count": sum(grade > 0 for grade in _relevance(query).values()),
                "metrics": metrics,
                "latency_ms": prediction.latency_ms if prediction else None,
                "missing_prediction": prediction is None,
            }
        )
    aggregate: dict[str, Any] = {
        name: sum(values) / len(values) if values else 0.0 for name, values in metric_rows.items()
    }
    aggregate["zero_result_rate"] = _mean(
        [float(row["metrics"]["zero_result"]) for row in per_query]
    )
    aggregate["duplicate_results_total"] = sum(
        int(row["metrics"]["duplicate_result_count"]) for row in per_query
    )
    aggregate["mean_unjudged_results_per_query"] = _mean(
        [float(row["metrics"]["unjudged_result_count"]) for row in per_query]
    )
    aggregate["unjudged_results_total"] = sum(
        int(row["metrics"]["unjudged_result_count"]) for row in per_query
    )
    aggregate["query_count"] = len(dataset.queries)
    aggregate["prediction_count"] = len(predictions)
    aggregate["mean_latency_ms"] = sum(latencies) / len(latencies) if latencies else None
    aggregate["p50_latency_ms"] = percentile(latencies, 50)
    aggregate["p95_latency_ms"] = percentile(latencies, 95)
    aggregate["p99_latency_ms"] = percentile(latencies, 99) if len(latencies) >= 100 else None
    intervals = {
        name: bootstrap_mean_interval(values, seed=seed)
        for name, values in metric_rows.items()
        if name in {"mrr", *(f"ndcg@{k}" for k in k_values), *(f"recall@{k}" for k in k_values)}
    }
    return {
        "run_name": safe_label(run.name),
        "configuration": safe_configuration(run.configuration),
        "aggregate": aggregate,
        "confidence_intervals": intervals,
        "per_query": per_query,
        "metric_definitions": ranking_metric_definitions(unjudged_policy),
    }


def ranking_metric_definitions(unjudged_policy: str) -> dict[str, str]:
    return {
        "precision@k": "Relevant results in top K divided by K; an empty slot lowers precision.",
        "recall@k": "Relevant results in top K divided by all judged relevant files for the query.",
        "mrr": "Mean reciprocal rank of the first relevant result, averaged over all queries.",
        "ndcg@k": "Graded NDCG using gain 2^grade-1 and logarithmic rank discount.",
        "map": "Average precision per query divided by the number of judged relevant files.",
        "hit_rate@k": "Whether at least one judged relevant file appears in the first K results.",
        "unjudged_policy": (
            "Unjudged results count as irrelevant and occupy ranks."
            if unjudged_policy == "irrelevant"
            else "Unjudged results are removed before ranking metrics are calculated."
        ),
    }


def offline_lexical_run(
    dataset: EvaluationDataset,
    *,
    repetitions: int = 1,
    warmups: int = 0,
) -> RankingRun:
    """A deterministic token-overlap baseline; it does not call ScoutX production retrieval."""
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmups cannot be negative")
    corpus = {
        document.logical_file_id: (
            document.filename + " " + document.extension + " " + document.content
        )
        for document in dataset.documents
        if document.supported
    }

    def rank(query: EvaluationQuery) -> tuple[list[str], float]:
        started = time.perf_counter()
        terms = [term.casefold() for term in _TOKEN.findall(query.text)]
        scored = []
        for logical_id, text in corpus.items():
            haystack = text.casefold()
            score = sum(haystack.count(term) for term in terms)
            if score:
                scored.append((logical_id, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [logical_id for logical_id, _ in scored], (time.perf_counter() - started) * 1000

    for _ in range(warmups):
        for query in dataset.queries:
            rank(query)
    predictions = []
    for query in dataset.queries:
        observed: list[float] = []
        result_ids: list[str] = []
        for _ in range(repetitions):
            result_ids, elapsed = rank(query)
            observed.append(elapsed)
        predictions.append(
            {
                "query_id": query.query_id,
                "result_ids": result_ids,
                "latency_ms": sum(observed) / len(observed),
            }
        )
    return RankingRun(
        name="offline_lexical_token_overlap",
        predictions=predictions,
        configuration={
            "provider": "synthetic corpus token-overlap baseline",
            "repetitions": repetitions,
            "warmups": warmups,
            "production_scoutx_pipeline": False,
        },
    )


def offline_agent_routing_run(dataset: EvaluationDataset) -> dict[str, Any]:
    """Evaluate ScoutX's deterministic query planner without Ollama or graph execution."""
    allowed = set(ToolName.__args__)
    records = []
    for query in dataset.queries:
        started = time.perf_counter()
        understanding = deterministic_understanding(query.text)
        plan = plan_retrieval(understanding)
        tools = [item.tool for item in plan.tools]
        valid_tools = set(query.valid_tools)
        records.append(
            {
                "query_id": query.query_id,
                "tools_invoked": tools,
                "routing_valid": bool(set(tools) & valid_tools),
                "authorized": set(tools) <= allowed,
                "retry_count": int(plan.retry),
                "planner_latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    count = len(records)
    return {
        "sample_count": count,
        "aggregate": {
            "routing_accuracy": sum(row["routing_valid"] for row in records) / max(1, count),
            "authorized_plan_rate": sum(row["authorized"] for row in records) / max(1, count),
            "mean_planner_latency_ms": sum(row["planner_latency_ms"] for row in records)
            / max(1, count),
            "p50_planner_latency_ms": percentile(
                [float(row["planner_latency_ms"]) for row in records], 50
            ),
            "p95_planner_latency_ms": percentile(
                [float(row["planner_latency_ms"]) for row in records], 95
            ),
        },
        "per_query": records,
        "scope": "Deterministic query understanding and planning only. No graph nodes, retrieval tools, Ollama, database, or reranker were invoked.",
    }


def offline_personalization_runs(dataset: EvaluationDataset) -> list[RankingRun]:
    """Apply production score adjustments to synthetic profiles over one fixed candidate set."""
    profiles = {
        "cold_start_disabled": UserProfile(preferences={"personalization_enabled": False}),
        "code_focused": UserProfile(
            preferences={"personalization_enabled": True},
            preferred_extensions=(".py", ".c", ".cpp", ".java", ".js", ".ts", ".tsx", ".rs", ".go"),
        ),
        "document_focused": UserProfile(
            preferences={"personalization_enabled": True},
            preferred_extensions=(".pdf", ".docx", ".pptx", ".md", ".txt", ".html"),
        ),
        "spreadsheet_focused": UserProfile(
            preferences={"personalization_enabled": True},
            preferred_extensions=(".xlsx", ".csv"),
        ),
        "multiple_project_user": UserProfile(
            preferences={"personalization_enabled": True, "preferred_root_ids": ["aurora"]}
        ),
        "negative_feedback": UserProfile(
            preferences={"personalization_enabled": True},
            file_feedback={"aurora-design": -1.0},
        ),
        "changing_interest_to_presentations": UserProfile(
            preferences={"personalization_enabled": True}, preferred_extensions=(".pptx",)
        ),
        "excluded_audio_type": UserProfile(
            preferences={"personalization_enabled": True, "excluded_extensions": [".wav"]}
        ),
    }
    corpus = {item.logical_file_id: item for item in dataset.documents if item.supported}
    output = []
    for profile_name, profile in profiles.items():
        predictions = []
        for query in dataset.queries:
            terms = [term.casefold() for term in _TOKEN.findall(query.text)]
            initial = []
            for logical_id, document in corpus.items():
                text = f"{document.filename} {document.extension} {document.content}".casefold()
                score = sum(text.count(term) for term in terms)
                if score:
                    # Strictly monotonic mapping preserves the source candidate order when
                    # the profile is disabled; avoid score clipping ties.
                    base_score = 0.2 + 0.75 * score / (score + 5)
                    initial.append(
                        {
                            "file_id": logical_id,
                            "chunk_id": None,
                            "extension": document.extension,
                            "root_id": str(document.metadata.get("project", "")) or None,
                            "relevance_score": base_score,
                            "final_score": base_score,
                        }
                    )
            initial.sort(key=lambda item: (-item["final_score"], item["file_id"]))
            started = time.perf_counter()
            personalized = personalize_ranked_candidates(initial, profile)
            elapsed = (time.perf_counter() - started) * 1000
            predictions.append(
                {
                    "query_id": query.query_id,
                    "result_ids": [item["file_id"] for item in personalized],
                    "latency_ms": elapsed,
                }
            )
        output.append(
            RankingRun(
                name=f"synthetic_profile:{profile_name}",
                predictions=predictions,
                configuration={
                    "provider": "synthetic lexical candidate set + app.personalization.scoring",
                    "profile": profile_name,
                    "database_activity_used": False,
                    "production_retrieval_used": False,
                },
            )
        )
    return output


def evaluate_reranking(
    dataset: EvaluationDataset, records: Sequence[Mapping[str, Any]], k_values=(1, 3, 5, 10)
) -> dict[str, Any]:
    before = RankingRun(
        name="before_reranking",
        predictions=[
            {"query_id": str(row["query_id"]), "result_ids": list(row.get("before_ids", []))}
            for row in records
        ],
    )
    after = RankingRun(
        name="after_reranking",
        predictions=[
            {"query_id": str(row["query_id"]), "result_ids": list(row.get("after_ids", []))}
            for row in records
        ],
    )
    left = evaluate_rankings(dataset, before, k_values=k_values)
    right = evaluate_rankings(dataset, after, k_values=k_values)
    left_by_id = {row["query_id"]: row for row in left["per_query"]}
    right_by_id = {row["query_id"]: row for row in right["per_query"]}
    changes = []
    for query_id, left_row in left_by_id.items():
        right_row = right_by_id[query_id]
        changes.append(
            {
                "query_id": query_id,
                "before_ids": left_row["result_ids"],
                "after_ids": right_row["result_ids"],
                "changed": left_row["result_ids"] != right_row["result_ids"],
                "mrr_delta": right_row["metrics"]["mrr"] - left_row["metrics"]["mrr"],
            }
        )
    return {
        "before": left["aggregate"],
        "after": right["aggregate"],
        "per_query_changes": changes,
        "candidate_counts": [
            int(row.get("candidate_count", len(row.get("before_ids", [])))) for row in records
        ],
        "reranker_status_counts": dict(
            Counter(safe_label(row.get("status", "unspecified")) for row in records)
        ),
        "fallback_frequency": sum(
            str(row.get("status", "")).endswith("fallback") for row in records
        )
        / max(1, len(records)),
        "reranking_latency_ms": {
            "mean": _mean(
                [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
            ),
            "p50": percentile(
                [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None],
                50,
            ),
            "p95": percentile(
                [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None],
                95,
            ),
        },
        "interpretation": "Quality differences are measured on paired queries; score values are not compared across models.",
    }


def evaluate_rag(
    dataset: EvaluationDataset, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    query_map = {item.query_id: item for item in dataset.queries}
    rows = []
    for record in records:
        query_id = str(record["query_id"])
        query = query_map[query_id]
        missing = {"answer", "answerable", "context_file_ids"} - record.keys()
        if missing:
            raise ValueError(f"RAG observation for {query_id} is missing {sorted(missing)}")
        answer = str(record.get("answer", ""))
        context_ids = set(map(str, record.get("context_file_ids", [])))
        expected_ids = {item.logical_file_id for item in query.evidence}
        answer_terms = {x.casefold() for x in _TOKEN.findall(answer)}
        reference_terms = {x.casefold() for x in _TOKEN.findall(query.reference_answer or "")}
        overlap = len(answer_terms & reference_terms)
        answer_f1 = (
            (2 * overlap / (len(answer_terms) + len(reference_terms)))
            if query.reference_answer is not None and (answer_terms or reference_terms)
            else None
        )
        context_recall = (
            len(expected_ids & context_ids) / len(expected_ids) if expected_ids else None
        )
        judged_relevant = {item.logical_file_id for item in query.judgments if item.grade > 0}
        context_precision = (
            len(judged_relevant & context_ids) / len(context_ids)
            if context_ids
            else 0.0
            if "context_file_ids" in record
            else None
        )
        supported_claims = record.get("claims", [])
        faithfulness = (
            sum(bool(claim.get("supported", False)) for claim in supported_claims)
            / len(supported_claims)
            if supported_claims
            else None
        )
        declared_answerable = record.get("answerable")
        rows.append(
            {
                "query_id": query_id,
                "answerable_correct": declared_answerable == query.answerable
                if query.answerable is not None and declared_answerable is not None
                else None,
                "answer_token_f1": answer_f1,
                "expected_context_recall": context_recall,
                "judged_context_precision": context_precision,
                "annotated_claim_support_rate": faithfulness,
            }
        )
    numeric_keys = (
        "answerable_correct",
        "answer_token_f1",
        "expected_context_recall",
        "judged_context_precision",
        "annotated_claim_support_rate",
    )
    aggregate = {
        key: _mean([float(row[key]) for row in rows if row[key] is not None])
        for key in numeric_keys
    }
    return {
        "sample_count": len(rows),
        "aggregate": aggregate,
        "per_query": rows,
        "limitations": "Token F1 is a lexical proxy, not semantic answer correctness; context precision uses limited query/file judgments; claim support requires trusted annotations.",
    }


def evaluate_citations(
    dataset: EvaluationDataset, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    query_map = {item.query_id: item for item in dataset.queries}
    documents = {item.logical_file_id for item in dataset.documents}
    output = []
    totals = Counter()
    for record in records:
        query = query_map[str(record["query_id"])]
        if "citations" not in record or "retrieved_file_ids" not in record:
            raise ValueError("citation observation must include citations and retrieved_file_ids")
        expected = {item.logical_file_id: item for item in query.evidence}
        retrieved = set(map(str, record.get("retrieved_file_ids", [])))
        citations = list(record.get("citations", []))
        cited_expected = set()
        valid_count = 0
        supported_count = 0
        provenance_count = 0
        provenance_expected_count = 0
        duplicate_count = 0
        seen_citations = set()
        seen_citation_ids = set()
        valid_citation_ids = set()
        for citation in citations:
            file_id = str(citation.get("logical_file_id", ""))
            identity = (
                str(citation.get("citation_id", "")),
                file_id,
                citation.get("page"),
                citation.get("section"),
                citation.get("chunk_id"),
            )
            citation_id = str(citation.get("citation_id", ""))
            if identity in seen_citations or (citation_id and citation_id in seen_citation_ids):
                duplicate_count += 1
                continue
            seen_citations.add(identity)
            if citation_id:
                seen_citation_ids.add(citation_id)
            valid = bool(citation_id) and file_id in documents and file_id in retrieved
            if valid:
                valid_count += 1
                valid_citation_ids.add(citation_id)
            reference = expected.get(file_id)
            if reference:
                supported_count += 1
                cited_expected.add(file_id)
                has_expected_location = reference.page is not None or reference.section is not None
                loc_ok = (reference.page is None or citation.get("page") == reference.page) and (
                    reference.section is None or citation.get("section") == reference.section
                )
                if has_expected_location:
                    provenance_expected_count += 1
                    provenance_count += int(
                        loc_ok and file_id in retrieved and file_id in documents
                    )
        claim_rows = list(record.get("claims", []))
        covered_claims = (
            sum(
                bool(set(map(str, row.get("citation_ids", []))) & valid_citation_ids)
                for row in claim_rows
            )
            if "claims" in record
            else None
        )
        totals.update(
            citations=len(citations) - duplicate_count,
            valid=valid_count,
            supported=supported_count,
            expected=len(expected),
            cited_expected=len(cited_expected),
            provenance=provenance_count,
            provenance_expected=provenance_expected_count,
            duplicates=duplicate_count,
        )
        if covered_claims is not None:
            totals.update(claims=len(claim_rows), covered_claims=covered_claims)
        output.append(
            {
                "query_id": query.query_id,
                "citation_precision": _rate(supported_count, len(citations) - duplicate_count),
                "citation_validity_rate": _rate(valid_count, len(citations) - duplicate_count),
                "citation_recall": _rate(len(cited_expected), len(expected)),
                "citation_coverage": _rate(covered_claims, len(claim_rows))
                if covered_claims is not None
                else None,
                "provenance_accuracy": _rate(provenance_count, provenance_expected_count),
                "invalid_or_unretrieved": len(citations) - duplicate_count - valid_count,
                "duplicate_citation_count": duplicate_count,
            }
        )
    return {
        "sample_count": len(output),
        "aggregate": {
            "citation_precision": _rate(totals["supported"], totals["citations"]),
            "citation_validity_rate": _rate(totals["valid"], totals["citations"]),
            "citation_recall": _rate(totals["cited_expected"], totals["expected"]),
            "citation_coverage": _rate(totals["covered_claims"], totals["claims"]),
            "provenance_accuracy": _rate(totals["provenance"], totals["provenance_expected"]),
            "duplicate_citation_count": totals["duplicates"],
        },
        "per_query": output,
        "limitations": "Citation support must be judged against the dataset's independent evidence annotations.",
    }


def evaluate_agent(
    dataset: EvaluationDataset, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    query_map = {item.query_id: item for item in dataset.queries}
    rows = []
    for record in records:
        query = query_map[str(record["query_id"])]
        tools = list(map(str, record.get("tools_invoked", [])))
        valid = set(query.valid_tools)
        correct = bool(tools) and (not valid or any(tool in valid for tool in tools))
        authorized = set(map(str, record.get("authorized_tools", [])))
        retry_count = int(record["retry_count"]) if record.get("retry_count") is not None else None
        rows.append(
            {
                "query_id": query.query_id,
                "routing_valid": correct,
                "tool_invocation_success": record.get("tool_success"),
                "unnecessary_tool_count": sum(tool not in valid for tool in tools)
                if valid
                else None,
                "unauthorized_tool_count": sum(tool not in authorized for tool in tools)
                if authorized
                else None,
                "retry_count": retry_count,
                "retry_bound_respected": retry_count <= int(record.get("max_retries", 1))
                if retry_count is not None
                else None,
                "latency_ms": record.get("latency_ms"),
                "model_tool_calls": record.get("model_tool_calls"),
                "graph_completed": record.get("graph_completed"),
                "fallback_succeeded": record.get("fallback_succeeded"),
            }
        )
    count = len(rows)
    success = [
        row["tool_invocation_success"] for row in rows if row["tool_invocation_success"] is not None
    ]
    graph = [row["graph_completed"] for row in rows if row["graph_completed"] is not None]
    retries = [row["retry_count"] for row in rows if row["retry_count"] is not None]
    retry_bounds = [
        row["retry_bound_respected"] for row in rows if row["retry_bound_respected"] is not None
    ]
    unnecessary = [
        row["unnecessary_tool_count"] for row in rows if row["unnecessary_tool_count"] is not None
    ]
    unauthorized = [
        row["unauthorized_tool_count"] for row in rows if row["unauthorized_tool_count"] is not None
    ]
    fallbacks = [row["fallback_succeeded"] for row in rows if row["fallback_succeeded"] is not None]
    latencies = [float(row["latency_ms"]) for row in rows if row["latency_ms"] is not None]
    return {
        "sample_count": count,
        "aggregate": {
            "routing_accuracy": sum(row["routing_valid"] for row in rows) / count
            if count
            else None,
            "tool_invocation_success_rate": sum(map(bool, success)) / len(success)
            if success
            else None,
            "unnecessary_tool_invocations": sum(unnecessary) if unnecessary else None,
            "unauthorized_tool_invocations": sum(unauthorized) if unauthorized else None,
            "mean_retry_count": sum(retries) / len(retries) if retries else None,
            "retry_bound_rate": sum(map(bool, retry_bounds)) / len(retry_bounds)
            if retry_bounds
            else None,
            "graph_completion_rate": sum(map(bool, graph)) / len(graph) if graph else None,
            "fallback_success_rate": sum(map(bool, fallbacks)) / len(fallbacks)
            if fallbacks
            else None,
            "mean_latency_ms": _mean(latencies),
        },
        "per_query": rows,
        "limitations": "Routing is accepted against each scenario's explicit valid-tool alternatives, not one preferred plan.",
    }


def evaluate_recommendations(
    dataset: EvaluationDataset, records: Sequence[Mapping[str, Any]], k_values=(1, 3, 5, 10)
) -> dict[str, Any]:
    scenarios = {item.scenario_id: item for item in dataset.recommendation_scenarios}
    output = []
    for record in records:
        scenario = scenarios[str(record["scenario_id"])]
        metrics = query_metrics(
            list(map(str, record.get("result_ids", []))),
            {key: 3 for key in scenario.relevant_file_ids},
            k_values,
        )
        output.append(
            {
                "scenario_id": safe_label(scenario.scenario_id),
                "profile": safe_label(scenario.profile),
                "category": safe_label(scenario.category),
                "metrics": metrics,
                "result_count": len(record.get("result_ids", [])),
                "expected_empty": scenario.expected_empty,
                "returned_empty": not record.get("result_ids", []),
            }
        )
    keys = sorted(
        {
            key
            for row in output
            for key in row["metrics"]
            if key.startswith(("precision@", "recall@", "ndcg@", "hit_rate@"))
        }
    )
    aggregate = {
        key: sum(float(row["metrics"][key]) for row in output) / max(1, len(output)) for key in keys
    }
    returned_ids = {str(file_id) for row in records for file_id in row.get("result_ids", [])}
    aggregate["recommendation_coverage"] = len(returned_ids) / max(1, len(dataset.documents))
    aggregate["empty_case_accuracy"] = (
        sum(row["expected_empty"] == row["returned_empty"] for row in output) / len(output)
        if output
        else None
    )
    aggregate["category_coverage"] = len({row["category"] for row in output}) / max(
        1, len({item.category for item in dataset.recommendation_scenarios})
    )
    aggregate["duplicate_recommendation_count"] = sum(
        int(row["metrics"].get("duplicate_result_count", 0)) for row in output
    )
    total_returned = sum(len(row.get("result_ids", [])) for row in records)
    aggregate["recommendation_diversity"] = (
        len(returned_ids) / total_returned if total_returned else 0.0
    )
    aggregate["unavailable_file_count"] = sum(
        int(row.get("unavailable_count", 0)) for row in records
    )
    aggregate["deleted_file_count"] = sum(int(row.get("deleted_count", 0)) for row in records)
    latencies = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
    aggregate["mean_latency_ms"] = sum(latencies) / len(latencies) if latencies else None
    acceptance = [bool(row["clicked"]) for row in records if "clicked" in row]
    novelty = [float(row["novelty"]) for row in records if row.get("novelty") is not None]
    freshness = [bool(row["fresh"]) for row in records if "fresh" in row]
    aggregate["controlled_acceptance_rate"] = (
        sum(acceptance) / len(acceptance) if acceptance else None
    )
    aggregate["mean_novelty"] = _mean(novelty)
    aggregate["freshness_rate"] = sum(freshness) / len(freshness) if freshness else None
    return {"sample_count": len(output), "aggregate": aggregate, "per_scenario": output}


def evaluate_observations(records: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    """Summarize explicit integration observations without treating missing data as a pass."""
    count = len(records)
    if mode == "ingestion":
        names = (
            "eligible",
            "discovered",
            "correctly_discovered",
            "indexed",
            "supported",
            "extracted",
            "mime_checked",
            "mime_correct",
            "unsupported_checked",
            "unsupported_handled",
            "duplicates",
            "changed_detected",
            "deleted_reconciled",
            "stale_index_detected",
            "version_consistent",
            "changed_expected",
            "deleted_expected",
            "stale_index_expected",
            "version_checks",
        )
        totals = {name: sum(int(row.get(name, 0)) for row in records) for name in names}
        aggregate = {
            "discovery_precision": _rate(totals["correctly_discovered"], totals["discovered"]),
            "discovery_recall": _rate(totals["discovered"], totals["eligible"]),
            "indexing_success_rate": _rate(totals["indexed"], totals["eligible"]),
            "extraction_success_rate": _rate(totals["extracted"], totals["supported"]),
            "mime_accuracy": _rate(totals["mime_correct"], totals["mime_checked"]),
            "unsupported_handling_rate": _rate(
                totals["unsupported_handled"], totals["unsupported_checked"]
            ),
            "duplicate_events_count": totals["duplicates"],
            "changed_file_detection_rate": _rate(
                totals["changed_detected"], totals["changed_expected"]
            ),
            "deleted_reconciliation_rate": _rate(
                totals["deleted_reconciled"], totals["deleted_expected"]
            ),
            "stale_index_detection_rate": _rate(
                totals["stale_index_detected"], totals["stale_index_expected"]
            ),
            "version_consistency": _rate(totals["version_consistent"], totals["version_checks"]),
            "eligible_file_count": totals["eligible"],
            "discovered_file_count": totals["discovered"],
            "supported_file_count": totals["supported"],
            "mime_checked_count": totals["mime_checked"],
            "unsupported_checked_count": totals["unsupported_checked"],
        }
        return {
            "sample_count": count,
            "aggregate": aggregate,
            "limitations": "Supply explicit expected and observed counts from an isolated corpus run; unsupported types are measured separately from extraction success.",
        }
    if mode == "performance":
        latency = [float(row["latency_ms"]) for row in records if row.get("latency_ms") is not None]
        cold = [
            float(row["latency_ms"])
            for row in records
            if row.get("phase") == "cold" and row.get("latency_ms") is not None
        ]
        warm = [
            float(row["latency_ms"])
            for row in records
            if row.get("phase") == "warm" and row.get("latency_ms") is not None
        ]
        cpu = [float(row["cpu_percent"]) for row in records if row.get("cpu_percent") is not None]
        memory = [
            int(row["memory_bytes"]) for row in records if row.get("memory_bytes") is not None
        ]
        stage_values: dict[str, list[float]] = {}
        for row in records:
            for stage, value in row.get("stage_latency_ms", {}).items():
                stage_values.setdefault(safe_label(stage), []).append(float(value))
        aggregate = {
            "mean_latency_ms": sum(latency) / len(latency) if latency else None,
            "p50_latency_ms": percentile(latency, 50),
            "p95_latency_ms": percentile(latency, 95),
            "p99_latency_ms": percentile(latency, 99) if len(latency) >= 100 else None,
            "cold_mean_latency_ms": sum(cold) / len(cold) if cold else None,
            "warm_mean_latency_ms": sum(warm) / len(warm) if warm else None,
            "throughput_per_second": _rate(
                sum(float(row.get("items", 0)) for row in records),
                sum(float(row.get("duration_seconds", 0)) for row in records),
            ),
            "bytes_per_second": _rate(
                sum(float(row.get("bytes", 0)) for row in records),
                sum(float(row.get("duration_seconds", 0)) for row in records),
            ),
            "chunks_per_second": _rate(
                sum(float(row.get("chunks", 0)) for row in records),
                sum(float(row.get("duration_seconds", 0)) for row in records),
            ),
            "max_memory_bytes": max(memory) if memory else None,
            "mean_cpu_percent": sum(cpu) / len(cpu) if cpu else None,
            "stage_mean_latency_ms": {
                stage: sum(values) / len(values) for stage, values in stage_values.items()
            },
            "stage_p95_latency_ms": {
                stage: percentile(values, 95) for stage, values in stage_values.items()
            },
        }
        metadata = [
            {
                key: safe_label(row[key]) if key == "phase" else row[key]
                for key in (
                    "phase",
                    "latency_ms",
                    "items",
                    "duration_seconds",
                    "memory_bytes",
                    "cpu_percent",
                    "stage_latency_ms",
                )
                if key in row
            }
            for row in records
        ]
        return {"sample_count": len(latency), "aggregate": aggregate, "sample_metadata": metadata}
    if mode == "reliability":
        detection_rows = [row for row in records if "expected_failure_detected" in row]
        recovery_rows = [row for row in records if "recovered" in row]
        completed = sum(bool(row.get("expected_failure_detected")) for row in detection_rows)
        recovered = sum(bool(row.get("recovered")) for row in recovery_rows)
        safe_rows = [
            {
                key: safe_label(row[key]) if key == "scenario" else row[key]
                for key in (
                    "scenario",
                    "expected_failure_detected",
                    "recovered",
                    "reported_success_on_failure",
                )
                if key in row
            }
            for row in records
        ]
        return {
            "sample_count": count,
            "aggregate": {
                "failure_detection_rate": _rate(completed, len(detection_rows)),
                "recovery_rate": _rate(recovered, len(recovery_rows)),
                "incorrect_success_count": sum(
                    bool(row.get("reported_success_on_failure", False)) for row in records
                ),
            },
            "scenarios": safe_rows,
        }
    if mode == "extraction":
        successful = 0
        status_count = 0
        matched_terms = 0
        expected_terms = 0
        cer_values = []
        wer_values = []
        for row in records:
            if row.get("status") is not None:
                status_count += 1
                successful += row.get("status") in {"success", "partial"}
            actual = str(row.get("extracted_content", "")).casefold()
            terms = [str(term).casefold() for term in row.get("expected_terms", [])]
            matched_terms += sum(term in actual for term in terms)
            expected_terms += len(terms)
            reference = row.get("reference_text")
            if (
                row.get("reference_trusted") is True
                and isinstance(reference, str)
                and reference.strip()
            ):
                hypothesis = str(row.get("extracted_content", ""))
                cer_values.append(edit_error_rate(reference, hypothesis, unit="character"))
                wer_values.append(edit_error_rate(reference, hypothesis, unit="word"))
        return {
            "sample_count": count,
            "aggregate": {
                "extraction_success_rate": _rate(successful, status_count),
                "expected_term_coverage": matched_terms / expected_terms
                if expected_terms
                else None,
                "cer": sum(cer_values) / len(cer_values) if cer_values else None,
                "wer": sum(wer_values) / len(wer_values) if wer_values else None,
                "trusted_transcription_samples": len(wer_values),
            },
            "limitations": "Expected-term coverage is not completeness. CER/WER are emitted only for explicitly trusted reference text and still depend on reference quality.",
        }
    raise ValueError(f"unsupported observation mode: {mode}")
