# ScoutX evaluation and benchmarking

This package is separate from FastAPI request handling and production ranking. It provides versioned judgments, deterministic metrics, saved run reports, paired comparisons, and a provider hook for observed ScoutX pipeline runs. It does not alter search, reranking, generation, or recommendation behavior.

## Audit findings and evaluation plan

Existing evaluation support was limited to `precision@k`, `recall@k`, graded NDCG, and category diversity in `app/personalization/evaluation.py`, exercised by personalization unit tests. Retrieval methods return validated `SearchCandidate` objects from keyword, semantic, hybrid, and metadata services. Reranking returns `RerankedCandidate` and a status (`qwen3`, `fallback`, `disabled_fallback`, or `failed`); file aggregation produces `FileRecommendation`. The LangGraph final state contains `tools_invoked`, retry count, execution trace, result, latency, reranking and generation status. RAG exposes `GroundedContext`, source locations, generated answer status, and citation records. Personalization produces bounded score adjustments and category recommendations. Normalized extraction records parser status and source provenance.

No versioned benchmark corpus, shared retrieval evaluation runner, Ragas integration, or end-to-end benchmark output existed before Phase 11. API telemetry has bounded aggregate request count/duration, while stage timing is available in agent state for searches; worker stages do not yet provide a complete standardized performance observation export. The `--provider module:function` hook adapts the actual services to these contracts without coupling the CLI to a database or requiring production data. A provider should capture outputs directly from the services under test and return the artifact shape shown below.

## Controlled dataset

`datasets/scoutx-small-v1/dataset.json` is synthetic and versioned as `scoutx-small@1.0.0`. It contains stable logical file IDs, synthetic content or extracted-text references, query scenarios for filenames, exact terms, semantic/hybrid queries, code, spreadsheets, OCR, audio/video transcripts, metadata, ambiguity, multi-relevant and no-result cases, and recommendation profiles. Grades are 0 (not relevant) through 3 (highly relevant); judgments are authored for this synthetic dataset and do not come from a search or embedding model.

Several multimodal entries are expected extracted-text records rather than byte-for-byte source assets. The dataset is suitable for metric, routing, citation, and adapter smoke tests; it does not establish parser/OCR/transcription accuracy or filesystem scanner quality. Add isolated real-format fixtures with trusted expected output before reporting those measures. Do not add personal files. The corpus and labels are small, manually authored, and incomplete; they are not representative user behavior.

Unlisted query/file pairs are unjudged. Ranking metrics default to `--unjudged irrelevant`: such results occupy rank positions and count as non-relevant. `--unjudged exclude` removes them before ranking calculations. This assumption is recorded in each retrieval report. Duplicate result IDs are deduplicated at first occurrence for ranking and separately counted. Precision divides by K (so a short result list is penalized); recall divides by the count of judged relevant files. No relevance judgments means recall, MRR, MAP, and hit rate are zero.

## Commands

Run in `backend/`:

```bash
uv run python -m evaluation.cli --mode retrieval
uv run python -m evaluation.cli --mode all --output-dir evaluation/results
uv run python -m evaluation.cli --mode retrieval --k 1 3 5 10 --repetitions 3 --warmups 1
uv run python -m evaluation.cli --mode all --dataset-version 1.0.0 --configuration evaluation/examples/local-config.json
uv run python -m evaluation.cli --mode retrieval --artifacts evaluation/examples/my-run.json
uv run python -m evaluation.cli --mode all --artifacts evaluation/examples/my-run.json --baseline evaluation/results/BASELINE.json
```

The default retrieval run executes a deterministic token-overlap baseline on the synthetic corpus. It is labelled `offline_lexical_token_overlap` and is **not** production ScoutX lexical/BM25 retrieval. It is useful for validating the runner and dataset only. Timings cover only local Python ranking and are not application latency.

Supported `--mode` values: `retrieval`, `reranking`, `rag`, `citations`, `agent`, `personalization`, `recommendations`, `ingestion`, `extraction`, `performance`, `reliability`, and `all`. `all` marks missing captured groups as NOT RUN. A requested artifact-only group exits with code 2 if there are no observations. `agent` without captured output evaluates the real deterministic `understand_query`/`plan_retrieval` route only; it does not execute graph nodes or tools. Slow/model-backed evaluation is not part of normal tests.

Reports contain a run ID and timestamp, dataset/version, Git SHA if available, Python/OS/runtime information, configuration, aggregate and per-query results, latency, limitations, and confidence intervals for selected query ranking metrics. JSON is the source of truth; Markdown is a readable summary and CSV contains flattened per-query/per-scenario rows. Generated results are ignored by Git. The report intentionally does not persist answers, document content, source paths, or arbitrary provider payloads.

### Captured artifact format

Start from `evaluation/examples/artifacts-template.json`. For example, `rankings` contains named configurations with one prediction per query:

```json
{
  "dataset": "scoutx-small",
  "dataset_version": "1.0.0",
  "rankings": [{
    "name": "hybrid_rrf",
    "configuration": {"candidate_limit": 40, "reranker": false},
    "predictions": [{"query_id": "q-semantic-cache", "result_ids": ["aurora-design"], "latency_ms": 12.4}]
  }],
  "reranking": [{"query_id":"q-semantic-cache", "before_ids":["vector-notes","aurora-design"], "after_ids":["aurora-design"], "candidate_count":2, "status":"qwen3", "latency_ms":5.2}],
  "rag": [{"query_id":"q-semantic-cache", "answer":"...", "answerable":true, "context_file_ids":["aurora-design"], "claims":[{"supported":true}]}],
  "citations": [{"query_id":"q-semantic-cache", "retrieved_file_ids":["aurora-design"], "citations":[{"citation_id":"S1","logical_file_id":"aurora-design"}], "claims":[{"citation_ids":["S1"]}]}]
}
```

Other artifact groups accept explicit observations: `agent` records `query_id`, `tools_invoked`, `tool_success`, `graph_completed`, retries, duration and optional authorized-tool set; `personalization` is paired `RankingRun` output to compare with a matching unpersonalized ranking; `recommendations` maps `scenario_id` to returned logical IDs; `ingestion` holds expected/observed counts; `performance` holds duration, phase, item count and optional CPU/memory measurements; `reliability` holds named isolated failure scenarios and their detection/recovery outcomes. `extraction` uses ingestion observations with `status`, trusted `expected_terms` and `extracted_content`; reports include only term coverage and status rate, not extracted text.

Providers are Python callables accepting keyword arguments `dataset`, `mode`, and `options` (including K values, repetitions, warmups, concurrency, seed, and a JSON configuration object); return an `EvaluationArtifacts`-compatible mapping for the same dataset/version. Use providers to call the real database/vector/retrieval/agent/personalization services with isolated fixtures, capture only logical IDs and measured outputs, and close all clients. The CLI does not run provider calls concurrently; the provider owns and must honor the requested workload. Do not use production user libraries.

## Metrics and comparisons

- Precision@K = relevant results in first K / K.
- Recall@K = relevant results in first K / all judged relevant files.
- MRR = reciprocal rank of first relevant result, averaged across queries.
- NDCG@K uses graded gain `2^grade - 1` with logarithmic rank discount.
- MAP averages precision at each relevant result, divided by the number of judged relevant files.
- Hit Rate@K is 1 when a relevant result appears in the first K, otherwise 0.
- Zero-result rate is the fraction of query outputs with no results.
- Latency summary includes mean, p50 and p95; p99 is emitted only for at least 100 observations.
- Reranking reports before/after ranking on the same query IDs; candidate and status counts are included when supplied.
- RAG deterministic checks include answerability agreement, expected evidence-file recall, judged context precision, and answer/reference token F1. These lexical and annotation-based measures are not Ragas faithfulness or semantic correctness. Extraction observations may include CER/WER only when `reference_trusted` is explicitly true.
- Citation validity checks known logical file IDs and retrieval membership. Citation support precision/recall and provenance accuracy are checked separately against independent evidence annotations; claim coverage requires a claim citation ID that exists in the citation list.
- Agent routing accepts the dataset's explicit valid-tool alternatives. The deterministic smoke route is not a full LangGraph/tool-selection benchmark.
- Personalization/recommendation use the same query/scenario IDs and ranking metrics. Cold start, profile kinds, empty results and categories are explicit scenarios.
- Ingestion/performance/reliability groups summarize measured observations; denominators must be supplied by the controlled integration run.

Bootstrap intervals use deterministic nonparametric resampling over per-query metric values (default 2,000 samples, 95%, seed 17). Small datasets produce unstable intervals; the interval is descriptive and is not a blanket significance claim. Baseline comparison calculates paired query differences when per-query values exist and matches retrieval configurations by run name. It does not flag regressions unless caller-supplied thresholds are provided. Threshold JSON can use `{"retrieval.hybrid.mrr":{"min_delta":-0.03},"performance.p95_latency_ms":{"max_delta":10}}`; use `min_delta` for higher-is-better metrics and `max_delta` for costs.

## RAGAS and model judging

Ragas is not installed in this repository and no version-specific Ragas call is assumed. Model evaluation remains optional: a provider may integrate a compatible Ragas version after checking that installed release's API, and should record judge model/version/configuration, prompt version, samples, and manual review. Keep LLM-as-judge output separate from deterministic checks; judge disagreement and shared model weaknesses remain possible. Normal CI does not download models or start Ollama.

## Reproducibility and privacy

Set the same dataset version, query set, K values, unjudged policy, provider configuration, model identifiers, model settings, warmups and repetitions for comparisons. Record GPU/runtime details in provider configuration when applicable; the CLI records host OS, CPU count, architecture, Python, selected package versions and Git SHA. `--seed` controls bootstrap sampling and provider guidance; it does not make a nondeterministic LLM deterministic. Local hardware changes latency materially. Never include file contents, private paths, user activity, credentials, database volumes or model weights in datasets or reports. Use synthetic or explicitly curated data only.

## Current measurements and limits

The deterministic runner's first committed baseline should be generated by the command above from the implementation and dataset version named in its report. No ScoutX production retrieval, Qwen reranker/embedding, RAG model, OCR, transcription, filesystem scan, database/Qdrant/Redis service, or resource benchmark is claimed by the offline smoke run. Those are NOT RUN until a local provider runs against isolated fixtures and the required services/models. No Ragas, dashboard, API, database schema, or production ranking dependency was added.

