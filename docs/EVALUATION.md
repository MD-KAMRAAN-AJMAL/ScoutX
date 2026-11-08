# Evaluation and benchmarks

The Phase 11 framework lives in `backend/evaluation`. It validates the
versioned, synthetic `scoutx-small@1.0.0` dataset; computes deterministic
ranking metrics; accepts explicitly captured observations; compares runs; and
writes JSON, Markdown, and CSV reports. Run from `backend/`:

```sh
uv run python -m evaluation.cli --mode all --output-dir evaluation/results
uv run python -m evaluation.cli --mode retrieval --k 1 3 5 10 --repetitions 3 --warmups 1
```

Results generated under `evaluation/results` are ignored by Git. JSON is the
source of truth. Provider/artifact commands, dataset scope, metrics, and
limitations are specified in `backend/evaluation/README.md`.

The default lexical-overlap retrieval result is an offline synthetic-corpus
baseline. It is not ScoutX production BM25, hybrid retrieval, or latency. The
deterministic CLI does not run a live LangGraph, reranker, Ollama, OCR, Whisper,
or Qwen embedding evaluation unless a provider supplies observed artifacts.
Ragas integration is not implemented. Never report those as measured or use
offline scores as real-world effectiveness claims.

Record dataset version, model IDs, Git SHA, configuration, sample counts,
skipped groups, and failures with any exported report. Do not include source
content, paths, or private user libraries in captured artifacts.
