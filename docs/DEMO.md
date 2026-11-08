# ScoutX synthetic end-to-end demo

This repeatable demonstration uses only `demo/sample-files`, a small corpus of
synthetic Markdown, Python, TSX, CSV/XLSX, PNG, WAV, and MP4 files. The audio is
a short system-synthesized voice sample describing the public demo workflow;
the image is a generated bitmap with synthetic OCR text. No private library or
external copyrighted sample is included. The repository contains the media
assets so the demo does not synthesize them at runtime.

## Prepare and start

```sh
cp .env.example .env
# Set a random URL-safe POSTGRES_PASSWORD in .env.
docker compose up --build -d
docker compose exec ollama ollama pull qwen3:8b
docker compose ps
```

Open `http://127.0.0.1:5173`; check `http://127.0.0.1:8000/api/v1/ready` and
`/api/v1/health`. Register `/demo/sample-files` from Settings. The synthetic
root is mounted read-only into the API and worker containers. In Indexing, run
Reconcile and wait for jobs to finish. Extraction and embeddings need optional
runtime packages and may download model assets on first applicable task. The
image/media parser paths have not been verified in this environment.

## Walkthrough

| User action | Expected result/evidence | Main implementation path |
| --- | --- | --- |
| Register `/demo/sample-files`, then Reconcile | Root appears and file/job counts increase; files outside that mount are not visible. | Roots API, scanner, Redis, Celery, PostgreSQL |
| Search `database normalization` in keyword mode | `notes/database-normalization.md` is a likely direct result. | Lexical retrieval and file aggregation |
| Search `Which note explains separating root ownership from file records?` in automatic/hybrid mode | The wording need not match the filename; evidence in `notes/database-normalization.md` describes this schema rule. | LangGraph planner, lexical/dense retrieval, RRF, reranker |
| Ask `What does first normal form require in the synthetic database note?` | A grounded answer should state one value per cell and stable row identity, with a citation to the database note. | Context builder and Ollama structured answer |
| Search `How do file contents and vector references move through ScoutX?` | Relevant passages can come from the release brief and database note; inspect individual file evidence. | Hybrid retrieval and file aggregation |
| Ask `What color is the office elevator?` | No reliable evidence exists in the synthetic collection; inspect the answer/status for insufficient evidence rather than treating a guess as sourced. | Evidence threshold and grounded generation |
| Search `validate signed bearer sessions` | The synthetic Python module is expected to match the HMAC signing and comparison helpers even though its filename says session tokens. | Tree-sitter extraction and code retrieval |
| Search `labels from a scanned image` | Expected sources include the vision note and OCR image when the optional image parser runs. | Image preprocessing, PaddleOCR, semantic/lexical retrieval |
| Search `monthly expenses` | CSV result contains hosting, testing, hardware, and documentation amounts; their synthetic total is USD 510. | Spreadsheet parser and metadata/lexical retrieval |
| Inspect `project-metrics.xlsx` and ask about release readiness | Expected evidence includes 44 files discovered, 42 indexed, and 63 tests passed. | XLSX parser, source references, citations |
| Ask what the audio says about storage | WAV/MP4 transcript should mention PostgreSQL extracted text and Qdrant vector references if Faster Whisper model setup succeeds. | FFmpeg, faster-whisper, timestamped media chunks |
| Open a file, mark a result relevant/not relevant, and revisit recommendations | Activity and feedback persist; personalization settings can be toggled and preferred extensions edited. A new profile can show cold-start behavior. | Feedback/activity APIs and recommendation services |
| Run evaluation CLI | A local report is written; offline ranking baseline is labelled synthetic and not production retrieval. | `backend/evaluation` |

Search ranking can vary with model availability and extraction status. The
expected source files above are factual fixture expectations, not a promise
about exact rank order. UI interactions, OCR, media transcription, and the
RAG question were not all live-tested in this environment.

## Short presentation script

1. Show the configured root and explain that the demo corpus is synthetic.
2. Reconcile and show indexing progress.
3. Search the HMAC session code with different wording from its filename.
4. Ask the first-normal-form question and open its source citation.
5. Ask the elevator question to show the insufficient-evidence boundary.
6. Open the expense sheet, submit relevance feedback, and show settings and
   recommendations.
7. Run the deterministic evaluation and explain that it is not a live model
   benchmark.

Stop the app with `docker compose down`; persistent volumes remain. Do not
delete volumes during the presentation shutdown.
