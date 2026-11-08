# Backend foundation

## Local setup

Use Python 3.12 and uv (the repository pins the interpreter in `.python-version`):

```sh
uv sync
cp .env.example .env
```

From the repository root, start PostgreSQL 18, Qdrant, and Redis:

```sh
docker compose up -d
```

Then, from `backend/`, apply the schema:

```sh
uv run alembic upgrade head
```

Install libmagic on the host (`brew install libmagic` on macOS or `apt install libmagic1` on Debian/Ubuntu), then copy `.env.example` to `.env`. The scanner uses libmagic to identify MIME types from file headers.

The Compose PostgreSQL service uses trust authentication for local development and binds to loopback only. Use proper password authentication and private networking for shared or production deployments. Set application overrides in `backend/.env`; no credentials are committed.

## Integration interfaces

- `app.config.get_settings()` returns the validated Pydantic `Settings` object.
- `app.database.engine.engine`, `SessionLocal`, and `get_db()` provide the SQLAlchemy engine, session factory, and FastAPI dependency.
- `app.database.models.Base` owns all ORM metadata. The canonical entities are `User`, `IndexRoot`, `File`, `DocumentChunk`, `IndexJob`, `FileActivity`, `UserPreference`, `Project`, `ProjectFile`, `AgentRun`, and `RecommendationFeedback`.
- `app.database.schemas` contains create/read transfer schemas; it does not duplicate ORM models.
- `app.database.repositories` contains user-scoped repository classes. Each repository accepts a SQLAlchemy `Session`; data access to user-owned records also takes the authenticated `user_id` explicitly.
- Call `session.commit()` at the application transaction boundary. Repository methods flush so generated IDs and constraint failures are available without committing a caller's transaction.
- `app.ingestion.scanner.register_directory()` validates the selected folder with the current process user's permissions before creating an `IndexRoot` record. Set extra excluded paths with the JSON array in `EXCLUDED_PATHS` or the root's `settings.excluded_paths`.
- `app.ingestion.watcher.FilesystemWatcherManager` owns live Watchdog observers. Start enabled roots for the authenticated user during application lifespan and start a newly registered root immediately; stop all observers during shutdown. Watchdog handlers only publish `discover_file` and `delete_file` Celery tasks.
- Run inventory workers with `uv run celery -A app.workers.celery_app:celery_app worker --loglevel=INFO`; run periodic root reconciliation with `uv run celery -A app.workers.celery_app:celery_app beat --loglevel=INFO`.

The extracted chunk text is stored in PostgreSQL (`document_chunks.content`). Qdrant point IDs are references only; Qdrant is not the canonical source for file or chunk records.

## Phase 8 API and frontend

Start the local dependencies from the repository root, then apply the database schema and start the API:

```sh
docker compose up -d
cd backend
uv sync
cp .env.example .env
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, configure and start the Vite application:

```sh
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env.local` if the API is hosted elsewhere. Backend configuration is read from `backend/.env`; commonly adjusted values include `DATABASE_URL`, `QDRANT_URL`, `REDIS_URL`, `SCOUTX_OLLAMA_BASE_URL`, `SCOUTX_OLLAMA_MODEL`, and `SCOUTX_CORS_ORIGINS`. The example files contain local development defaults and no credentials.

### API endpoints

- `GET /api/v1/health` reports API availability and the PostgreSQL, Qdrant, Redis, and Ollama reachability checks.
- `POST /api/v1/search` runs the existing LangGraph retrieval pipeline. The request supports `query`, `limit`, `mode`, `generate_answer`, `extensions`, `root_id`, `modified_after`, and `modified_before`.
- `GET /api/v1/files` lists files with pagination, folder, filename, extension, status, modification-time, and sort filters.
- `GET /api/v1/files/{file_id}` returns user-scoped file metadata and active extracted snippets. It accepts a file ID, never a filesystem path.
- `GET /api/v1/indexing/status` returns indexing counts and recent jobs. `/api/v1/index/status` remains an alias.
- `POST /api/v1/feedback` records relevant/not-relevant and opened/selected events.
- `GET/POST/DELETE /api/v1/roots` manages explicitly authorized folders; `POST /api/v1/index/reconcile` queues reconciliation for registered roots.

Example retrieval-only request:

```sh
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Find the budget forecast","generate_answer":false,"extensions":[".xlsx"]}'
```

Run checks from the repository root:

```sh
cd backend && uv run ruff check app tests && uv run pytest -q
cd ../frontend && npm test && npm run build
```

The health endpoint remains available when a dependency is down and reports `degraded` without returning connection strings. Search can return retrieval results even when answer generation fails. File paths shown in file/search responses are relative to the authorized folder; details and snippets are resolved through the authenticated file ID.

Troubleshooting: confirm Docker services are running, `backend/.env` points to their addresses, Ollama is serving the configured model for answer generation, and `SCOUTX_CORS_ORIGINS` includes the frontend origin (`http://127.0.0.1:5173` by default). The frontend displays an API or dependency error when a service cannot be reached.

## Phase 9 personalization

Personalization reuses `file_activity`, `user_preferences`, `recommendation_feedback`, active PostgreSQL chunks, and the existing Qdrant `file_content_chunks` collection. No additional database table, model, or environment variable is required. The profile is derived on demand from at most 1,000 access events and feedback records from the previous 180 days; raw activity details and document content are not included in the profile or activity API. Search-run records continue to use the existing `agent_runs.query` field.

Activity is recorded for search submission, file opening/selection, recommendation dismissal, and positive/negative feedback. Search activity retains mode, result count, and the associated run ID, but not a copy of the query. Best-effort search activity is written after the HTTP response. Repeated same-file activity within five seconds and repeated searches for the same run within fifteen seconds are coalesced. Explicit feedback supersedes the most recent explicit feedback for a file within five minutes; later feedback can change the signal.

The profile derives preferred extensions and frequently used roots from access events. Access weights use a 30-day exponential half-life; explicit feedback uses a 60-day half-life; dismissal suppression is short-lived (30 days). The scoring adjustment is:

```text
delta = clamp(-0.07, +0.08,
  0.025 * preferred_extension
  + 0.025 * preferred_root
  + 0.015 * decayed_extension_feedback
  + 0.015 * decayed_root_feedback
  + 0.03 * decayed_file_access
  + 0.06 * latest_decayed_explicit_feedback)
personalized_score = clamp(0, 1, reranker_score + delta)
```

Only existing retrieval candidates with reranker score at least `0.20` receive a boost or penalty. User exclusions are applied before scoring. With personalization disabled, search preserves the Phase 8 retrieval/reranking order. Grounded context and citations are then assembled from the same retrieved chunks in the resulting order.

`GET /api/v1/recommendations` supports `recommended_for_you`, `recently_accessed`, `continue_working`, `frequently_used`, `related`, and `recently_indexed`. Related files use Qdrant recommendations seeded by active chunk points from files the user recently accessed or marked useful; the query reuses stored vectors and filters results by user and active status. PostgreSQL then hydrates the source file and checks its enabled root and current filesystem permissions. Cold-start users receive recently indexed files when available. Recommendation paths are relative to their authorized root.

Personalization controls and data routes:

- `GET /api/v1/personalization/profile` returns aggregate counts and preferences, not raw history.
- `GET/PUT /api/v1/personalization/preferences` controls the feature, preferred/excluded file types, preferred/excluded authorized roots, and recommendation limit.
- `GET/DELETE /api/v1/activity` pages through and clears activity plus explicit feedback.
- `POST /api/v1/personalization/reset` resets the derived profile cutoff; pass `{"clear_activity_history": true}` to also delete activity and feedback.
- `GET /api/v1/recommendations` returns stable file IDs, safe relative paths, categories, scores, and signal-based explanations.

The Settings page exposes the enable switch, type/folder preferences, recent activity, profile reset, and a confirmation before history deletion. When disabled, the UI and search still use the existing search, file library, and reranker. Search-run query text remains in the pre-existing `agent_runs` record; clearing activity removes `file_activity` and `recommendation_feedback`, not those search-run records.

Offline ranking metrics (`precision@k`, `recall@k`, MRR, NDCG, and category diversity) are tested on controlled fixtures. They do not establish that personalization improves relevance for real users; that requires representative relevance judgments and a measured baseline.

The derived profile is held in a bounded process-local cache for up to 30 seconds and invalidated when activity, feedback, preferences, or history are changed. This avoids repeating aggregation queries for rapid searches while bounding staleness across local worker processes.

## Evaluation and benchmarks

Phase 11's reproducible evaluation CLI, synthetic dataset, metric definitions, artifact/provider format, and limitations are documented in [evaluation/README.md](evaluation/README.md). The default smoke run is an offline lexical baseline and must not be interpreted as a production retrieval benchmark.

## Phase 10 security, performance, and reliability

### Security assumptions and controls

ScoutX's development API is single-user and local-only. `SCOUTX_API_LOCAL_ONLY=true` is enforced by middleware; setting it to false fails settings validation until a network authentication provider exists. Run Uvicorn on `127.0.0.1`; the middleware also rejects non-loopback clients if the server is accidentally bound to a public interface. CORS uses explicit localhost origins and permits only the methods used by the UI, including `PUT`. Do not expose the API through a reverse proxy or tunnel without adding an authentication boundary first.

Request bodies are bounded by `SCOUTX_MAX_REQUEST_BODY_BYTES` (1 MiB by default), including streamed requests without `Content-Length`. Validation errors return field locations and error types without echoing rejected values. File APIs resolve a stable file ID to its registered root, require that root to remain enabled, and return a safe relative path. Discovery, recommendation, and detail checks reject symlink components and paths that resolve outside the authorized root. Retrieval hydration excludes disabled roots. These checks are re-run before scanning and parsing; a path can still race after validation, and this is not an `openat`-based filesystem sandbox.

Parser inputs default to a 512 MiB maximum (`SCOUTX_PARSER_MAX_FILE_BYTES`). ZIP-based office formats are checked for unsafe member paths, entry count, member size, expanded size, and extreme compression ratio before parser libraries inspect them. External media tools are invoked with argument arrays, no shell, restricted environment variables, and bounded timeouts (`SCOUTX_PARSER_TIMEOUT_SECONDS`, 300 seconds by default). Celery uses two worker processes by default and applies a 55-minute soft and 60-minute hard task limit; tune with `SCOUTX_WORKER_CONCURRENCY`, `SCOUTX_WORKER_TASK_SOFT_LIMIT_SECONDS`, and `SCOUTX_WORKER_TASK_HARD_LIMIT_SECONDS` for the host. Parser instances are reused within each worker process to avoid reloading heavyweight media models for each file.

These controls are not an OS-level parser sandbox. Docling, OCR, Tree-sitter, spreadsheets, and Tika run in the worker process; Python libraries cannot be forcibly stopped by Celery's task timeout until the worker process is terminated. Tika's own server process lifecycle is not fully controlled by the Python timeout. Keep indexing limited to files you trust and run the worker as the normal desktop user, never as root. No general archive extraction endpoint exists.

SQLAlchemy uses parameterized statements, connection pre-ping, a five-second PostgreSQL connect timeout, and a bounded pool wait. Qdrant calls have a five-second default timeout. Existing candidate, reranker, context, activity-history, and recommendation limits remain in place. The profile cache remains bounded and invalidated on writes. No schema migration or indexing behavior change was required for these controls.

### Health, logs, and traces

- `GET /api/v1/live` reports process liveness without contacting dependencies.
- `GET /api/v1/ready` requires PostgreSQL and Qdrant; Redis and Ollama are reported as optional dependencies.
- `GET /api/v1/health` preserves the existing all-dependency status response.

API logs are emitted as JSON with a server-generated `X-Request-ID`, route template, method, status, and duration. They omit request bodies, raw query text, interpolated log arguments, and absolute filesystem paths. Existing agent, reranker, and parser logs add operation and duration/status context. Request validation errors do not reflect user-provided values. React renders file-derived text through ordinary text nodes; it does not use raw HTML rendering or construct filesystem URLs.

OpenTelemetry API traces and aggregate request count/duration metrics are opt-in. Leave `SCOUTX_OTEL_ENABLED=false` for local-only operation with no telemetry export. To enable export, configure `SCOUTX_OTEL_ENABLED=true` and `SCOUTX_OTEL_EXPORTER_ENDPOINT` with a trusted OTLP/HTTP collector base URL (for example `http://127.0.0.1:4318`). FastAPI, SQLAlchemy, HTTPX, the agent invocation, extraction/chunking, and recommendation generation are instrumented. No collector or third-party telemetry service is started by Docker Compose. Never configure an untrusted exporter; traces can contain route and database operation metadata, though ScoutX does not add raw queries, file contents, or paths as custom span attributes.

### Audit notes and verification limits

The audit found concrete gaps in local API exposure enforcement, body-size bounds, PUT CORS support for preferences, symlink-parent/root revalidation in file reads, unbounded default Celery worker concurrency/task time, parser ZIP handling, and dependency health checks sharing one route. Those are addressed above. Existing controls retained include Pydantic request schemas, user-scoped SQL/Qdrant retrieval, query and pagination limits, content-based MIME detection, `O_NOFOLLOW` for final file hashing, no-shell FFmpeg invocation, bounded graph retries/tool allowlists, citation ID validation, and React text rendering.

No external rate limiter was added: the API is restricted to loopback and search/body limits already apply. No CSP header was added because Vite serves the UI separately and its development server needs its own header policy. The local test suite verifies the new middleware, path checks, archive checks, and subprocess timeout; it does not prove race-free filesystem access or isolate parser memory/CPU. OpenTelemetry export and live PostgreSQL/Qdrant/Redis/worker startup require external services and were not verified by the default unit suite.
