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
