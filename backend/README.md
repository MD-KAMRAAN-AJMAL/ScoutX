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
