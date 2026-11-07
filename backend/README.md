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

The Compose PostgreSQL service uses trust authentication for local development and binds to loopback only. Use proper password authentication and private networking for shared or production deployments. Set application overrides in `backend/.env`; no credentials are committed.

## Integration interfaces

- `app.config.get_settings()` returns the validated Pydantic `Settings` object.
- `app.database.engine.engine`, `SessionLocal`, and `get_db()` provide the SQLAlchemy engine, session factory, and FastAPI dependency.
- `app.database.models.Base` owns all ORM metadata. The canonical entities are `User`, `IndexRoot`, `File`, `DocumentChunk`, `IndexJob`, `FileActivity`, `UserPreference`, `Project`, `ProjectFile`, `AgentRun`, and `RecommendationFeedback`.
- `app.database.schemas` contains create/read transfer schemas; it does not duplicate ORM models.
- `app.database.repositories` contains user-scoped repository classes. Each repository accepts a SQLAlchemy `Session`; data access to user-owned records also takes the authenticated `user_id` explicitly.
- Call `session.commit()` at the application transaction boundary. Repository methods flush so generated IDs and constraint failures are available without committing a caller's transaction.

The extracted chunk text is stored in PostgreSQL (`document_chunks.content`). Qdrant point IDs are references only; Qdrant is not the canonical source for file or chunk records.
