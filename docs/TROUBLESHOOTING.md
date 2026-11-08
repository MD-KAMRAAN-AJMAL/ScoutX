# Troubleshooting

## Docker Compose command is missing

Install/enable the Docker Compose v2 plugin for the selected Docker Engine,
then confirm `docker compose version`. The standalone `docker` command alone
does not provide Compose.

## Containers do not become healthy

Run `docker compose ps` and `docker compose logs --tail=200 postgres qdrant
redis ollama backend worker`. PostgreSQL must pass its password-based health
check; API startup waits for core dependencies and runs Alembic. A migration
error stops the backend. Do not clear the PostgreSQL volume to troubleshoot.

## UI cannot reach the API

Use the bundled UI at `http://127.0.0.1:5173`. It proxies `/api` through the
configured local frontend container. Check `docker compose logs frontend
backend` and `http://127.0.0.1:8000/api/v1/live`. Host local-development Vite
uses `frontend/.env.local` and a host API on port 8000.

## Health is degraded or search is empty

`/api/v1/health` returns individual states for PostgreSQL, Qdrant, Redis, and
Ollama. `/api/v1/ready` requires PostgreSQL and Qdrant. Ollama may be healthy
with no model pulled; run `docker compose exec ollama ollama list` and pull the
configured identifier. Confirm roots are enabled, reconciliation has completed,
and jobs have not failed in Indexing. Look at worker logs for parser/model errors.

## A root is inaccessible

The API process must be non-root and have read/traverse permissions. In
Compose, the host directory must be explicitly mapped into both backend and
worker containers. Register the container-side mount path, not the host path.
Mount removable drives only while available; reconciliation will not treat an
unavailable root as an empty tree. Symlinks are excluded by default.

## Parser or model is missing

Lightweight local `uv sync --group dev` does not install OCR, Docling, Whisper,
or embedding packages. Install only the required optional group, check system
libmagic/FFmpeg/Java, and retry. The Qwen embedding/reranker caches and the
Ollama model are separate. Model initialization is lazy, so the first indexing
or query may take longer. Never commit the cache.

## Backup and restore

See [Deployment](DEPLOYMENT.md). Preserve source files and current backups.
Inspect Compose logs and the indexing UI, then run reconciliation. Do not use
`docker compose down -v` as a generic recovery command.
