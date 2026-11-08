# Deployment, data, backup, and restore

The supported deployment is local Docker Compose. It starts the API, static
frontend, Celery worker and beat, PostgreSQL 18, Qdrant, Redis, and Ollama.
Database, Qdrant, Redis, Ollama, and Hugging Face data use named volumes. Only
app ports and optional local dependency ports are published, and all published
ports bind to loopback. The frontend and worker run non-root; the API image also
runs as UID 10001. The sample file mount is read-only.

PostgreSQL's `POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256` and role creation
apply only when the database volume is empty. A volume created by an earlier
trust-auth configuration is not rewritten by changing Compose environment
variables. Preserve that data and inspect/migrate its role password and
`pg_hba.conf` with PostgreSQL's supported administration procedures before
assuming SCRAM is active; never delete the volume to apply this setting.

Start with `cp .env.example .env`, set a random URL-safe `POSTGRES_PASSWORD`,
then run `docker compose up --build -d`. API startup applies the Alembic upgrade
and fails if it cannot complete. Readiness is at `/api/v1/ready`.

## What to back up

- PostgreSQL: users, roots, file metadata, extracted chunk content, jobs,
  personalization, feedback, and graph checkpoints.
- Qdrant: vector points and payloads; can be rebuilt from original files and
  PostgreSQL records by reindexing when the source files remain available.
- Redis: transient broker, task, and result data; not the canonical index.
- Ollama volume and Hugging Face cache: downloaded model artifacts; these can
  be re-fetched if network and storage permit.
- `.env` and any explicit Compose mount configuration: secrets and host-to-
  container root mappings. Store backups securely.
- Original files: remain in the user's configured host folders; this is the
  source-of-truth copy and must be backed up separately.
- Evaluation reports: under `backend/evaluation/results`; generated local
  results are ignored by Git unless intentionally exported elsewhere.

## PostgreSQL backup/restore

```sh
mkdir -p backups
docker compose exec -T postgres pg_dump -U "$(sed -n 's/^POSTGRES_USER=//p' .env)" -Fc scoutx > backups/scoutx.dump
```

Restore is destructive to the selected database. First stop API and workers and
preserve a current backup. Then restore into an empty or explicitly selected
database; do not run this against a database whose contents must be retained.
For a fresh local recovery database:

```sh
docker compose stop backend worker worker-beat
docker compose exec -T postgres createdb -U scoutx scoutx_restore
docker compose exec -T postgres pg_restore -U scoutx -d scoutx_restore --no-owner < backups/scoutx.dump
```

These commands illustrate PostgreSQL backup/restore but have not been exercised
against this workspace's Compose environment. Set the username/database to
match `.env`. Validate a restore in an isolated database before relying on it.

## Qdrant backup

The Compose Qdrant volume is `scoutx_qdrant_data` under the default project
name. For a cold file-level copy, stop Qdrant, archive the volume, then restart:

```sh
docker compose stop qdrant
mkdir -p backups
docker run --rm -v scoutx_qdrant_data:/source:ro -v "$PWD/backups:/backup" alpine \
  tar -czf /backup/qdrant-data.tar.gz -C /source .
docker compose start qdrant
```

This copy procedure has not been verified here. For a production-quality
backup, use Qdrant's supported snapshot/restore API for the deployed version
and test restoration. If the Compose project name or volume name is changed,
inspect `docker volume ls` and substitute the actual volume name.

## Recovery

An interrupted worker leaves durable job/file status in PostgreSQL; inspect
`docker compose logs worker` and the indexing page, then use **Reconcile** to
re-discover registered roots. The original files are not deleted by index
cleanup. If the Qdrant index is lost, preserve the database, restore Qdrant or
reindex source roots. Do not remove volumes as part of ordinary shutdown.

For model failures, inspect `docker compose logs ollama backend worker`, confirm
the configured model exists with `docker compose exec ollama ollama list`, and
pull the exact configured Ollama model explicitly. Embedding and reranker
weights are separate Hugging Face models and may be lazily downloaded by the
first applicable indexing/search request.
