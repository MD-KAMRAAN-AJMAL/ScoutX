# API reference

The FastAPI application publishes OpenAPI at `/openapi.json` and Swagger UI at
`/docs` when running locally. The API is a local single-user workspace, not a
multi-user authenticated service. Compose exposes it only on loopback and the
frontend proxy is the normal UI entry point.

## Routes

| Route | Purpose |
| --- | --- |
| `GET /api/v1/health` | Reachability checks for PostgreSQL, Qdrant, Redis, and Ollama. Returns `degraded` when a dependency is unavailable. |
| `GET /api/v1/live` | Process liveness without dependency probes. |
| `GET /api/v1/ready` | Readiness requiring PostgreSQL and Qdrant; reports optional dependency states too. |
| `GET, POST, DELETE /api/v1/roots` | List, register, or disable explicitly authorized roots. A deletion disables the root and preserves file history. |
| `POST /api/v1/index/reconcile` | Queue reconciliation for registered roots, optionally a selected root. |
| `GET /api/v1/indexing/status` | User workspace indexing summary and recent job state. `/api/v1/index/status` is retained as a compatibility alias. |
| `POST /api/v1/search` | Agent-routed search; supports query, bounded limit, mode, answer generation, root/extension filters, and date filters. |
| `GET /api/v1/files` | Paginated and filtered user-scoped file metadata. |
| `GET /api/v1/files/{file_id}` | File metadata and indexed snippets by stable ID, not arbitrary path. |
| `GET /api/v1/recommendations` | Personalized or fallback recommendation items. |
| `/api/v1/personalization` | Preferences, profile, activity, and personalization controls. See generated OpenAPI for individual methods. |
| `POST /api/v1/feedback` | Record result/file feedback. |

Search response includes result files, source citations/locations, tools used,
retrieval plan, warning and reranking/generation status fields, and stage timing
where available. The frontend displays paths relative to authorized roots.

## Validation and errors

Pydantic rejects unknown or invalid request fields. Request bodies are limited
by `SCOUTX_MAX_REQUEST_BODY_BYTES`. Common responses use an `error` object with
a stable `code`, safe `message`, and optional `details`; database outages return
503 and unexpected errors do not expose exception text. Search, pagination,
candidate, reranker, context, activity, and recommendation limits are enforced
in the existing routes/services.

Use IDs returned by the API for file, root, and feedback operations. Never
construct a filesystem path from an API result or expose an arbitrary path
endpoint. Ollama failures may leave retrieved files available while answer
generation reports a fallback/error status.
