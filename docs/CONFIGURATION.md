# Configuration

Compose reads the root `.env`; copy `.env.example` and replace the PostgreSQL
password with a URL-safe random value (`openssl rand -hex 24`). The example is
not a secret and must not be used as a shared deployment credential. Host
application ports bind to `127.0.0.1` by default. Do not change that binding
without first adding a supported authentication boundary.

| Group | Variables and behavior |
| --- | --- |
| PostgreSQL | `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_PORT`. Fresh Compose volumes initialize with SCRAM; existing database volumes retain their initialization-time auth configuration. |
| Qdrant | `QDRANT_HTTP_PORT`, `QDRANT_GRPC_PORT`, `QDRANT_COLLECTION`, `QDRANT_TIMEOUT_SECONDS`; Compose uses the internal service URL. |
| Redis | `REDIS_PORT`; Celery uses the internal Compose URL. |
| API/UI | `API_PORT`, `FRONTEND_PORT`; both published on loopback. `VITE_API_BASE_URL` is `/api/v1` in the packaged frontend. |
| Ollama | `OLLAMA_PORT`, `OLLAMA_IMAGE`, `SCOUTX_OLLAMA_MODEL`. The configured default is `qwen3:8b`, from `backend/app/config.py`. Pull the model explicitly. |
| Embeddings | `SCOUTX_EMBEDDING_DEVICE` accepts auto/cpu/cuda/mps; CPU threads and batch size are configurable. The code pins `Qwen/Qwen3-Embedding-0.6B` and a repository revision. |
| Reranking | `SCOUTX_RERANKER_ENABLED`, `SCOUTX_RERANKER_MODEL`, local-files-only, fallback, and bounds are validated in `backend/app/config.py`. The configured model is `Qwen/Qwen3-Reranker-0.6B`. |
| Filesystem | `SCOUTX_DEMO_ROOT` is a host path mounted read-only at `/demo/sample-files`. Add other host roots explicitly to both backend and worker Compose volume lists, then register their container paths. |
| Worker | `SCOUTX_WORKER_CONCURRENCY` and task time limits bound Celery work. Celery beat schedules reconciliation every six hours. |
| Security | `SCOUTX_API_LOCAL_ONLY` must remain true. `SCOUTX_TRUSTED_PROXY_IPS` is an internal Compose-only address for its loopback-published frontend proxy. |
| Observability | `SCOUTX_OTEL_ENABLED` defaults false. If enabled, set a trusted `SCOUTX_OTEL_EXPORTER_ENDPOINT`; no collector is bundled. |
| Other limits | Request size, parser file size/time limits, CORS, RAG context size/confidence, SQL echo, and database pool settings are listed in `backend/.env.example` and validated by `Settings`. |

Local host processes read `backend/.env`. The frontend reads `frontend/.env.local`;
it contains only public `VITE_` configuration. Never put credentials in a
frontend environment file. Compose maps selected root variables into each
container; editing `backend/.env` does not configure Compose.
