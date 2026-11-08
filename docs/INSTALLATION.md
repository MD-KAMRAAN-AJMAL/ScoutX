# Installation

## Recommended: local Docker Compose

Prerequisites: Docker Engine with Compose v2, and enough local resources for
PostgreSQL, Qdrant, Redis, Ollama, the extraction worker, and optional model
caches. Image builds install the optional extraction and embedding dependency
groups. They do not download model weights or index files at startup.

```sh
cp .env.example .env
openssl rand -hex 24
# Replace POSTGRES_PASSWORD in .env with the generated URL-safe string.
docker compose up --build -d
docker compose ps
docker compose logs -f backend worker worker-beat
docker compose exec ollama ollama pull qwen3:8b
docker compose exec ollama ollama list
```

Open `http://127.0.0.1:5173`. In Settings, add `/demo/sample-files`, then
request reconciliation on the Indexing page. Wait for processing before
searching. API docs are at `http://127.0.0.1:8000/docs`.

The API container runs `uv run alembic upgrade head` before serving. It exits
if an upgrade fails. PostgreSQL, Qdrant, Redis, Ollama data, and Hugging Face
cache use named volumes. No host user folder is mounted by default; the included
demo root is mounted read-only. To add a host root, edit both API and worker
volumes in Compose, map it to a fixed container path read-only, and register
that container path in the UI.

## Local development

Prerequisites: Python 3.12, uv, Node.js 22+, npm, Docker Compose for dependencies,
and system libmagic. FFmpeg and Java are used by media/Tika parsers. Linux
distributions may use `libmagic1`, `ffmpeg`, and a headless Java runtime; macOS
can install these through Homebrew. PaddleOCR, Docling, Faster Whisper, and
embedding/reranker packages are optional and have significant platform-specific
dependencies.

The Compose worker has no GPU device passthrough configured, so automatic
device selection uses CPU in the standard deployment. A host-run backend can
select supported CPU, CUDA, or Apple MPS devices through
`SCOUTX_EMBEDDING_DEVICE`; enabling container GPU access requires a separate
host-specific Docker runtime configuration and was not verified here.

The Compose service is the default Ollama runtime. For a host-managed runtime,
install Ollama with its platform installer, start the local server, and run
`ollama pull qwen3:8b` followed by `ollama list`. Set
`SCOUTX_OLLAMA_BASE_URL=http://127.0.0.1:11434` in `backend/.env` when the API
also runs on the host. Do not use `127.0.0.1` from inside the Compose backend;
inside Compose the service address is `http://ollama:11434`.

Ollama provides generation only. Embeddings use the fixed,
revision-pinned `Qwen/Qwen3-Embedding-0.6B` Sentence Transformers model, while
the cross-encoder defaults to `Qwen/Qwen3-Reranker-0.6B`. Their Hugging Face
cache is under `HF_HOME` (the Compose `model_cache` volume). They are not
available through Ollama's model registry and are loaded/downloaded lazily on
first use. Set `SCOUTX_RERANKER_LOCAL_FILES_ONLY=true` to prohibit reranker
network downloads; no equivalent embedding path override is implemented.
Check the Ollama model with `docker compose exec ollama ollama list`. Check
whether the Hugging Face caches already exist without triggering a download:

```sh
docker compose exec backend python -c "from pathlib import Path; p=Path('/home/scoutx/.cache/huggingface/hub'); print('\n'.join(x.name for x in p.glob('models--Qwen--Qwen3-*')) or 'No cached Qwen models')"
```

Start data services and Ollama (the Compose file also contains API/UI services,
so select only dependencies):

```sh
cp .env.example .env
docker compose up -d postgres qdrant redis ollama
cd backend
uv sync --extra extraction --extra embedding --group dev
cp .env.example .env
```

Edit `backend/.env` so its `DATABASE_URL` password matches the root `.env` and
its database host is `localhost`. Then initialize and run:

```sh
uv run alembic upgrade head
uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, install the frontend and run it:

```sh
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Start a Celery worker and beat process from `backend/` in separate terminals:

```sh
uv run celery -A app.workers.celery_app:celery_app worker --loglevel=INFO
uv run celery -A app.workers.celery_app:celery_app beat --loglevel=INFO
```

Host-based local development runs the API on loopback, so it does not use the
trusted reverse-proxy setting used only by the Compose frontend.

## Lightweight checks

For docs, deterministic metrics, and unit tests, do not install optional ML
groups or pull model weights:

```sh
cd backend
uv sync --group dev
uv run pytest -q
uv run python -m evaluation.cli --mode all --output-dir evaluation/results
cd ../frontend
npm ci
npm test
npm run build
```
