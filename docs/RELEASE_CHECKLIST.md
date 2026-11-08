# Phase 12 release checklist

Status reflects checks actually run in this workspace. `NOT RUN` means the
check is not verified; it is not a pass. This checklist is local and does not
claim a public deployment or production release.

| Area | Check | Status | Evidence/notes |
| --- | --- | --- | --- |
| Application | Backend formatter/linter and unit/contract suite | PASS | `uv run ruff format --check app tests`; `uv run ruff check app tests`; `uv run pytest -q` → 165 passed, 2 skipped. Local Qdrant test emitted its documented local-mode payload-index warning. |
| Application | Backend type checker | NOT APPLICABLE | No type-checker is configured in `backend/pyproject.toml`. |
| Application | Frontend tests, TypeScript build, production bundle | PASS | `npm test` → 8 files/17 tests passed, including root disable interaction; `npm run build` passed TypeScript and Vite production build. |
| Application | API request/response contracts and loopback restriction | PASS | Backend tests cover request validation, scoped routes, local-only behavior, and trusted-proxy spoof rejection; no live API probe. |
| Application | Search, hybrid, RAG, citations, file details | NOT RUN | Requires live PostgreSQL/Qdrant and, for generated answers, an Ollama model. |
| Application | Recommendations, feedback, personalization | PASS | Deterministic backend service/API contract tests passed; not a live database workflow. |
| Infrastructure | Compose YAML syntax | PASS | Ruby YAML parser loaded `docker-compose.yml` and found all eight services. |
| Infrastructure | Compose rendered configuration | NOT RUN | Docker Compose v2 is not installed in this workspace. |
| Infrastructure | Clean container startup and health checks | NOT RUN | Docker CLI exists, but `docker compose` is unavailable. |
| Infrastructure | Celery worker and beat startup | NOT RUN | Requires Redis, PostgreSQL, and runnable containers. |
| Infrastructure | Alembic clean upgrade and downgrade | NOT RUN | No PostgreSQL service/client is available in this workspace. |
| Infrastructure | Persistent data across restart | NOT RUN | Requires a live Compose deployment. |
| Models | Ollama generation connectivity/model presence | NOT RUN | Ollama CLI is present on host; configured model availability was not checked. |
| Models | Qwen embedding/reranker inference | NOT RUN | Optional model packages/weights are not installed. |
| Filesystem | Explicit-root limits, symlink and permission checks | PASS | Backend discovery, permission, path, and symlink unit tests passed; container mount flow not exercised. |
| Filesystem | Synthetic demo fixture integrity | PASS | PDF/PNG/WAV/MP4 signatures checked; XLSX opened with openpyxl; media streams probed with FFprobe. Parser extraction itself remains model-dependent and not run. |
| Security | No tracked `.env`, credentials, model cache, or user library | PASS | Tracked-file review found only `.env.example` templates; added demo assets are synthetic. Repeat check on staged files before future pushes. |
| Evaluation | Dataset schema and deterministic report generation | PASS | `uv run python -m evaluation.cli --mode all --output-dir evaluation/results` wrote JSON/Markdown/CSV for `scoutx-small@1.0.0`; offline ranking baseline only. |
| Evaluation | Live retrieval/model benchmark | NOT RUN | Requires captured live artifacts; offline baseline is not production search. |
| Observability | Structured logs and health/liveness response | NOT RUN | Live service probes/log inspection need a running stack. |
| Documentation | README and installation/configuration/deployment/demo guides | PASS | Added guides linked from the root README; recheck commands after final edits. |
| Git | Dedicated branch, focused commits, no unrelated files | PASS | Phase 12 work is committed on `feature/phase-12-final-release`; staged-file review found only this phase's implementation, tests, docs, and synthetic fixtures. |
| Git | Feature branch published to origin | PASS | `origin/feature/phase-12-final-release` was created and tracks the local finalization branch. |
| Git | Merged into stable integration branch/main | NOT RUN | This phase leaves integration to the maintainer; `main` remains unchanged. Exact merge commands are in the final report. |
