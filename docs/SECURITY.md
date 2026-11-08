# Security and privacy

ScoutX currently targets a local, single-user workspace. `SCOUTX_API_LOCAL_ONLY`
is enforced by middleware and cannot be disabled until an authentication
provider exists. Bind the API/UI to loopback. Do not expose them through a
reverse proxy, tunnel, or shared network. Compose's frontend proxy forwards a
loopback client marker only from its fixed internal address, while the published
frontend port itself is bound to `127.0.0.1`.

Only register directories the user intends to index. Scanner and file APIs
enforce allowed-root boundaries, permissions, exclusions, and symlink checks.
Docker cannot read arbitrary host paths: add an explicit read-only volume mount
to both backend and worker and register its container-side path. The demo mount
contains synthetic data only. Compose runs backend, workers, and Nginx as
non-root users; do not run Celery or API processes as root.

PostgreSQL stores extracted text and metadata. Qdrant stores embeddings and
references; neither service contains original files. Redis carries task state
and results. Local model caches and volumes can contain derived data and should
be protected like local application data. Keep backups private.

Never commit `.env`, user libraries, generated model weights, database/Qdrant
volumes, private evaluation artifacts, or credentials. Frontend `VITE_`
variables are public. Do not place backend passwords or tokens there.

Parser time/size limits and archive checks reduce risk but are not an OS-level
sandbox. Document parsers and ML libraries run in worker processes. Index only
files the operator trusts. Race-free `openat` confinement and network user auth
are not implemented; this is not a multi-tenant or public deployment.
