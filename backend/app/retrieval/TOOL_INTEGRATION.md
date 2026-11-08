# Retrieval tool contract for Person B

The retrieval functions are ordinary synchronous Python callables. They do not
read LangGraph state, create a graph, or accept model-selected filesystem paths.
Tool wrappers should validate only model-visible arguments with the corresponding
`*Input` model exported by `app.retrieval`, take `user_id` only from authenticated
backend context, and open a fresh SQLAlchemy session for each invocation. The
input schemas intentionally reject a `user_id` field.

| Tool | Callable | Input model |
| --- | --- | --- |
| Exact filename | `search_exact_filename(session, user_id, filename, limit=20)` | `FilenameSearchInput(filename=...)` |
| Fuzzy filename | `search_fuzzy_filename(session, user_id, query, limit=20, minimum_similarity=0.2)` | `FilenameSearchInput(filename=...)` |
| File metadata | `search_metadata(session, user_id, ..., limit=100)` | `MetadataSearchInput` |
| Keyword/BM25 | `search_keyword(session, user_id, query, ..., limit=20)` | `QuerySearchInput` |
| Dense semantic | `search_semantic(session, user_id, query, ..., limit=20)` | `QuerySearchInput` |
| Hybrid | `search_hybrid(session, user_id, query, ..., limit=20)` | `HybridSearchInput` |
| Source code | `search_code(session, user_id, query, ..., limit=20)` | `CodeSearchInput` |
| OCR/transcripts | `search_media(session, user_id, query, ..., mode="hybrid", limit=20)` | `MediaSearchInput` |

Every search returns `list[SearchCandidate]` from `app.retrieval.common`. Fields
include PostgreSQL file/chunk UUIDs, filename and canonical path, excerpt,
relevance score, modality, source page/line/time values, persisted
`source_references`, and the retrieval method. Qdrant payloads are used to find
point IDs; PostgreSQL supplies canonical identity and content.

## Safety and limits

- The authenticated `user_id` is mandatory. Qdrant filters always include
  `user_id` and `is_active`; SQL retrieval statements include `File.user_id`.
- Optional facets only add filters. Callers cannot provide or replace arbitrary
  Qdrant filters. An explicitly empty file-ID scope matches no points.
- Candidates pass `require_readable_file` under the current process user's
  filesystem permissions. Inaccessible or missing files are omitted.
- Input models reject blank queries, unknown fields, malformed UUIDs, and limits
  outside 1–1000. Date bounds must be ordered; media modes/types are enumerated.
- Invalid inputs raise Pydantic `ValidationError` (or `ValueError` for invalid
  hybrid candidate limits). Backend failures raise `RetrievalBackendError`.
- BM25 and dense methods are separate callables. `search_hybrid` is already
  available if the supervisor chooses one fused tool; do not wrap it in a second
  retrieval orchestration layer unless the graph intentionally combines tools.

## Concurrency

Do not share a SQLAlchemy `Session` between concurrent tool calls. Give each
invocation its own session; the Qdrant client can be shared and the embedding
provider serializes access to its model. The synchronous callables can run in
worker threads or as independent LangGraph tool calls. Their return values are
plain Pydantic candidates and do not depend on graph state.

```python
from app.retrieval.common import QuerySearchInput
from app.retrieval.keyword import search_keyword

args = QuerySearchInput.model_validate({"query": query, "limit": 10})
with SessionLocal() as session:
    candidates = search_keyword(session, authenticated_user_id, args.query, limit=args.limit)
```
