# ScoutX synthetic release brief

This fictional project note is part of the public ScoutX demonstration corpus.

## Product goals

ScoutX searches files from explicitly permitted folders. PostgreSQL stores file
metadata and extracted text; Qdrant stores vector points that refer back to
PostgreSQL chunks. The original files remain in the user's folder.

## Local processing

The filesystem scanner records file identity and content hashes. Celery workers
extract text and source structure, split extracted content into attributed
chunks, and create embeddings. The web API and worker use Redis for task
dispatch. A root is never mounted unless the operator adds that path to the
Compose configuration.

## Safety

The local demo binds application ports to loopback. Search results display safe
relative paths. The sample corpus contains no real people, credentials, or user
documents.
