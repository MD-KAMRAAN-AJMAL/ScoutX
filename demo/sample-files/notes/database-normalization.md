# Database normalization notes

This synthetic note explains the example inventory schema.

## First normal form

Store one value in each cell and give each row a stable identity. A file record
contains one canonical path, one content hash, and one current version.

## Second normal form

When a table has a composite key, every non-key field should depend on the whole
key. In a project-file join, project-specific metadata belongs to the join
record, not to the file.

## Third normal form

Non-key fields should describe the key rather than another non-key field. Keep
root ownership on the root record and refer to it from each file.

## ScoutX example

PostgreSQL is the canonical store for file metadata and extracted chunks.
Qdrant stores vectors and payload references. A vector point is not the only
copy of extracted text.
