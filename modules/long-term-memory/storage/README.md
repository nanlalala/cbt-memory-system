# Memory storage

Persist accepted memories and retrieval indexes with strict user isolation.

The storage layer should support:

- structured records plus embeddings;
- provenance and version history;
- active, superseded, expired and deleted states;
- correction without destructive loss of audit history;
- hard deletion when requested;
- rebuildable local indexes that are excluded from Git.

Real user memories, credentials and production indexes must never be committed to this repository.
