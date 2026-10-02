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

## Reference implementation

`sqlite_store.py` provides a local SQLite implementation with user-scoped reads, indexed lifecycle/type fields and content-free audit events. Multi-record lifecycle changes and their audit events use one transaction, so a failed correction or confirmation cannot leave an intermediate state. User deletion removes the memory content rather than retaining a soft-deleted copy. The database file and any real memories must remain outside Git.
