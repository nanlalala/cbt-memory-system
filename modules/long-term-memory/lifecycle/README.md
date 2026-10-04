# Memory lifecycle

Manage memory after extraction and storage.

Required operations:

- consolidate duplicates;
- preserve provenance across merges;
- detect contradictions;
- prefer confirmed and newer information without erasing history;
- mark old records as superseded rather than silently overwriting them;
- apply expiry or importance decay where appropriate;
- support user confirmation, correction and deletion;
- prevent deleted or superseded records from retrieval.

## Memory Manager v1

`memory_manager.py` now implements the deterministic write path:

- exact duplicate rejection;
- slot-based conflict detection with user confirmation;
- all model-extracted summaries held as candidates until separately confirmed;
- correction and superseding with audit events, with corrections restricted to the current active version;
- atomic correction, confirmation, expiry and deletion transitions;
- explicit confirmation evidence recorded through a consent event ID and optional confirmation turn IDs;
- time-based expiry;
- user-requested hard deletion;
- active-context filtering by user, validity and sensitivity.

Validity uses one shared half-open interval rule: `valid_from <= time < valid_to`. Slot reads, conflict decisions and writes execute inside one immediate SQLite transaction, preventing two workers from independently creating conflicting active versions. Semantic duplicate detection and retrieval ranking are handled by the Long-term Memory RAG stage. The v1 manager never silently resolves a semantic conflict.
