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
