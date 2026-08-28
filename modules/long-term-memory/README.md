# Long-term memory RAG

This is the core research module of the CBT Memory System. It owns both the **write path** and the **retrieval path** for user-specific memory.

Unlike the relatively stable CBT knowledge corpus, long-term memory is dynamic, user-scoped, time-sensitive and correctable. The module must preserve provenance and must never treat model inferences as confirmed user facts.

## Responsibilities

1. extract candidate memories from the current session;
2. validate, classify and store accepted memories;
3. retrieve memories relevant to the current task;
4. rank by semantic relevance, recency, importance, confidence, validity and sensitivity;
5. consolidate duplicates while preserving provenance;
6. update, supersede, correct, expire or delete memories;
7. expose user controls for inspection and correction.

## Submodules

- `schema/`: typed memory records and lifecycle metadata;
- `extraction/`: conversation-to-memory candidate extraction;
- `storage/`: user-scoped persistence and indexes;
- `retrieval/`: query building, candidate search, reranking and relevance gating;
- `lifecycle/`: consolidation, conflict resolution, correction and forgetting.

## Retrieval contract

The retriever may return zero memories. A memory can enter the response context only when it is relevant, currently valid and permitted for the active task. Superseded or deleted memories must never be injected.

Each returned item should include:

- memory identifier and type;
- content or a safe summary;
- provenance and timestamps;
- confidence and user-confirmation state;
- lifecycle status;
- retrieval score and score components;
- sensitivity and access decision.
