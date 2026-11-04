# Long-term memory retrieval

Retrieve user-specific memories for the current conversation.

## Implemented v1 pipeline

1. build a task-focused memory query from the current conversation;
2. apply user, lifecycle and sensitivity filters;
3. retrieve semantic and structured candidates;
4. rerank using relevance, recency, importance, confidence and validity;
5. apply relevance gating and allow an empty result;
6. return a small, provenance-aware memory context.

Knowledge-RAG scores and memory-retrieval scores must remain separate. Professional-source authority matters for CBT knowledge; temporal validity, user correction and confirmation matter for personal memory.

The dialogue application's conversational mode applies `memory_use_policy.py`
after relevance retrieval. Audit-verified quotations can be used as dated
self-reports without per-item approval; changing, conflicting or stale information
may instead need a relevant conversational check. Unsupported content is excluded.
Evidence quality, freshness and currentness are shown separately; model confidence
is not a probability or a permission to read candidates. The default retriever
continues to admit active records only. Candidate-history access is an explicitly
scoped service policy, never a global relaxation of lifecycle or privacy filters.

`memory_retriever.py` implements the access filters, a local TF-IDF baseline, an optional multilingual E5 backend, transparent score components, temporal reranking and relevance gating. Recency and importance are applied only after an item passes the semantic relevance gate.

The retriever is deliberately allowed to return zero memories. The E5 adapter loads `sentence-transformers` lazily, keeping the core manager usable without PyTorch.
