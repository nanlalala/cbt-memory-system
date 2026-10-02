# Memory extraction and retrieval evaluation

`memory_extraction_test_set.jsonl` contains ten synthetic, multi-session cases for Memory Schema v1. Each case specifies the source turns, expected memory types, information that must not be stored and one or more future retrieval checks.

The initial set tests:

- goals, CBT assignments and outcomes;
- explicit corrections and latest-state handling;
- emotion change across sessions;
- unconfirmed cognitive-pattern inference;
- exclusion of transient small talk and unsupported diagnosis;
- restricted handling of safety context;
- expiry and user-requested deletion.

These cases are schema-level fixtures, not clinical-quality ratings. Week 2 will add extraction and lifecycle predictions; Week 3 will add retrieval-ranking metrics.

Evaluate whether the correct user memories are retrieved across sessions.

Core measures include Recall@K, temporal/latest-state accuracy, correction compliance, contradiction rate, superseded-memory leakage, provenance accuracy, empty-result accuracy and sensitivity-policy compliance.
