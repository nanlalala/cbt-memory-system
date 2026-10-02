# Memory extraction and retrieval evaluation

`memory_extraction_test_set.jsonl` contains ten synthetic, multi-session **development fixtures** for Memory Schema v1. They verify project rules but are not a formal benchmark and will not be used to claim research effectiveness.

The initial set tests:

- goals, CBT assignments and outcomes;
- explicit corrections and latest-state handling;
- emotion change across sessions;
- unconfirmed cognitive-pattern inference;
- exclusion of transient small talk and unsupported diagnosis;
- restricted handling of safety context;
- expiry and user-requested deletion.

Formal evaluation uses the official LongMemEval and LoCoMo datasets, which provide externally released questions, answers and evidence annotations. See `BENCHMARK_PROTOCOL.md` and `benchmark_adapters.py`.

Evaluate whether the correct user memories are retrieved across sessions.

Core measures include Recall@K, temporal/latest-state accuracy, correction compliance, contradiction rate, superseded-memory leakage, provenance accuracy, empty-result accuracy and sensitivity-policy compliance.
