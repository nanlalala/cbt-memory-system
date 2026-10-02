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

## Week 3 retrieval runner

`run_memory_retrieval_benchmark.py` converts LongMemEval sessions or LoCoMo dialogue turns into retrieval documents and scores predictions against the released evidence IDs. It supports lexical, multilingual E5 and hybrid backends. The accompanying Colab notebook runs the neural backends on a GPU without a paid generation API.

The first reproducible CPU baseline on all 1,986 LoCoMo questions achieved:

| Metric | TF-IDF baseline |
|---|---:|
| Evidence Recall@1 | 0.241 |
| Evidence Recall@5 | 0.449 |
| Evidence Recall@10 | 0.528 |
| MRR@10 | 0.359 |

Four questions without evidence annotations are excluded from evidence metrics, leaving 1,982 scored questions. This is a retrieval baseline, not an end-to-end dialogue-quality result.

Evaluate whether the correct user memories are retrieved across sessions.

Core measures include Recall@K, temporal/latest-state accuracy, correction compliance, contradiction rate, superseded-memory leakage, provenance accuracy, empty-result accuracy and sensitivity-policy compliance.
