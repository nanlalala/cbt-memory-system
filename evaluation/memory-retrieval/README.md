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

The complete v1 comparison on all 1,986 LoCoMo questions produced:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.241 | 0.449 | 0.528 | 0.359 |
| Multilingual E5 | 0.255 | 0.459 | 0.543 | 0.381 |
| TF-IDF + E5 Hybrid | **0.280** | **0.520** | **0.598** | **0.415** |

The hybrid method improved Recall@5 and Recall@10 by about 0.071 absolute and MRR@10 by 0.056 over TF-IDF. Four questions without evidence annotations are excluded from evidence metrics, leaving 1,982 scored questions. Neural runs used a Google Colab Tesla T4.

These are retrieval-only results. They show that the hybrid retriever finds the annotated historical evidence more reliably, but do not yet establish an improvement in final Agent dialogue quality. That comparison belongs to the Week 4 end-to-end evaluation.

Evaluate whether the correct user memories are retrieved across sessions.

Core measures include Recall@K, temporal/latest-state accuracy, correction compliance, contradiction rate, superseded-memory leakage, provenance accuracy, empty-result accuracy and sensitivity-policy compliance.
