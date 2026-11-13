# Three-arm architecture evaluation

This directory connects the context-v4 memory architecture to separately frozen evaluations. The current comparison has exactly three conditions: **no RAG, raw-history RAG and managed-memory RAG**. Every condition uses the complete shared [CBT counselor policy](../../docs/context-assembly-v4/counselor-policy.md), the same response model and the same within-study source and request-budget rules.

## Completed LoCoMo result

The original **1,540-question** LoCoMo comparison and all **5,078** planned reference-grading entries are complete.

| Condition | Primary: source-family-equal accuracy | Secondary: question-micro accuracy |
| --- | ---: | ---: |
| No RAG | 2.56% | 2.34% |
| Raw-history RAG | 41.87% | 41.30% |
| **Managed-memory RAG** | **55.75%** | **55.26%** |

The primary managed-minus-raw gain is **13.88 percentage points** (95% family-cluster interval **10.15–17.32**). Managed memory scored higher in all four included question categories. Four historical technical missing answers remain in the fixed denominator; no grades are pending in this endpoint. Repeated-label agreement was **99.35%** on 462 valid pairs.

[Methods, category breakdown and coverage](results/locomo1540_completed_20261113.md) · [Exact aggregates](results/locomo1540_completed_20261113.json)

These are independent adapted Qwen reference scores for DeepSeek answers, not official benchmark scores, human ratings or CBT clinical outcomes.

## Comparison and context contract

| Condition | Shared recent context | Older memory supplied to the generator |
| --- | --- | --- |
| `no_rag` | Same admitted recent window | No retrieved older-memory block content |
| `raw_history_rag` | Same admitted recent window | Retrieved original source statements |
| `managed_memory_rag` | Same admitted recent window | Source-supported processed reports and necessary evidence/version dependencies |

The primary comparison tests the full managed pipeline against capable raw-history retrieval. No extra representation-ablation condition is included. The role policy, model, sampling, source cutoff and complete outbound ceiling stay fixed within an experiment.

The shared [renderer](../../modules/agent-integration/cbt_context.py) constructs system policy, admitted recent conversation, a user-role sourced-memory reference block, then the current question. The generator sees this actual per-turn selection, not the database contents or external audit trace. [Implementation paths and full fictional request](../../docs/context-assembly-v4/IMPLEMENTATION_PATHS.md) distinguish the current processed-memory path from older quote-only adapters.

For general-memory benchmarks, the common policy is followed by an identical factual-recall instruction in all arms. It asks for a direct factual answer rather than a CBT exercise. Processing preserves the supplied speaker/time scope and explicit corrections; it does not infer present-day status from an old report.

## Published result scope

| Evaluation | Question roster | Scope |
| --- | ---: | --- |
| LoCoMo | 1,540 | Completed factual-answer comparison above, with 10 source conversation families |
| MemoryAgentBench | 800 | Conflict-task accuracy under its recorded experiment protocol |

The CBT-specific fact, continuity and R1–R5 protocols remain separate from the completed general-memory result. This update publishes no new CBT-specific scores or execution-status snapshots.

The [MAB 800-question conflict-task result](../objective-memory/results/full-20261105/README.md) includes its own frozen methods and implementation references. MAB data and scores are preserved. Controlled-development fixtures remain software/development evidence rather than additional independent benchmark results.

## Reproducibility entry points

| Module | Responsibility |
| --- | --- |
| [`formal_sources.py`](formal_sources.py) | Original released source revisions, rosters and reference preparation |
| [`formal_full.py`](formal_full.py) | Question-independent supported memory preparation, matched retrieval and real message assembly |
| [`formal_continuation.py`](formal_continuation.py) | Receipt-bound continuation without relabelling prior outputs |
| [`formal_judge.py`](formal_judge.py) | Frozen adapted independent reference scoring and complete-denominator summaries |
| [CBT architecture retest](../cbt-memory-effect/architecture_retest/README.md) | Domain-specific fact/continuity evaluation |
| [`quality_span_judge_v3.py`](quality_span_judge_v3.py) | Separately versioned R1–R5 automatic assessment |
| [Migration matrix](../ARCHITECTURE_MIGRATION_20261113.md) | Which artifacts can be reused after architecture changes |

Models, embeddings, full datasets and paid requests run only in genuine hosted Colab. Source-only preparation is distinct from paid execution. Existing runtime modules, checkpoints, receipts and in-flight requests must be checked before continuation; a file on disk alone does not identify the code objects already loaded by a kernel. Isolated credential-free child workers preserve selected source versions while approved provider transports remain in the parent. Call caps are ceilings, not new permission or automatic-retry instructions.

Original protocols, costs, failures and outputs remain retained. The [earlier architecture execution guide](https://github.com/nanlalala/cbt-memory-system/blob/ea21751e17066aa768361313c08695251a31af0c/evaluation/architecture-rerun/README.md) records its historical entry points; it is not an instruction to rerun preserved MAB data or overwrite existing runs. The [evaluation index](../README.md) separates completed findings, endpoint methods and historical records.
