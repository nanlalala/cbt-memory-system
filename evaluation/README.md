# Evaluation

The primary question is whether managed personal memory improves cross-session tracking over retrieval from the **same raw history**. Each versioned comparison retains its original question roster, model, role policy, source permissions and request budget. Memory control, factual answering, CBT response quality and clinical benefit are distinct endpoints.

## Completed evaluation highlight

The context-v4 LoCoMo evaluation completed all **1,540 questions × three conditions** and its planned independent reference-grading entries.

| Condition | Primary: source-family-equal accuracy | Secondary: question-micro accuracy |
| --- | ---: | ---: |
| No RAG | 2.56% | 2.34% |
| Raw-history RAG | 41.87% | 41.30% |
| **Managed-memory RAG** | **55.75%** | **55.26%** |

The primary managed-minus-raw gain is **13.88 percentage points** (95% family-cluster interval **10.15–17.32**). Managed memory scored higher in each of the four included question categories. The fixed denominator retains four technical missing answers as zero; repeat-label agreement was **99.35%** on 462 pairs.

[Methods, all category rates and denominators](architecture-rerun/results/locomo1540_completed_20261113.md) · [Exact aggregates](architecture-rerun/results/locomo1540_completed_20261113.json)

### MemoryAgentBench conflict resolution (800 questions)

| Condition | Accuracy | Token F1 |
| --- | ---: | ---: |
| No RAG (short-term only) | 10/800 (1.25%) | 1.3 |
| Raw-history RAG | 419/800 (52.38%) | 52.6 |
| **Managed-memory RAG** | **480/800 (60.00%)** | **60.1** |

Managed memory gained 68 questions and lost 7; multi-hop questions rose from 44 to 96 correct. [Full record](objective-memory/results/full-20261105/README.md)

### CBT dialogue pilot

On synthetic CACTUS-based CBT dialogues, a blinded pairwise judge preferred the managed-memory response for use of the client's history (net win **+0.42**, 95% interval 0.17–0.65). [Union run and judge](cbt-memory-effect/architecture_union/README.md)

Benchmark scores are adapted independent Qwen reference scores for DeepSeek-generated answers, not official benchmark scores or clinical outcomes. The homepage highlights completed findings; historical records are retained separately.

## Current experiment design

| Endpoint | What it tests | Interpretation |
| --- | --- | --- |
| General-memory factual answering | LoCoMo questions over the same admitted history; no RAG / raw-history RAG / managed memory | Completed result above; 10 source conversation families |
| CBT fact memory and continuity | Released CBT dialogue: necessary facts, problematic assertions, goal/homework follow-up | Separate domain-specific endpoint |
| CBT response quality | Independent blinded R1 factual context, R2 CBT method, R3 next-step suitability, R4 empathy and R5 safety | Dimensions reported separately, with emphasis on R1 and R3 |
| Memory lifecycle | Correction, supersession, pause, expiry, deletion, permissions and isolation in event replay | Software checks; not evidence of clinical benefit |
| Professional-knowledge retrieval | Recall/ranking, citation support and grounded professional evidence | Separate from user memory and from response effectiveness |
| User experience and clinical outcomes | Real experience, usability, symptoms and functioning | Not established by automatic benchmark scores |

All three current arms share the full [CBT counselor policy](../docs/context-assembly-v4/counselor-policy.md). The managed arm must send supported processed content through the actual [context renderer](../modules/agent-integration/cbt_context.py), rather than relying on a summary stored only in the database. Question-independent preparation, source cutoff and complete outbound-request budgets remain frozen within a run.

The primary comparator is capable raw-history RAG. A recent window is shared auxiliary context; it is not an additional experimental arm. The current comparison measures the full managed pipeline, with no additional representation-ablation arms.

## Protocols and preserved records

| Record | Purpose |
| --- | --- |
| [Architecture reruns](architecture-rerun/README.md) | Current implementation, original rosters and separately frozen executions |
| [CBT memory study](cbt-memory-effect/architecture_retest/README.md) | Domain-specific facts, continuity and source-bound automatic evaluation |
| [Managed-union CBT pilot](cbt-memory-effect/architecture_union/README.md) | Raw-superset managed arm, knowledge channel, blinded pairwise judging |
| [Published-label calibration](architecture-calibration/README.md) | Calibration only for the dimensions covered by released expert labels |
| [MAB 800-question comparison](objective-memory/results/full-20261105/README.md) | Conflict-task accuracy, methods and complete result record |
| [Objective-memory protocols](objective-memory/README.md) | Released-answer scoring and historical control comparisons |
| [Memory-control replay](controlled-memory-v2/README.md) | Versioned software evidence for lifecycle and access controls |
| [Shared dialogue pipeline](end-to-end-dialogue/README.md) | Preparation, provider and scoring code used by the current runs |

Existing results, failures and preparation costs are not overwritten to create a new score. Different dataset versions and scoring formats are not pooled into a preferred aggregate. Identical requests can share a source-bound receipt, while genuinely different requests require their own recorded execution. Automatic judgments are not described as human ratings or clinical efficacy.

Models, full datasets and paid runs execute only in genuine hosted Colab; small mocked checks verify software behavior only. No new human rating study is required by the current automatic-evaluation protocol.
