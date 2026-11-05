# Full objective memory comparison: 2026-11-05

This folder records the completed, frozen within-v3 comparison on 800 MemoryAgentBench conflict cases and 589 PersonaMem-v1 32k cases. All model calls and full-source scoring ran in genuine hosted Colab. The reporting step used only controlled derived metadata.

- Executable: `1511706b3a4b0ac71416998b5b4d752a13c4e313`.
- Suite plan: `940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568`.
- Retained census: 1389 cases × 3 arms = 4167 terminal arm rows. Technical failures remain in every denominator.
- Primary contrast: managed-memory RAG minus raw-history RAG. Model, sources, question, access policy and character budget are shared within each dataset. A character budget is not an exact token budget.

| Dataset | Arm | Official correct / all | Micro accuracy | Equal-group accuracy | Exact-text/option diagnostic | Strict-format diagnostic | Technical failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| memoryagentbench_conflict | Short-term | 10/800 | 1.25% | 1.25% | 1.25% | 1.25% | 0 |
| memoryagentbench_conflict | Raw-history RAG | 419/800 | 52.38% | 52.38% | 52.25% | 52.25% | 0 |
| memoryagentbench_conflict | Managed-memory RAG | 480/800 | 60.00% | 60.00% | 59.75% | 59.75% | 0 |
| personamem_v1 | Short-term | 364/589 | 61.80% | 62.24% | 61.63% | 61.63% | 0 |
| personamem_v1 | Raw-history RAG | 416/589 | 70.63% | 71.18% | 70.63% | 70.63% | 0 |
| personamem_v1 | Managed-memory RAG | 416/589 | 70.63% | 71.18% | 70.63% | 70.63% | 0 |

| Dataset | Managed − raw micro | Managed − raw equal-group | Win / tie / loss | Identical-input shared answers | Inference boundary |
| --- | ---: | ---: | --- | ---: | --- |
| memoryagentbench_conflict | +7.62 pp | +7.62 pp | 68 / 725 / 7 | 22 | not_estimable_fewer_than_two_groups |
| personamem_v1 | +0.00 pp | +0.00 pp | 0 / 589 / 0 | 589 | blocked_by_identical_inputs_and_shared_responses |

## Interpretation

- MemoryAgentBench variants are conservatively one fact-consolidation family, not 800 independent people. There is no estimable independent-family uncertainty interval. The managed representation is a deterministic source-order version-slot view, not a general natural-language semantic extractor or a CBT state model.
- PersonaMem has 20 connected persona groups. Its two RAG arms are lossless quotation controls with identical inputs and shared canonical responses. Their zero difference is by construction; it is not evidence of treatment efficacy, statistical equivalence or effective semantic/version management.
- Released official substring/option scoring is primary. Exact-text/option and strict-format diagnostics are descriptive literal rules, not independent semantic gold. Official substring positives may include refusal or contradictory text.
- Source parse, retrieval and clipping traces are observations. Content errors retain `answer_incorrect_stage_unresolved`; no independently annotated causal stage is invented.
- Development-exposed questions and question-held-out cases retain the frozen partition. Held-out questions still share histories; these are not independent unseen users or histories.
- Earlier pilots remain separate negative or insufficient evidence. Changed prompts, budgets and cohorts prevent a causal old-to-new improvement claim.

## Actual cost and missingness

- Recorded provider attempt intents: 3550; successful unique canonical calls: 3550; failed attempts: 0; unresolved intents: 0.
- Reported tokens: prompt 8768813, completion 8270, total 8777083. Missing fields remain unknown; no failure is assumed free.
- Costs count actual canonical attempts, never duplicated logical-arm usage. Cross-shard requests are not deduplicated. API latency excludes preparation/retrieval, some failure time and parallel wall-time effects. No monetary total is claimed without verified prices.

## CBT and user-benefit boundary

Independent CBT professional ratings: 0. Actual four-visit participants: 0. Clinical symptom/function outcomes: 0. Natural-check and repeated-question burden: not measured. These public tasks do not establish CBT goal progress, homework completion, natural correction quality, deletion handling, user burden or clinical effectiveness.

## Files and controlled material

- `aggregate_evidence.json`: recomputed conditions, primary contrast, exposure partitions, uncertainty boundary, actual call/token costs and source hashes.
- `per_question_metadata.jsonl`: all terminal case/arm scores and failure states, with only explicit metadata fields.
- `SHA256SUMS.json`: integrity hashes of the public output files.

Question text, histories, options, released reference answers, original model replies, private source traces and credentials are excluded from this public folder. Full answers and the independently recomputable controlled audit remain separate cloud artifacts. Hashes establish file identity, not independent authenticity.
