# Research-guided state-memory experiment — 12 November 2026

The immediate test asks whether implemented memory state management reduces
incorrect CBT action-plan answers relative to the same original-source RAG.
It uses a separately frozen open reader in genuine hosted Colab. It neither
changes the earlier DeepSeek experiment nor consumes a new paid API allowance.
The reader was loaded on CPU before any new answers were generated; loading
alone is not a completed experiment or a measured gain.

## Research translated into testable changes

| Primary source | Applicable idea | Implementation/test boundary |
| --- | --- | --- |
| Zep, §2.2.3 | Retain historical facts while selecting the version valid at the query time | Use the existing source-bound task projector and explicit dates/periods; do not invent dates or treat an LLM update as confirmed user state. |
| Mem0, §2.1–2.2 | Distinguish addition, update, removal and no change | Keep target corrections, cumulative reports, participation, agreement and outcome as separate axes, with original source dependencies. |
| SimpleMem, §2.3 | Retrieve according to query intent and metadata constraints | Resolve task/period/as-of scope and preserve relevant original evidence under the same context/source limits; ambiguity remains unknown. |
| Beck Institute CTRS-R and action-plan guidance | Review the previous action plan, updates, goals, agreement, attempts and barriers | Controlled tasks check whether the system recalls these reports correctly; they do not produce a clinician competency or treatment score. |

The existing raw source ranking and both arms' direct recent history are held
constant. The management treatment is the actual product's rule-verified state
projection, rather than a more favourable prompt title or a reference answer.
Ordinary RAG can reason from every retrieved quotation under the same response
instructions. Both arms receive the same nine-source opportunity, recent-six
buffer, complete request ceiling, model and output limit. Any specialist
knowledge channel must also be common to the two arms.

The software already separates outcome from completed-attempt counts and
agreement from a therapist suggestion. The new study must test those behaviours
with actual reader answers, not count passing unit tests as an effect. Paused or
withdrawn topics keep the normal control path; they are not reopened merely to
obtain a comparable model response.

## Separately identified reader and evidence

- Reader: `Qwen/Qwen2.5-1.5B-Instruct`, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, declared Apache-2.0.
- Execution: genuine hosted Colab, existing CPU backend, float32; no GPU credit
  switch or external inference endpoint. Model files and their hashes are
  retained under the new model-root protocol.
- Model download/loading: 9 files, 3,098,971,928 bytes; observed 44.3046 seconds.
  These are measured setup costs, not answer latency or accuracy.
- Hard cap: 64 actual local generations across the model-root journal; greedy
  generation, at most 4,096 input tokens and 64 new tokens. Identical canonical
  requests share one response, but each logical use rechecks current controls.
- Every attempt, failure, original answer, token count and duration must be
  retained. Unknown interrupted attempts are not automatically retried.

Controlled cases are designed mechanism examples, not sampled patients,
professional dialogue transcripts or independent clinical gold. Numeric
variants of one template stay in one family. Targets are constructed separately
from the source-template specification before answers, never copied from the
projector being evaluated. Report all cases, failures, ties and previous
versions. This development experiment can identify an actionable failure or a
narrow measured benefit; an independent final comparison still needs its own
frozen data and review.

## Parallel natural-dialogue preparation

### Frozen before/after comparison

Before the first reader answer, freeze the same 24 constructed cases and literal
targets for two product versions: `controlled-memory-v2.2` from parent commit
`bd28eb05f7f02213fca40ef0bfd4ced4bc598825`, and
`controlled-memory-v2.3-english-explicit-state`. The latter adds general English
correction inflections and binds explicit first-person cumulative reports across
an adjacent comma. It does not infer agreement or relax provenance controls.
The harness, shared prompt, raw retrieval, case bytes and targets are identical.

Prepare each in a fresh process, verify identical raw requests, and count the
union of canonical request payloads before generation. That union must fit the
existing 64-attempt global ceiling. Identical requests can reuse one original
answer after fresh source admission; the two version totals must not be added
as separate costs. Report all cases by arm, version, language and four template
families, including ties, losses, format failures and actual token/latency costs.
These exposed development cases are not an independent held-out benchmark.

The v1 pre-generation guard stopped with 72 distinct payloads and zero model
answers. Fresh product databases assigned different internal task UUIDs even
when their task contents were identical. V2 keeps the original v1 artifacts,
uses deterministic wire aliases for identifier metadata only, and preserves
actual UUIDs and version dependencies in the private preparation/gate records.
Values, quotations, offsets, dates and source IDs do not change. Both product
versions use this same amended harness; their new directories and hashes are
frozen before the first generation. The 64-attempt ceiling is unchanged.

MusPsy's official MIT-licensed synthetic dialogue files were frozen in Colab at
`eaae4df12808f4cbda97d8bc8b243b34cc69f613`: 1,199 files, 31,918,882 bytes;
manifest SHA256
`7a0c774524f43b30411f8d9520bf517037a28e346df2633c71259deb3dc9c67a`.
Only dialogue files were checked out as generation material. Post-session
memories, user profiles and future plans are excluded. Released separators have
not been certified as chronological visits; a boundary audit is required before
creating next-session test inputs. The material remains explicitly synthetic.

CBT-Bench's 156 professional exercises and published human preferences remain
separate quality-calibration material. Neither their reference answers nor a
whole-session treatment plan enters the controlled reader's memory. No new
human review is implied by either data preparation or this open-reader study.

## Primary references

1. Rasmussen et al. Zep: A Temporal Knowledge Graph Architecture for Agent
   Memory. 2025. https://arxiv.org/html/2501.13956v1
2. Chhikara et al. Mem0: Building Production-Ready AI Agents with Scalable
   Long-Term Memory. 2025. https://arxiv.org/html/2504.19413v1
3. Liu et al. SimpleMem: Efficient Lifelong Memory for LLM Agents. ICML 2026.
   https://proceedings.mlr.press/v306/liu26dx.html
4. Beck Institute. CTRS-R Scoring Rubric and CBT Checklist. 2022.
   https://beckinstitute.org/wp-content/uploads/2022/09/Beck-Institute-CTRS-R-Scoring-Rubric-and-CBT-Checklist.pdf
5. Beck Institute. Homework. https://beckinstitute.org/blog/homework/

The Beck document's CTRS-R 0–3 scale is not interchangeable with the older CTRS
0–6 scale, nor with this project's R1–R4 1–5 rubric. No score is generated here.
