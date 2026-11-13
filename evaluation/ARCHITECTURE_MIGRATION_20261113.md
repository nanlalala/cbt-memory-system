# Evaluation migration for the revised memory architecture

Status on 2026-11-13: this migration specifies what must be rebuilt for the revised architecture. It does not declare a completed full-system evaluation. Existing frozen studies and failures remain unchanged.

## Version boundary

Three distinct context paths currently exist: the legacy `context_builder.build_context`, product `evidence_bundles.build_outbound_messages`, and PR41 source-only retrieval renderer. PR42 introduces a fourth experimental processed-memory renderer. None can be treated as an interchangeable implementation of the others. The latest branch is not automatically the deployed product.

Freeze one intended product revision and its exact extraction, state, retrieval, assembly, send and scoring adapters before the new end-to-end run. The new renderer currently consumes externally verified projections; it does not yet implement general semantic support verification, query-aware memory ranking or the live product send boundary.

## Migration matrix

| Existing evaluation | What remains useful | Required new work | Reuse boundary |
| --- | --- | --- | --- |
| memory-retrieval development extraction fixtures | Original synthetic histories and expected lifecycle behavior | Map current schema; add claim support, current/historical distinction and uncertainty checks | Engineering fixtures only; old pass counts do not validate the new schema |
| memory-retrieval external retrieval benchmark | Pinned datasets and released evidence annotations | Reindex new record types; score retrieval and post-budget source coverage separately | Old retrieval results remain results for the old retriever |
| end-to-end-dialogue frozen LoCoMo/LongMemEval v2 | Retained answers, journals, source identities and historical study design | New adapters and competent raw-history comparator; regenerate whenever actual request/model changes | Never append new architecture responses to frozen v2; identical requests alone are insufficient without full model/settings identity |
| objective-memory MemoryAgentBench | Public answers, source-family mapping and deterministic scoring rules after audit | Run current full pipeline and matched raw-history RAG; freeze held-out families separately from development | Old positive results are historical, not proof of PR42 or current product |
| objective-memory PersonaMem | Public task definitions may support secondary profiling diagnostics | Inspect actual requests and keep only contrasts that test the intended intervention | Identical raw/managed requests cannot establish management benefit; do not restore excluded claims |
| controlled-memory-v2 lifecycle replay | Event histories, permission/negation/deletion regressions | Rebind to new store and assembly path; test state changes in actual messages and final answers | Mock replay shows software behavior, not CBT response quality |
| cbt-memory-effect CACTUS source quotation recall | Original source IDs, literal quote scoring and frozen raw baseline | Preserve as a separate retrieval task; add a semantic continuation cohort with processed representations | Do not score an abstraction task only by verbatim quote reproduction |
| controlled care-tracking model experiments | Constructed histories and predeclared targets | Rebuild actual requests for new renderer; compare raw, processed and processed plus evidence, then full pipeline | Constructed cases remain development/mechanism evidence; no invented dates for ordinal corpora |
| cbt-response-review archived five-dimension ratings | Historical review receipts and original rubric | Regenerate answers for changed pipeline, then independently blind-score R1–R5 | Re-rating an old answer does not test a new memory architecture |
| auto_eval_v2 evidence and quality | Fact precision/coverage accounting, R1–R5 packet validation, repeat/position checks | Bind new responses to exact request/model receipts; support three representation contrasts plus the main pipeline contrast | No single combined quality number; R1 and R3 remain primary quality interests |
| auto_eval_v2 published-label calibration | Published expert labels and source manifest | Rerun if judge, rubric, prompt, preprocessing or exposed split changes | Stable judge calibration can be reused with exact identity; CTRS/safety calibration does not validate project-specific memory R1/R3 automatically |
| dialogue-benefit four-visit workflow | Fictional follow-up task structure and process measures | Convert appropriate items to automatic source-supported continuation tests; retain user-effort work as unperformed optional research | No new human recruitment or human scoring as completion conditions |
| knowledge-retrieval CBT references | Separate professional corpus and relevance/citation targets | Rerun only components affected by routing, packing, embedding or generation changes | Unchanged KB retrieval can remain a frozen component baseline; final answer citation support must be rechecked |
| safety and access controls | Existing policy cases and expected denial/deletion outcomes | End-to-end checks for cross-subject leakage, revocation between retrieval/send, stale summaries and instruction-bearing memories | Data-only labels and source IDs do not establish safety |

## New study sequence

1. Interface fidelity: assert real source events become processed projections, dynamic blocks and actual model messages. Verify exact roles, cutoff, versions, deletion propagation, omitted items, full request hashes and token accounting. Include recent real dialogue; the tiny mechanism pilot has no recent history and does not satisfy this gate alone.
2. Representation ablation: identical selected sources, recent context, model/settings and complete request ceiling. Compare raw, processed and processed plus evidence. Freeze cases and references before generation. Report preparation failures and represented-source coverage, without arm-specific source substitution.
3. Full pipeline: competent raw-history RAG versus the complete memory system on public CBT continuation histories. Evaluate automatic extraction/support verification, changes across turns, retrieval and context assembly together. Recent-window-only is auxiliary. Use ordinal timing honestly; test dated state mechanisms in a separately identified cohort.
4. Independent automatic assessment: factual precision, required-information recall, stale/contradicted facts, correct goals/homework/pause/update continuation; independent R1–R5 scores, calibrated on applicable published expert labels. Report R1/R3, safety, AB/BA consistency and repeated judgments separately.
5. Audit and report: all frozen rows and failures, source-family paired intervals, per-item outputs/error attribution, extraction/update/index/verification/generation/judge usage and latency. New results receive new directories and explicit architecture identities. Abstract, figures and conclusions must name the version actually tested.

## Reuse decision

Changed history, cutoff, memory representation, message roles/order, model, generation settings, tokenizer/chat template or assembled payload requires new generation. Stable source data and gold can be reused if the task remains valid and exposure is disclosed. Stable scoring code can be reused after binding validation. Stable historical outputs remain historical artifacts. A failed or unknown old intent is never silently resent. A positive effect is a research target, not an acceptance rule for deciding which rows to report.

## Current execution

A separate hosted-only pilot is limited to three invented development cases and nine Qwen2.5-1.5B-Instruct generations, with no paid model API or Secret access. Its purpose is model compatibility and representation reading; manually supplied projections prevent it from measuring automatic memory-management benefit. Natural CBT semantic extraction and the live product adapter remain work to complete before the main comparison.
