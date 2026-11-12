# Memory readout and context assembly

Status: experimental renderer and invented integration example. No new model-effect result. PR41 remains unchanged. The general semantic memory extractor, support verifier and hosted answer transport are not connected to this renderer yet.

## Three separate objects

1. Stored memory retains summary, subject, version, operational validity and source dependencies.
2. The dynamic block contains the admitted projection, epistemic status and source IDs; the evidence arm also includes exact source spans.
3. The external trace records rejected memories, missing representation coverage, source hashes, versions and actual request hash/length. It is never included in messages.

Actual order is fixed policy (`system`), recent original dialogue (`user`/`assistant`), reference block (`user`), and current question (`user`). The reference block is application-inserted data rather than a fabricated user utterance. No tool role or call ID is fabricated. Consecutive user messages require compatibility validation on the selected hosted model. Labels alone do not prevent injection or guarantee correctness.

The existing product evidence-bundle renderer places claims in a second system message. PR41 instead places raw sources in a JSON user message. This new opt-in path deliberately does neither silently: its message position and policy require a new frozen protocol. It does not rewrite historical experiments.

## Admission and packing

The current caller supplies a permission-checked registry snapshot and selected source IDs, then passes active, nondeleted, valid and nonsuperseded projections produced by a frozen verifier. The renderer rejects subject mismatches, future/ineligible sources, incomplete dependencies, mismatched spans and unverified interpretations. A literal span binding does not establish semantic entailment; the producer's support verifier remains a required interface, not an implemented general semantic guarantee.

For the isolated representation comparison, source selection is fixed and whole-request overflow fails preparation; no arm receives a different replacement source. The next full-pipeline study must freeze query-aware ranking, control-state priority, conflict closure and indivisible-item packing separately. A summary's dependencies must include the correction or pause that changes its interpretation. Ordinal sources receive no invented dates or weeks.

## Prespecified next comparison

Use the same selected sources, recent messages, answer model/settings and maximum complete request budget for raw, processed and processed-plus-evidence. Equal budget means equal cap, not padded equal length. Freeze case roster and development/held-out families before generating answers. Report representation coverage and preparation failures for every arm. Do not optimize against held-out reference answers.

Primary endpoints: source-supported factual accuracy, required-information coverage, stale/incorrect facts, and correct goal/homework/pause/update continuation. Independent automatic R1–R5 scoring is secondary, with particular attention to R1 and R3; calibrate applicable tasks using published expert labels and keep calibration results separate. Include AB/BA/repeat checks, source-family paired intervals, failures and full extraction/verification/answer/judge costs. No new human raters or clinical-effect claims.

Before sends: recover existing hosted journals, reconcile unknown intents, verify current permission, freeze code/input/model/request hashes and explicit authorized attempt ceilings. Never reuse old answers under changed request hashes. Provider token usage is unknown until received. Optional tokenizer counts must document tokenizer, chat template and scope; JSON character counts are not model tokens.

## Reproduce the invented example

The three `*.request.json` files are actual renderer outputs. `*.trace.json` files are external audit records. `invented_input.json` is a fictional thought-record correction, not a clinical case or natural CBT result. Run the renderer with those arguments and an explicit `arm`. The `model` field is a placeholder and is not an executed model identity.

Validation on 2026-11-13: 96 fixture regressions passed (91 inherited plus five new assembly tests). No real inference, model loading, Secret access or paid calls occurred during this development.

## Cloud recovery observation

The genuine hosted notebook was opened read-only. Its UI showed Connect rather than an active runtime. Saved output showed the independent retrospective judge completed 69 attempts (66 assessments, three technical failures); a separate state study's last saved monitor showed 31 published / 32 admitted and no final score. These are saved outputs, not current filesystem verification. Do not retry an unresolved generation solely from this snapshot.
