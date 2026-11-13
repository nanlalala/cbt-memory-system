# Non-MAB continuation, 13 November 2026

This continuation uses the original case rosters and three conditions: no RAG,
raw-history RAG and managed memory. DeepSeek generates answers; the independent
Qwen model evaluates them. The complete shared CBT counselor policy and the
original scoring rubrics remain in place. Historical MAB data and results are
outside this entry point.

## What is being tested

| Cohort | Fixed roster | Outcome |
| --- | ---: | --- |
| Professional CBT | 7 cases, 21 planned answers | Source-grounded memory and blind R1–R5 quality, especially factual fidelity and next-step appropriateness |
| CACTUS | 600 prefixes from 200 source families | Required-fact coverage, stale or contradicted facts, and blind R1–R5 quality |
| LoCoMo | 1,540 questions, 4,620 planned answers | Original reference-based answer scoring, with the original repeat-assessment policy |
| LongMemEval | 500 questions, 1,500 planned answers | Source and budget inspection only until its separately pending paid budget is resolved |

The five dimensions are R1 context facts, R2 CBT method, R3 next-step
appropriateness, R4 empathy and R5 safety. Automatic ratings are not human
experience or clinical outcomes. Existing expert-label calibration and its
limitations still apply; this continuation adds no human raters.

## Source-bound reuse

The recovered checkpoint contains 17 professional-CBT answers, 540 CACTUS
answers and 2,978 LoCoMo answers. Other received calls include extraction,
support checks and assessment; they must not be counted as generated answers.
An inherited response is reused only when its original protocol, provider
identity and actual serialized request match. Original receipts and source
files remain unchanged; new requests use independent journals and protocols.
Consequently this is an explicitly reported continuation with reused answers,
not an all-fresh generation run.

The opt-in CBT continuation binds candidates individually. An invalid quotation
or role binding rejects that candidate; valid candidates still undergo the
original semantic support check. Duplicate IDs or corrupt batch envelopes
remain failures. The old default binding API and the historical formal
benchmark rendering path are preserved. Professional-CBT quality grading
starts only after all 21 source-bound answer packets exist.

Two historical CACTUS unknown sends are post-answer memory adjudications.
They remain unavailable and are never resent; their existing targets and
answers are retained. Four historical LoCoMo unknown final answers remain
technical missing rows for the exact question and arm. All planned rows remain
in the reports. Any newly unknown send stops continuation without retry.

## Remaining request bounds

| Cohort | DeepSeek | Fact/assertion Qwen | Quality/reference Qwen |
| --- | ---: | ---: | ---: |
| Professional CBT | 8 | 4 | 24 |
| CACTUS | 2,111 | 2,128 | 1,980 |
| LoCoMo | 1,638 | — | at most 5,078, within the original 5,082 cap |
| LongMemEval, not started here | 39,232 | — | 1,650 |

These are upper bounds, not observed costs or new budget authorization. The
hosted controller reconciles original local paid intents against their frozen
envelopes before admitting paid work; inherited legacy receipts are shown
separately and are not charged as new sends. The separate professional quality
repair allowance is 24. LongMemEval's original plan explicitly deferred its paid
stage, so this entry performs no LongMemEval model calls.

## Hosted execution

`cloud_continue_non_mab.run(...)` requires genuine hosted Colab, an exact clean
published checkout, the original source directories and the existing
parent-owned transports. It starts a separate relay and credential-free
workers without reopening paused legacy jobs. The fixed default run directory
is reserved exclusively; repeating the entry reports that invocation instead
of launching another one. Progress, finite budgets, immutable imports, new
receipts and scientific summaries are retained for review.

Software tests and zero-call source audits do not establish effectiveness.
No new comparison score should be claimed before actual hosted execution and
valid assessment receipts are available.
