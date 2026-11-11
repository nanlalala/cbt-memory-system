# Natural CBT memory source annotation contract

This extends the existing `cactus_materials.py` material compiler. Do not create a
second natural-continuation runner or overwrite any old frozen package.

Existing inputs are `generation/cases.jsonl`,
`assessor_only/pending_annotations.jsonl` and
`controller_only/source_map.jsonl`. `draft_from_pending()` creates a new annotation
file joined by exact case ID/hash, preserving the pending source package. All
annotations and source-answer labels remain assessor-only. The original current
Client input remains the generation question; do not supply quotation prefixes,
four-word cues or future Counselor replies.

## Fields and evidence

- `memory_targets` identify permitted historical information required for this
  particular response. `required_memory`, `optional` and `context_only` prevent
  rewarding irrelevant fact dumping. Facts given in the current input are
  `current_input`, not historical-memory coverage.
- `truth_status` is `known`, `unknown` or `not_annotated`. Unknown is a reviewed
  absence of sufficient evidence in the released prefix; not annotated is
  unfinished work. Unknown values remain null and require a reason.
- Every supported fact has exact message ID, source-text SHA256, source order and
  Unicode start/end offsets. The retained controller source map connects parsed
  messages back to the released dialogue spans. Include the full assertion,
  negation, attribution and qualification needed to support the label.
- `expected_current_state` separates agreement, participation, progress, target,
  preference and outcome. A pause does not reset reported progress. A correction
  references its later basis facts and earlier superseded facts. Unknown benefit
  stays unknown; an assistant suggestion cannot establish user consent.
- `expected_response_policy` identifies unsupported claims on which to abstain
  and whether a relevant missing detail warrants clarification. It does not
  instruct the generator or force extra questions during ordinary conversation.

Keep the released ordinal cutoff and original roles. Three compilation cutoffs
are points inside a dialogue, not treatment visits. Preserve explicit released
session IDs if a future dataset provides them; otherwise session boundaries stay
unavailable. Do not derive dates, weeks, elapsed durations or treatment outcomes
from source order. Corrections, pauses and deletion cases invented for mechanism
tests remain separately labelled fixtures.

## Human review workflow

1. Freeze source-family membership, source hashes, original history boundary and
   the task question before annotation. Reserve entire families for holdout.
2. A source annotator drafts necessary facts, evidence spans, current-state axes,
   unknown targets and response alternatives while viewing only the released
   prefix. A model may propose a draft only with `label_origin=model_proposal`;
   this is not independent truth.
3. Two independent humans review sources, necessity, speaker attribution,
   corrections and abstention separately, without seeing memory-system arms or
   generated answers. Disagreement enters `adjudication_required`; an independent
   third reviewer resolves it with a retained review record.
4. An external verified-human registry and record references are required before
   `independently_reviewed` / `semantic_gold_available=true`. Structural validation
   cannot certify that a person reviewed the evidence or that the semantic label
   is correct. All bundled examples are invented, unreviewed fixtures.
5. Only reviewed labels enter scoring. Pending/unresolved items retain their
   planned identity and are reported separately, never silently counted correct
   or dropped. Keep the approved annotation version/hash for every score.

The repository already provides the broader blinded CBT-quality workflow in
`evaluation/dialogue-benefit/study_analysis.py`, including qualified independent
human attestations and third-rater adjudication. Reuse it for CBT response
appropriateness; source-fact validation does not replace professional R1–R5
review or participant experience measurement.

## What subsequent measurements verify

| Measure | What it verifies |
| --- | --- |
| Required known historical facts correctly used | Whether more necessary information is remembered and used |
| Unsupported, contradicted, stale or misattributed output claims | Whether the system remembers fewer things incorrectly |
| Latest state and correction adherence by axis | Whether an earlier state or suggestion displaces a later user report |
| Correct abstention on independently labelled unknowns | Whether greater coverage is achieved without guessing |
| Unnecessary clarification and repeated-question counts | Whether information already supplied reduces conversational burden |
| Five separate blinded CBT dimensions | Whether memory improves response quality, particularly factual context and next step |

Use the same history, recent window, response model, permissions, complete-request
budget and clarification policy across raw and managed arms. Keep retrieval,
packing, reading, state and review failures distinct. Measure participant time or
experience only through actual user testing.

## Files and tiny fixture validation

`schema.json` is the closed annotation format; `validate.py` checks source/hash,
cutoff, roles, chronology and review-status cross constraints. `fixtures.jsonl`
contains three invented cases: progress plus pause, explicit correction, and a
proposal without user agreement. None is real CACTUS material or independent gold.

```bash
python -m unittest discover -s tests -p test_natural_cbt_annotation_contract.py -v
```

For real cases, annotation preparation and validation belong in genuine hosted
Colab under the existing source protocol. No model/provider calls are implemented
by this contract. Keep every formerly assessed family out of future untouched
holdout claims, as recorded in `RESEARCH_DECISIONS_20261111.md`.
