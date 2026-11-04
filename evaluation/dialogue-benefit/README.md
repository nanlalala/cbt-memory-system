# Dialogue benefit and CBT process evaluation

This directory implements collection, independent review and analysis for the
[four-visit nonclinical protocol](../../docs/research-validation-protocol.md).
It does not start a participant service, make model calls, recruit people or
establish clinical effectiveness. The
[evidence checkpoint](../../docs/evidence-status-20261105.md) and
[machine-readable registry](evidence_registry.json) preserve the actual findings:
no participant observations, independent CBT ratings or clinical outcomes yet.

## What is measured

One person stays in one randomized memory condition across four visits. The
primary contrast is managed memory versus competent raw-history RAG; short-term
memory is a secondary comparator. The fixed feasibility design has 60 planned
slots, not 60 enrolled people. Four visits and 12 follow-up tasks per person do
not turn into 720 independent participants.

The [three-variant task draft](task_materials_draft.json) and
[four-visit cards](../../docs/four-visit-task-cards.md) contain 36 planned tasks
and 48 source turns. Exact text, dates, roles, visible-history boundaries and
scoring references are verifiable; independent human validation is still zero.
This combined scoring ledger contains gold and forbidden/deleted reference text:
**never send it to a model**. A production source-only generation adapter, full
stage-trace connection and independently annotated check opportunities remain
unimplemented launch requirements. The collector is not that adapter.

Two co-primary operational outcomes are independently source-adjudicated task
success out of 12 and failure-sensitive restricted task time. Missing/rescued
visits receive a prespecified 900-second penalty; that is not observed time.
Report task failures, actual observed time, natural checks, repeated checks,
edits, facilitator assistance and all model attempts alongside the endpoints.
Completed answers awaiting independent assessment remain unresolved.

CBT review has seven separate dimensions: source fidelity, current goal,
homework review, guided discovery, collaborative action, progress integration,
and scope/safety. Two qualified independent humans score responses before any
adjudication. `NA` and `UNRESOLVED` are distinct from 0/1/2. No validated CBT score,
therapist-equivalence grade or overall clinical total is computed. The
[CBT evidence note](../../docs/cbt-process-outcome-evidence.md) explains why
appropriate process, actual skill learning, symptoms and functioning require
different observations.

## Prepare and inspect an empty collection

Use a new private directory in the independent cloud. Preparation alone allocates
slots and freezes identities; it enrolls nobody and does not authorize collection.

```bash
python evaluation/dialogue-benefit/study_capture.py prepare \
  --study-dir /absolute/cloud/path/new-study
python evaluation/dialogue-benefit/study_capture.py status \
  --study-dir /absolute/cloud/path/new-study
python evaluation/dialogue-benefit/study_analysis.py analyze \
  --study-dir /absolute/cloud/path/new-study \
  --output-dir /absolute/cloud/path/empty-study-audit
```

An empty/open study must report zero people and no comparative intervals. Actual
enrollment requires the approved launch record, exact independently reviewed task
materials and arm-fidelity/privacy/consent gates in `protocol.json`. Small offline
verification uses explicitly frozen `--mode fixture`; it cannot become real
participant evidence. Full datasets, models, embeddings and paid calls remain
in genuine hosted Colab under the existing guard.

Capture accepts structured, validated events through `record --event-json PATH`;
`enroll` and `withdraw` are explicit aliases with event-type checks. Free text
belongs only in restricted fictional task-response artifacts, not the event
journal. Consent withdrawal can append a separate revocation after collection
is sealed, erase local response artifacts and invalidate stale analysis/packets.
Previously distributed copies and provider logs require the approved human/data
controller process; the tool does not claim remote erasure.

After lawful collection, `seal` freezes the event snapshot without pretending
missing tasks completed. `study_analysis.py export-packets` creates controlled
single-answer forms and a separate restricted allocation key. It requires the
actual frozen scoring criterion and source ledger, not an arbitrary key hash.
Keep gold references out of generation. Review import binds labels to source,
response, task and snapshot identities. Modified or revoked packets fail closed.
Use `--help` for exact command arguments. Do not publish raw human-task artifacts,
allocation keys or credentials in GitHub or the public benchmark exporter.

The analyzer retains randomized-arm flow and missingness, participant-level
metrics, cost completeness and independent pre-adjudication review records.
Person-level bootstrap intervals are exploratory, with two 97.5% co-primary
intervals under the prespecified Bonferroni rule. No inference is available from
fixtures, zero people, open collection or pending source assessments. The sample
is for feasibility, not a powered clinical-benefit claim.

## Existing CBT answers available for professional review

The archive exporter uses **32 actual old replies from 16 fictional scenarios**:

```bash
python evaluation/dialogue-benefit/prepare_legacy_cbt_review.py \
  --output /absolute/cloud/path/new-archived-review
```

It reads the literal case bank without importing legacy model/API code, checks
complete case/arm coverage and preserves every reply byte. Two differently
ordered reviewer folders contain HTML/JSONL replies, blank CSV ratings and an
unfilled qualification attestation. The restricted condition key stays separate.
Do not send a reviewer the whole output directory or prior model-judge scores.
Original style/citation clues remain, so reviewers record possible unmasking.

These are already exposed **knowledge-RAG development outputs**, not a new
memory experiment. Model identity and retrieved reference context cannot be fully
verified from that archive. No professional ratings have been received. The
exporter does not fill scores or infer clinical benefit.
