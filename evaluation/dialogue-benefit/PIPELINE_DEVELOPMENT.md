# Fictional CBT follow-up: source-candidate pipeline development

This is a new, separately frozen development comparison. It is not the human
four-visit study, a clinical intervention, or an independent professional review.
The original objective run, its 1,389 questions and all prior negative findings
stay unchanged. No model-effect result exists merely because these programs pass
their software checks.

## Concrete follow-up questions

The existing researcher-authored material has 12 task types in three parameter
variants: initial goal and practice, partial homework, thoughts versus observed
events, an unaccepted assistant suggestion, a corrected goal, successive practice
versions, prediction versus one observed outcome, unresolved changes, deletion,
unrelated permitted memory, temporary information and a new-session boundary.
There are 36 tasks, not 36 independent people. There are zero independent human
reference validators, participants, CBT professional ratings or clinical outcomes.
The short histories may fit the raw baseline; zero managed improvement is an
admissible finding. Budgets and histories must not be weakened to manufacture a gain.

## What actually differs

All arms receive the same task query, fixed response model, source/access policy,
6,000-character context ceiling and 1,024-token answer ceiling. The two retrieval
arms share the real MemoryRetriever policy and top-k 12. No professional knowledge
base is loaded in any arm. These are character limits, not exact tokenizer budgets.

The managed arm adds chronological natural-language extraction: each eligible
user source is processed before later sources or deletion. Real MemoryManager,
SQLite store, conflict/correction-candidate writes, source pooling, retrieval and
context assembly are exercised. Semantic summaries index memories; final context
renders dated original supporting quotes. All extracted interpretations remain
unconfirmed. No model output fabricates user consent or confirms its own summary.
Ambiguous changes are not silently made into current facts. Extra extraction is
measured preprocessing cost; total compute is not matched across arms.

The raw baseline retains all permitted original source quotes and uses the same
ranker and context budget. The short-term arm uses the same current-visit window.
Source-level pooling prevents many summaries of one turn consuming every slot.
Actual retrieved candidates, score components, selected/dropped IDs and source
spans are traced. Path observations do not independently establish causal answer
sufficiency or a unique stage responsible for an incorrect answer.

## Source, scoring and privacy separation

`prepare_generation_materials.py` validates original dates, hashes, prefixes,
roles, spans and controls, producing:

- `generation_materials.json`: per-task eligible source snapshots, no assessor
  criteria, variant parameters or reference answers.
- `replay_events.jsonl`: engine-only chronological events; never send the whole
  file to a model. It permits actual ingestion of an allowed source followed by
  deletion rather than treating a missing record as a successful deletion test.
- `scoring_references.jsonl`: separate assessor-only criteria and references.

Opaque deletion/do-not-store/new-session controls are supplied equally to all
arms. They remove actual quote/semantic records and related raw/session/cache
state. They do not test autonomous natural-language identification of a deletion
target. No expiry event occurs in these 36 tasks; expiry fixture checks are not
an executed expiry outcome. Historical controlled research journals are retained;
remote provider-log erasure is not claimed.

Deleted, do-not-store or otherwise excluded scoring references never enter a
later model-review request. Such task judgments remain `unresolved` in the full
denominator; local source/context observations are reported separately. Default
new reviewer packets also withhold those references. No forbidden reference is
fed back into generation, indexing, retrieval or a subsequent model assessment.

The common answer envelope retains a natural Chinese reply, supporting source
IDs and an optional source-bound clarification. Emitted check counts are model
behavior, not a validated measure of interruption, necessity or participant burden.
The common scripted source history is fixed; generated answers do not invent new
user replies or alter another arm's history. This is event replay with real model
extraction/answers, not a closed-loop user trial.

## Finite actual execution

Use a clean, exact commit in genuine hosted Colab. Keep all run/source/model work
there; do not spoof a Colab environment or connect a local runtime.

```bash
python evaluation/dialogue-benefit/prepare_generation_materials.py \
  --ledger evaluation/dialogue-benefit/task_materials_draft.json \
  --output-dir /content/NEW_SOURCE_PACKAGE
python evaluation/dialogue-benefit/run_followup_pipeline.py prepare \
  --source-dir /content/NEW_SOURCE_PACKAGE --run-dir /content/NEW_RUN
python evaluation/dialogue-benefit/run_followup_pipeline.py generate \
  --run-dir /content/NEW_RUN --max-new-calls 160 --workers 4
python evaluation/dialogue-benefit/run_followup_pipeline.py review \
  --run-dir /content/NEW_RUN --max-new-calls 160
```

Read the already authorized Colab Secret inside its notebook, pass the credential
only in a subprocess environment, and discard that environment afterward. Never
print it, put it in a command argument, export it, or copy it to another host.
The provider remains the fixed DeepSeek request/returned-alias policy. A returned
name is not an attestation of underlying model weights.

The new run freezes code, source-file hashes, model, prompts and budgets, with a
global maximum of 320 attempts and no automatic retry. Exact identical requests
share one canonical response or same-case model assessment; they are not charged
or counted as independent calls twice. Every intent is durable before transmission.
An interrupted intent without an outcome blocks further calls rather than sending
a possible duplicate. Supplied usage on failed responses is retained; missing
usage and currency costs stay unknown. Partial batches cannot produce comparative
results. The 108 planned arm rows remain in the denominator despite failures.

The model reviewer sees one masked answer and source-bound draft criteria. It
uses the same model family, not an independent professional. Quotation validation
checks identity, not semantic truth. These draft criterion checks and actual
mechanical controls may guide development; they establish neither independently
assessed CBT task success nor clinical benefit. A separate human study retains
its original launch, consent, facilitation and independent review requirements.
