# Controlled open-reader task-state comparison v2

This is a new, finite development experiment on invented dated CBT-homework
reports. It is separate from CACTUS, CBT-Bench, LongMemEval and every previous
paid-provider run. It tests whether the implemented product task projector makes
a fixed reader more accurate than original-source RAG when interpreting explicit
corrections, cumulative totals, proposals and reported outcomes.

It does not measure therapist response quality, clinical outcome, patient
experience or independent held-out generalization. Twenty-four bilingual cases
are variants of **four** related templates, not 24 independent patients.

## Fixed comparison

- Model identity is owned by `open_reader_colab.py`: pinned
  `Qwen/Qwen2.5-1.5B-Instruct`, CPU float32, greedy decoding, at most 64 new
  tokens and 4,096 input tokens. The shared model-root ledger allows at most 64
  admitted generation attempts across runs. Loading weights is not inference.
- The actual reader runs only in genuine hosted Colab. No paid provider or Secret
  is used. The preparation path uses the product's source-only bounded worker;
  there is no LLM extraction or inferred confirmed state.
- Raw uses the existing `StableSourceRetriever` lexical original-source RAG over
  the full currently eligible history, top nine distinct older sources. This is
  the named frozen comparator, not every possible dense/reranked RAG system.
- Managed shares that exact original-source ranking, and adds only actual
  source-bound task fields computed by the product rules/projector. Both arms
  use the same six direct recent messages, current question, system instruction,
  complete escaped 6,000-character request cap, and nine-source retrieval cap.
  Raw retrieval does not waste slots repeating those already-visible recent
  messages. Both arms may reason from explicit dated user reports.
- The common prompt explains correction, cumulative-count, consent and outcome
  distinctions. It asks for exactly `{"answer": ...}`; counts are integers,
  outcomes are lower-case words, and unavailable agreed quantities are null.
- State witnesses count against the common retrieved-source cap. All source
  dependencies, including current question and recent buffer, enter actual
  durable send/publication fences. No source, permission or task version may
  change between preparation, admission and publication.

## Material and gold boundary

`fixed_materials()` declares four families, three numerical variants and two
languages per family, yielding 24 cases and 48 arm rows. The literal expected
answers are construction labels separate from the generation cases; no expected
answer is obtained from the projector or included in model requests. Cases and
assessor-only targets are frozen before context preparation or generation.

The families are target correction, repeated cumulative reports, proposal versus
explicitly agreed quantity, and outcome versus progress. English and Chinese
sources express the same controlled facts. Source dates are explicit fictional
inputs, not fabricated calendar labels on an undated public dataset. Pauses,
refusals, deletion and risk routes are not forced through the answer generator;
the product's separate local-control gates remain in effect.

## Commands and API

```python
import controlled_open_memory as material
material.prepare('/content/NEW_PREPARED_DIRECTORY')
gates = material.make_gates('/content/NEW_PREPARED_DIRECTORY')
```

The gate object implements `identity()`, `preflight(row)`, `begin(row)` and
`finish(row, admission, outcome=..., response=None, usage=...)`. Its preflight
binds exact request bytes, source text/role/time, current permissions, withdrawal
generation and task versions. `begin` admits the actual durable attempt;
interrupted attempts cannot be resent automatically. `finish` uses the actual
publication fence and retains no extra response cache. Loaded product modules
from another checkout are rejected; use a fresh product import set.

After **all** original answer rows have been fixed, run:

```bash
python evaluation/cbt-memory-effect/controlled_open_memory.py score \
  --prepared-dir /content/NEW_PREPARED_DIRECTORY \
  --answers /content/NEW_READER_OUTPUT/answers_unscored.jsonl \
  --model-root /content/ACTUAL_PINNED_MODEL_ROOT \
  --output /content/NEW_SCORE_FILE.json
```

The assessor reconciles the completed answer archive with its reader protocol,
record checksums, shared generation ledger and canonical model-output caches
before reading the separate targets. A response with a rewritten local hash
cannot replace its canonical generation receipt. The scorer reports strict
per-question values, errors, paired wins/losses/ties, both arm denominators and
reader fixture status. Model output shape failures remain visible; they are not
silently converted into factual errors or dropped rows.

## Verification status before reader generation

Fourteen tiny invented-fixture tests passed, including shared recent context,
retrieval-source caps, actual source-backed correction, gold isolation, mutated
requests, withdrawal after admission, unknown-send blocking, replay of a settled
attempt, frozen receipt reconciliation and forged-answer rejection. A separate
all-case fixture pass prepared 48 requests, passed all 48 source preflights and
checked all 24 rule projections against construction labels after the English
source-grammar repair. Those checks made zero model calls and are software
validation; only the later genuine reader answers can establish this experiment's
observed mechanism result. Do not promote such a result to professional CBT
quality or general superiority over all RAG baselines.

## Stable wire identifiers

Version 2 addresses an execution-identity problem observed before any real reader
answers: fresh product databases assign random internal task UUIDs, which made
otherwise equivalent managed requests look different. A deterministic view
alias now replaces only task, bundle, claim and evidence-span identifier
metadata before request packing. The stable key includes the user, goal,
activity, explicit period and first explicit source event, so distinct task
instances are not conflated. Quotation text, values, original source IDs, dates,
roles and offsets remain unchanged. Actual UUIDs, alias mappings and task
versions remain in the private preparation archive and durable fences.

Two independent preparations of the same 24 controlled cases produced identical
hashes for all 48 requests; all 96 source preflights passed and the largest full
request remained 5,803 characters. This is zero-model-call fixture verification.
Version 1 prepared artifacts and its zero-answer budget-stop record remain
untouched. Both compared product versions must use the same version 2 harness;
request equivalence may then reflect evidence/state differences rather than
random database identifiers.
