# CBT memory optimization and automatic evaluation v2

This opt-in development version combines source-level memory retrieval,
source-grounded factual accounting, independent R1–R5 review, and calibration
against existing published expert labels. It does not recruit new human raters.
Old CACTUS, generic-memory and professional-roleplay runs retain their original
code, protocol, requests and scores.

## What changes in the memory system

`auto_eval_v2.retrieval.select_evidence` combines the competent raw source
ranking with source-bound semantic/facet rankings using weighted reciprocal
rank fusion. It returns **original messages**, never inferred index prose.
Useful facets can include current concerns, goals, prior barriers, agreed tasks
and explicit corrections. Facet rankings must be generated without target
answers and their extraction/index costs belong to the managed condition.

- The same recent six messages, roles and source IDs are directly included in
  both conditions. Ineligible messages are removed without older backfill.
- Both conditions use the same complete request renderer, system prompt,
  response model/options, top-k and character cap. The raw comparator receives
  the whole eligible original source pool; it is not a short-history baseline.
- Future, deleted, use-denied and external-denied sources are excluded. The
  cutoff is inclusive in every v2 module.
- Literal source binding is mandatory for each index candidate. Neither an
  assistant suggestion nor an inferred summary becomes user agreement.
- There is no role-based ranking bonus. Stable source keys replace job-UUID
  tie breaks. Equal selected source sets restore the raw order and request.
- Duplicate pointers to one source vote once per channel. Distinct original
  messages remain distinct even when their text matches, preserving repeated
  attempts and confirmations. Eligible adjacent original context can be added
  when the frozen budget permits it.
- Budget checks cover the returned complete ASCII-escaped canonical JSON request, including
  provider options. Use its exact `serialized_request` at a separately admitted
  send boundary, or recheck the actual final serialization before sending.

This helper consumes existing index rankings. It does not implement a new
embedding model, infer a clinical state, change weekly task projection, or
authorize an outbound send. The live store's current permission/withdrawal and
version checks still apply to all sources **including the current input**.
Do not route a new v2 body through an old frozen snapshot's admission receipt.

## What each evaluation verifies

| Evaluation | Question | Output |
| --- | --- | --- |
| Evidence accuracy and coverage | Does managed memory use more required information accurately? | Precision, required-fact recall, F1, omissions, unsupported/contradicted/stale claims |
| R1–R5 response quality | Does the memory difference improve contextual facts and the next step? | Four separate 1–5 scores; safety pass/fail; exact evidence |
| Published-label calibration | Can the independent judge recover existing expert CTRS/safety labels? | Per-dimension Spearman, MAE, bias, missing counts and safety confusion matrices |
| Consistency | Are grades stable under repetition and candidate position? | Deterministic repeat slots; separate AB/BA pairwise plans, ties and abstentions |
| Resources | What did all attempts cost? | Intents, native receipts, known tokens/latencies, missing usage and unresolved sends |

`evidence.score_memory` consumes explicit semantic verdicts, validates their
literal source bindings and computes the metrics. It is not an entailment
oracle. Reference and verdict provenance must be declared as `published_gold`,
`rule_verified` or `model_proposed`; proposed labels remain identifiable.
An old fact incorrectly used as current is still wrong if the correct new fact
also appears. Repetition cannot inflate precision. Missing outputs stay in the
fixed denominator; conditional and completion-adjusted results are separate.

The older CACTUS four-word quotation task keeps its own strict quote+source-ID
score and original identity. Within-recording educational roleplay is not
silently renamed a natural multi-visit experiment. Dated corrections, pauses,
expiry and deletion mechanisms remain separately identified tests.

## Independent automatic review

`quality` fixes R1 contextual facts, R2 CBT method, R3 next-step fit, R4 empathy
and R5 safety. The judge receives only allowed history, current input and one
anonymous response. Case/family/model metadata and future/reference responses
are excluded from the prompt. Every grade includes a literal source ID, role,
source quote and response quote. No visible risk cue is a valid safety pass;
unassessable text can remain unresolved. No total score is fabricated.

The declared judge family must differ from the answer family. Model identities
are reported configurations/receipts, not authenticated model weights. The
deterministic default repeats ceil(10% of logical response packets); repeats
incur independent attempts and do not increase the effect sample size.
`quality.plan_pairwise` prepares AB/BA requests separately; these are **not**
automatically sent or included in the default absolute-rating call cap.

`calibration` uses the nine retained CTRS aspects and four safety categories
from TherapyJudgeBench. `calibration_prompt` supplies gold-free whole-dialogue
judge requests. Explicit profile-to-source-family provenance is required before
splitting; exposed sibling cases stay in development. Null labels are excluded
per dimension. A safety class with no positive examples has undefined recall.
Existing expert labels calibrate those defined tasks; they do not themselves
validate project-specific R1/R3 memory judgments.

The code recognizes the already frozen TherapyJudgeBench v2 source identity
through `verify_source_manifest`. Reuse the checked bank in hosted Colab rather
than downloading a replacement. No acquisition is performed by this package.

## Hosted Colab entry point

Use a fresh checkout/kernel for this version. Add
`evaluation/cbt-memory-effect` to `sys.path` in the existing **hosted** Colab.
Do not use a local runtime. No Secret lookup, provider SDK or default transport
is supplied, and no historical 5/5,000/7,200 allowance transfers to this run.

```python
from auto_eval_v2 import run_colab

# packets: frozen allowed source history/current input/actual response objects.
# comparisons: every planned question; use null response IDs for failed answers.
# judge_identity: explicit provider/model/model_family, independent of answers.
run_colab.prepare(
    packets, judge_identity, "/content/NEW_CBT_AUTO_V2_RUN",
    comparisons=comparisons,
    maximum_attempts=0,  # preparation only; freeze a bounded authorized run first
    authorization_reference="not-authorized-for-new-calls",
)
```

Quality packet fields are exactly `case_id`, `source_family`,
`answer_model_family`, `history` (list of `id/role/content`), `input`, `response`.
Comparison rows are `case_id/source_family/raw_case_id/managed_case_id`. Missing
generation IDs are explicit null, not omitted questions. Both existing answers
must have the same judge-visible history, current input and answer family.

For calibration, set `task="calibration"`, omit comparisons, and provide packet
fields `case_id/source_family/answer_model_family/messages`; messages are the
ten visible alternating role/content objects. Supply `calibration_bank` pointing
to the existing checked source bank. Its actual bytes and manifest are bound to
the protocol and rechecked before execution. Only after every primary model
attempt is frozen does `score_calibration(run_folder)` read the published labels
from that bank and verify that each rated dialogue matches the frozen input.
Caller-supplied labels are accepted only for explicitly invented fixture tests.
The new version does not change the old five-dialogue calibration run.

When a bounded new run is authorized, the notebook supplies a no-retry callable
with `transport.identity == judge_identity`. It receives the exact payload and
returns `text`, `provider_model`, and normalized `usage` with input/output/total
tokens. Call `run(..., max_new_calls=N)` explicitly; the default is zero. The
transport must not make hidden SDK retries or change the outbound body. Current
source eligibility must be rechecked by `transport.preflight(payload, packet)`
returning exactly `True` immediately before every real send intent; missing or
denied preflight blocks execution. This notebook hook must use the current
authoritative permissions, not a guessed consent flag. Provider access must
stay within hosted Colab; no key is copied to this package.

Each actual callback has a durable, fsynced intent before execution. Identical
primary requests share a canonical receipt; repeat slots are separate attempts.
Unknown/interrupted attempts block every later call. Known invalid outputs are
retained with their cost and never automatically retried or repaired. Resume
and summary both verify protocol, code, input and native-score bindings.
Completed scores, unresolved reviews and merely received bodies are separate.
One absent source family never produces a fabricated cluster interval.

## Development and validation

Improve retrieval/state behavior on development families first, then freeze
the system, comparison, rubric, sample and endpoints before held-out testing.
Retain all planned questions and all failures. Check precision as well as
coverage and R1/R3; report safety and resource changes separately. If a proposed
change does not improve the held-out comparison, retain that result and use a
new development version rather than relabeling it as a win.

```bash
python3 -m unittest discover -s tests -p 'test_auto_eval_v2_*.py' -v
```

These tests use invented fixtures only. They verify source recovery/packing,
gold isolation, scoring/accounting, family splits, durable calls, repeated-cost
accounting and integrity. **No new CBT dataset effect or model-quality result
has been generated by this change.**

## Research inputs

- [MACBT](https://arxiv.org/abs/2609.30939): separate session/longitudinal scores
  and memory ablations.
- [CACTUS](https://aclanthology.org/2024.findings-emnlp.832/): anchored CBT skill
  evaluation of multi-turn dialogue.
- [TherapyGym](https://arxiv.org/html/2603.18008v1) and
  [TherapyJudgeBench v2](https://huggingface.co/datasets/therapygym/therapyjudgebench):
  reuse published expert labels to audit an automatic judge.
- [LongMemEval](https://arxiv.org/html/2410.10813v2): reference-based answer
  correctness and separate retrieval measurement.

These are combined design inputs, not a reproduction of one paper's full
method. No new human-review requirement is introduced.
