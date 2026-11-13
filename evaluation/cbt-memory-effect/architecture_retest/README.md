# CBT context architecture replication

This runner applies the shared product CBT projection contract and renderer to
the previously fixed CACTUS source roster. It keeps original source material,
natural continuation quality and quotation identity as separately named tasks.
Real source preparation, embeddings and provider calls require genuine hosted
Colab. Importing the package does none of those operations.

## Natural continuation study

`cactus_materials.compile_rows` selects 200 source families and three released
dialogue-prefix cutoffs per family. All 600 natural continuations remain in the
planned denominator. Cutoffs are positions in a synthetic dialogue, with no
invented session calendar. Hidden intake, CBT plan, cognitive-pattern metadata,
future counselor responses and benchmark targets stay outside generation.
Pass the recovered historical `family_ids` list to `runner.prepare` when the
previous private overlay selected a different family roster. Preparation binds
that exact family set back to the immutable released corpus and preserves the
original row indexes. The protocol names whether a recovered historical roster
or the public deterministic compiler supplied the selection.

Exactly three answer conditions are frozen:

| Condition | Older context |
| --- | --- |
| `no_rag` | No retrieved older memory; the same six recent original messages remain ordinary conversation context |
| `raw_history_rag` | Whole original messages selected by BM25 plus pinned E5 rank fusion, with adjacent-source expansion |
| `managed_memory_rag` | Query-selected, source-supported processed CBT memory with dependency closure and key evidence |

All three use the identical complete counselor policy exported by
`cbt_context.CBT_POLICY` from `cbt_counselor_policy.py`, whose role statement is
`You are a CBT psychological counselor within an AI research prototype.` The actual system message, answer model,
recent dialogue and complete-request/output ceilings are shared. The primary
comparison measures the entire memory pipeline, including selection and
representation. No additional representation-ablation conditions are run. Equal
ceilings do not imply padded equal input lengths, and two conditions may
legitimately produce identical requests when no older context is admitted.

The shared production module `cbt_projection_contract.py` defines extraction,
literal source binding and a separate automatic source-entailment assessment.
The runner reuses those functions. Accepted paraphrases still have an automatic
support verdict, distinct from human confirmation. Inferences remain tentative
working formulations or proposed guidance. The shared `cbt_context.py` places
the processed content in the actual model request, with source identities,
scope, versions and an external trace.

Raw retrieval combines BM25 and `intfloat/multilingual-e5-small` at immutable
revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. Each arm has six recent
messages, 24,000 characters for the complete escaped answer request and 1,024
output tokens. Extraction and verification have 48,000-character complete
request ceilings and 4,096 output tokens. The finite answer/preparation ceiling
is five sends per case (two preparation and three answers), or 3,000 for the natural roster. Exact canonical
duplicates share completed receipts while every logical case remains recorded.

## Direct memory endpoints and independent R1–R5

Before any answer for a case, a separate Qwen-family assessor proposes necessary
and stale state-bearing facts from the visible source prefix and current input.
A separate source-support/relevance call checks that target set. These targets
are frozen in `assessor_only` and never enter extraction, retrieval or answer
messages. They are `model_proposed` labels.

After each answer, the assessor binds actual answer assertions to exact answer
substrings and eligible source quotations. The existing
`auto_eval_v2.evidence.score_memory` computes supported, unsupported,
contradicted and stale assertion counts, assertion error rate and coverage of
the frozen necessary facts. Counts keep planned and scored answer denominators.
These semantic verdicts are automatic assessments. R1 ordinal scores are
reported separately from assertion error rates and fact coverage.

Version 2 rejects invalid target candidates individually, preserving every
original proposed ID and its validation status. Exact-source binding never
rewrites an unknown source ID, altered quotation or abbreviated span. Only
retained candidates proceed to the separate semantic support/relevance check.
Reports include proposed, binding-retained and verified counts, per-case partial
or unavailable status and the retention fraction. Coverage concerns the verified
retained model-proposed targets; it is not complete necessary-fact gold.
Unavailable direct-metric targets do not prevent the three answer conditions or
independent R1–R5 assessment. Empty answer exports preserve the planned quality
grid and perform zero quality calls.

A repair uses a separate new run directory. The optional
`memory_assessment.prepare(..., inherited_folder=original_memory_run)` freezes
unchanged completed pre-answer `fact_targets` receipts with their original
protocol, exact request wire hashes, returned models and raw responses. Exact
matching requests read those recorded receipts without resending. Inherited
receipts stay outside the new paid-intent journal and reduce its remaining
technical cap. Fixture receipts cannot enter a real run. Reports separately
count inherited requests/usage and new attempts/received usage, avoiding double
counting old tokens as new spending. Raw target
requests are never repeated after answer generation.

The independent memory-assessment ceiling is five calls per case, or 3,000.
Its complete-request character cap is separately frozen at 96,000 to accommodate
full source history, targets and the actual answer. Known source-only request
sizes are written before inference. Later target-check and answer-adjudication
payloads are checked before any send intent. Missing or invalid assessments
contribute zero to coverage of every frozen required fact. Conditional coverage
and completion-adjusted coverage are reported separately, with unavailable
target preparation recorded independently. Paired precision/recall/F1 reports
reuse the fixed-grid family-bootstrap helper.
R1–R5 judging reuses the existing anonymous source-bound rubric and validation.
The requested judge model and the actual returned model alias are frozen
separately. The judge family must differ from DeepSeek. Ten percent of quality
packets receive actual repeated judgments for consistency assessment. The full
1,800-response quality roster has at most 1,980 judge sends. Repeats never count
as extra effect cases.

## Historical quotation task

`quotation.prepare` reuses the historical `choose_probes` cue, target-length and
ambiguity rules on the same source roster. The planned 600 probes and all
exclusions are recorded, with the resulting eligible count computed from the
frozen corpus. It reuses completed, question-independent latest-prefix
projections by file hash, avoiding duplicate extraction costs.

Raw, managed-with-evidence and recent-only requests answer an explicit exact
quotation question. For this task, managed context can restore full original
source quotations in addition to processed items, because the requested output
is a quotation. No target quotation is supplied to retrieval or generation.
The original `score_quote` rule checks the complete quotation and actual source
identity. The source namespace is consistently bound to the same original
message in every arm. This endpoint remains literal recall, separate from
natural CBT follow-up. Its cap is three sends per eligible probe.

## Hosted usage

Freeze all code and inputs before preparing a run. The notebook supplies existing
approved credentials inside the hosted kernel without printing or exporting
them. The following functions contain no credential lookup.

The complete orchestration entry point is now versioned in the repository:

```python
import runpy
cbt = runpy.run_path(str(checkout / "evaluation/architecture-rerun/cbt_hosted_controller.py"))
cbt["prepare_all"](checkout, new_run_root, inherited_memory_roots=inherited)
cbt["run_all"](checkout, answer_transport, judge_transport, new_run_root,
               stage="all", inherited_memory_roots=inherited)
```

Here `inherited` is an explicit cohort-to-original-memory-directory mapping, or
an empty dictionary for a fresh cohort. The controller freezes and validates its
own code SHA, checks repository module origins before source work, and preserves
the source protocol's real/fixture gate. The lower-level interfaces are:

```python
import sys
sys.path.insert(0, str(checkout / "evaluation/cbt-memory-effect"))
from architecture_retest import runner, memory_assessment, judge, quotation
from architecture_retest.transport import HostedGatewayTransport

answer_transport = HostedGatewayTransport(runner.IDENTITY, key)
judge_identity = {
    "provider": "https://tokendance.space/gateway/v1",
    "model": "qwen3.5-plus",
    "returned_model": "qwen3.5-plus-2026-04-20",
    "model_family": "qwen",
}
judge_transport = HostedGatewayTransport(judge_identity, key)

runner.prepare(verified_cactus_json, natural_run, family_limit=200,
               family_ids=historical_family_ids)
memory_assessment.prepare(natural_run, memory_run, judge_identity)
runner.run(natural_run, answer_transport, max_new_calls=3000, workers=4,
           assessment_folder=memory_run, assessment_transport=judge_transport,
           assessment_max_new_calls=3000)

runner.export_quality(natural_run, judge_identity, quality_run,
                      maximum_attempts=1980)
judge.run(quality_run, judge_transport, max_new_calls=1980, workers=4)

quotation.prepare(natural_run, quotation_run,
                  frozen_generation=old_quote_generation,
                  frozen_gold=old_quote_targets, frozen_inventory=old_inventory)
quotation.run(quotation_run, answer_transport,
              max_new_calls=3 * eligible_quote_count, workers=4)
```

For the recovered professional seven-case or controlled 24-case cohorts, use
`runner.prepare_case_pool(original_source_cases, output, source_identity)`.
The normalizer retains only original history, current input, identity and real
source dates. Reference answers and old scores remain outside generation.
The original deterministic controlled-task scorer can evaluate those new
answers separately from independent contextual assessment.

`maximum_attempts` is a per-run technical ceiling bound to the current user
request. Historical code limits never authorize this run. The run journal
stores exact serialized request strings, canonical payload hashes, wire hashes,
actual returned aliases, finish reasons, response IDs, token usage and failures.
Source/request canonical hashes and transport wire-byte hashes have explicitly
different scopes. An unknown send stops new admission. Completed receipts and
individual arm answers survive a later failure and are retained on resume.
There is no automatic resend of an unknown or rejected canonical request.
