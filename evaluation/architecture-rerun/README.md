# Full original-cohort architecture replications

The latest user instruction fixes exactly three conditions everywhere:
`no_rag`, `raw_history_rag` and `managed_memory_rag`. All retain the same ordinary
recent conversation. Raw RAG adds retrieved older sources; managed RAG adds
source-supported processed memory. There is no fourth condition or representation
ablation. Every condition receives the complete shared `CBT_POLICY` from
`modules/agent-integration/cbt_counselor_policy.py`, plus an identical
benchmark-specific answer-format instruction. The policy dependency is frozen
in each executable protocol.

| Cohort | Original eligible questions | New answer rows |
| --- | ---: | ---: |
| MemoryAgentBench Conflict Resolution | 800 | 2,400 |
| LongMemEval-S cleaned | 500 | 1,500 |
| LoCoMo, category 5 excluded | 1,540 | 4,620 |
| Constructed dated care tracking | 24 | 72 |

`inventory.json` records exact earlier source revisions, hashes and separate
historical result identities. PersonaMem's previously excluded management-benefit
claim remains excluded. Full source preparation, every model call and real
scoring run require genuine hosted Colab. No import reads a Secret or invokes a
model. All old protocols, costs, failures and outputs remain intact.

## Hosted entry point

Start from a fresh hosted runtime when switching checkouts or changing executable
code. A matching file on disk does not establish which module objects the kernel
already loaded. Entry points now reject every loaded repository module whose
origin belongs to another checkout, even when its source bytes match.

If reusing an idle kernel, first join every worker and finish all outstanding
requests. Clear **all** repository-owned modules and old checkout paths, including
generic names such as `colab_provider`, `durable_memory`, `controlled_dialogue`,
`controlled_replay_adapter` and `run_three_arm`, together with package parents.
Identify ownership by the project-specific `modules/agent-integration` and
`evaluation/cbt-memory-effect` directories; generic dependency directories named
`modules` and `evaluation` do not establish project ownership.
Filtering only names or paths starting with `/content/cbt` is insufficient.
Reimport the pinned checkout and run the origin guard before preparation or send
admission. The guard never purges modules automatically while workers run. For
in-place code changes, prefer restarting the runtime, since origin equality alone
cannot prove that existing objects were reloaded.

When other frozen jobs are still active, `hosted_relay.py` and
`hosted_worker.py` run a clean Python child in the same genuine hosted Colab VM.
The parent retains the approved provider transports and credentials. A private
Unix socket preserves exact request bytes and receipts, with immutable per-job
call caps and no retry or response cache. Helpers are copied outside the selected
clean checkout and recorded with their hashes. Source-only preparation has a
separate zero-call job. The scientific protocol and its selected checkout remain
unchanged, including when grading a completed older generation run.

Use the live kernel's already approved DeepSeek transport. Preparation freezes
full rosters and finite caps before execution. The controller reuses an existing
run only after checking its frozen code and artifacts.

```python
import runpy
controller = runpy.run_path(str(REPO / 'evaluation/architecture-rerun/hosted_controller.py'))
controller['prepare_cohort'](REPO, 'controlled24')
controller['run_batch'](REPO, 'controlled24', deepseek_transport, max_new_calls=72)

controller['prepare_cohort'](REPO, 'memoryagentbench800')
controller['run_batch'](REPO, 'memoryagentbench800', deepseek_transport, max_new_calls=120)

controller['prepare_cohort'](REPO, 'longmemeval500')
controller['prepare_cohort'](REPO, 'locomo1540')
controller['run_batch'](REPO, 'longmemeval500', deepseek_transport, max_new_calls=120)
controller['run_batch'](REPO, 'locomo1540', deepseek_transport, max_new_calls=120)
```

Continue finite execution batches until the full planned cohort completes. The
default root is `/content/cbt-v4-general`; controller progress is written as
`<cohort>.status.json`. Individual directories retain protocols, actual requests,
receipts, failure traces and scores. No automatic provider retry occurs. Unknown
sends block continuation. Four workers stop admission after failure and drain
already admitted calls into their own journal records.

The natural CBT continuation, original professional seven-case cohort and
historical CACTUS quotation endpoint use the separate pinned-checkout controller
below. Its own source SHA is frozen in `controller_manifest.json` and checked
before resume. Existing independent target receipts are accepted only when their
source protocol has the same real/fixture mode and their complete request bytes
match. They remain distinct from newly sent requests.

```python
cbt = runpy.run_path(str(REPO / 'evaluation/architecture-rerun/cbt_hosted_controller.py'))
NEW_ROOT = '/content/cbt-v4-repair-v2'
inherited = {'professional7': '/content/cbt-v4-cbt-full/professional7_memory'}
cbt['prepare_all'](REPO, NEW_ROOT, inherited_memory_roots=inherited)
cbt['run_all'](REPO, deepseek_transport, independent_judge_transport, NEW_ROOT,
               stage='professional7', inherited_memory_roots=inherited)
cbt['run_all'](REPO, deepseek_transport, independent_judge_transport, NEW_ROOT,
               stage='all', inherited_memory_roots=inherited)
```

## MemoryAgentBench

The [completed 800-question rerun](results/20261113-mab800-summary.json) has
2,400 received DeepSeek answers. The released substring metric is 9/800,
431/800 and 469/800 for no RAG, raw RAG and managed memory respectively
(managed minus raw: +4.75 percentage points). This original metric remains the
requested reporting and comparison method; an extra exact-string diagnostic is
not an effectiveness threshold or a replacement score. Already computed
diagnostic records remain archived.

The earlier full run recorded 480/800 (60.00%) for managed memory; the bounded
repair grid recorded 483/800 (60.375%), combining 206 new replies with 2,194
unchanged frozen replies. The new full run's 469/800 is 11 below the earlier
full-run result and 14 below the repair grid. The corresponding raw scores are
419/800 previously and 431/800 now. This is a regression against the older
managed-memory scores, even though the new within-run managed–raw gap remains
positive. Changed role policy, retrieval/readout and execution time require
case-level investigation before attributing that difference to one component.
These source-grammar results
come from one conservative source family; they do not establish a CBT or
clinical benefit.

`mab_full.py` preserves the eight declared factconsolidation strata, 100 questions
each. It reuses the existing candidate store, deterministic source-grammar version
projection and source-edge retrieval. Both RAG conditions receive the same
capable retrieval algorithm. Parsed subject/relation/object fields actually reach
`cbt_context`; unparsed source grammar remains explicitly source text. Synthetic
source assertions retain the `context` role and benchmark provenance.

The common recent channel holds the latest 16 source reports within 6,000
characters. Older selection uses 12,000 characters; the complete request cap is
24,000 characters and output cap 128 tokens. Preparation freezes all 2,400 logical
packets and a finite unique-wire attempt count. The supplied kernel transport is
supported directly, with exact returned-model, finish and wire-hash checks.
Scoring preserves released substring matching plus the stricter response
commitment diagnostic. All missing rows stay in the denominator. The 800 related
questions belong to one conservative source family and do not establish a CBT
or clinical effect.

## LongMemEval and LoCoMo

`formal_sources.py` reuses pinned complete releases, original eligible question
IDs, source speakers, dates and cutoff rules. It normalizes repeated session
occurrences with the earlier adapter and keeps scoring references separate.
LongMemEval contains 500 questions, not 1,500 questions; 1,500 is its three-arm
answer count. LoCoMo contributes 1,540 eligible questions after its earlier
category-5 exclusion.

`formal_full.py` partitions every cutoff-eligible history into complete contiguous
fragments with parent source and exact offset mapping, then chronological chunks.
Extraction receives no question, reference answer, category or evidence label.
Each chunk gets a separately journaled model support check. Verified dated
subject/attribute reports are grouped into complete historical version series
across chunks. Later mention alone never establishes supersession or current
validity. Speaker names and original dates survive both raw and processed
readout. Failed chunks remain visible, while verified chunks can still support
an answer; missing material is never filled with invented memory.

The shared recent context uses up to 16 turns from the latest eligible session
within 6,000 characters. The dynamic memory cap is 12,000 characters and full
request cap 24,000, with 4,096 output tokens. Full-pipeline raw and managed retrieval
are separately auditable; every ranked, selected and budget-rejected unit is
recorded. Caches are reconstructed from extraction/verifier receipts. Answer and
completion records are checked against the actual provider-bound request and
complete receipt. The finite upper bound is two calls per unique preparation
chunk plus three answers per question, without automatic retry.

After the entire reader grid has answers or explicit terminal failures,
`formal_judge.py` freezes blind independent-family grading packets. It reuses the
earlier pinned LongMemEval reference templates and their declared transfer to
LoCoMo, with strict yes/no parsing, all planned rows, category and abstention
summaries, and equal-weight history-family means with 10,000 cluster bootstrap
resamples. A fixed outcome-blind 10% repeat is included. Maximum logical grading
attempts are 1,650 for LongMemEval and 5,082 for LoCoMo; exact-wire receipt reuse
can reduce actual calls. These are adapted reference-model scores, not official
benchmark-model scores, human ratings or clinical outcomes.

```python
controller['judge_batch'](REPO, 'longmemeval500', independent_judge_transport,
                           max_new_calls=120)
```

## Dated controlled 24-case replay

`controlled.py` imports the exact archived 24 dated cases and literal targets.
It uses real durable source ingestion, current product task rules and projector,
then the new shared context renderer. It performs no semantic extraction calls.
The model gets actual rule-supported task state, original source evidence and
real dates. All three conditions share the role, factual JSON instruction and
recent conversation. Source/object versions are checked again before sending.

The original recent-six, distinct-source cap nine and output cap 64 tokens are
retained. The new complete counselor policy makes the previous 6,000-character
full-request ceiling too tight for some mandatory contexts. Source-only
preflight against that old ceiling is recorded per case. All three conditions
therefore share a declared 24,000-character ceiling; the policy is never clipped.
This change is separate from the historical Qwen model/decoder results.

Scoring retains the original strict JSON object with only `answer`, exact value
and type matching, and no repair or answer extraction. Missing rows remain in
the planned denominator. The original scorer source identity is recorded. These
24 exposed constructed variants are a development mechanism regression, not
independent natural CBT cases or clinical observations.

## Separately versioned CBT quality grading

`quality_span_judge.py` regrades retained answers using the original R1–R5 rubric.
It does not regenerate answers or change the counselor policy. The earlier
professional-seven grader returned 19 receipts, but none satisfied its evidence
binding contract: copied quotations were not always literal and some source IDs
or fields were invalid. Those attempts and their costs remain archived; invalid
reviews are not interpreted as zero quality scores.

The new protocol presents losslessly indexed segments of the complete source,
current request and answer. A review selects segment IDs instead of reproducing
quotations. The audit retains parent IDs, roles, offsets, exact content, original
reader request and receipt hashes. It validates all five evidence objects and
keeps missing answers in coverage reports. Scores remain conditional on available
valid reviews; missing responses are not replaced with invented ratings.

Preparation is source-only. An immutable manifest fixes model identity, source
hashes, blind packets, a ten-percent independent repeat and call caps before
grading. Exact wire bytes and real receipts are journaled; unknown sends stop
the run without automatic retry. `hosted_worker.py` exposes separate `quality-v2`
prepare and run actions in fresh credential-free Colab children. This protocol
repair does not establish new CBT effectiveness results until grading completes.
