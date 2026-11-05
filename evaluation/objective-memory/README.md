# Objective memory comparison

This is a separate evaluation from the frozen LoCoMo / LongMemEval v2 run. It
uses released answers and fixed deterministic scoring rules. It does not use an
LLM judge as the primary outcome, and it does not measure CBT treatment benefit.

The main question is whether managed memory adds value over retrieval from the
**same original history**, rather than whether more history helps a reader.

## Completed full run: 2026-11-05

The [full results and all per-question score metadata](results/full-20261105/README.md)
retain 1,389 questions and all 4,167 terminal arm rows. Full inputs, generation and
source-based scoring ran in genuine hosted Colab under executable
`1511706b3a4b0ac71416998b5b4d752a13c4e313` and suite plan
`940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568`.

MAB official accuracy is short-term 10/800, raw 419/800 and managed 480/800:
the primary managed-minus-raw contrast is **+7.625 percentage points** (68 wins,
725 ties, 7 losses). This tests a deterministic source-order version view within
one conservative fact-consolidation family; no independent-family interval is
estimable. Managed multi-hop accuracy remains only 96/400 (24%).

PersonaMem official accuracy is 364/589, 416/589 and 416/589 respectively. Its
two RAG arms share all 589 inputs and canonical responses. Their zero difference
is a quotation control by construction, not management efficacy or statistical
equivalence. There are 20 connected groups, not 589 independent people.

The run records 3,550 actual successful canonical calls, 8,777,083 reported
tokens, zero API failures and no missing usage fields. It preserves old runs and
the 61-call negative/insufficient pilots separately. Strict-format diagnostics
and source-path observations are retained; all 2,062 officially incorrect rows keep
causal stage unresolved. No human CBT ratings, participant benefit, natural-check
burden or clinical outcomes have been established. The
[post-run reporting scripts](reporting/README.md) reproduce the controlled
derived report and its public metadata projection without more model calls.

[Open the objective comparison in Colab](https://colab.research.google.com/github/nanlalala/cbt-memory-system/blob/main/evaluation/objective-memory/Objective_Memory_Comparison_Colab.ipynb)
or inspect the [notebook](Objective_Memory_Comparison_Colab.ipynb). It checks out
the immutable executable revision recorded in its metadata and code cell.
Use a fresh hosted runtime; preparation and paid batches are disabled by default.

## Conditions

| Arm | Information available |
| --- | --- |
| `short_term` | A fixed recent-message window |
| `raw_history_rag` | That window plus retrieval over eligible older original messages |
| `managed_memory_rag` | That window plus source-preserving records imported through the real memory manager |

All arms share the question, model identity, generation settings, knowledge
condition and context budget. Both RAG arms retain the same released history,
including its original roles. Benchmark `system` messages are quoted history
data, never instructions. Synthetic benchmark documents are not genuine user
statements or consent. Static QA supplies no extra answers to confirmation checks.

The first managed representation is **verbatim history, not semantic goal or
homework extraction**. The manager preserves provenance and lifecycle policy;
it does not independently prove the truth of a summary. The raw baseline gets
ordinary time, deletion and access controls too.

Before any API call, the runner compares the actual messages supplied to the
model. Identical prompts share one recorded canonical response. The journal
identifies derived arm outputs and charges the actual request once. If raw and
managed inputs are identical, a zero difference is a **control-equivalence result
by construction**, not empirical proof of no effect, statistical equivalence or
better memory management. Changing a heading just to manufacture a difference
would not answer the research question.

In the original quotation-control version both RAG arms receive the same
quotations and use the same retriever, so their prompts are identical. That
version is a control baseline for the pipeline and older history. The completed
v3 comparison applies a deterministic source-order version view to MAB before
shared retrieval; PersonaMem retains the original quotation-control contrast.
Neither representation evaluates natural-language CBT goal/homework extraction.

The shared budget is measured in characters; actual provider prompt/completion
tokens are reported separately. This is not an exact tokenizer-matched budget.

## Frozen external tasks

The machine-readable [dataset registry](dataset_registry.json) fixes revisions,
release identities, licensing evidence and eligible task families. Full files
are downloaded and loaded only in hosted Colab. Local tests use small fixtures.

| Dataset | Released labels and scoring | Interpretation |
| --- | --- | --- |
| PersonaMem-v1, 32k | Four-option answers; author's fixed answer parser | Synthetic changing personas and preferences, not real CBT participants |
| MemoryAgentBench conflict resolution, `fact_sh` / `fact_mh` | Released aliases; author's normalized substring match | A constrained fact-conflict task; a match does not rule out every contradictory statement |

Colab's first verified PersonaMem census found **589 questions, 20 personas and
20 connected persona/shared-context groups**. The 37 referenced contexts do not
create 37 independent participants. Source messages are sliced at the released
exclusive `end_index_in_shared_context` before import or retrieval.

MemoryAgentBench context-length variants are grouped conservatively as one
fact-consolidation family. More questions or length variants do not establish
more independent histories. Its results are exploratory, with no participant
population inference. The scorer keeps datasets and task types separate.

## Scoring and cost

Only a terminal full cohort may produce a final summary. Pending work produces
a progress record. Valid-response accuracy and request availability are separate;
exhausted technical failures remain zero in the planned-cohort denominator.
There is no weighted overall score across these datasets or software controls.

Report question counts, independent history groups, micro accuracy, equal-group
accuracy, task breakdowns, paired differences and cluster uncertainty where
meaningful. A cluster interval is not a power analysis. Without a frozen sample
size/precision plan, results remain exploratory. Identical-input contrasts have
no inferential effect estimate. Report actual calls, failed attempts, latency and
tokens; never count a shared response as three independent paid generations.

Gold answers are in scoring references, separately hashed and never passed to
history preparation, retrieval or generation. Protocol, prompt, model identity,
resume journal and output-directory locks are checked on continuation. Do not
change a frozen run to accommodate a negative result.

Public QA errors are currently marked `answer_incorrect_stage_unresolved` unless
there is a recorded provider failure. The public labels do not identify which
source evidence should survive each stage. The synthetic replay checks storage,
visibility, retrieval and context assembly separately, but does not run natural
language extraction or answer generation. A full source-to-answer error report
remains pending independently annotated evidence.

## Lifecycle acceptance

`replay_memory_controls.py` runs the real store, manager, retriever and context
builder against separately fixed synthetic events and an expected-state ledger.
The independent raw-history baseline receives the same explicit events.

It reports goals, homework, state, correction, expiry, deletion and isolation.
These are **software acceptance checks**, not independently annotated natural
conversations. They do not test semantic extraction, the usefulness of a reply,
or the burden of answering a clarification. Equal success is a valid result.

```bash
python evaluation/objective-memory/replay_memory_controls.py --help
```

## Execution and artifacts

Use the [hosted Colab notebook](https://colab.research.google.com/github/nanlalala/cbt-memory-system/blob/main/evaluation/objective-memory/Objective_Memory_Comparison_Colab.ipynb)
for full data and API work. Start with a finite
call budget and inspect progress before another batch. No automatic download of
neural models or local-GPU requirement is introduced here. API credentials stay
in the Colab secret/environment and are excluded from files and exports.

Keep the frozen protocol, prompts, references, responses, attempt journal,
canonical-response links, source identities, executable scoring code and
`SHA256SUMS` in a controlled audit archive. Public results may contain aggregate
and per-question scores with dataset IDs; do not publish full benchmark-derived
prompts/responses by default. Real user content requires a separate retention
and access policy.

## What still requires evidence

- Source-supported **semantic** goals, homework and state extraction, evaluated
  against independently annotated spans and status labels.
- A four-visit task study measuring continuity, corrections, deletion, unnecessary
  checks, completion time and user effort.
- Independent review of free dialogue and response claims.

These outcomes stay unmeasured until the corresponding study is done. Passing
the deterministic comparison or adding more generated QA cannot replace them.


## Separate v2 semantic-index experiment

The current entry pins the review-fixed executable **1511706b3a4b0ac71416998b5b4d752a13c4e313**.
The earlier notebook and 7832f38a639eb5b5e3c27d6c564db5db25f3999c executable remain available in Git history for their original frozen runs.
Do not resume a v1 run using the v2 checkout. Keep its directory and executable
snapshot. V2 adds `prepare --semantic-index FILE`; it requires a new run directory
and freezes the index bytes in `semantic_index.jsonl`.

The index format is one `{"case_id": ..., "candidates": [...]}` per case, including
empty candidate lists. Each candidate requires `subject_id`, `source_document_id`,
`source_text_sha256` (SHA-256 of the eligible whitespace-trimmed source),
`memory_type`, `summary`, and an exact contiguous `supporting_text` quote.
The index must be generated **without scoring references or question answers**.
This preparer does not authenticate that upstream generation process: until its
request/response journal and costs are supplied, provenance and total pipeline
costs remain unverified/unreported. `semantic_source_index.SYSTEM_PROMPT` supplies
the extraction contract; it is not itself an executed extractor experiment.

Semantic summaries influence retrieval ranking only. The same ranker and context
budgets apply to the raw and managed arms; the answer model still sees eligible
original quotes with their original source roles. Source hashes, ownership,
visibility and lifecycle checks reject deleted, expired, restricted, future,
unknown or altered source references. Summaries do not confirm themselves, change
homework status, supersede versions, or become answer facts. Thus the new contrast
can test **semantic indexing with source-grounded rendering**, once a separately
logged index is supplied. It cannot isolate or prove version-management benefit.
Do not present its software-fixture input differences as model effectiveness.

## Evidence-stage ledger

`diagnose_evidence_stages.py` writes a separate per-request ledger, including
pending requests. It does not modify frozen scoring code, supply gold to a model,
or turn a partial cohort into a final accuracy result. Run it with the matching
frozen runner code; write output outside the live run.

Optional scoring-side annotation JSONL requires `case_id`, `protocol_sha256`,
`history_inventory_sha256` (the exact current case-visible source history),
`reviewer_id`, `annotation_origin` (`released`, `human_review`, `developer_fixture`,
or `model_proposed`), `coverage: complete_sufficient_alternatives`, and
`alternatives`: a list of sufficient sets of spans. Each span requires
`document_id`, `source_text_sha256` (original inventory text), and Unicode character
`start`/exclusive `end`. The tool binds spans to source bytes and subjects; it
cannot authenticate reviewer independence or semantic sufficiency. Model-proposed
labels stay explicitly labelled, never silently become independent review.

The v2 diagnostic reader validates the actual rendered source text, role/date and
exact span offsets against this case-visible source version. Conflicting versions
under one document ID are rejected. Unknown renderers or insufficient span traces
produce unknown coverage instead of an ID-only positive. Diagnostics use a
coherent read-only snapshot: no lock file is created and missing failure mirrors
are derived only in memory. Ordinary writer recovery remains a separate action.
`arm_row_counts`, `canonical_request_counts` and actual attempt counts are separate.

The ledger distinguishes source coverage in retrieval and context from causal
attribution. Joint evidence requires all documents in at least one sufficient
alternative. Missing annotations remain unresolved. A wrong answer despite source
documents in context is **not automatically a generation error**. Quote-control
runs mark extraction/update as not run. Static QA has no measured participant
burden; time and repeat-question outcomes remain null.

## Review fixes and frozen-run migration

The follow-up [review response](../../docs/review-response-20261105.md) records
F01–F03 and L01–L07 separately from benchmark effectiveness. Patched objective
protocols are v2.1 / v3.1; the extraction stage now uses a complete visible-history
identity. Only cases with identical subject, source versions and complete visible
history may share extraction results. Compiling by final cited source ID alone is
not sufficient to establish temporal isolation.

Use a fresh directory and the newly pinned executable for new paid runs. Preserve
the old protocols, source snapshots and results under their original Git SHAs;
do not edit their hashes or silently resume them with patched code. The earlier
61-call pilot results remain unchanged, and the synthetic review counterexamples
do not establish that those specific pilots leaked future information.
