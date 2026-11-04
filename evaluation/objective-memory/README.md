# Objective memory comparison

This is a separate evaluation from the frozen LoCoMo / LongMemEval v2 run. It
uses released answers and fixed deterministic scoring rules. It does not use an
LLM judge as the primary outcome, and it does not measure CBT treatment benefit.

The main question is whether managed memory adds value over retrieval from the
**same original history**, rather than whether more history helps a reader.

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

Use the hosted Colab notebook for full data and API work. Start with a finite
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
