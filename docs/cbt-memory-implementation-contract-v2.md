# Controlled CBT memory implementation contract v2

This page describes the implemented service APIs and their remaining limits.
The new core is an explicitly selected, file-backed research prototype. It
reuses the existing manager, SQLite store, retrieval and guarded provider rather
than introducing another agent framework. It is not a production deployment,
an independently evaluated semantic-memory system, or evidence of CBT benefit.

The entry point is
[`ControlledDialogueSession`](../modules/agent-integration/controlled_dialogue.py),
separate from the retained legacy `DialogueSession` and frozen research contracts.
The current UI factory selects this controlled class only after explicit storage,
user and run configuration; an injected legacy session retains its previous UI.
The separate [Colab entry](../app/Controlled_Memory_V2_Colab.ipynb) starts with paid
calls disabled. The [finite CLI](../app/controlled_memory_demo.py) makes no real
model/network call. This change does not execute a paid experiment.

## Implemented files and APIs

| Responsibility | Implementation | Entry points |
| --- | --- | --- |
| Canonical records | [`memory_contract.py`](../modules/long-term-memory/schema/memory_contract.py), [`memory_record_v2.schema.json`](../modules/long-term-memory/schema/memory_record_v2.schema.json) | `construct_memory`, `validate_json_memory`, `legacy_metadata`, `split_legacy_memory`, `downgrade_memory` |
| Existing record and CLI ingress | [`memory_models.py`](../modules/long-term-memory/schema/memory_models.py), [`validate_memory_data.py`](../modules/long-term-memory/schema/validate_memory_data.py), [`extraction_contract.py`](../modules/long-term-memory/extraction/extraction_contract.py) | Version dispatch, canonical validation and `parse_extraction_output(..., schema_version="2.0")` |
| Transactions and durable control | [`sqlite_store.py`](../modules/long-term-memory/storage/sqlite_store.py), [`durable_memory.py`](../modules/long-term-memory/storage/durable_memory.py) | Explicit `savepoint`, `submit_source`, `claim_job`, `put_effect`, `begin_attempt`, `finish_attempt`, `revoke`, `pause_user/is_paused`, maintenance/restore APIs |
| Rule-supported task views | [`task_projector.py`](../modules/long-term-memory/lifecycle/task_projector.py), [`task_input_rules.py`](../modules/long-term-memory/lifecycle/task_input_rules.py) | `verify_claim_binding`, `validate_task_event`, `project_task`, dated-attempt and correction rules |
| Stable original-source retrieval | [`stable_retrieval.py`](../modules/long-term-memory/retrieval/stable_retrieval.py) | `StableSourceRetriever.search` |
| Minimal outbound evidence | [`evidence_bundles.py`](../modules/agent-integration/evidence_bundles.py) | `EvidenceSpan`, `EvidenceClaim`, `EvidenceBundle`, `build_outbound_messages` |
| Common natural-check policy | [`natural_memory_policy.py`](../modules/agent-integration/natural_memory_policy.py) | `permissions_for_input`, `refusal_for_input`, `bound_task_choice`, `decide_check` |
| Knowledge channel | [`knowledge_adapter_v2.py`](../modules/agent-integration/knowledge_adapter_v2.py) | `KnowledgeAdapter.retrieve`, `KnowledgeSelection.has_support` |
| Source-first orchestration | [`controlled_dialogue.py`](../modules/agent-integration/controlled_dialogue.py) | `ControlledDialogueSession.open/send_message`, `propose_task`, `delete_content`, `run_pending`, `snapshot`; `BoundedMemoryWorker.run_batch` |
| Actual outbound-provider check | [`colab_provider.py`](../app/colab_provider.py) | `preview_request`, versioned extraction instructions and the existing guarded send callable |
| Prospective evaluation | [`v2_followup_contract.py`](../evaluation/dialogue-benefit/v2_followup_contract.py) | Material grouping, protocol freezing, observation statuses and offline adoption-gate validation |

## Explicit storage and frozen limits

`ControlledDialogueSession.open(database, approved_root=..., user_id=..., ...)`
requires an absolute, nonsymlink, file-backed SQLite path under an explicitly
selected existing approved root. The durable store rejects an in-memory
prototype; `allow_memory_for_tests=True` is a separate storage-test exception.
An approved path is a caller/operator configuration, not a guarantee that a
Colab VM will survive or that its filesystem is backed up across runtimes.

The defaults below are actual implementation values. They are frozen with the
run's protocol and cannot be silently enlarged under the same `run_id`.

| Limit | Default | Meaning |
| --- | ---: | --- |
| `DurableLimits.max_run_calls` | 12 | Durable admitted external attempts across the run, including response and extraction |
| `DurableLimits.max_session_calls` | 12 | Durable admitted attempts for one user/session in that run |
| `DurableLimits.max_queue_length` | 32 | Queued, claimed and sending jobs counted against the run's queue ceiling |
| `DurableLimits.max_batch_size` | 4 | Maximum size permitted for an explicit worker batch |
| `PrototypeLimits.max_worker_jobs_per_turn` | 1 | Maximum jobs scheduled by the automatic single-worker turn path |
| Background executor workers | 1 | One finite background worker with its own SQLite connection |
| `PrototypeLimits.max_outbound_chars` | 18000 | Complete escaped outbound JSON request-body ceiling, not a token count |
| `PrototypeLimits.max_response_tokens` | 4096 | Requested output ceiling checked against the live provider |
| `PrototypeLimits.max_session_turns` | 24 | Configurable bounded runtime turn list, not a long-term content category |

`run_pending()` collects an already scheduled result and executes one further
bounded batch; it does not loop until the queue is empty. A full queue retains
an otherwise permitted source and returns `unprocessed_queue_limit`; it does
not extend the queue or budget. Starting a new chat session does not reset the
run cap, erase outstanding jobs, lift a refusal, or release a safety pause.

A persistent user-level `pause_user` gate serializes with ordinary source,
worker, effect and send admission. Old connections and a new run/session cannot
bypass it. Pausing also changes the control generation, so an already in-flight
normal response is discarded at publication; explicit local deletion remains
available. There is no automatic unpause API. This demonstration matcher is not
clinical triage.

The provider retains its own existing call/output reservations and exact model
identity checks in addition to the durable admission cap. A live controlled run
requires a clean pinned checkout. Its protocol records code and prompt hashes,
provider settings, the shared clarification-policy identity, knowledge policy,
permissions, timezone and limits. An unknown or missing usage report remains
unknown; the output reservation is not a billing total or monetary guarantee.

## One record constructor and two content roots

The canonical constructor validates both Python and JSON ingress without
coercing numeric strings, integer IDs or truthy permission values. Structural
and format checks precede shared semantic checks for classification pairs,
business payloads, timestamps, duplicate IDs and related fields. The bundled
JSON evaluator implements the closed vocabulary used by the checked-in schemas,
not an arbitrary JSON Schema engine. See
[`SCHEMA_2_CONTRACT.md`](../modules/long-term-memory/schema/SCHEMA_2_CONTRACT.md)
for exact fields and the optional reference-library differential check.

Assigned content has exactly the requested roots: `fixed_historical` with
F1–F4, or `mutable_state` with V1–V5. Pending classification has null class and
category; it is not a third content type. Source quotation, state, summary and
index describe representation or management responsibilities. The existing
business `memory_type` values remain interoperable rather than being equated
one-to-one with fixed/current content.

Version 1.0 is read as 1.0. Missing source role, source time or permission stays
unknown; the implementation does not invent an upgrade, consent or confirmation.
Explicit split migration returns input/output mappings and unconfirmed outputs.
A downgrade that would discard v2 semantics is refused. These are migration
primitives and validation contracts, not a completed automatic migration of
every existing external database or old experiment archive.

## Product task state is a multi-axis projection

Task events retain `goal_id`, stable `task_instance_id`, `period_id`, source and
event identities, state versions, and physical `attempt_id` where justified.
`project_task` produces independent acceptance, participation, outcome and
due-review axes, plus progress fields such as target count, reported count,
attempt IDs and count conflict. Agreement, one of two attempts, a pause and an
unknown outcome can coexist. A pause does not erase progress, and reaching a
deadline only adds the due-review condition.

Explicit corrections append a new event and change the effective projection;
they do not rewrite the original attempt source. Re-reporting one physical
attempt adds source evidence without incrementing its count. Cumulative reports
replace the covered total rather than being summed as increments. Dated late
reports select the corresponding historical weekly period. Reassignment in a
new week uses a new task instance rather than attaching progress to a same-name
task. Deletion/stop-use barriers block an object and prevent a corrected value
from exposing an older superseded value again.

The chat path now calls the deterministic date/attempt and explicit-correction
helpers. An explicitly dated unqualified task/day reference obtains a stable
attempt identity; an ambiguous day with several known attempts is not resolved
by counting another report. Calendar-day references carry day precision and a
sorting boundary, not an invented exact occurrence time. Explicit dated ordinals
identify one physical attempt through a separately verified ordinal binding;
“second time” is not treated as a completion count of two. Undated ambiguous attempts, contradictory quantities and unsupported language
fall back to original quotations or an unresolved field. A numeral in "second
attempt" is not automatically two completions.

Current-use eligibility is computed by the product validator from actual
source spans, permitted roles, object identity, quantity/negation/choice rules,
time and current permissions. A quotation match alone does not verify meaning.
General model extraction remains `candidate/unconfirmed`, including V5 model
interpretations, and cannot update factual task axes through assessor labels
or serialized support flags. The initial natural-language rules cover a bounded
set of explicit Chinese/English forms and activity labels, not unrestricted
semantic understanding or clinical formulation.

## Four permissions and source-first receipts

Each source has separate `save`, `long_term_use`, `response_external` and
`extraction_external` permissions. Only literal `True` grants an operation;
false and unknown are not authorization. The controlled fictional-prototype
configuration currently defaults these four axes to true. Operators must supply
the permission scope actually authorized for their use; this default is not a
participant consent mechanism and does not turn unknown legacy permissions into
true. Runtime refusal/opt-out instructions can narrow the configured scope.

`send_message` evaluates risk and local controls before ordinary source commit,
queueing or external response/organization. A risk-paused ordinary path saves no
normal source, creates no extraction job and calls no provider. Explicit scoped
local deletion can still be processed without releasing that pause. An input
such as "this week pause, do not save this message" can be acknowledged in the
current reply without retaining its text, inferred pause, summary or generated
paraphrase for later context or organization. An existing current object may
receive a body-free stop-use barrier so the older contrary state does not return.

Ordinary source admission and any source-supported task/control updates happen
before reading state for the answer. Source and queued-job creation share one
transaction. The response distinguishes `source_commit`, `answer_status` and
`organizing_status`: a successful answer does not claim a failed write succeeded;
an answer failure does not roll back an already committed source/control; a
queued job is not a successfully organized memory. `input_id` and persistent
effect/request identities provide idempotent replay rather than implicit retries.

## Transaction and withdrawal fences

The base store retains `BEGIN IMMEDIATE` admission and now provides explicit
`savepoint()` scopes. Nesting `transaction()` alone is not a savepoint. Dependent
candidate groups are committed or rolled back together; independently retained
groups require their own explicit savepoint. Durable claims carry tokens,
effects use unique keys and `expected_version` conditional updates, and a worker
rechecks generation and source dependencies before committing.

`begin_attempt` is the serialized send-admission boundary against withdrawal.
A request already handed to a provider cannot be recalled. `sending` or
`attempt_unknown` is not automatically resent after lease expiry or a crash.
`finish_attempt` checks the generation and permissions again before publishing
or retaining a response; invalidated output and caches are discarded.

Withdrawal distinguishes representation-only deletion, source/span deletion,
content-object deletion and stopping use. Deleting only a summary is not a
claim that the underlying content was forgotten. Object withdrawal retains a
body-free blocked barrier, clears dependent primary payloads and invalidates
jobs/representations without reviving an older version. Receipts separately
report logical blocking, primary/source payload cleanup, and the fact that
uncontrolled external replicas have not been claimed erased.

The shared check ledger distinguishes shown, answered, declined, unanswered and
suppressed checks without storing the answer body in the ledger. Unrelated
progress does not reset a field's substantive identity. Topic pauses also filter
original quotations and semantic pointers through stable topic associations and
exact object-reference routing; pausing only structured state would be insufficient.
The default bounded language rules reject unsupported historical dates, quotations
and attributed speech as current factual updates. They preserve the source as a
historical quotation instead. This conservative scope is not general semantic
understanding; arbitrary model prose does not become a registered task proposal.

## Control anchors maintenance and restore boundary

A registered file database requires both its path binding
`<database>.control.json` and the current checkpoint in
`approved_root/.cbt-memory-control/<storage_id>.json`. The checkpoint includes
storage identity, database path, control watermark/hash and storage revision.
Missing or mismatching bindings/anchors, a stale restored DB, or failure to
persist a control checkpoint fail closed; a missing current anchor is not
recreated from an older database. Registered v2 storage also refuses ordinary
manager reads/writes without its verified durable hooks.

Use `begin_maintenance`/`close_for_maintenance` before coordinated migration,
close old connections, and retain the latest control ledger independently.
`finish_maintenance(latest_control_ledger=...)` and `open_restored(...)` reject a
missing or different latest ledger. These APIs verify the restored control
boundary; they are not a general backup uploader, source-data recovery service
or automatic replay of external deletion records.

The separate control namespace is still a local file namespace under the
approved root. If the database, binding and anchor are all rolled back together,
those files alone cannot prove that they are the latest control domain. An
independently retained latest checkpoint is required outside that rollback
domain. Likewise, a file-backed DB does not by itself provide cross-runtime
Colab persistence. Deployment must supply the approved durable location and
backup/control retention. Logical blocking and primary cleanup receipts never
claim completion of cleanup of all replicas or already transmitted data.

## Stable source spans outbound evidence and knowledge

`StableSourceRetriever` constructs lexical statistics only after current user,
source-visible cutoff, permitted scope, role and current-use/externally allowed
source filtering. Its statistics pool uses whole original sources once each;
soft category routes, selected spans and newly added semantic paraphrases do
not refit that pool's IDF. Permission or source-visibility changes can correctly
change the legal pool. Lexical and optional dense channels recall independently,
then fuse; a lexical miss is not a prefilter that prevents vector recall.

Multiple necessary spans from the same source survive as separate span hits
while the independent-source count remains one. `source_cutoff` constrains when
the report became visible, `as_of` describes state applicability, and actual
read permission is not rewound by a historical question. A late report's earlier
event date cannot cross a source-visible cutoff. The default controlled chat
path does not download an embedding model; dense recall requires an explicitly
supplied encoder.

The managed semantic channel searches its unconfirmed candidates separately and
maps hits back to original legal quotations. It does not change the original
pool's IDF or promote its generated summary into current factual context.
Minimal `EvidenceClaim` alternatives preserve the required exact spans for an
individual field. A short supported pause can be retained when a long quantity
history does not fit; the missing quantity stays unknown instead of dropping
the whole task and asking for an already supported pause again.

`build_outbound_messages` measures the complete escaped JSON payload including
all messages and request metadata. `ColabChatProvider.preview_request` includes
the actual versioned extraction instructions, settings and output ceiling and
is checked again at send. Authentication headers/credentials are not exported
in previews. The 18000-character ceiling is not a precise token budget; actual
reported usage remains separate and absent usage is not filled with zero.

`KnowledgeAdapter` keeps textbook material out of personal memory and uses a
shared versioned task/query policy across K0/K1 and memory conditions. It checks
the existing corpus/source registry, fixed bytes, approved uses, external-use
permission, chunk/page identity and relevance, with explicit no-hit and
unavailable exits. A channel without admitted chunks does not claim textbook
support. The seven old sources remain disabled. Project-authored fixture success
does not establish a usable real K1 corpus or a new knowledge-benefit result.

## Natural checks and scoped refusals

`decide_check` keys cooldown on a stable subject, field, substantive value and
uncertainty, rather than unrelated progress/version increments. It suppresses
checks after a sufficient explicit current report, irrelevant ambiguities,
another question already shown in the turn, an applicable refusal, or the
frozen seven-day substantive-field cooldown. Today-only topic pauses use the
configured local day; indefinite check refusals persist until changed. Task
rejection, check refusal and no-save permissions remain separate concepts.

`bound_task_choice` accepts a yes only after a unique valid, single-action,
single-question proposal bound to the same user, session and reply turn. Multiple
proposals, compound questions or an unbound yes do not accept any item. A valid
yes creates a task-choice witness, not `user_confirmed` old memory. These pure
functions are reusable by both raw and managed comparisons; the controlled chat
currently implements only bounded task/check routes, not every possible CBT
dialogue or a completed independent burden study.

## Evaluation and evidence contract

[`v2_followup_contract.py`](../evaluation/dialogue-benefit/v2_followup_contract.py)
freezes identities and validates observations. It does not generate data, call a
model, set product memory state or establish benefit.

The existing v1 material compiler, runner, protocols and result directories stay
under their original identities. This v2 module is not permission to relabel or
resume a frozen v1 run with changed code.

## Connected materials and exposure

Every case supplies explicit string identities for its sources, material mothers
and parent cases, plus its intended split and known exposure. Parent cases must
also be inventoried. Shared source groups, shared mothers and derivative chains
are transitively connected before a split is accepted. A historical exposure
ledger can connect additional mothers or sources; its exposure propagates to the
whole resulting group.

`freeze_materials(cases, exposure_ledger=..., exposure_ledger_complete=True)`
produces a stable identity-only manifest. Blind-test groups require an explicit
complete ledger, even when that ledger is empty. A connected group cannot cross
development, validation and blind-test splits. A newly worded derivative does
not become an unexposed test simply because it has a new case ID. The contract
rejects missing parents, cyclic lineage, contradictory exposure declarations and
duplicate IDs. It never treats the number of questions or variants as a number
of independent people.

The manifest contains no source text, questions, answers or assessor gold. Such
generation and scoring packages retain their separate access boundaries. Human
semantic labels may be used by the scorer and separately identified oracle
diagnostics; they must not initialize or repair product memory state.

## Primary comparison and common interaction policy

`freeze_protocol(config, manifest, clarification_policies=...)` freezes a
managed-memory minus competent raw-history comparison at one fixed knowledge
condition. The optional short-term arm remains a secondary history-access
comparison. This contract does not introduce a six-cell factorial sweep. At most
two named mechanism ablations may be frozen as diagnostics.

Each arm supplies the actual clarification callable as well as its identity.
`policy_identity` hashes the function source and containing source file; the full
run code manifest must also freeze imported dependencies. The declared hash must
match the actual function. A management-only estimand requires raw and managed
to share the same callable and identity, with no managed-only UX features.
Different policy functions or managed-only UX change the frozen estimand to
`package_effect`.

Sharing a policy does not require identical counts of displayed questions.
Different lawful current views can lead the same function to show, suppress or
resolve different checks. Those events still require actual display/reply logs
and independent necessity assessment. Equal answer scores also do not imply
equal prompts or shared provider requests. Canonical-response reuse remains
based on exact request identity in the existing execution layer.

K0 means no additional textbook retrieval, not that the base model knows no CBT.
K1 requires a pinned knowledge-manifest identity. That identity does not by itself
authorize the corpus: product retrieval must still pass the existing source,
permission, relevance, no-hit and page/chunk checks. The seven legacy sources
remain disabled unless their independent governance requirements are satisfied.
An in-project knowledge fixture is not a real K1 corpus.

## Separate truth execution review and cause

`score_observation` preserves these separate fields:

| Field | Meaning |
| --- | --- |
| `truth_status` | `known`, a legitimately labeled `unknown` target, or `not_annotated` |
| `execution_status` | `success`, `technical_failure`, `timeout`, or `missing` |
| `review_status` | `complete`, `pending_review`, `unresolved`, or `not_required` |
| `semantic_correctness` | An independent boolean verdict, or `None` |
| `operational_score` | A scored 0/1 outcome, or `None` while review is unresolved |
| `cause` | `unresolved` unless independent cause adjudication and evidence IDs exist |

A technical failure contributes zero under the operational definition without
inventing a semantic answer verdict. An answer awaiting independent review
remains null rather than being filled with zero. A gold target of unknown can
be answered correctly and score one. An unannotated reference is a different
state and cannot support a completed semantic verdict.

`summarize_observations` keeps all supplied planned rows in the denominator and
blocks comparative intervals while any successful answer still has a pending
or unresolved score. It rejects stored scores that disagree with their status
fields. Neither a substring match nor a model's source-use assertion resolves a
failure cause. Independent evidence is required, and oracle diagnostics remain
separate from deployed-arm scores.

## Prospective meaningful-benefit and resource gates

Every protocol is frozen before any new answer is generated. Confirmatory
offline use additionally requires independent blind groups and all of the
following caller-supplied, justified gates:

- A minimum meaningful absolute improvement, rather than any positive change.
- The independent grouping unit, interval method, level, replicates, seed and
  favorable-bound rule.
- Named critical-error categories and their maximum permitted counts.
- Total provider-attempt, token and wall-time limits covering preparation,
  failed attempts, source commit, extraction, updates, indexing, retrieval,
  context assembly and generation.

There are no default scientific thresholds. Missing gates reject confirmatory
freezing. `assess_offline_adoption` applies the exact frozen identity, requires
the group interval to favor improvement, checks the minimum meaningful magnitude,
and applies every critical-error and resource cap. Unknown token usage or
unreviewed critical-error counts keep the decision unresolved; they are not
treated as zero. A small positive point estimate alone cannot pass this gate.
Passing it describes the prespecified offline criteria only, not treatment
efficacy or a validated clinical adoption decision.

## Human design remains a different analysis

The existing four-visit protocol remains parallel randomized. Each participant
uses one assigned condition throughout the visits. The independent unit is the
randomized participant, not a task, source group, session or rating. The two
co-primary endpoints remain `task_success_fraction` and
`restricted_elapsed_seconds`, each with its original 97.5% interval rule.
For managed-minus-raw differences, success is favorable upward and restricted
time downward; the contract separately freezes a positive success lower bound
and a negative time upper bound rather than reusing one direction for both.

The new contract validates those identities rather than replacing the existing
participant-analysis implementation with the offline group helper. A future
confirmatory human configuration must justify meaningful differences for both
co-primary endpoints; it cannot choose only the favorable outcome. Human task,
burden and independent CBT-process data still have to be collected through the
actual approved research process. No clinical observations are created here.

## Historical evidence boundaries

The older main Persona quotation control and the separate PR31 management
experiment have distinct executable, protocol and artifact identities. The PR31
public figures are reported findings, not new row-by-row recomputation by this
module. Equal-score pairs do not establish shared-request counts, and connected
source groups are not independently recruited users. Existing MemoryAgentBench
and Persona results remain under their original versions, including unfavorable
findings and unresolved error attribution.

The synthetic contract tests validate grouping, freezing and status semantics.
They use no participants, models, embeddings or paid API calls and are not
evidence of improved CBT process, user burden or clinical outcomes.

## Verification checkpoint

The [cloud engineering replay](../evaluation/controlled-memory-v2/README.md)
records four fictional scenarios and 17 passed stage assertions with zero real
model/API calls. The release suite ran 953 tests (952 passed, one optional
reference-library differential skipped); the separate reference environment
passed all 21 schema tests with that check enabled. Compilation and both pinned
historical artifact verifiers passed. These checks do not establish memory
benefit, naturalness, reduced burden or CBT/clinical effectiveness.

### Automated review follow-up before merge

PR #32's later automated review identified predicate-negation errors. The fix
uses the same local predicate rule in public validation and object binding,
preserves genuine task refusal/pause, and does not infer agreement or activation
from a negated opposite. User questions are retained as quotations rather than
current assertions; the supported unknown-outcome self-report remains valid.
Original replays and protocols remain unchanged. The notebook now pins corrected
executable `9166f233e1ba0305ec1c558da77f9afade476e7d`. The follow-up full cloud
suite ran 964 tests (963 passed, one optional reference-validator skip), and
independent read-only review closed the reproduced issue. See the new replay
linked in the engineering report; no real model or participant study was run.
