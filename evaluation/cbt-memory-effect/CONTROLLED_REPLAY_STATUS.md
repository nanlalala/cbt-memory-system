# Controlled-module replay and ordinal-only research boundary

`controlled_replay_adapter.prepare_contexts` replays released source records
without asking a model to answer any historical turn. It uses the current v2.1
durable source store, explicit task rules/projector, bounded memory worker and
manager, original-source retriever, semantic candidate pointer channel and exact
outbound evidence packer. Assistant turns remain assistant evidence; neither a
matching quote nor a source-only worker makes a record user-confirmed.

All conditions share the same eligible original-source pool, current
question, response system prompt, output ceiling and complete escaped request
character ceiling. Raw-history retrieval uses the current original-source
retriever at the same `top_k=6`; managed adds only the actual product candidate
and verified state channels. The short-term secondary comparator restricts that
retriever to the last four released turns. It is reported as window-scoped
retrieval, not a full-history baseline. The same pure natural-check policy is
applied to each condition's available view. Proposed checks have not been shown
to anyone and cannot establish an observed burden reduction.

The frozen `raw_relevance_rank_then_semantic_supplement-v1` packing policy assigns
original quotes priority by their actual retriever rank in every arm, including
managed's shared raw channel. Source ID alphabetic order cannot replace that
ranking when the budget is tight. Semantic-only quotes follow all raw-channel
candidates; they have no bonus that can displace a higher-ranked original quote.
This is a declared conservative research selection policy, not a change to the
production evidence packer. Dated rule-verified task fields retain their product
priority separately from historical quotations.

The adapter prepares contexts and offers separate durable send-admission helpers.
It never calls the response provider, never
reconstructs an unobserved historical proposal/check display, never reads a
scoring ledger, or sends a received body without its final publication check.
Callers must freeze the real provider metadata and use durable generation/version
admission immediately at the actual send boundary. Score
the complete displayed response plus any actual check, not just its base text.

## Explicit clock modes

`ReplayConfig(clock_mode="dated-sources-v1")` is the default. It retains the
product's explicit date, task and period rules. Undated input is rejected before
database creation; no missing date is filled from today's date.

`ReplayConfig(clock_mode="ordinal-only-v1")` is a separate research protocol for
CACTUS-style null dates. It explicitly requires `calendar_time_available=false`,
null source dates, released increasing `source_order` values and an explicit
current cutoff order. A fixed epoch plus this order encodes the store's required
timestamp internally. The encoding is not a source date, a reported elapsed
duration or an inferred appointment time. It is never rendered to the model.
The standard worker's extraction payload is already only turn ID/role/text and
does not contain the internal timestamp.

In ordinal mode, the adapter skips task event/projection creation and all
calendar-dependent refusal/expiry/current-week interpretation. It cannot produce
structured current task-state claims. Candidate retrieval reuses the actual
manager/store and retriever over unconfirmed candidates, but removes calendar
validity fields only from the temporary indexing view; persisted candidate
records remain unchanged. Relevance still only locates currently permitted
original source quotes. The final wire replaces each evidence `source_time`
metadata field with its original `source_order`. Original quotation text,
including any dates the speaker actually wrote, remains byte-for-byte intact.
The final escaped full request is counted again against the same character cap.
Initial packing is conservative because it includes internal span metadata;
this is a maximum-character contract, not exact token matching or a claim that
every unused character has been optimized away.

Ordinal comparisons evaluate research semantic-index/quote memory over CBT
dialogue order. **They are not the full online product treatment**, do not test
weekly goal/attempt/due-date semantics and cannot establish treatment outcomes.
Those properties remain in separate explicitly dated fixtures/tasks.
Product-recognized explicit refusals (`never ask`, `stop asking`, `not today` and
the corresponding supported Chinese forms) block the entire ordinal case before
any database write, queue or external call. The product recognizer is used only
as a boolean gate; its mandatory internal clock argument does not create a
retained expiry or claim that an unknown refusal has expired. Blocked cases stay
in the outer experiment denominator. Assistant quotations do not become a user's
refusal. This avoids silently skipping either permanent or date-scoped controls.

## Durable send boundary

`preflight_outbound(prepared, arm, database=..., approved_root=...)` checks the
frozen body-free snapshot binding, exact serialized payload/hash, full request
cap/output ceiling, current permissions, literal evidence/role/order, current
question, withdrawal generation and any actual task versions. It makes no call.

`begin_response_attempt` performs that recheck and the real durable
`begin_attempt` under one immediate transaction. Exact identical request payloads
share one canonical research key across conditions; an unfinished admitted send
cannot be resent. Response-attempt limits are separate from unused extraction
allowance. The caller should send only a new non-replayed admission. The
research route requires ordinary retained/use/external eligibility for every
source, including the current question, and grants no ephemeral-current bypass.

`finish_response_attempt` uses the product's current-generation/dependency
publication check. A withdrawn/stale response is not publishable and its body
must be discarded by the outer journal. This adapter retains no answer bodies
or complete outbound cache in its extra snapshot table; only hashes, IDs,
limits and control metadata are stored there. Missing usage remains unknown.
All network attempts/costs and any eligible response-body retention must still
be captured by the separately frozen outer experiment runner.
An unresolved extraction attempt also stops prefix replay before any additional
background calls; it is not retried or silently replaced with a new job.

Explicit local deletion controls and risk-triggered normal paths require their
own live/local-control route and are blocked by this narrow context adapter.
Current no-save, refusal and external-transmission controls cannot be bypassed by
its prepared outbound path. All blocked cases need denominator/status entries
in the outer experiment; they are not successful answers.

## Current evidence

The associated tests use tiny invented dated and ordinal cloud fixtures. They
verify exact source roles/text, no prefix answer calls, candidate nonconfirmation,
shared original-source statistics, complete request budgeting, no-save/risk
boundaries, no internal dates on the ordinal wire, no ordinal task-state claims,
durable canonical admission, response-attempt caps and post-send withdrawal
discard. These are software checks.
They are not natural-language accuracy, capacity, independent CBT ratings,
participant observations, or clinical results.
