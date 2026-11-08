# Review remediation: controlled memory v2.1

Review baseline: `c757b0d8dca89b39e996fa16c9c40bfc99ec0d1a`.
The review supplied by the project owner identified 16 findings: 15 product
or API defects and one engineering-acceptance blind spot. Its probes and logs
were described in the supplied text but were not included in the attachment.
The fixes below were checked with new independent reproductions in an isolated
Linux cloud container, using fictional data, temporary file SQLite, and offline
providers. No real model/API execution or user-computer access was used.

## Finding-by-finding changes

| Finding | Implemented change | Regression evidence |
| --- | --- | --- |
| F01 | Durable revocation erases same-database manager replicas in its transaction; get/list also apply the persistent ledger to pre-existing leftover copies. | `test_review_durable_fences.py`: source/span/object, transaction interruption, reopen, cross-user and non-overlapping content |
| F02 | Complete predicate polarity and negative complements are checked before current-state admission; ambiguous wording stays a quotation without reversing it. | `test_review_task_semantics.py`: six supplied compound cases, genuine negative choices, public validation and chat/reopen |
| F03 | Unknown source IDs acquire a persistent source/stop-use barrier; future submit/use/admission cannot bypass it. | Durable regressions: both scopes, reopen and two connections |
| F04 | Bounded unpunctuated interrogatives do not assert current state. | Task regressions: English inversion, Chinese A-not-A, with/without punctuation, positive assertions |
| F05 | An unqualified same-task/period/day report reuses a unique existing physical attempt, including an existing second attempt; multiple identities remain unresolved. | Task regressions: second then unqualified, genuine first/second, multiple-ordinal ambiguity and reopen |
| F06 | Cumulative projections and correction targeting use coverage intervals; receipt time governs visibility rather than replacing current progress. | Task regressions: late Wednesday 2 preserves Thursday 3; Wednesday 4 conflicts; explicit correction and cutoff |
| F07 | Requests persist the versions of task objects actually represented in the outbound package; admission, finish and cached-body access recheck them. Only dependent caches are invalidated. | `test_review_dialogue_fences.py` and durable tests: correction before admission/after handoff, same as_of, unrelated-object control and old-schema migration |
| F08 | Bare yes consumes only the top-level proposal witness calculated over all live proposals. Explicit task/period parameters cannot narrow that set. | Dialogue regressions: no selector, either object, period-only and valid single proposal |
| F09 | Body-free operation receipts are separate from answer replay. Run/session and current source permissions govern cached bodies; changed explicit input fields conflict. | Dialogue regressions: new/reopened session without long-term permission, same-session retry, changed permissions/time, no duplicate provider calls |
| F10 | Semantic candidates are pointers only. Their substring locator restores the complete permitted source with exact Unicode offsets; an oversized quote is dropped whole. | `test_review_evidence_integrity.py`: negation, attribution, condition and quantity; raw miss/shared hit, tight actual message budget |
| F11 | Source hashing and PDF page parsing use the same immutable bytes snapshot; real source paths are revalidated after ranking and packet construction. | Evidence regressions: replace/delete during rank, replace during evidence-card construction, exact bytes reader and final messages |
| F12 | UI pending identity includes session/epoch and survives local receipt-render failure; a new session never automatically resends it. Replayed receipts show original identity and only authorised bodies. | `test_review_ui_evaluation.py`: completed-send and render interruptions, same/new session, permission/text changes, safety pause and receipt-only display |
| F13 | One strict RFC3339 parser is used by schema, product policy and durable ingress. Invalid explicit as_of/occurred_at is rejected before writes, queueing or external admission. Unknown source times remain unknown. | `test_review_schema_boundaries.py`: space/minute-only/empty/False rejected with zero side effects; seconds/Z/offset/fraction accepted |
| F14 | Typed case/source/mother identities from current materials and every historical exposure row form one transitive graph before split checks and exposure propagation. | `test_review_group_ledger.py`: historical bridges, one group in same split, cross-split refusal, namespace collision, ordering and blind exposure |
| F15 | Replay adds historical-only retrieval and final outbound body/value/offset/role/time assertions, with unrelated, top-k and zero-hit controls. State and quote counts are named separately. | `test_review_ui_evaluation.py`: zero retrieval, ignore-query, ignore-top-k and unchanged-ID/body-corruption mutants must fail |
| F16 | Direct MemoryRecordV2 rejects a dict source_evidence; the canonical constructor still converts JSON into a validated SourceEvidence. | Schema regressions: object/dict, dataclasses.replace, manager duplicate consumption and strict permissions/types |

Independent read-only review additionally found and corrected two integration
windows within these contracts: old caches with unknown object dependencies
must fail closed during migration, and UI pending identity must survive receipt
rendering failure as well as send failure. These are covered explicitly rather
than inferred from a green aggregate count.

## Version, persistence and replay contract

The service identity is now `controlled-memory-v2.1`; Schema 2 content classification
is unchanged. Old experimental protocols, hashes, data and replay records are
preserved. A changed protocol/code/provider requires a new run identity; do not
resume an old experiment directory with changed code. The additive storage
migration authenticates the existing independent control anchor first. Old
response caches whose task dependencies were never recorded cannot be treated
as known empty dependencies; receipts, attempts, usage and budget counts remain.

The v2.1 replay uses five fictional software scenarios, 22 stage assertions and
six local mock-provider calls, with zero real API/model calls. The previous
four-scenario/17-assertion records remain historical records at their own SHAs.
A retrieval mutant failing an engineering assertion is not a clinical endpoint.

File persistence still depends on a separately retained current control domain
and an approved cloud location. Whole-domain rollback, external copies already
received by a provider, arbitrary external source writers after a completed
knowledge snapshot, actual hosted-Colab UI and production recovery remain
outside these local assurances. K0 remains default; the seven old knowledge
sources remain disabled. No teacher, therapist or participant review is created.

## Effect evidence

This patch is a repair of counterexamples and engineering acceptance. It does
not rerun MemoryAgentBench or PersonaMem, change their historical scores, or
establish improved CBT process, user effort or symptoms. The positive and
negative benchmark results keep their earlier executable identities. Answer-level
semantic stage attribution remains unresolved without independent annotation.

## Recorded verification

The final isolated-cloud run executed **1,046 tests: 1,046 passed, zero failed,
zero errors and zero skips**, with the optional independent JSON Schema reference
validator enabled. Runtime was 51.100 seconds. Compilation, the existing scripted
memory demo and both pinned historical verifiers passed. The first full run
(1,042 tests) exposed one real concurrent-open checkpoint race; its failing log
is retained separately, and four deterministic concurrency regressions were added.
The validation summary records log hashes. Published CI and the fresh replay
are linked in the release follow-up; they are separate from these local results.
