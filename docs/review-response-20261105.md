# Response to the 5 November source and state review

The attached review assessed `52ea8b6d721013fb5322e2f7be7719488bc9eabf`
against main `80551b7edc0199b8870b174e3f144155b0d60c85`. Passing its existing
tests did not cover the supplied counterexamples. The corrections below address
those counterexamples; they are software checks, not new model effectiveness,
independent human annotation, user-experience or clinical results.

| Review item | Correction | Regression evidence |
| --- | --- | --- |
| F01: future explanations influence an earlier semantic index | Extraction requests and compilation bind to each complete visible history and source version; cohort-wide source unions cannot share a case's extraction output | Actual prepare → fake-provider response → compile → retrieval is identical for early-alone and early/late cases, for both as-of and released-prefix boundaries; same-ID conflicting source versions are refused |
| F02: evidence labels and coverage use global IDs instead of case/version | Labels require the current case's history digest, source bytes and spans; actual rendered prompt text, date, role and offsets are checked | Full `diagnose()` tests reject same-subject future labels, wrong hashes, conflicting same-ID versions and body replacement that retains IDs |
| F03: a read-only diagnostic repairs its input artifacts | A coherent read-only snapshot uses an existing shared lock without creating one, detects observable concurrent changes and derives missing valid mirrors only in memory | File lists and byte hashes remain unchanged; instrumented offline copies without locks and with missing/short mirrors work; contradictory mirrors are rejected |
| L01: saved response followed by failed success commit becomes unrecoverable | Provider/validation failures are separated from persistence errors in both objective runners and the index generator; a valid saved response remains an unresolved recoverable intent | Injected post-append and success-outcome errors preserve one intent, one response and one provider call; explicit recovery commits the saved response without a paid retry; malformed partial lines still fail closed |
| L02: returning all candidates passes relevance acceptance | New v2 replay adds irrelevant distractors, a zero-hit query and more relevant sources than top-k | Returning everything and ignoring the query while respecting top-k both fail acceptance |
| L03: changing prompt bodies while retaining IDs passes acceptance | Decoded final source text and positive metadata are independently bound to the expected ledger | Keeping IDs while replacing the body, type, observation time or confirmation fields fails context source-integrity checks |
| L04: explicit storage opt-outs are missed | Common Chinese storage prohibitions and normalized apostrophes are handled; manual extraction rechecks sources and excludes opt-out text from its input | Both automatic capture and later manual processing reject the reported Chinese/curly-apostrophe examples; ordinary statements about forgetting or saving money remain distinct |
| L05: explicit goal updates cause redundant checks | Explicit updates use source-bound topic/quantity matching; ambiguous or named unrelated topics cannot attach to an old goal | Running and previously unlisted activities accept an unambiguous update without reasking; unrelated-topic and ambiguous multi-goal counterexamples are exercised |
| L06: a pending check is not version/time bound | Pending checks bind record/source/content/payload/lifecycle versions, asked session/turn and expiry; stale links are invalidated | Same-ID revisions, direct store edits, delayed replies, unrelated affirmations and multi-topic responses cannot reuse an old check; a linked reply is never automatic confirmation |
| L07: refresh silently discards an unsaved edit | Drafts are retained by memory/action with source-version bindings; stale saves/confirms are blocked and discard is explicit | Editing A then operating B or sending a message preserves A; source/status changes retain the draft while blocking stale actions; deletion purges detached draft text |

The strengthened replay uses new immutable v2 fixtures. Its v1 fixtures remain
available unchanged. Normal v2 software replay has 26 queries and 56 operations
per arm; these counts are not natural-language user study outcomes.

Independent follow-up review additionally exercised unrelated quantity changes
and successive goal updates. Reported counterexamples are fixed before merge;
the final validation count and immutable executable are recorded in the PR.

## Protocol and state boundaries

- The patched quotation-index objective runner is v2.1 and the full-suite runner
  is v3.1. The visibility-isolated extractor has its own v2 stage protocol.
- New paid work must use new directories and the fixed executable. Existing v1,
  v2 and v3 artifacts are neither rewritten nor relabelled. Their original Git
  source snapshots remain the compatible route for historical recomputation.
- The review's synthetic temporal counterexample is not proof that the earlier
  four-question pilot actually leaked future data. No trial is relabelled on that
  basis, and none is rerun merely to replace a negative outcome.
- Twenty-four-hour check expiry and the seven-day repeat-check cooldown are
  engineering settings, not calibrated clinical thresholds.
- The MAB source-grammar version view remains a synthetic benchmark adapter.
  It does not certify a general CBT semantic extractor or user-confirmed state.
- Full-scale model results are still pending. The cloud-browser control transport
  failed during attempted development preparation; no new v3 provider calls were
  started. These repairs and their tests do not turn that interruption into an
  effectiveness result.
