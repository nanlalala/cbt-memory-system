# Controlled memory replay

These are six **synthetic software acceptance scenarios**, not a clinical dataset
or evidence that memory improves real conversations. Their expected states are
specified in `control_expected_v1.json`, separately from the input events in
`control_events_v1.json`. The evaluator reads the fixed ledger; it does not ask
the system under test to write its own answers.

Run this small dependency-free check from the repository root:

```bash
python evaluation/objective-memory/replay_memory_controls.py \
  --output evaluation/objective-memory/control_replay_v1_results.json
```

The published run uses the real SQLite store, Memory Manager, lexical memory
retriever and context builder. No model, paid API, GPU or full benchmark download
is involved. The output includes fixture, ledger and implementation hashes.

| Family | Fixed expectations | Query checks per arm |
| --- | --- | ---: |
| Goals | An unconfirmed proposal stays hidden; confirmation changes the current goal; budget-dropped items do not appear in the prompt trace. | 5 |
| Homework | No completion is invented without a report; an explicit partial report stays partial. | 3 |
| State | A confirmed newer state replaces the current value while preserving each source turn. | 2 |
| Corrections | A pending correction does not replace a fact; a confirmed correction does; a stale target is rejected. | 3 |
| Expiry | Visibility ends at the exact exclusive expiry boundary, even before cleanup; cleanup cannot reintroduce a hard-deleted record. | 4 |
| Deletion and isolation | User scope and default restricted filtering hold; explicit access is separate; another user cannot delete the record; deletion remains final. | 6 |
| Total | Six scenarios, not 23 independent users. | 23 |

Both arms also undergo the same 32 control operations, including two expected
rejections. Each query checks storage state and declared fields/source IDs,
lifecycle and access eligibility, retrieval, budget drops, and actual complete
items in the rendered prompt. Failures are assigned to the first failing stage.
Forbidden IDs and distinctive source markers are checked against the real
context, rather than trusting a selected-ID list alone.

The raw-history baseline stores the original text and a minimal version/access
log. It receives exactly the same typed source metadata and explicit
confirmation, correction, expiry, deletion and authorization events. It is
expected to implement those ordinary controls too. Both arms share retrieval
and rendering so this comparison tests control compliance; it does not isolate
an advantage in semantic retrieval. **Both passing is a valid parity result.**

This replay does not test language-to-memory extraction, confidence calibration,
natural-language answers, real user burden, retrospective time-travel retrieval
or concurrent workers. Existing concurrency regression tests remain separate.
The typed goal/status/slot labels are supplied by the fixture author, not
inferred from a user's wording. These checks must stay separate from released
benchmark question-answer scores and from any clinical or user-benefit claim.
