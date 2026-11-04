# F01: isolate semantic extraction by complete visible history

The previous extraction stage could combine sources from early and late questions
into one model request. Filtering the final candidate by its cited source ID did
not remove future explanations that had already affected the generated summary.

The new `source-only-semantic-extraction-visible-history-v2` protocol constructs a
visibility identity from the subject and the complete ordered source versions.
Only cases with identical visible histories share extraction results. Request keys
bind that complete identity as well as the actual model messages; the prompt hash
still describes the actual message bytes. Compilation selects outcomes from the
case's own visibility identity before validating cited source IDs and quotes.

The shared source ledger remains a deduplicated storage artifact, not a shared
model input. Each request is rebuilt and checked against its exact case-history
mapping when loaded. Same-ID sources with conflicting versions are rejected rather
than overwritten. Registered Persona prefixes are reconstructed using their source
context and exclusive boundary, then checked against the frozen history hash.
Unreconstructable boundaries fail closed.

Two end-to-end synthetic regressions use a future-sensitive fake provider: the
early quote `I wrote it down.` gains a `thought record` summary only if the future
explanation is visible. Both date-based and exclusive-prefix tests confirm that
early-only and early-plus-late preparation produce identical early requests,
candidate summaries and retrieval results. Late results may differ, but cannot be
compiled back into the early index. Additional cases check exact-history sharing,
same-ID version conflict rejection, and refusal to resume the old union-scope
protocol.

These are local software regressions in the independent cloud environment, with
no real provider calls or full dataset loading. They do not establish that the old
four-question pilot was contaminated, nor do they measure model effectiveness.
Keep old v1 stages with their original executable SHA. Any affected extraction and
QA rerun requires new directories and frozen protocol identities; do not rewrite
old results.

## Related index-response persistence correction

The source-index generator also separates provider/response-validation errors
from response persistence. If a valid raw response has been appended but its
flush reports an error, or if the later success-outcome append fails, the intent
stays unresolved with its valid response and token usage preserved. No conflicting
technical-failure outcome is appended. Normal continuation does not retry that
intent; explicit `use_recorded_response` recovery can commit the saved response
without another provider request. Fault injection covers both boundaries, with
one paid-attempt fixture and one valid response retained throughout. This is a
software persistence fix, not an effect result or a claim about all other runners.
