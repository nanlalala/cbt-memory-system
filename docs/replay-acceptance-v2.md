# Replay acceptance coverage after L02 / L03 review

The v2 controlled replay retains the original v1 fixtures unchanged. Its separate
`control_events_v2.json` and `control_expected_v2.json` add a fixed 12-record
retrieval scenario: ten equally relevant sources exceed `top_k=3`, two unrelated
sources remain eligible, a specific query has one hit, and an unrelated query has
zero hits. Both arms receive the same explicit synthetic events and policy.

The final rendered memory text is decoded and hashed against the independent
ledger. ID tags alone no longer establish source inclusion. The ledger also
binds rendered `type`, `observed`, `confirmation` and `data_only` attributes.
Escaped `<`, `&` and other source characters are checked as their exact original
text. A new `context_source_integrity` stage separates body/version integrity
from ID tracing and storage integrity.

The targeted replay suite contains 21 tests. The genuine raw-history and managed
pipelines each pass 26 query assertions and 56 operation assertions. Mutation
checks confirm that returning every record, ignoring the question while obeying
top-k, replacing all final prompt bodies while preserving IDs, and changing
confirmation metadata make acceptance fail.

These are deterministic software checks. They do not evaluate natural-language
extraction, final model responses, participant effort or clinical benefit.
Previously reported v1 counts remain historical results of their frozen code and
fixtures; they are not relabelled as v2 outcomes.
