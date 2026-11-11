# Secondary response-format diagnostic declaration

Declaration recorded at 2026-11-11 18:24:59 UTC (12 November 2026, 02:24:59
UTC+08). Reader generation had already begun. The motivation was aggregate
output-token lengths only; no actual response text or gold score had been
inspected. This is a blinded addition during the run, not the original
pre-generation primary endpoint.

The frozen strict JSON scorer, reader, source cases and targets remain unchanged.
The separate `audit_open_reader_response_format.py` runs only after every
planned answer has a completed, authenticated canonical receipt and the strict
primary score file exists. Its expected file SHA256 is supplied explicitly.
It reports all rows, preserving their original answer text for audit.

Fixed recovery rules:

1. Recover exactly one JSON object containing only the key `answer`. Reject
   unparseable brace objects, duplicate keys and multiple JSON objects,
   including repeated identical ones.
2. Do not guess a number from free text or extract an inner answer from a valid
   enclosing JSON schema. Reject unsupported scalar types.
3. Normalize only canonical unsigned integer strings and the exact
   case-insensitive word `unknown` (outer whitespace allowed) to integer and
   null respectively. Do not normalize category spelling or capitalization.
4. Reject example-like exterior text, explicit negation/incorrectness outside the JSON and
   exterior numeric/status indicators conflicting with the recovered typed value.
   This conservative rule may leave
   otherwise correct reasoning unresolved; it never uses the expected answer.
5. Keep the strict primary result unchanged. Report recovered typed-value
   correctness and unresolved formatting separately. Never describe a difference
   caused only by wrapping or scalar type as established memory comprehension.

After goal-independent parsing, a separate scoring classification compares the
typed value with the frozen reference: asserted non-null when the reference is
unknown is `unsupported_assertion`; null when a known value is required is
`incorrect_abstention`; other unequal values are `wrong_value`. A correct value
recovered from a primary format/type failure is `recoverable_format`; unresolved
parsing remains `unresolved_format`. Every original primary reason is retained.
These categories are descriptive mechanism diagnostics, not clinical judgments.

Source and task gates, canonical ledger/cache, full answer/completion hashes,
primary score bindings and frozen target hashes are checked. No model call,
professional review or clinical-quality claim is created by this diagnostic.

`--planner-path` explicitly selects the frozen preparer/scorer checkout for
each compared product version. Its complete code identity must match the
prepared protocol. Use a fresh subprocess for each version; do not copy this
helper into an old checkout or change its frozen source files.
