# Memory Schema 2 contract

`memory_contract.construct_memory` is the version-aware constructor used by
Python, JSON ingress, extraction and the validation CLI. The new schema does
not coerce IDs, booleans or numeric strings and rejects unknown top-level and
source-evidence keys. Direct v1 `.validate()` and v2 construction execute the
same structural, format and shared semantic rules.

## Version boundaries

- A `1.0` record is read as `MemoryRecord` with `schema_version == "1.0"`.
  No classification, source role, source time, permission or confirmation is
  invented. `legacy_metadata(record)` explicitly reports unknown provenance
  and four unknown permissions without modifying the original record.
- A `2.0` record is a `MemoryRecordV2`, retaining the existing record attributes
  and adding content classification and provenance metadata. It is also accepted
  by `MemoryRecord.from_dict` through explicit version dispatch.
- `split_legacy_memory` is an explicit migration operation. It returns new
  candidate/unconfirmed outputs with source-to-output mappings. It never copies
  a confirmation to altered content and does not update the database.
- `downgrade_memory` refuses any downgrade that would discard classification,
  role, known permissions, support information, spans, task identity or migration
  mapping. It does not silently drop v2 fields.

## Classification and permissions

The only assigned content classes are `fixed_historical` with `F1`–`F4` and
`mutable_state` with `V1`–`V5`. Pending classification has null class/category;
it is not a third content class. A source quotation, structured record, summary
and index are representations rather than additional content roots.

Every v2 JSON record explicitly supplies `source_role`, nullable `source_time`
and the complete `permissions` object:

```json
{
  "save": true,
  "long_term_use": true,
  "response_external": null,
  "extraction_external": false
}
```

Only exact boolean `true` grants a permission. Null means unknown and does not
grant access. Missing fields, `1`, `"allow"` and `"unknown"` are not alternate
representations of the contract.

## Source binding does not establish meaning

`claim_bindings` identifies each necessary claim by `claim_id`, `field`, JSON
scalar `value`, `turn_id`, Unicode `start`/`end` offsets and exact `quote`.
The offsets use a half-open interval and must match the quote length. Extraction
also checks the quote against the actual cited user turn. General deserialization
cannot independently authenticate an external source and does not claim to do so.

`semantic_support` is separate metadata with `status`, `method` and `version`.
Verified statuses require method/version and claim bindings. These serialized
labels are not authority to use a current state: product eligibility must still
be computed from the trusted source, deterministic rules, current permissions
and state version. Evaluator labels are never fed into this constructor to
repair a product projection. V5 model interpretations remain inference
candidates and cannot become active factual states.

`parse_extraction_output(..., schema_version="2.0", permissions=...)` uses this
constructor. `ConversationTurn.source_time` carries optional system-supplied
source time. Model-created candidates remain unconfirmed, with semantic support
`not_checked`, even when source spans match exactly. The legacy extraction API
keeps its version and its separate unconfirmed-candidate contract.

## JSON validation

JSON clients must enforce formats and call the shared semantic validator:

```python
from memory_contract import validate_json_memory
record = validate_json_memory(payload)
```

The bundled evaluator supports only the closed keyword vocabulary used by the
checked-in schemas and rejects unsupported keywords, including nested ones.
It is not a general JSON Schema implementation. Cross-field time ordering,
payload requirements, strict count/offset types and migration relationships
are evaluated in shared semantic validation after structural/format checks.

`tests/test_memory_contract_v2.py` includes an optional differential check using
`jsonschema.Draft202012Validator` with `FormatChecker`, followed by the same
semantic rules. If the reference library is absent, that comparison is explicitly
skipped; all product validation remains available without a new dependency.

These are engineering contracts, not evidence of improved CBT dialogue or
clinical benefit. Frozen runs remain bound to their original code and schema.

Task and goal state in v2 is the separate multi-axis task projection. V2 goal
and assignment source records require a description, not a single status word;
an optional legacy status remains source metadata. The constructor does not
turn such a label into a current-state update or erase known progress.
