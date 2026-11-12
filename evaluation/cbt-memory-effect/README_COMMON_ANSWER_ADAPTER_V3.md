# Prospective output adapter v3

`common_answer_adapter_v3.adapt(text)` receives only the complete raw response.
It accepts a single-key `answer` JSON object, a bare integer from 0 to 9999,
a bare or JSON-quoted fixed lowercase status, bare `null`, or exact lowercase
`answer:` followed by one such scalar. Outer whitespace is stripped for parsing.
The 37 status strings are fixed in `STATUSES`; no question-specific domain is used.

It rejects prose, fences, multiple values, extra/duplicate keys, booleans,
floats/exponents, negatives including `-0`, leading zeros and unknown statuses.
Quoted `"null"` remains an unsupported string. Prefixes cannot contain an object
or another prefix. Standard JSON escapes in strings and keys retain JSON meaning.

The result contains `accepted`, `raw_response_text`, `raw_response_sha256`,
`canonical_response_text`, `canonical_response_sha256`, `rule` and `reason`.
Accepted values become compact `{"answer":...}`. Rejections use
`rule="adapter_rejected"`, an explicit reason, and `None` canonical fields;
they never become `{"answer":null}`. Raw text and its UTF-8 SHA remain unchanged.
Callers must supply a UTF-8-encodable string. `identity()` returns version,
code SHA, policy SHA and the complete fixed policy.

This adapter changes neither logits nor factual values, reads no question, arm,
gold, state or sources, and establishes no factual correctness. A new runtime
must freeze its identity and declare any adapted metric prospectively. Existing
v2 archives and scores stay unchanged. Known v2 free outputs are not independent
validation of v3; the following invented fixtures test parser behavior only:

```bash
python -m unittest discover -s tests -p test_common_answer_adapter_v3.py -v
```
