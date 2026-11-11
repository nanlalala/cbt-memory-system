# Actual controlled-state results, 12 November 2026

The English state update change repaired six generated answers that had retained
an obsolete count of seven. The frozen strict response score did not establish
an advantage over original-source RAG because every response failed the required
single-field JSON interface. Both findings are retained.

## Comparison and scope

The completed grid contains 24 constructed cases: four related templates, three
variants and two languages. It is a development mechanism experiment, not 24
independent patients or a natural-dialogue clinical study. Original-source RAG
and managed memory used the same six directly concatenated recent messages,
eligible original sources and ranking, answer model, input/output limits and
permission/publication checks. Managed memory additionally supplied the actual
product's source-bound task state.

Both product versions completed all 48 logical rows. Equivalent requests were
reused with fresh source checks: the 96 logical rows required 57 unique model
generations. All 24 raw-RAG requests and responses were unchanged; nine managed
requests changed after the English projection repair.

| Product version | Raw RAG, strict correct | Managed, strict correct |
| --- | ---: | ---: |
| Before repair | 0/24 | 0/24 |
| After repair | 0/24 | 0/24 |

The outputs were prose, bare values, fenced scalars or objects with additional
fields. All ended normally at EOS. The conservative format diagnostic declared
during generation also recovered zero eligible responses. Neither scorer was
changed after observing these results.

## Individually inspectable content repairs

Post hoc inspection of the archived original text identified the following
changes. These are content examples, not a replacement primary endpoint.

| Template and language | Variant | Fixed reference value | Managed before | Managed after |
| --- | ---: | ---: | ---: | ---: |
| Target correction, English | 1 | 2 | 7 | 2 |
| Target correction, English | 2 | 3 | 7 | 3 |
| Target correction, English | 3 | 4 | 7 | 4 |
| Cumulative total, English | 1 | 2 | 7 | 2 |
| Cumulative total, English | 2 | 3 | 7 | 3 |
| Cumulative total, English | 3 | 4 | 7 | 4 |

The other three changed managed outputs rephrased an already correct practice
outcome. Full original responses and all losses/ties are retained in the private
checkpoint and accompanying results report.

## Separate evaluator and output-interface pilots

Five whole published TherapyJudgeBench dialogues were scored with the same
small open model as a calibration attempt against released clinician ratings.
Every output was an invalid 13-item array rather than the required 15-item
array; safety positions also contained six, outside their binary domain. There
were zero usable predictions for 45 nonmissing CTRS labels, with ten released
labels missing. This is evaluator failure, not a measured CBT quality score.
It supplies no R1–R5 result and no new human review.

Two new non-CBT fixtures tested a common constrained JSON decoder. Syntax was
valid in 2/2; basic factual answers were correct in 1/2. The count fixture
expected eleven and received null; the outcome fixture correctly received
unchanged. The failed count remains in the record. A tokenizer-only audit did
not support the proposed colon/null token-merge explanation. A separate
prospective decoding diagnosis must preserve these results and both fixtures.

## Actual resource accounting

| Purpose | Unique generations | Input tokens | Output tokens | Total tokens | Answer-path seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Controlled memory comparison | 57 | 100,146 | 1,047 | 101,193 | 4,057.45 |
| Released clinician-label calibration | 5 | 9,969 | 200 | 10,169 | 452.36 |
| Independent JSON/readability fixtures | 2 | 176 | 13 | 189 | 12.19 |
| Original ledger total | 64 | 110,291 | 1,260 | 111,551 | 4,522.01 |

All 64 admitted generations have published outcomes. There were no external
inference API calls, no secret access and no automatic retries. Model setup is
recorded separately. Source ingestion, indexing, organisation and packing were
not originally timed and are not assigned invented durations. Logical rows and
cross-version cache references are not added to unique generation cost.

## Archive identity

- Complete 62-generation memory/calibration checkpoint: 6,277,389 bytes;
  SHA256 `8cee7b783915c8da05f428c2c45ab5e0e1f8c9507a21c7946ebcf317f5887e33`.
  All 522 manifest-listed files passed byte-length and SHA verification.
- The two-fixture supplement closes the original ledger at 64: 30,569 bytes;
  SHA256 `6f67748b220724bb4b510092e229c42e262b458ef67896fd196633e0757174c1`.
- Complete original-byte paper evidence export: SHA256
  `22f4d410c6fcc34b4f68f8afee89d4a656b4fa5e73ff3d4844627d3347e81217`.

The original protocol, product versions, scores, responses and generation cap
remain unchanged. No result here is a clinical efficacy claim.
