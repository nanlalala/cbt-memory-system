# Quotation and source binding development results

A completed hosted-Colab pilot found a specific output error: the reader copied
the correct complete Client quotation but attached the ID of another visible
message. The raw-history and managed-memory conditions each made this error on
the recent probe, returning `turn-025` for a quotation from `turn-027`.

The new validator checks the answer against the original Client quotations
actually visible in that request. It corrects an ID only when the complete
quotation identifies one visible source uniquely. It preserves valid bindings
and explicit unknowns; ambiguous, partial, unsupported or malformed quotations
become explicit unknown. It does not use the question, locator cue, reference
answer or hidden history to repair a response. The same rule applies to every
condition.

## Completed answer results

The pilot contains three related development probes from one synthetic CACTUS
source family. The existing question, history, model request and scoring rule
were retained. Correctness requires both the complete quotation and its exact
source ID.

| Condition | Original recorded outputs correct | After common binding validation |
| --- | ---: | ---: |
| Recent context | 1/3 | 1/3 |
| Original-history retrieval | 2/3 | 3/3 |
| Managed memory | 2/3 | 3/3 |

The hosted replay repaired two source IDs and reduced source-binding errors
from two to zero. Quotation text was retained. All nine original answer rows,
provider receipts and earlier experiment artifacts remain unchanged; transformed
outputs were written to a separate archive before the assessor opened targets.
The replay added no provider calls.

This is a retrospective development result for a common output validator. Both
retrieval conditions improve equally, so it is not evidence of a managed-over-raw
advantage or an independent held-out effect. The original quotation benchmark
also retains its four-word locator and original task identity.

## Observed answer-call resources

The paid pilot completed nine published provider calls with no unknown sends.
It used 11,239 reported tokens: 10,821 prompt tokens and 418 completion tokens.
Cumulative API time was 18.524 seconds; cumulative observed invocation time was
19.132 seconds. These are invocation measurements, not participant task time.

| Condition | Reported tokens | Cumulative API time |
| --- | ---: | ---: |
| Recent context | 2,131 | 5.914 s |
| Original-history retrieval | 4,590 | 5.654 s |
| Managed memory | 4,518 | 6.956 s |

The source-binding replay incurred zero additional provider calls. Prior memory
extraction and local embedding remain reused development inputs, with their
costs retained separately. Token differences are not attributed to memory
management or to the validator. Monetary charges were not reconciled to a bill.

## Verification and reproduction

Twenty-eight targeted software tests passed across quotation binding,
retrospective replay, canonical-cost reconciliation and the annotation contract.
The hosted replay also verified that the original artifacts remained unchanged.
These software tests are separate from the three-probe answer results.

- Implementation and replay executable: `1b263f8883509229e37ef6d95486355e24438781`.
- Transformation rule: `visible-full-quotation-source-binding-v1`.
- Frozen transformed-answer SHA256:
  `b123c1e283d41e76ff24b7fe117c367c30e4c2442c55f3904477c4fc319d7309`.

The detailed private receipts retain per-question scores, transformation reasons,
full request identities and resource measurements. The original pilot protocol
and paid journal are preserved alongside the new retrospective replay.

## Adoption in a new live protocol

The pure binder is available to future structured-quotation senders. It is not
inserted into either frozen sender, and it must not be applied to ordinary CBT
free-text replies. A new sender must declare the quotation schema in advance,
construct its catalog from complete user originals actually present on the wire,
and use the same policy for all arms. It must preserve the provider receipt and
store the validated output as a derived record bound to the request, original
response, catalog and binder-code hashes. Recheck source permissions before
persisting or displaying the derived result; an explicit unknown does not
trigger another provider attempt.
