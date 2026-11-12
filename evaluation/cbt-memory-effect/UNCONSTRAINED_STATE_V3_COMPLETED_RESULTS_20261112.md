# Completed neutral-reader development comparison

This experiment checks whether the same answer model uses sourced state updates
more accurately than capable original-quotation RAG, with the same recent six
messages, history, retrieval ranking, permissions, request budget and checking
policy. It is a development regression on four exposed constructed families,
three variants and two history languages: 24 questions, not 24 independent
patients, natural CBT conversations or clinician-reviewed observations.

## Observed result

All 96 logical answers authenticated before scoring. All 57 admitted unique
generations settled successfully in genuine hosted Colab. The completed
one-shot continuation did not resend an ambiguous request or increase the cap.

| Product version | Arm | Original strict JSON | Predeclared common canonical pipeline | Accepted outputs |
|---|---|---:|---:|---:|
| Before | Original-quotation RAG | 0/24 | 6/24 | 13/24 |
| Before | Managed memory | 0/24 | 6/24 | 9/24 |
| After | Original-quotation RAG | 0/24 | 6/24 | 13/24 |
| After | Managed memory | 0/24 | 6/24 | 9/24 |

Both arms answered the same six unknown-value questions correctly. Neither
answered any of the 18 known-value questions correctly under the frozen typed
pipeline. The comparison has zero managed-only wins and zero raw-only wins;
the common output adapter therefore supplies no evidence of greater fact
coverage or superiority over quotation RAG. Rejected output is not a null
answer, and correct abstention is not counted as remembered known information.

| Family, six variants each | Raw after | Managed after |
|---|---:|---:|
| Target correction | 0/6 | 0/6 |
| Repeated cumulative total | 0/6 | 0/6 |
| Proposal is not agreement | 6/6 | 6/6 |
| Outcome is not progress | 0/6 | 0/6 |

## Original-answer error attribution

A post hoc content inspection is retained separately from the fixed scores. It
is not an independent human or clinical semantic accuracy metric.

* Three English cumulative-total answers changed from the obsolete value 7 to
  the referenced totals 2, 3 and 4. One English target answer changed from 7 to
  2. Their extra fields, prose or unfinished objects still fail the fixed
  answer contract.
* Two English outcome answers regressed from `unhelpful` and `mixed` to counts
  3 and 4. The other English outcome remained a wrong count; all three Chinese
  outcome answers also contain counts rather than outcome values.
* All three Chinese cumulative-total answers remain incorrect abstentions.
* For the outcome example, the actual managed request already contains the
  correct `outcome` and its original quotation. Its answer instead copies
  `reported_count`. This is a reader field-selection failure, not absent
  storage evidence, and is the concrete target of the next code change.

The full private checkpoint and accompanying report preserve all 24 paired
original responses, all four conditions, labels, frozen scoring, request and
response identities, sources, journals and both product versions. The raw
control requests and responses are identical across product versions.

## Measured resources

| Item | Observed |
|---|---:|
| Unique admitted and published generations | 57 |
| Prompt tokens | 98,265 |
| Completion tokens | 1,620 |
| Total tokens | 99,885 |
| Answer-path time | 4,559.722166795 seconds |
| Two actual model setups, separate | 42.653621892 seconds |
| Common adaptation over 96 logical rows, separate | 0.007753995 seconds |
| New paid provider calls / secret access | 0 / false |

Answer-path time includes generation, source fences and publication, rather
than pure forward-pass latency. Ingestion, state projection, extraction,
organization/indexing and cache-reuse validation were not timed; missing values
remain null. Monetary cost is unmeasured. The original 64-attempt study and the
failed four-attempt v2 gate remain separate historical runs, not additions to
this study's generation cost and not transferable authorization.

## Reproducibility identities

* Frozen runtime checkout: `8d1b0f7e6688dc480ad4ef282a80b8c1fd4916fb`.
* Study protocol SHA256:
  `118bccbf288105cafbe4c88e6b582f0742b1b28056c8d06a5cf7860296841d16`.
* Derived complete-analysis SHA256:
  `1ba2e45cd59fb6335bafc6ea7d9f208ef229963e4abbd1426e7ca3a3f72563df`.
* Full private checkpoint: 2,136,101 bytes, SHA256
  `2f136bf204935bcc69a790ec973bb69c4435ec24f94a62ae3c52e511a163379c`.
* Delivery archive including the checkpoint, 24-pair CSV/JSON, analysis and
  manifests: 1,921,014 bytes, SHA256
  `a1251e452abf3bf79dcb404bc884ab1f1e5c9c2ad51a42465359f2ae8a1e2be1`.

The original v3 protocol, outputs and scores are unchanged. The separately
versioned question-field selection and question-only output-schema code is a
prospective reader improvement. Its software tests are not an effect result;
an eight-generation invented reading gate must pass before admitting any
follow-on full comparison. Professional CBT quality and independent natural
memory labels are not supplied by this constructed regression.
