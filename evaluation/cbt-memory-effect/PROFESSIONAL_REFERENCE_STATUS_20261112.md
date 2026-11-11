# Professional CBT reference acquisition — 12 November 2026

The genuine hosted-Colab run completed reference acquisition with zero new
model calls and no Secret access. CBT-Bench revision
`bbd37514ba2e4db46bd9633d9207a57828c2ec9d` declared CC-BY-NC-4.0 in its dataset
card. Twelve exact source files, totalling 1,249,348 bytes, were archived before
parsing; references and published human labels remain assessor-only.

| Material | Actual count | Evidence role |
| --- | ---: | --- |
| Professional CBT practice/reference exercises | 156 | Quality anchors; ten skill categories |
| Existing human pairwise comparison rows | 468 | Prospective judge calibration material |
| Rows passing the current structural checks | 466 | Structurally usable, not semantic revalidation |
| Rows with an invalid/missing one-hot label | 2 | Preserve original bytes; explicit pending diagnosis |
| Released longitudinal history records | 0 | Not eligible for the primary memory comparison |
| New model calls / new human reviews | 0 / 0 | No new quality-effect result |

Exercise counts by category 1–10 are 14, 16, 15, 17, 16, 15, 17, 16, 15 and 15.
Difficulty counts are beginner 49, intermediate 55, advanced 52. Counts come
from the pinned files, rather than a contradictory cell in the paper's table.
During source inspection 25 exercises were exposed directly or through an
identical statement group and are registered as development material; the
remaining 131 are not automatically certified independent holdouts.

These are professional author references and existing human expert judgments.
The paper does not establish that every annotator was a licensed physician.
The references are not unique correct answers. Short practice statements are
not converted into invented patient visits, and their responses are not placed
in user memory or model generation inputs.

## Verified acquisition identity

- Collector/code checkout: `5e4d280d64e1d8b3128c2deedb966fa475882c76`.
- Runtime directory: `/content/cbt-professional-quality-anchor-20261112`.
- Source manifest SHA256:
  `a5c7feb2543b6969cd3399f8d124cbfb0910c8e6d782bfea768cdb4eeacbd1b1`.
- Statement-only input SHA256:
  `e09e310ff4bea19dfa190de3684a4ee567a848b378d1b91f7e4ed4e2fa056e2f`.
- Assessor-only reference SHA256:
  `6cddd74084b288e01c4f55a3b41410f08d722d483b090efb3da3d975e72aca01`.
- Released comparison-file SHA256:
  `bda3cf0c862b98b13045c8134b0e60b91ebf5d3bbee5b799044f4791c91579ec`.

The original nine-answer journal still contains nine intents, nine outcomes and
nine answer rows; its paid-checkpoint SHA256 remained
`3e438dffd795b73929f7167c209e09cf0d31905e01d3dcb29d4a19a289d3aa7b`.
The notebook subsequently displayed **All changes saved**, including its native
JSON checkpoint output. The earlier failed copy request was not retried.

## Implementation status and next gate

Distinct-source candidate selection is connected to the live memory evidence
path. It prevents duplicate candidates or candidates already covered by raw
retrieval from exhausting the supplemental source allowance. Complete original
user quotations, current permissions and all multi-source dependencies remain
required; candidate summaries never become confirmed facts.

A single aggregated run of 105 targeted tests passed across source selection,
live dialogue, dates, durable review fences, controlled replay and reference
material preparation. This is software validation, separate from CBT response
quality. No human-professional memory-effect score has been produced.

The new primary comparison is specified in
[EXPERT_CBT_COMPARISON_V1.md](EXPERT_CBT_COMPARISON_V1.md). Raw retrieval must
receive the same nine-source ceiling and budget-filling opportunity; comparing
raw-six to managed-six-plus-three would confound capacity with management.
The original fixed-top-six replay is preserved with its old identity.

RealCBT's official repository was inspected at
`bf5b7914831d439d00a10e54c8abb1dad1a9bde4`. Its 4 August 2026 access notice says
raw transcripts are no longer publicly distributed and researchers should
contact the authors. No removed transcript, old revision or third-party mirror
was acquired. Authorized professional multi-turn transcripts and independent
new CBT reviewers are still required before the new main study can be run.

## Primary sources

- https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench
- https://aclanthology.org/2025.naacl-long.196.pdf
- https://gitlab.com/xiaoyi.wang/realcbt-dataset
- https://aclanthology.org/2026.lrec-1.372/
