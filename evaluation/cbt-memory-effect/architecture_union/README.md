# Managed-memory union (draft, not yet evaluated)

A variant of the CACTUS natural-continuation request builder in which the managed
arm always contains the raw-history RAG selection and adds memory on top of it.
The frozen `architecture_retest` package is only read: its code hashes bind earlier
runs. A run of this variant needs its own protocol and result directory.

## Layers

| Layer | Content | Switch (`UnionConfig`) | Default |
| --- | --- | --- | --- |
| 1 | Raw-RAG whole messages, identical to the raw arm; never dropped | always on | on |
| 3 | Pinned profile: user corrections, then case state and task/outcome items (latest first); hypotheses and background never pinned | `pin_profile`, `profile_limit` | off |
| 2 | Fact-key channel: up to N extra whole messages, ranked by "message + values of items that cite it", drawn only from the eligible older pool | `fact_key_sources` (design value 3) | 0 |
| 4 | Question-ranked items in the frozen relevance order | always on | on |
| 4 | Optional re-rank: relevance stays primary; recency (ordinal, mutable sections only), importance by kind and evidence-grade reliability adjust neighbours | `rerank`, `*_weight` | off |
| — | Renderer: v4 or the v5 draft that pins user corrections | `renderer` | `v4` |

Addition order is 1, 3, 2, 4, so the layer added last loses first when the
complete-request ceiling binds. A candidate counts as added only if the context
builder actually rendered it (`included_claim_ids`); silent builder drops are
recorded with their reason. The default config reproduces the earlier tested union.

Reliability is computed at read time from the evidence grade (user correction >
user statement with user-role evidence > assistant proposal > hypothesis) plus
independent user-role mentions. Repetition never raises a hypothesis, and a model
confidence number is not used. It is written to the selection audit only; nothing
new is shown to the answer model.

## Not implemented here

- Linked "possibly related" memory (layer 5).
- Conflict down-weighting: these extracted items carry no slot key, so conflicts
  cannot be detected reliably at this stage.
- New profile fields (background kind, core goal, risk, follow-ups): these need a
  new extraction contract version, because the contract file is hash-pinned.
- A run harness. `prepare_case_requests_union(..., config=UnionConfig(...))` builds
  the three requests; protocol, journal and scoring still have to be written.

Weights default to the product retriever's values and must be tuned on a
development split only, then frozen before any test run.

Tests: `python -m unittest tests.test_architecture_union`.
