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
| — | Compact quotes (v5 only): each whole message rendered once with a short claim ID, instead of value + identical evidence span + long IDs | `compact_quotes` | off |
| — | Evidence once: items cite whole quoted messages by source ID instead of repeating spans (`processed` representation) | `evidence_once` | off |

A zero-cost diagnosis on real cases showed the first union rendering each quoted
message twice with long IDs, so on average less than half of the frozen managed
arm's items still fitted under the ceiling. The compact options remove that
duplication; the quote-equals-source check is unchanged.

Addition order is 1, 3, 2, 4, so the layer added last loses first when the
complete-request ceiling binds. A candidate counts as added only if the context
builder actually rendered it (`included_claim_ids`); silent builder drops are
recorded with their reason. The default config reproduces the earlier tested union.

Reliability is computed at read time from the evidence grade (user correction >
user statement with user-role evidence > assistant proposal > hypothesis) plus
independent user-role mentions. Repetition never raises a hypothesis, and a model
confidence number is not used. It is written to the selection audit only; nothing
new is shown to the answer model.

## CBT knowledge (textbook RAG) channel

`knowledge.py` connects the existing knowledge adapter
(`modules/agent-integration/knowledge_adapter_v2.py`, over the reviewed corpus in
`modules/cbt-knowledge-rag/`) to this run. `build_packets` asks the adapter once
per case, using the current user message as the query, and freezes the cards.
`prepare(..., knowledge_packets=...)` then gives the **same** cards to no-RAG, raw
RAG and every managed config, as `cbt_knowledge` inside the sourced data block.
`fit_knowledge` keeps the longest prefix of cards that fits every arm under the
24,000-character limit, so the arms still differ only in memory. Because the raw
and no-RAG requests change, those answers are generated again in the union run
rather than reused from the prior, and the frozen managed answer is not compared.

The adapter returns cards only for sources whose retrieval and external-model use
are approved and pinned in `source_registry.json`. All seven sources are disabled
today, so packets are empty and `prepare` refuses them instead of reporting
textbook grounding that did not happen. Without packets every request is byte-for-byte
what it was before this channel existed (pinned by a request digest test), and
answers prepared before it stay readable.

## Not implemented here

- Linked "possibly related" memory (layer 5).
- Conflict down-weighting: these extracted items carry no slot key, so conflicts
  cannot be detected reliably at this stage.
- New profile fields (background kind, core goal, risk, follow-ups): these need a
  new extraction contract version, because the contract file is hash-pinned.

## Running

If no completed prior run is available, `prepare_prior` freezes a new roster from
the verified public CACTUS file and `run_prior` runs the unchanged frozen
three-arm pipeline (`runner._case`) for one split only, without the independent
pre-answer assessor. Without the historical family list this roster can differ
from earlier runs and is reported as a new run.

`run.py` reuses a completed `architecture_retest` CACTUS run: its roster, its checked
memory items and its no-RAG, raw and frozen managed answers. Only new managed
answers are generated, one per config in `DEFAULT_LADDER`, through the same
journal (identical requests share one receipt; nothing is resent). A case is skipped and
counted unless its prior raw answer is a verified receipt and the rebuilt raw
request serializes to exactly the bytes sent for it (`serialized_request`,
`wire_sha256`). Families are split deterministically into dev and test. `export_quality`
puts the prior raw and frozen managed answers and every config into one blinded
judge batch; `compare` reports paired differences against raw per R1–R5 with a
family-clustered bootstrap. Use `Managed_Union_Layers_Colab.ipynb`; paid steps are
off by default.

Weights default to the product retriever's values and must be tuned on a
development split only, then frozen before any test run.

Tests: `python -m unittest tests.test_architecture_union`.
