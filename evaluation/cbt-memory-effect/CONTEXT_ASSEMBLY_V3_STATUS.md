# Context assembly v3 development status

`context_assembly_v3.py` is a pure, unintegrated helper. It is not imported by the
frozen CACTUS v2 runner, replay adapter or live service. Its tests use invented
fixtures only. No dataset accuracy or model effect is established by these tests.

Implemented contracts:

- All arms use the same direct recent dialogue in original speaker order and the
  same current input, system prompt, metadata and complete JSON character budget.
- Raw uses the supplied original-source ranking. Managed uses equal-weight
  reciprocal rank fusion (k = 60) of original-source rank and candidate-index rank
  projected to source IDs. Duplicate IDs contribute once per channel.
- Sources from other users, future source orders or without explicit save,
  long-term-use and response-external permission are excluded before fusion.
- A compact packet renders each retrieved original quotation once with its source
  ID, role and order. Sources already in direct recent context are not repeated.
- Quotes are indivisible. Oversized sources are skipped with a reason; recent
  overflow removes oldest exchanges equally across arms. Current input is never
  truncated. Returned dependencies and snapshot hashes identify historical text
  actually used; the current input source ID is returned separately for admission.

Before any real run, integrate a verified source-binding adapter, the same pinned
encoder for raw and managed, durable send and publication checks for all returned
historical dependencies plus the current input, and a separately frozen v3
protocol with a finite authorized invocation budget. Candidate summaries must not
be relabelled as original or confirmed evidence. Encoder, indexing and answer
costs must all be recorded. The frozen v2 directory, protocol and results remain
unchanged.

Validation command:

```bash
python -m unittest discover -s tests -p test_context_assembly_v3.py -v
```

## Existing-checkpoint dry run

`prepare_context_revision_v3.py` now connects the helper to a **zero-provider**
checkpoint diagnostic. It uses the existing complete SQLite case's source family
and its three frozen generation inputs, validates the unchanged v2 product code,
reads original sources through v2's control methods on a read-only connection,
and ranks existing candidate summaries with TF-IDF. No dense encoder is present.
It does not read reference answers or generate scores. This does not integrate a
v3 answer runner or grant model-call authority.

After restoring the v2 checkpoint at its original bound path in genuine hosted
Colab, place the script and `context_assembly_v3.py` in the same new code directory:

```bash
python prepare_context_revision_v3.py \
  --old-run /content/cbt-cactus-source-quote-200families-20261109-v2-authorized100 \
  --product-repo /content/cbt-cactus-v2-private-9f7568116bf3 \
  --output /content/cbt-context-v3-development-20261111
```

The new directory contains its own frozen `protocol.json`,
`context_rank_pack_traces.jsonl` and `summary.json`. All generated contexts are
marked `not_send_ready`. Original checkpoint input hashes must remain unchanged.
SQLite fixture integration tests run with:

```bash
python -m unittest discover -s tests -p test_prepare_context_revision_v3.py -v
```

A separate optional diagnostic may add `--encoder multilingual-e5-small` and a
new output directory. It uses the pinned `intfloat/multilingual-e5-small` snapshot
through the unchanged product `MultilingualE5Encoder` adapter, and passes one
shared encoder instance to both raw-source and candidate-summary retrieval.
Snapshot or loading failures stop the diagnostic; no lexical fallback occurs.
The default remains TF-IDF with no encoder. Local encoding text counts and time,
model identity and loading time are recorded separately from provider calls
(which remain zero). This option does not generate answers or accuracy scores.
