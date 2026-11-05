# Candidate shared retrieval repair and source-only diagnostic

The [completed v3 result](results/full-20261105/README.md) stays frozen. Its
managed MAB arm has 320 officially incorrect rows; 213 are the literal response
`UNKNOWN`. The raw arm has 381 officially incorrect rows, including 196 literal
`UNKNOWN` responses. All seven raw-to-managed regressions still retrieved and
included 12 older documents, and no older retrieved MAB document was dropped by
the character budget anywhere in either arm. These are source-path observations,
not independent evidence that retrieval, versioning or generation caused a
specific error. PersonaMem's 173 incorrect RAG answers also remain unresolved;
its observed chunk clipping alone is not a causal diagnosis.

## Demonstrable algorithm defect

With the original lexical threshold `.05`, `top_k=12`, and these invented sources:

1. `Orion is a citizen of Aurelia.`
2. `The capital of Aurelia is Nova.`

the query `Orion citizen` followed by 35 unrelated word tokens retrieves source
1, then records one exact source-object/subject neighbor but admits none, despite
11 unused result slots. The generic whole-query cosine threshold erased a
source edge that had already been identified from original text. This tiny
counterexample establishes algorithm behavior; it does not identify the cause
of any particular released benchmark error.

`v4_source_retrieval.py` keeps the original direct-seed threshold, exact entity
matching, ownership scope, original quotations, expansion rounds, total top-k
and character budget. For already validated source-edge neighbors only, lexical
scores rank candidates without applying that threshold a second time. The
candidate supports the original lexical backend; it rejects dense/hybrid rather
than silently changing their semantics. Apply the same repair to raw and managed
conditions. This shared repair is not the memory-management treatment.

## Run the full offline diagnostic in genuine hosted Colab

Use a separate clean candidate checkout at an exact new commit. Keep the old
151170 checkout and full-run directory unchanged. The diagnostic imports the
candidate source-edge function while retaining the old shared grammar, ranker,
renderer and all budgets. It does not patch a module, bypass a guard, resume the
old run, call a provider or spend tokens.

```bash
python NEW_CHECKOUT/evaluation/objective-memory/compare_v4_retrieval_colab.py \
  --old-repo /content/cbt-objective-v3.1-authorized-1511706b3a4b \
  --candidate-repo NEW_CHECKOUT \
  --candidate-commit EXACT_NEW_COMMIT \
  --suite /content/cbt-v3.1-authorized-runall-20261105/full-800-589 \
  --out FRESH_COLAB_OUTPUT
```

Before comparison, existing shared locks and read-only readers verify the old
frozen inputs and journals. The helper reconstructs source histories from the
original inventory, verifies their identities, and reproduces the original
v3 prompt hashes and message bytes exactly. It compares **all 800 MAB cases in
both RAG conditions**. All changed inputs are selected from source/prompt
identity before looking up or recomputing old correctness scores. The latter
are subsequently recomputed solely for descriptive audit, never generation.

Outputs:

- `summary.json`: actual changed-case/pair counts and source-only selection hash.
- `per_request_observations.jsonl`: 1,600 rows with original/new prompt hashes,
  selected/included/dropped source IDs and old official scores. No source text.
- `diagnostic_manifest.json`: source and output checksums.
- `controlled_candidate_requests.jsonl`: **original source text in candidate
  prompts; keep in Colab and do not publish or load outside the controlled
  source-processing environment**. It contains no scoring references.

Only the first three files are derived-only reporting exports. Zero changed
cases must be reported as zero. A changed prompt is not a fixed answer. Any
later model comparison requires a separate frozen protocol, finite call budget,
actual attempt/usage journals and an explicitly development-exposed evaluation
label. This candidate has no new benchmark effectiveness result yet.
