# Common output adapter and unconstrained state regression v3

This prospective study tests whether the existing source-bound state repairs
improve answers to corrected targets, repeated cumulative reports, proposal versus
agreement, and progress versus outcome. It compares capable original-quotation
RAG with managed memory before and after the product repair.

The material is the same exposed development set: four constructed families,
three variants and two languages, giving 24 questions. There are 96 logical rows
across two memory arms and two product versions, with 57 unique request payloads.
These are mechanism regressions, not independent patients or natural CBT trials.

## Independent study identity

The new hosted root is
`/content/cbt-unconstrained-state-study-20261112-v3`. Its own maximum is **57
admitted generations**, including failed or unknown attempts. It does not spend
unused allowance from another study. The original 64-generation experiment and
the rejected four-generation v2 pilot remain immutable historical references.
Initialization authenticates their settled ledgers and the rejected v2 gate,
then binds their protocol, answer and score hashes in the new protocol.

No new pilot or independent output-interface validation is claimed. The output
adapter was developed with already exposed diagnostics. Its complete policy and
code are frozen before any v3 generation. No old output or score is rewritten.

All execution requires genuine hosted Colab. The model is the same pinned
Qwen2.5-1.5B-Instruct CPU float32 reader, with greedy generation, at most 4,096
actual input tokens and 64 output tokens. The complete serialized request is at
most 6,000 characters. There is no assistant prefix or logits constraint. The
runtime loads only verified cached files and does not access credentials.

## Common inputs and source authority

Both arms receive the same recent six original messages, permitted historical
source pool, source retrieval cap, answer model, output adapter, budgets and
checking policy. The common neutral headers permit reconstructing the latest
explicitly reported state from original quotations. Missing structured fields
alone do not imply that the answer is unknown.

The source product checkouts remain the frozen before/after versions. Each side
gets 24 fresh authoritative databases at their original newly created paths.
New neutral request hashes are registered through real product store bindings.
Static prepared files are frozen separately from the mutable database authority;
database paths and storage identity are not copied or rewritten. Each version's
source/permission/version audit runs in a fresh subprocess before the complete
grid is frozen. The original evidence text, source identifiers, offsets, order,
recent messages and literal reference labels are preserved.

## Output and metrics

The model's original text, text hash, actual input tensor hash and token usage
remain in its raw generation cache. Each logical answer separately records the
common adapter's output and measured adapter time. Authentication recomputes the
adapter from the cached raw text and verifies every source-bound request.

The adapter reads only the whole output string. Its fixed grammar accepts an
unambiguous single scalar or one-field answer object; ambiguity and explanatory
prose are rejected. It does not inspect a question, memory arm, evidence, expected
type or reference answer. Rejection has its own status and cannot become null.

The complete 96-row archive must authenticate before main labels are opened.
The report keeps two metrics separate:

| Metric | Scoring input | Rule |
| --- | --- | --- |
| Raw strict accuracy | Unmodified model text | Original single-key JSON parser and exact typed equality |
| Canonical pipeline accuracy | Accepted output of the prospectively frozen common adapter | Same typed equality; adapter rejection is an error |

Adapter acceptance, error categories and paired raw/managed wins are also
reported. A valid output format alone does not establish a correct answer. The
four constructed source families are retained as grouping identifiers; 24
variants do not establish 24 independent clinical observations.

Generation cost is counted once per canonical request, including all admitted
attempts. Within-study response reuse still executes each logical row's source
gates. New actual model reload events, token usage and answer-path timing are
reported separately from inherited historical receipts. Adapter time is measured
once per logical row and separately summed. Unmeasured source ingestion,
indexing, state projection, retrieval packing and cached-row validation timings
remain null. No monetary cost is invented.

## Hosted execution

The helper checkout must remain at one immutable reviewed commit for this study.
Run the following commands as separate steps, inspecting each result. Never run
an existing output directory again or resume a failed/unknown generation.

```bash
python unconstrained_state_comparison_v3_colab.py initialize \
  --old-root /content/cbt-open-reader-20261112-v1 \
  --failed-v2-root /content/cbt-constrained-state-study-20261112-v2

python unconstrained_state_comparison_v3_colab.py freeze-main \
  --before-checkout /content/cbt-open-state-before-20261112-v2 \
  --before-reference /content/cbt-open-state-prepared-before-20261112-v2 \
  --after-checkout /content/cbt-open-state-after-20261112-v2 \
  --after-reference /content/cbt-open-state-prepared-after-20261112-v2

python unconstrained_state_comparison_v3_colab.py run-side --side before
python unconstrained_state_comparison_v3_colab.py run-side --side after
python unconstrained_state_comparison_v3_colab.py score
```

The before side must settle all 48 unique generations before the after side
begins. The after side reuses unchanged canonical requests and admits nine new
generations. Scoring requires all 57 admissions and all 96 authenticated rows.
There is no intermediate scoring, outcome-based stopping, automatic retry,
question replacement or automatic budget increase.

## Engineering checks

The new runtime tests use tiny invented metadata, fake model returns and fake
source gates only. They exercise budget closure, cache isolation, raw versus
adapter preservation, tamper rejection, source withdrawal, full-grid freezing and
the prohibition on opening labels before complete authentication. They do not
execute a real tokenizer, model or dataset locally.

```bash
python -m unittest discover -s tests -p 'test_unconstrained_state*.py' -q
```

At this document's creation, no v3 model generation has occurred. Executed results
must be added separately while preserving this prospective declaration.
