# Whitespace-prefill diagnostic v2

This is a new, bounded prospective diagnostic. It does not change the already
executed v1 grammar, pilot, memory responses or scores. The real v1 pilot had
schema validity 2/2 and basic readability 1/2: its explicit count question was
answered null. Hosted zero-generation token inspection found **no colon/null
merge**; a BPE-merge explanation of that error has not been established.

The only new format intervention is a fixed assistant prefix with a space after
the colon: `{"answer": `. The reader renders the unchanged source messages using
the same chat template with `add_generation_prompt=True`, then appends the fixed
prefix token IDs. It generates only a generic value and closing brace. Prefix
tokens are recorded as input, never as model-generated tokens. Actual tensor
hashes and counts include the prefix; generated suffix IDs are retained in full.

The generic domain remains 0–9999, all fixed status strings from v1, and null.
It is identical for every question and arm. No question, reference answer,
product state or expected type selects a narrower domain. EOS is allowed only
at a complete accepted suffix. Syntax does not establish factual correctness.

## Fixed four-pilot sequence

The two original non-CBT questions are retained, with unchanged system/user
messages: an explicit returned-book count of 11 and a machine outcome of
`unchanged`. The prospective order is:

1. Count question, unconstrained reference.
2. Outcome question, unconstrained reference.
3. Count question, whitespace-prefill v2.
4. Outcome question, whitespace-prefill v2.

All four requests and the separate fixed labels must be frozen before outputs.
The unconstrained responses are retained as raw references; no post-hoc format
or semantic score is invented for them. Only the two prefill responses enter the
predeclared strict JSON/type/value gate. If either fails, the study stops before
the main comparison. It does not replace questions, remove null or retry.

The separately declared study budget is at most **61 new admitted generations**:
four pilots plus 57 expected unique main-regression generations representing 96
logical rows. The old root's 64 attempts remain unchanged and are referenced
separately; the cumulative bound is 64 + 61. Main cases are already exposed
development regressions, not new holdout evidence. Actual loader, ledger and
source-gated orchestration belong to a separately reviewed hosted-only runtime.

## Module interfaces

`constrained_json_prefill_v2.py` exposes `policy()`, `AnswerTrie`,
`input_preflight(tokenizer, messages, trie)`, and
`generate(model, tokenizer, messages, trie, max_new_tokens=64)`.

`constrained_prefill_pilot_v2.py` exposes `requests()` with four rows,
`prepare(output)` for a fresh hosted directory, `validate_prepared(root)`, and
`score_authenticated(answers, prepared_dir=...)`. The runtime must authenticate
all four answers against its new ledger/cache/input identities before calling
the scorer. The scorer opens fixed labels only after complete logical coverage.

No real tokenizer or model was executed locally. Six tests use invented
character tokens and metadata only:

```bash
python -m unittest discover -s tests -p test_constrained_prefill_v2.py -q
```

At document freeze, the new four-pilot model run has not occurred. Results must
be added as a separate record, preserving this prospective declaration.
