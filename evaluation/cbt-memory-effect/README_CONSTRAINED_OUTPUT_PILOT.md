# Independent constrained-output pilot v1

This is a new format/readability pilot, not a rerun or revised score of the
completed memory experiment. The old reader, preparation, original responses,
strict scorer and secondary diagnostic remain unchanged. No real model result
exists for this new pilot at the time this document is frozen. Six tiny tests
use an invented character tokenizer and no model or network.

The two newly invented, non-CBT questions concern an explicit library count of
11 and an explicit machine outcome of `unchanged`. Both requests and their
separate assessor labels are frozen by `prepare` before any output. They do not
reuse any of the 24 previously examined mechanism cases. They test JSON format
and basic readability only; they are not memory, CBT quality, clinical benefit,
independent holdout or new human-review evidence.

## Common constraint

`constrained_json_grammar.py` accepts exactly one compact JSON object with the
sole key `answer`. Its value is an integer from 0 through 9999, one of the fixed
general status strings listed in `STATUSES`, or null. The status domain includes
helpful/unhelpful/mixed, approved/planned/paused/completed/ongoing and other common
states. This is a bounded output language, not arbitrary JSON or arbitrary text.

The same full domain applies to every example and any future comparison arm.
The grammar does not receive gold, arm identity, question content or product
state. A tokenizer trie permits only valid continuations of the generated token
prefix and permits EOS only after a complete accepted object. It leaves the
caller-supplied chat messages unchanged and uses the pinned model and tokenizer,
greedy decoding and the existing 64-output-token budget. This guarantees syntax
for successfully completed generations; it does not establish factual truth.

The full decoding identity enters each new canonical key and run protocol.
Original unconstrained cache keys are never reused. Results retain all original
generated token IDs, decoded raw text including special tokens, the exact JSON
response and the chosen typed value. All candidate strings must round-trip
through the tokenizer before inference; both pilot input tensors are checked
against the 4096-token ceiling before the first generation admission.

## Hosted Colab only, fresh Python processes

Use the new checkout's actual absolute path for `REPO` below. The fixed shared
model root remains `/content/cbt-open-reader-20261112-v1`. The runner requires
both global locks idle, exactly 62 settled published attempts in its unchanged
ledger, and successful completion of the existing five-example calibration.
These two new attempts are the last two slots under that root's existing cap64.
Failures and interrupted attempts remain counted. Existing output or any prior
pilot-purpose intent stops execution; there is no automatic retry or recovery.

```bash
python "$REPO/evaluation/cbt-memory-effect/constrained_open_reader.py" prepare \
  --output /content/constrained-format-pilot-prepared-20261112-v1

python "$REPO/evaluation/cbt-memory-effect/constrained_open_reader.py" run \
  --prepared-dir /content/constrained-format-pilot-prepared-20261112-v1 \
  --output /content/constrained-format-pilot-results-20261112-v1

python "$REPO/evaluation/cbt-memory-effect/constrained_open_reader.py" score \
  --prepared-dir /content/constrained-format-pilot-prepared-20261112-v1 \
  --output /content/constrained-format-pilot-results-20261112-v1
```

Each command is a fresh CLI process, avoiding cached product/module identities
in a notebook kernel. Do not run it concurrently with model loading or inference.
`score` performs no model call, authenticates the completed outputs against the
shared ledger and exact canonical cache, and only then opens the frozen labels.
It writes a new `pilot_score.json`, keeping raw output and old scores intact.

Repeating the main memory comparison with constrained decoding would require a
separately authorized, versioned protocol and budget. Creating a new model root
or renaming a run does not authorize bypassing the existing cap64.

## Fixture verification

```bash
python -m unittest discover -s tests -p test_constrained_json_grammar.py -q
```

Current result: six tests passed. This is engineering verification, not a model
experiment or effect score.
