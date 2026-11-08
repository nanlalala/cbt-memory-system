# CBT dialogue memory measurement

This directory contains a separately versioned research replay, not a clinical
trial or a replacement for the live dialogue entry point. The first experiment
asks whether memory conditions recover the correct original Client message and
its source ID from a released synthetic CBT dialogue.

## Frozen comparison

- Source: `LangAGI-Lab/cactus`, revision
  `237cb781cb6cbab609bc11f02a24ea6bc408ea4b`; 270,309,432 bytes;
  SHA256 `be3421495f9dd76dd47d5fd4abd9fdabed9c97fcb7f7bba34ecfc78fe07d3d18`.
- Source preparation groups related material before selecting 200 source
  families. Three probes per family select early, middle and recent Client
  messages from the **same** latest frozen released prefix. The 600 planned
  probes are not 600 independent people or therapy sessions.
- Each condition receives the same current question, four-word locator cue,
  eligible source pool, permission policy, provider and escaped full outbound
  ceiling of 6,000 characters. The short-term secondary comparator restricts
  retrieval to the last four released turns. Raw and managed retrieval share
  original-source statistics and preserve their common relevance ranking.
- Managed replay uses the actual source store, bounded extraction worker,
  manager, semantic pointer retrieval, evidence packing and durable send
  admission. Extracted candidates are not user-confirmed facts. This undated
  corpus uses the explicit `ordinal-only-v1` contract: no invented source dates,
  weekly task state, calendar updates or refusal expiry are asserted.
- A probe has at least 24 characters and five words beyond the disclosed cue.
  Correct recall requires the complete original message, after whitespace-only
  normalization, **and** its actual source ID. Gold is assessor-only; it is
  never installed in product state or sent to the extraction worker.

The main contrast is managed versus raw retrieval; short-term is secondary.
The prospective research decision rule requires at least five percentage points
of recall improvement, a positive paired family-cluster 97.5% interval lower
bound, and an error-rate increase interval upper bound no greater than one
percentage point. Resampling uses 10,000 draws and seed 20261109. Critical
permission errors must be zero. Unknown or unattributed costs cannot pass the
separate, currently unresolved resource adoption gate. These are engineering
research criteria, not clinical thresholds, sampling rules or stopping rules.

## Execute only in genuine hosted Colab

Use a fresh checkout/kernel so cached modules cannot come from a previous run.
The preparation step can use a subprocess; paid execution must run in the
notebook kernel to access the already authorized Colab Secret `Capstone`.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "evaluation/cbt-memory-effect"))
import run_cactus_effect as experiment

experiment.prepare(
    Path("/content/cbt-cactus-effect-20261109/cactus.json"),
    Path("/content/NEW_INDEPENDENT_RUN_DIRECTORY"),
    family_limit=200,
)
experiment.run(
    Path("/content/NEW_INDEPENDENT_RUN_DIRECTORY"),
    max_new_calls=100,
    max_cases=1,
)
```

The fixed provider/model policy is encoded in `settings()`. Extraction and
answer generation together have a global limit of 5,000 provider attempts,
one worker, extraction output ceiling 4,096 tokens and response ceiling 1,024
tokens. The extraction outbound ceiling is 18,000 characters. Character ceilings
are not exact token budgets. Identical requests can share an original canonical
receipt; shared responses are not independent model observations. A voluntarily
small batch stops before an incomplete prefix. Unknown sends block further
execution and must not be retried automatically.

Run directories retain frozen protocol/code/data identities, all planned probes,
assessor-only targets, source diagnostics, actual request/response journals,
per-case SQLite state, context traces, original answers and deterministic scores.
Keep these controlled artifacts outside public GitHub. Never export a Secret.
Do not overwrite an old run when code, protocol or dataset identities differ.

## Interpretation and diagnosis

Report correct recall, wrong quote/source binding, explicit unknown, technical
failures, policy blocks, preparation exclusions, pending rows, identical raw/
managed requests, all attempts, missing usage and monetary cost uncertainty.
The runner keeps every planned denominator and labels incomplete recall as a
completion-adjusted lower bound, not a completed observed effect.

Source receipt, extraction, candidate retrieval and final evidence traces allow
stage observations. Presence of an answer substring alone does not establish
semantic correctness or a causal failure stage. Unresolved attribution stays
unresolved. A wrong response despite sufficient literal evidence is a response-
stage observation; a missing target in the final wire needs retrieval/packing
traces before any more specific diagnosis.

This first measurement is cue-assisted quotation recall. It does not establish
free recall, semantic fact/update correctness, cross-session care tracking,
overall dialogue benefit or clinical efficacy. Dated follow-up tasks and human
CBT assessment remain separate evaluations. Tiny fixtures and passing tests are
engineering checks, not evidence that users benefit.
