# PersonaMem full-manager development protocol

Run the complete frozen **589-question, 20-connected-group** cohort in two arms:
`raw_history_rag` and `managed_memory_rag`. Both receive fresh answers in a new
run: **1,178 logical answer rows**, with exact-request deduplication within this
run. Original-run replies are never reused. This document specifies the protocol;
it reports no experimental results.

The managed arm uses the actual SQLite store, `MemoryManager` candidate writes,
memory-use checks and `MemoryRetriever`. Model-extracted, unconfirmed indexing
hypotheses can change retrieval; the answer model receives **original source
quotations only**, not generated summaries presented as facts. The retrieval
bridge is lexical TF-IDF, not a learned embedding. Pooling by original source
chunk prevents multiple indexes from consuming multiple top-k source slots.

## Fairness and access gates

- Preserve the original questions, all four options, admitted histories, response
  instructions and response/context budgets. Both retrieval queries use the
  question plus all released options, never the correct-option label.
- Extraction sees source windows only: no question, options or gold. Candidates
  must quote exactly one current **user** chunk. Earlier supplied context is
  interpretation-only; correction/assignment links must identify a strictly
  earlier supplied user source. No user confirmation or calendar date is invented.
- Enforce the released **exclusive history cutoff** and eligibility controls.
  Cache/replay scopes bind persona, source context and private history branch;
  connected statistical groups never authorize cross-context access. Recheck each
  case's exact eligible prefix; conflicting source versions fail closed.
- **Before any provider call**, all 589 cases must pass the empty-semantic gate:
  new raw prompts equal the frozen original raw prompts, and managed prompts with
  synthetic empty candidates equal raw. Any mismatch stops preparation. This is
  a source/context parity control, not evidence about extraction quality or effect.
- Keep source/query/option/answer/extraction text and gold in controlled Colab
  artifacts. Preparation/extraction/answering never open gold references; scoring
  opens their frozen hashes only after all 1,178 answers reach terminal states.

## Frozen computation and accounting

| Setting | Policy |
| --- | --- |
| Response model | `deepseek-v4.1-flash`; allowed returned alias `deepseek-v4-1-flash-260910` |
| Provider/settings | Existing gateway; temperature 0, thinking disabled; four workers |
| Output limits | Extraction **4,096 tokens**; answers **64 tokens**, both arms |
| Extraction inputs | Up to **8,000 source characters** plus **4,000 prior-context characters**; at most 12 candidates/window |
| Source chunks/request ceiling | 1,200-character chunks; serialized request at most 24,000 characters |
| Whole-run cap | At most **2,500 actual provider attempts**, extraction and answers combined |

Preparation rejects a cap that cannot cover all planned extraction windows plus
1,178 possible answer calls. Every invocation also needs a finite `--max-new-calls`.
One attempt per canonical request; **no automatic retry**. Terminal failures stay
terminal, invalid/unavailable extraction remains in the assigned managed arm,
and all planned questions remain in the denominator. Unresolved in-flight requests
require manual reconciliation; changed code/inputs or truncated journals halt.
Preserve request/intent/outcome journals, protocol/code/input hashes, operation
and retrieval/context traces. Count extra extraction computation and every failed
attempt; report available token usage and missing usage separately. Currency cost
is unknown without verified prices. Equal answer budgets are not equal total cost.

## Exact phase commands

Execute from the committed, clean repository checkout **inside genuine hosted
Colab**. Set these variables to the actual frozen source suite, fresh `/content`
run directory outside the source suite/repository, and reviewed 40-character SHA.
The launcher inherits the already authorized Colab credential environment; it
does not read Secrets itself. Never put a credential in command arguments/files.

```bash
python evaluation/objective-memory/launch_persona_management_colab.py prepare \
  --run "$PERSONA_RUN" --original-suite "$PERSONA_SUITE" \
  --executable-commit "$PERSONA_SHA" --maximum-provider-attempts 2500

python evaluation/objective-memory/launch_persona_management_colab.py extract \
  --run "$PERSONA_RUN" --max-new-calls 250

python evaluation/objective-memory/launch_persona_management_colab.py answer \
  --run "$PERSONA_RUN" --max-new-calls 250

python evaluation/objective-memory/launch_persona_management_colab.py status \
  --run "$PERSONA_RUN"

python evaluation/objective-memory/launch_persona_management_colab.py score \
  --run "$PERSONA_RUN"
```

Inspect each phase's returned status. Continue a partial extraction/answer phase
with another explicitly bounded invocation within the frozen global cap; do not
run an unbounded loop or retry a recorded failure. Answers require terminal
extraction. Partial scoring reports progress, not final comparative scores.
`persona_management_scoring.py export --run-dir "$PERSONA_RUN" --out "$PERSONA_EXPORT"`
creates separate controlled reproducibility and public text-free result archives
in a fresh output directory, without provider calls; run it in genuine Colab.

## Inference boundary

Use the unchanged official Persona option parser; retain strict/exact diagnostics
as secondary checks. Primary comparison is the mean paired difference with equal
weight for the **20 connected source groups**; report micro accuracy, per-task
results, wins/ties/losses and the frozen paired cluster bootstrap (seed 42,
10,000 resamples). Identical-prompt/shared-response pairs remain construction
controls; if every pair is shared, management-effect intervals are blocked.

This is an exposed development benchmark, not 589 independent people or an
independent holdout. The contrast tests the **whole extraction/indexing/manager/
retrieval pipeline versus competent raw-history RAG**, not lifecycle management
alone. Source binding and structural validity do not establish semantic truth or
the causal stage of an incorrect answer; unsupported attribution stays
`answer_incorrect_stage_unresolved`. No participant burden, independent CBT
professional quality, symptom/function change or clinical effectiveness follows
from these scores. Keep earlier quotation-control results and protocols separate.
