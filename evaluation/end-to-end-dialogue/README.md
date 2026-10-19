# Three-condition conversation-memory experiment

| Arm | Accessible conversation history |
| --- | --- |
| `none` | No earlier sessions |
| `short_term` | Last 16 turns from the most recent eligible session |
| `short_plus_long_term` | The same short window plus query-retrieved turns from older sessions |

Questions, released reference answers, response model, instructions, response budget
and temperature stay fixed. **CBT knowledge is explicitly empty in all three arms.**
This first experiment isolates a raw-history memory-RAG baseline, not the full CBT
system or confirmed structured-memory extraction. Benchmark dialogue is not promoted
to confirmed user facts. Arbitrary knowledge-file input is disabled: later integration
must check both source retrieval approval and external-model transmission approval.

## Latest pilot — 19 October 2026

The scored run completed **20 questions across 9 LoCoMo conversations, with 60/60
successful responses**. Mean lexical token-F1 proxies were **0.01887** (no history),
**0.07130** (short window), and **0.20865** (short window plus retrieved older history).
The latter exceeded the short-window proxy on 16 questions, tied on 2 and was lower
on 2; the mean paired difference was +0.13735. These are non-official answer-overlap
proxies, not accuracy, CBT quality or clinical scores. Spot checks found correct short
answers penalized when embedded in longer wording, and wrong dates receiving higher
overlap; semantic correctness still needs reference-based review. Official evaluation
and model-assisted judging have not run; only judge inputs were prepared.

Reproduction metadata:

- Requested model: `deepseek-v4.1-flash`; every scored response reported
  `deepseek-v4-1-flash-260910`. The historical v1 protocol did not preapprove this
  alias. These remain identity-unverified legacy scores, not strict v2 results.
- Seed `20261014`; CPU lexical retrieval; latest 16-turn window; up to 12 older
  turns retrieved. CBT reference context was empty in every arm, and the history
  consisted of raw turns rather than confirmed structured memories.
- Temperature 0; response budget 4,096 tokens; `--thinking-mode disabled`.
  All 60 responses reported zero reasoning tokens and normal `stop` completion.
- Execution commit: `baf5df31ceda6117cbaf8c74b603fd27e0de3777`.
- Protocol SHA-256: `ce7f6c4f341f40fb1cb320ad8ff157ce2ca2872c1c8c86a739c092e3e3559c30`.

Two earlier provider-default thinking rounds stopped after empty, length-limited
responses: the 1,024-token round had 24 valid and 2 empty responses; the 4,096-token
round had 5 valid and 1 empty response. Those rounds are excluded in full, including
their valid responses, and are not mixed with the scored 60-response protocol.

The [per-question CSV](pilot_20261019_scores.csv) contains all 60 score rows: benchmark
ID, condition, category, lexical proxies, API latency and reported total tokens.
The [aggregate JSON](pilot_20261019_summary.json) contains group scores, paired
exploratory intervals and fixed settings. Neither file contains prompts, reference
answers, generated answers or credentials. F1 values are on a 0–1 scale and are not
percentages of correct answers; token counts cover successful scored responses only,
not the excluded failures. This pilot does not test general dialogue quality,
automatic memory extraction, user corrections, safety effectiveness or clinical outcomes.

## Audit the published scores offline

The [minimal audit ZIP](pilot_20261019_audit.zip) includes byte-preserved
`protocol.json`, `prompts.jsonl`, `references.jsonl`, `responses.jsonl`, original
proxy outputs and code snapshots. `SHA256SUMS` covers its members. The external
[audit manifest](pilot_20261019_audit.json) binds the ZIP, every member, published
CSV/JSON, original export time and execution commit. The original full debugging
archive is also retained separately; it is not needed for the scored-run audit.

```bash
python evaluation/end-to-end-dialogue/verify_pilot_audit.py
# Also check question/reference annotations against the pinned upstream release:
python evaluation/end-to-end-dialogue/verify_pilot_audit.py \
  --dataset evaluation/memory-retrieval/data/locomo10.json
```

The verifier runs offline with no API key, does not execute archived source, and
checks raw answers → all 60 per-question scores → the published aggregates. It
preserves `legacy_identity_unverified: true`. The raw protocol file's byte SHA
differs from its canonical JSON protocol hash; both are recorded separately.
Hashes protect integrity against the published manifest, not provider authenticity,
semantic correctness or clinical validity.

This named release contains public LoCoMo benchmark material and generated pilot
answers for noncommercial research. The upstream CC BY-NC 4.0 text, creator
attribution, pinned source links and modification notice are inside the ZIP.
This exception does not permit publishing real-user or patient prompts/responses.

## New runs use strict model identity

Protocol v2 freezes the requested model and an explicit reported-name allowlist
before generation. By default only the requested name is accepted. Repeat
`--accepted-provider-model NAME` to preapprove a provider alias; unknown, missing,
empty or inconsistent returned names fail generation/resume/scoring. This checks
provider-reported names, not cryptographic proof of backend weights.

The historical v1 protocol is not rewritten to add an alias after seeing the
answers. Default scoring refuses it; `score_three_arm.py --legacy-audit` can
recompute identity-unverified `legacy_audit_*` outputs without producing a strict
summary or preparing official/judge inputs. Use the audit verifier above for the
unchanged published pilot.

To prepare a new v2 run on the same 20-question sample after downloading LoCoMo:

```bash
export CBT_EVAL_API_BASE=https://tokendance.space/gateway/v1
python evaluation/end-to-end-dialogue/run_three_arm.py \
  --dataset locomo --input evaluation/memory-retrieval/data/locomo10.json \
  --model deepseek-v4.1-flash \
  --accepted-provider-model deepseek-v4-1-flash-260910 --backend lexical \
  --short-turns 16 --top-k 12 --minimum-relevance 0.05 \
  --short-chars 6000 --long-chars 12000 --max-context-chars 24000 \
  --max-response-tokens 4096 --thinking-mode disabled \
  --seed 20261014 --limit 20 \
  --output-dir evaluation/end-to-end-dialogue/outputs/locomo-pilot-v2
```

This command prepares prompts without API calls. For a new generation run, supply
`CBT_EVAL_API_KEY` through the environment and add `--generate --max-calls 60`;
then score that output directory with `score_three_arm.py`. The alias above was
observed in the historical pilot; a future provider change must fail until a new
identity policy is reviewed and frozen in a new directory. Preparing this v2 run
does not claim a repeat of the historical model responses or old protocol hash.

## Prepare a run without an API

Use the released files pinned in `evaluation/memory-retrieval/benchmark_manifest.json`:

```bash
python evaluation/end-to-end-dialogue/run_three_arm.py \
  --dataset locomo \
  --input evaluation/memory-retrieval/data/locomo10.json \
  --model YOUR_RESPONSE_MODEL --limit 20 \
  --output-dir evaluation/end-to-end-dialogue/outputs/locomo-pilot
```

This validates dataset bytes/SHA-256 and freezes `protocol.json`. It writes
`prompts.jsonl`, separate `references.jsonl`, and `preparation_summary.json`, without
generating responses. Output directories are ignored by Git; do not commit full
benchmark inputs or arbitrary generated histories. Only the explicitly licensed,
named audit release above is published. Fabricated fixtures in unit tests only test
code invariants and are never used as experimental questions.

For full eligible LoCoMo preparation, set `--limit 1540` (4,620 prompt records).
The original 1,986-example release is still checksum-validated; its **446 category-5
adversarial questions are excluded**, because `adversarial_answer` is a distractor,
not a standard answer. A seeded
shuffle selects a pilot sample; the exact question IDs are frozen before generation.

LongMemEval is also supported. The currently pinned file is **oracle cleaned** with
oracle-selected sessions, not its full distractor-heavy history. Label any such run
an **oracle-history generation pilot**; it cannot establish general retrieval or
long-term-memory effectiveness. LoCoMo uses complete released conversation history.

## Opt-in generation, checkpoints and budget

```bash
# Supply secrets via environment variables, never a file or CLI argument.
# CBT_EVAL_API_KEY: a key for the chosen OpenAI-compatible provider.
# CBT_EVAL_API_BASE: optional HTTPS /v1 base URL (no embedded credentials).
python evaluation/end-to-end-dialogue/run_three_arm.py \
  --dataset locomo \
  --input evaluation/memory-retrieval/data/locomo10.json \
  --model YOUR_RESPONSE_MODEL --limit 20 \
  --output-dir evaluation/end-to-end-dialogue/outputs/locomo-pilot \
  --generate --max-calls 60
```

Replace the placeholder with the chosen response model **before** preparing the
generation run. Use a new directory if model/provider/protocol settings change.
Default API base is `https://api.openai.com/v1`; this does not select or configure a
paid model. `--generate` is required, a key must be configured, and pending calls
must fit the explicit `--max-calls` cap (default 30). Three calls are needed per
example. Nothing is sent to a provider by a preparation-only run.

Each successful response checkpoints to `responses.jsonl` with prompt/protocol
hash, provider model, latency and token usage. Resume skips completed calls and
rejects mismatched or duplicate checkpoints. Empty/truncated provider responses
stop the run without a success checkpoint or low-score substitution.
`--thinking-mode provider_default|enabled|disabled` freezes the provider thinking
setting across all three arms; the default sends no thinking parameter, while an
explicit mode sends `thinking: {type: ...}` only where the selected provider supports
it. Changing this setting requires a new protocol/output directory; do not mix prior
reasoning-budget failures into the new run. Empty/truncated errors expose only finish
reason and completion/reasoning token counts, not response bodies or credentials.

## Leakage and chronology controls

- Every LongMemEval question has an isolated subject scope; every LoCoMo
  conversation has an isolated two-participant scope. Different conversations are
  never pooled. This is subject/conversation isolation, not authentication.
- Sessions are sorted chronologically. LongMemEval sessions with
  `session_date > question_date` are excluded; equal timestamps are eligible at the
  released date granularity. Missing or unknown dates fail closed. LoCoMo QA has no
  question dates, so the declared policy is QA after the complete conversation.
- Short-term means the latest session only, capped by turns/characters. There is
  no oracle summary of older sessions. Long-term retrieval uses only older sessions,
  excluding the latest short session.
- Ranking uses the question and eligible history, never gold answers or released
  evidence IDs. The reference-file checksum is frozen separately for scoring.
- Default CPU lexical retrieval is a labelled baseline. `--backend dense` or
  `hybrid` uses the reviewed, file-hashed E5 snapshot. New protocols bind the planned
  immutable snapshot and actual model/runtime identity; caches bind subject, text
  sequence, model bytes and preprocessing. Dense dependencies remain optional.
- System rules stay fixed. History is escaped and marked `data_only`, not
  executable instructions. Whole entries must fit their channel budget before their
  IDs appear in trace; dropped entries are recorded separately. Total context budget
  is measured in characters, not claimed to be an exact token budget.
- Arm order is seeded and shuffled within each example. API temperature is zero;
  provider-side nondeterminism can still exist.

## Scoring is separate

```bash
python evaluation/end-to-end-dialogue/score_three_arm.py \
  --run-dir evaluation/end-to-end-dialogue/outputs/locomo-pilot
```

The scorer requires all three successful answers for every frozen question. It
reports exact-match and token-F1 **proxies**, paired differences and
conversation-cluster bootstrap intervals (not treating correlated QA as independent),
latency/token usage, category-wise results and the released abstention subset.
LoCoMo category 5 is not put into lexical scoring or judge gold. Its upstream
[evaluation code](https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py)
uses abstention handling; a separate official abstention study is still needed.

These lexical proxies are **not** the official benchmark evaluator and poorly
capture some paraphrases, lists and abstention. LongMemEval outputs can be exported
in upstream `question_id`/`hypothesis` JSONL format, but its official evaluator must
be run separately with the upstream version/settings recorded.

The scorer also prepares anonymized `model_judge_pairs.jsonl` and a separate
`model_judge_mapping.jsonl`. Reference answers are visible to the judge, not to
response generation. **Preparation is not execution of a judge**: summaries explicitly
say official scoring/model judging have not run. Anonymity cannot hide all stylistic
cues, and model-assisted scoring is not human or clinical evaluation.

This measures general history-question answering. It does not evaluate CBT clinical
quality, homework adherence, risk recognition, consent, memory lifecycle correctness,
or prove better real mental-health dialogue. Safety tests and the confirmation /
correction / deletion demo are separate deliverables.
