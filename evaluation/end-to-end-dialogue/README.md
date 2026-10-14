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
benchmark history or reference answers. Fabricated fixtures in unit tests only test
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
  `hybrid` enables multilingual E5 through the existing encoder; dense dependencies
  are optional. Corpus caches bind subject, text sequence and model.
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
