# External memory evaluation protocol

The ten repository-specific multi-session cases are **development fixtures**, not evidence that the memory system improves conversation quality. Formal memory claims must use published data with benchmark-provided questions, answers and evidence locations.

`benchmark_manifest.json` freezes the source URLs, expected counts, file sizes and SHA-256 hashes used in this project. `validate_external_benchmarks.py` rejects a changed or incomplete download before an experiment begins.

## Primary benchmarks

### LongMemEval — ICLR 2025

- Official repository: <https://github.com/xiaowu0162/LongMemEval>
- Official cleaned dataset: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>
- Paper: <https://openreview.net/forum?id=pZiyCaVuti>
- Abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates and abstention.
- Required reporting: evidence Recall@K/MRR, official QA accuracy by question type, abstention accuracy, latency and context size.

Use the cleaned 2025 release. Generate `question_id`/`hypothesis` JSONL with `write_longmemeval_hypotheses`, then run the authors' official `evaluate_qa.py`. Do not replace the released answers with locally written answers.

### LoCoMo — ACL 2024

- Official repository and data: <https://github.com/snap-research/locomo>
- Paper: <https://aclanthology.org/2024.acl-long.747/>
- Tasks: long-conversation QA and event summarisation, with annotated evidence dialog IDs.
- Required reporting: evidence Recall@K/MRR, QA F1 or the authors' published evaluation method, category breakdown, latency and context size.

## Project-specific deterministic checks

The local fixtures test engineering invariants with objectively checkable outcomes:

- memories never cross user boundaries;
- unconfirmed inference is not retrieved as fact;
- correction prevents superseded-memory leakage;
- deletion removes content from storage and retrieval;
- expiry removes stale memory from active context;
- restricted safety context requires explicit permission;
- empty retrieval is allowed.

These checks may be reported as pass/fail unit tests. They must not be called clinical evaluation or used as a substitute for LongMemEval, LoCoMo or human review of psychotherapy dialogue quality.

## Experimental comparison

Keep the same response model, prompt, knowledge RAG and safety router across all arms:

1. no cross-session memory;
2. recent-session context only;
3. structured long-term memory retrieval.

Freeze the benchmark split and configuration before running the comparison. Report all cases, including failures, and keep memory retrieval metrics separate from final-response metrics.
