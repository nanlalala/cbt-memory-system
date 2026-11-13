# Drained CACTUS continuation after a terminal length response

The observed v4 CACTUS extraction received an exact provider response with
`finish_reason=length`. The old reader retained it as invalid and stopped
admissions. Already queued cases continued to prepare independent fact targets
before reaching the stopped reader. The old study ended at 273/600 processed
cases, with 485 local generation attempts and 327 retained `UnknownAttempt`
case errors. Those local errors are not 327 additional provider sends.

The new entry is separate from that immutable run. It authenticates the exited
C6 worker and source hashes, imports the entire received receipt chain, and
preserves the exact invalid extraction without resending it or treating its
truncated text as a fact. No active old worker, dispatcher or protocol is
patched. Old LoCoMo can continue under its independent allowance and checkout.

In a genuine hosted Colab parent kernel, use a new pinned clean checkout:

```python
api = runpy.run_path(str(new_repo / "evaluation/architecture-rerun/cloud_c6_terminal_recovery.py"))
c6_terminal_recovery = api["start"](
    globals()["_capstone_non_mab_continuation_v1"],
    repo=new_repo,
    expected_commit=new_commit,
)
```

One fixed output root per old nonce prevents duplicate admission. Workers are
credential-free; the new relay reuses initialized parent transports and their
shared four-call semaphore. No new Secret is read. The source-only phase is
followed by reconciliation of the original CBT finite budget against old paid
intents, the already debited three previous DeepSeek attempts, and all current
v4 CBT parent admissions, including the truncated response. LoCoMo's allowance
is never borrowed, and no new cap is introduced.

Generation resumes the unchanged 600-case, three-arm roster. The explicit new
`source-bound-terminal-length-v1` policy isolates only an exactly wire/model
bound terminal length response: preserve raw output, usage and invalid status;
do not retry, score it, or stop unrelated arms and cases. Missing responses and
assessments remain in full planned denominators. Unknown sends or mismatched
model/wire metadata retain stop-on-ambiguity behavior. Older protocols without
the policy flag retain their original behavior. Prompts, output limits and
ratings are unchanged by this failure policy.

The first CACTUS five-dimension run uses the separate v3 complete-format prompt
and at most its original 1,980-call allowance. It does not first spend another
full v2 grading run. P7 is then migrated through exact cached receipts with zero
new generation or assertion calls before a complete 24-call v3 grading run,
only if actual original Qwen allowance remains. A budget shortfall is reported
with required and available counts, not silently expanded.

V3 strict summaries preserve invalid grades and valid-score coverage. Original
v2 and post-hoc results remain untouched; no preferred scores are merged across
versions. Processing completion is distinct from complete valid coverage or an
improvement in memory quality. MAB and the frozen LoCoMo/LongMemEval checkouts
are unchanged.

The existing v2 post-hoc CLI must not be pointed at a v3 directory. A later
zero-call readout can explicitly pass the v3 API to the same frozen parser and
report its per-dimension coverage alongside, without replacing, the strict
summary.
