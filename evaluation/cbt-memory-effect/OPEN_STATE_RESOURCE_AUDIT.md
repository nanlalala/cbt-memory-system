# Read-only resource audit

`audit_open_state_resources.py` runs **only in genuine hosted Colab** against
completed controlled-state archives. It never loads a model, generates an
answer, opens scoring targets, retries a request, or opens a product database
through a writable product initializer. It writes one new JSON report.

```sh
python evaluation/cbt-memory-effect/audit_open_state_resources.py \
  --model-root /content/ACTUAL_MODEL_ROOT \
  --prepared-dir /content/BEFORE_PREPARED \
  --prepared-dir /content/AFTER_PREPARED \
  --reader-output /content/BEFORE_OUTPUT \
  --reader-output /content/AFTER_OUTPUT \
  --output /content/NEW_RESOURCE_REPORT.json
```

Run after the relevant readers finish and their databases are quiescent.
Include additional `--prepared-dir` arguments for any discarded preparation
passes whose CPU work should be in scope. These are controlled-state preparation
directories, not expert-rating calibration preparations. Separate calibration
generations in the same model-root ledger are still counted once in the global
total, under their explicit purpose and in `other_global_attempt_cost`.

The auditor checks journal-record checksums, completed answer-file hashes,
request/model identities, canonical response-cache bindings, frozen preparation
files, and source case hashes. Input hashes are rechecked before publication.
SQLite is queried with `mode=ro` and `query_only`; no source or response text is
included in the report. A changed or incomplete archive raises an error instead
of producing an apparent completed-run audit.

Use `global_unique_generation_cost` as the unique admitted-attempt total.
`owned_generation_cost` belongs to a run's newly admitted canonical generations;
`referenced_cost_nonadditive` describes receipts reused by that run and must not
be added across runs. Within an owning run, a shared receipt is allocated equally
across its distinct arms for accounting; this is not independently measured arm
latency. An admitted intent is not necessarily a completed `model.generate` call.
Failures, stale outcomes, missing outcomes, and missing token usage are retained.

Answer-path wall time includes generation plus publication checks/cache work.
Model setup/reload measurements retain their original field names and stay
separate. Failed reloads and unrecorded setup attempts are not recast as zero
latency. Source ingestion, source-only worker, indexing/organization, state
projection and retrieval/packing latency remain `null` because the original
preparer did not time these stages. Extraction tokens are reported as zero only
when source-only traces and zero durable extraction attempts jointly support it.
The script performs no new model/provider calls; monetary cost is not inferred.

Tiny metadata-only accounting tests can run outside Colab:

```sh
python -m unittest tests.test_audit_open_state_resources
```
