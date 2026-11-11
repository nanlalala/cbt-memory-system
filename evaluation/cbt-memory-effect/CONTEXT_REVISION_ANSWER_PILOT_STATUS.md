# Three probe development answer pilot

`run_context_revision_v3.py` adds a separately authorized answer pilot over a
completed context-revision diagnostic. It does not resume or alter v2, does not
sample new cases, and does not read assessor answers. It freezes exactly three
existing development probes and nine logical arm requests. Canonical requests
may share provider receipts; the global new-provider-attempt ceiling is nine.
This development pilot is not a new held-out dataset-level result.

Prepare in genuine hosted Colab, without Secret access or provider requests:

```bash
python run_context_revision_v3.py prepare-proposal \
  --diagnostic-dir /content/REVIEWED_DIAGNOSTIC_DIRECTORY \
  --old-run /content/cbt-cactus-source-quote-200families-20261109-v2-authorized100 \
  --product-repo /content/cbt-cactus-v2-private-9f7568116bf3 \
  --output /content/NEW_SEPARATE_ANSWER_PILOT_DIRECTORY
```

The generated proposal and request file retain every payload hash, the complete
request budgets, the diagnostic and code identities, and a fingerprint of the
old extraction-cost artifact. Existing extraction and local-embedding costs
remain reused development costs rather than free new work.

Only after new user authorization, write a separate authorization record with:

- `proposal_sha256`: the exact generated proposal hash;
- `user_authorized`: `true`;
- `maximum_new_provider_calls`: `9`;
- `authorization_source`: a faithful reference to the actual new user instruction.

Then run with `--proposal-dir` and `--authorization-json`. No CLI fixture bypass
is exposed. Unit fixtures call the Python function with an explicit fake
transport, and fixture/live proposals cannot be interchanged.

Each real run validates old source controls, source snapshots, original roles,
order, current input and all model-visible bytes before credential access. It
copies the permitted original records and the exact current question into a
new durable store per case, with no extraction jobs. Durable send and publication
admission bind the request to those records. Immediately before network use and
after generation, the original checkpoint's controls and snapshots are checked
again. Old files are never written. A changed, withdrawn or unavailable source
prevents response publication.

The new provider journal fsyncs its intent before each send, records available
token usage and latency, shares exact canonical responses, and blocks automatic
resend after interrupted, failed or stale requests. Generated answers remain
unscored in `answers_unscored.jsonl`; separate scoring must preserve the pilot's
development identity and all failures. Secret `Capstone` is read only inside the
genuine Colab execution path and is never written to an artifact.

Validation: seven tiny checkpoint/fake-transport tests cover proposal creation,
authorization rejection, finite complete execution with unchanged old files,
unknown sends without retry, provider identity checks including canonical shared responses, new-store source snapshot checks, and withdrawal/change after generation preventing
publication. No real provider call has been executed by these tests. A real
proposal must be prepared and reviewed against the completed diagnostic before
this sender is used.
