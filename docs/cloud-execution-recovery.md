# Recovering objective execution without cloud browser control

This is an execution-recovery record, not a new effectiveness result. The full
1,389-question objective comparison has not yet produced responses. Original
61-call pilot artifacts, protocols and negative findings remain unchanged.

## The available alternatives

Repeated cloud-browser calls failed before returning any page state: reading
control documentation, resetting the session, selecting an existing tab and
creating a new tab all reached the same 300-second tool-call timeout. This
identifies a nonresponsive control path, not a verified failure of Colab itself.
The available tools expose no service-administration/restart endpoint. No user
computer was used and no user-browser fallback was attempted.

Google's [official Colab CLI](https://github.com/googlecolab/google-colab-cli)
provides a browser-independent controller for real Colab runtimes. Google
[announced it on 5 June 2026](https://developers.googleblog.com/introducing-the-google-colab-cli/).
Version **0.7.5** was actually installed in the independent Codex Linux cloud;
`colab --help` and `colab version` both succeeded. This installed a controller,
not a Colab runtime, a model or an experiment. No cloud runtime was allocated.

The controller has no authorized Google login cache and no task-provider key.
The official CLI's remote OAuth flow requires account authentication; account
credentials, authorization codes and model keys must not be pasted into chat.
Do not reuse platform/service credentials for the user's Google account.
The [official Secrets issue](https://github.com/googlecolab/google-colab-cli/issues/48)
also records that `google.colab.userdata.get()` depends on the notebook frontend;
the implemented workaround is environment injection, not direct access to the
existing web Secret. Installation alone therefore does not authorize or enable
this project's paid execution. The official
[Colab MCP API](https://developers.google.com/colab/api/reference/mcp) likewise
requires authentication and is not a connected tool in this session.

## Fix the existing entry before retrying

The full notebook embedded the correct compact, gold-free 16-question exposure
metadata but wrote it as `previously_exposed_cases.jsonl`, without its original
protocol. Frozen executable `1511706b3a4b0ac71416998b5b4d752a13c4e313`
requires a file literally named `cases.jsonl` next to the original `protocol.json`,
and verifies their identities before preparation. The prior entry could not pass
that check even with a working browser.

The repair restores the exact two files from the verified original pilot audit:

| Original member | Bytes | SHA-256 |
| --- | ---: | --- |
| `run/cases.jsonl` | 11,736 | `d9cd969e9f9d22d3f9b35626433d6fd809e9ad3c4cf60616ddfd89651e6b29e9` |
| `run/protocol.json` | 9,180 | `bb71df0ba73f053d7a7fa0b839af51b7803d3a768b47429fdda3fc068af049d2` |

The protocol identity is
`9b798544eaf0f7b4355eb7b2772e199b45f80420dc648154942db54f0751d4fd`.
These exposure files contain metadata and hashes, not history bodies, released
answer text or credentials. Their original bytes and provenance must be verified;
do not invent a replacement protocol, weaken `_exposure`, spoof the Colab guard
or relabel the old experiment. The scientific executable, model and protocol stay
pinned; the corrected launcher uses a separate run directory.

## Minimal manual handoff when control is unavailable

The separate authorized Run-All entry is intended for the existing instruction
to run the experiment:
[open it in hosted Colab](https://colab.research.google.com/github/nanlalala/cbt-memory-system/blob/main/evaluation/objective-memory/Objective_Memory_Authorized_RunAll_Colab.ipynb).
Opening it does not start paid calls. Explicitly choosing
Colab's **Runtime → Run all** starts the bounded cloud workflow. Colab may ask
the account owner to grant that notebook access to the existing Secret named
`Capstone`; it must remain in Colab. No key is printed, embedded or exported.
The assistant does not operate the user's desktop or local browser.

The launcher must verify genuine Colab, check out the fixed executable, restore
the exposure files, freeze all 800 MAB and 589 PersonaMem questions, and use the
same fixed provider/model and shared budgets. Generation must finish before
final scoring. The maximum is four workers, ten rounds per execution shard and
sixty attempted calls per batch across twelve shards: at most **7,200 attempts**
per full invocation, with the existing two-attempt canonical-request ceiling and
no automatic inflight recovery. Identical prompts retain canonical deduplication.
The cap is not an expected bill; monetary cost remains unknown without verified
pricing. All recorded attempts, failures and usage remain in the journals.

Files under `/content` are real Colab runtime checkpoints but are **ephemeral**.
They do not survive runtime deletion. Preserve the runtime until a controlled
audit copy has been retained. No benchmark text is automatically downloaded to
the user's computer or published to GitHub, and no Drive mount is silently added.
Only completion of the real workload can establish new objective results; a
repaired notebook, installed CLI or successful offline bootstrap is not evidence
of memory benefit, CBT process quality or patient outcomes.
