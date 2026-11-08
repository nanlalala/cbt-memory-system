# Controlled v2 engineering replay

This directory records a finite fictional **software contract replay**, not a
benchmark, extraction-quality test, participant trial or clinical result.

The executable is `8099f98d3e9487db56e3b740a7bd2fbd47cdf5fa`; the subsequent
notebook/documentation commit only pins that executable and saves this output.
The run occurred from a clean checkout of that published GitHub commit in an independent cloud Linux environment without bypassing
any Colab guard. It made **0 real API/model/network calls** and 4 explicitly
local mock-provider calls. The file database and latest control anchor were
kept in a separate approved cloud scratch directory, outside the repository.
No real user/patient data or credentials were used.
The earlier [local-commit replay](engineering-replay-local-20261108.json) is
preserved separately. Its Git tree matches the published executable byte for byte
(`9ed28e2cd7b14ed9cdf7eef8bf8439982cdd89f7`); each run made zero real API calls
and four local mock calls. The published-commit replay uses its own fresh DB/run.

| Fictional scenario | Passed stage assertions |
| --- | ---: |
| Acceptance, partial progress, pause, unknown outcome and source-only worker | 5 |
| Append-only correction, content withdrawal and reopen without resurrection | 5 |
| Current-only no-save pause and no later retained body | 4 |
| Answer outcome unknown, committed source survives and no canonical resend | 3 |
| Total | 17 |

[Actual stage observations and protocol identities](engineering-replay-20261108.json)
are retained. The driver is
[`app/controlled_memory_demo.py`](../../app/controlled_memory_demo.py).
It refuses to overwrite an existing run. Its protocol hash binds the finite
fixture names, clock, limits, driver bytes and per-scenario product protocol
hashes. These hashes establish byte/configuration consistency; they are not
independent authentication of semantic correctness or provider behavior.

Release verification in the cloud ran 953 unit tests: 952 passed and one
optional reference-`jsonschema` differential check was skipped in the default
environment. A separate reference environment ran all 21 schema tests with that
check enabled and passed. Compilation, the original scripted memory demo, and
the two pinned historical artifact verifiers also passed. These are engineering
checks, and old experiments keep their original identities and limitations.

The new hosted-Colab entry has zero execution outputs and model calls disabled.
Its actual browser UI, real model extraction/answers, production recovery, CBT
professional review and participant outcomes have not been verified by this
replay. The prospective v2 evaluation contract must be frozen and a separate
fair model/participant comparison run before claiming memory or CBT benefit.

## Negation review follow-up

The original replays above are retained with their original executable identities.
A later automated review found negated pause/cancellation/refusal predicates being
accepted as opposite states. The corrected executable is
`9166f233e1ba0305ec1c558da77f9afade476e7d`; the Colab entry now pins that commit.
The [fresh published-code replay](engineering-replay-negation-20261108.json) again
passed four scenarios / 17 stage assertions, in a new file/run without overwriting
prior runs. This replay itself does not contain the negation test matrix.

The separate rule and actual dialogue regressions cover predicate-specific
negation, cross-object scope, Unicode/plain contractions, unknown self-reports,
and questions that must not update the state they ask about. All 964 release
tests ran in cloud Linux: 963 passed and one optional reference-validator check
was skipped in the default environment; that check passes in the separate
reference environment. These remain engineering checks, with zero real model/API
calls and no efficacy conclusion.
