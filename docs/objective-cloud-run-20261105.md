# Cloud experiment status — 5 November 2026

**Incomplete model experiment. No new response-effectiveness result.**

## Verified execution

The operator used an independent Codex Linux cloud container, whose execution
service identified itself as `codex-cloud-environments.chatgpt.com`. Repository
checkout returned main `80551b7edc0199b8870b174e3f144155b0d60c85`, matching the handoff.
No personal Mac, desktop browser, terminal or files were used.

A separate remote cloud browser connected to hosted Google Colab. Colab's guard
passed without modification. The UI identified a Python 3 Google Compute Engine
backend; the executed runtime receipt reported Python 3.13.18, pyarrow 23.0.1,
numpy 2.4.6 and ipywidgets 7.7.1. The executable checkout was exactly
`7832f38a639eb5b5e3c27d6c564db5db25f3999c`. No guard spoofing, neural model download,
package upgrade or local-computer runtime was used.

## Data and frozen pilot

Registered MemoryAgentBench data checksum:
`24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45`
(1,491,588 bytes). The executed inventory reported 800 questions, eight source
contexts and one conservative fact-consolidation family. Each of eight task /
length strata contained 100 questions. This does not represent 800 participants.

Before generation, select two questions per stratum by ascending SHA-256 of
`case_id`. All 16 selected IDs are frozen in the protocol. This finite pilot
does not replace the full benchmark. Its protocol identity is:
`9b798544eaf0f7b4355eb7b2772e199b45f80420dc648154942db54f0751d4fd`.

The pilot has 48 arm rows and 32 distinct canonical prompts. All 16 raw/managed
prompt pairs are equal by construction under the v1 quotation control. Their
shared response cannot estimate semantic or version-management benefit.

## Observed model status and costs

| Outcome | Recorded value |
| --- | ---: |
| Successful arm rows | 0 |
| Pending arm rows | 48 |
| Inflight arm rows | 0 |
| Exhausted arm failures | 0 |
| Actual provider attempts | 0 |
| Valid provider responses | 0 |
| Recorded provider prompt/completion tokens | 0 / 0 |
| Monetary bill | Not observed; no amount inferred |
| Participant effort, unnecessary/repeated checks | Not measured |

One operator preflight failed before any provider request: Colab displayed that
`Objective_Memory_Comparison_Colab.ipynb` lacked access to the existing Secret
named `Capstone`; the request returned `TimeoutException`. Secret contents were
not displayed or exported. This is an access preflight failure, not an API
failure, incorrect benchmark answer, or exhausted task outcome.

A cell-editing error initially concatenated an old line with new source code,
causing a SyntaxError before preparation. The first line was corrected and
verified; preparation then returned the protocol receipt above. Neither this
operator error nor the Secret timeout produced a model call. They remain part
of this execution record rather than being hidden by starting another pilot.

## Recovery artifact

The hosted exporter independently validated and recomputed the partial run and
returned `partial_recovery_checkpoint`, `final: false`, 27 members, 4,322,146 bytes.
Archive SHA-256:
`9a314f5fd04249d85efced59387a478f724515934c6d6084b01e5ef18c6bcb81`.
The cloud-downloaded bytes were checked against this receipt and retained as a
controlled checkpoint. No benchmark prompts, responses, full data or Secrets are
published in this repository status report. The older LongMemEval results were
not changed or relabelled.

## Research gaps that remain

There are no new answered questions to attribute. Public gold answers do not
independently annotate sufficient evidence spans, so retrieval/context coverage
must not be inferred from answer substrings. The new diagnostic ledger can
consume separately reviewed, byte-bound evidence alternatives; without these it
keeps attribution unresolved. A software-observed evidence gap is not proof of
causation.

The separate v2 semantic-source-index code makes source selection genuinely
different when an index is supplied, without rendering unconfirmed summaries as
facts. Its gold-blind extraction journal, model outputs, extraction costs and
semantic quality are still unmeasured. It does not implement or validate a new
semantic version-management benefit. Four-visit user testing and independent CBT
professional review remain outstanding. Software checks do not replace any of
these outcomes.

The next model invocation requires granting this hosted notebook access to the
existing `Capstone` Secret. It must continue this frozen pilot from its matching
executable SHA, with finite batches and canonical-response accounting. Any v2
semantic experiment must use a new protocol and directory, preserving this v1
checkpoint.
