# Evaluation

The main question is whether manageable long-term memory improves cross-session
tracking compared with retrieval over the **same raw history**. More historical
information, reliable state changes and helpful dialogue are distinct findings.

The [final evaluation protocol](../docs/validation-next.md) is the single source
for the next comparison, task definitions, analysis and remaining work.

The separate [four-visit dialogue-benefit workflow](dialogue-benefit/README.md)
implements a nonclinical feasibility protocol, source-bound independent review
and participant-level analysis. Its task materials remain drafts with zero
independent human reference validators. The
[evidence checkpoint](../docs/evidence-status-20261105.md) records actual pilot
results and missing full runs; [CBT process/outcome requirements](../docs/cbt-process-outcome-evidence.md)
separate appropriate responses, learning and patient benefit. No human study has
started.

| Layer | Evidence and measures | Interpretation |
| --- | --- | --- |
| External memory tasks | Selected MemoryAgentBench conflict tasks and PersonaMem-v1 choices, scored against released answers by deterministic rules | Current-fact and preference performance; new adapters/comparison pending |
| End-to-end memory control | Goals, homework, state, correction, expiry, deletion/isolation; expected-state event replay and source traces | Software/continuity compliance; synthesized interventions are not clinical gold |
| User value | Real participants complete a four-visit fictional workflow; completion, time, errors, correction and confirmation effort | Usefulness and usability; not conducted yet |
| Open-ended response quality | Independent blinded review; model-assisted ratings as a supplementary diagnostic | No model score substitutes for users or CBT-qualified reviewers |
| Knowledge retrieval and safety | Evidence Recall@k/MRR, citation support, risk/access and role-boundary checks, reported separately | Retrieval and safeguards do not establish treatment efficacy |

The primary paired comparison is **raw-history RAG versus managed-memory RAG**;
short-term history is a reference arm. Both receive the same admitted history,
user events, time/deletion/access filters, response model and context budget.
Report independent history/persona groups, failures and full preparation/update/
retrieval/generation costs. Do not combine all layers into an arbitrary total.

The existing [three-condition historical-QA workflow](end-to-end-dialogue/README.md)
keeps its frozen v2 protocol and original model-assisted endpoint. It is supporting
work, not the new overall evaluation. Its controller is paused; existing responses
are retained. Future model runs, embeddings and full datasets run only in hosted
Colab. Small local mocked checks test software behavior only.
