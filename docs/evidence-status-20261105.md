# Evidence checkpoint: CBT continuity, process and outcomes

The [full objective run](../evaluation/objective-memory/results/full-20261105/README.md)
completed in genuine hosted Colab on 2026-11-05 after the reviewed launch recovery
in PR [#25](https://github.com/nanlalala/cbt-memory-system/pull/25). Implementation
checks and the observed public-task findings remain separate. The machine-readable
[evidence registry](../evaluation/dialogue-benefit/evidence_registry.json) records
the corresponding counts and identities.

## Findings available now

| Question | Completed evidence | Conclusion supported |
| --- | --- | --- |
| Did the quotation-control memory arms improve? | 16 questions, 32 canonical calls, 48 logical rows; short 0/16, raw 2/16, managed 2/16 by the frozen official substring rule | Raw and managed prompts were identical. Both credited replies were UNKNOWN/refusal text listing candidate answers; no memory-management benefit demonstrated. |
| Did the semantic follow-up improve? | Four already exposed questions, 18 extraction + 11 answer calls; short 0/4, raw 1/4, semantic 0/4 officially credited | Negative development result. Most extraction requests returned no candidates; source-path observations identify missing support but do not prove a unique causal stage. |
| Did the full new objective comparison finish? | 800 MAB + 589 PersonaMem questions; all 4,167 three-arm rows complete; 3,550 actual successful canonical calls | Yes. MAB raw 419/800 versus managed 480/800, +7.625 percentage points; one conservative family, no estimable independent-family interval. Persona raw/managed both 416/589 with all 589 inputs shared; no management-effect inference. |
| Is CBT response quality independently assessed? | Thirty-two archived actual replies across 16 fictional scenarios exported with blank, separately ordered professional-review forms | No human ratings yet. This archive compares old knowledge RAG, not memory management, and was already used during development. |
| Does the system make CBT easier or more useful to people? | Four-visit collection/analysis tools and a separately versioned study protocol | No participant observations yet. Planned slots and fixture checks are not enrolled people or demonstrated benefit. |
| Does it improve symptoms or functioning? | No clinical participants, professional clinical assessments or follow-up outcomes | Not established. A separately approved, powered clinical comparison is necessary. |

The two objective pilots used **61 actual calls and 161,445 reported tokens** in
total, including extraction. No API failures were recorded in those runs. Money
cost remains unverified without a frozen provider price schedule. The four-question
follow-up is part of the earlier sixteen-question exposure, not four new histories.
Original protocols, raw outputs, archive checksums and negative results remain
unchanged. The old 1,500 LongMemEval answers still lack completed judging and are
not relabelled as new evidence.

## Full-run results, errors and cost

| Dataset | Short-term | Raw-history RAG | Managed-memory RAG |
| --- | ---: | ---: | ---: |
| MAB official substring rule | 10/800 (1.25%) | 419/800 (52.375%) | 480/800 (60.00%) |
| MAB strict-format diagnostic | 10/800 | 418/800 | 478/800 |
| PersonaMem official option parser | 364/589 (61.80%) | 416/589 (70.63%) | 416/589 (70.63%) |
| PersonaMem strict-format diagnostic | 363/589 | 416/589 | 416/589 |

The main MAB contrast has 68 managed wins, 725 ties and 7 losses; 22 pairs share
identical inputs. Single-hop results are raw 375/400 and managed 384/400, while
multi-hop results are 44/400 and 96/400. The latter improves by 13 percentage
points but still reaches only 24% accuracy. MAB's length/type variants remain one
conservative fact-consolidation family. The 17 development-exposed questions and
783 question-held-out questions retain their frozen partition; the latter still
share histories and are not independent unseen-history validation.

PersonaMem has 20 connected groups. Equal-group accuracy is 62.2435% for
short-term and 71.1816% for either RAG arm. Its RAG arms share every input and
canonical response, so their zero difference is by construction. The increased
score relative to short-term history concerns access to old history, not semantic
or version management. Literal exact/strict diagnostics are not independent
semantic correctness labels; official scores are retained unchanged.

All 4,167 planned rows remain in the result denominator, including any terminal
failures (none occurred in this run). The 2,062 rows scored incorrect by the
official rules keep
`answer_incorrect_stage_unresolved`. Source parse/version, retrieval and context
observations are available, but no independent evidence annotation establishes
which stage caused each error. MAB uses a deterministic syntactic source-order
version view, not a natural-language CBT goal/homework extractor. PersonaMem uses
lossless quotation spans. Natural-check and repeated-question burden were **not
measured**, rather than observed to be zero.

This run records 3,550 successful canonical calls, zero API failures/unresolved
attempts and no missing usage fields. Reported tokens are **8,768,813 prompt +
8,270 completion = 8,777,083 total**, separately from the earlier pilots. The
successful-call latency median is 1.48296457 seconds; the sum is 5,448.1383
seconds and is not parallel wall time or end-to-end preparation/retrieval time.
No monetary cost is inferred without verified prices. All provider responses
reported the allowed alias `deepseek-v4-1-flash-260910`; matching a returned model
name is not an independent attestation of underlying model weights.

The [public result folder](../evaluation/objective-memory/results/full-20261105/README.md)
contains aggregate evidence and all 4,167 score/status metadata rows. Original
model answers, question text, options, source histories and scoring references
remain in separate controlled cloud artifacts; credentials are excluded. The
[reporting code](../evaluation/objective-memory/reporting/README.md) independently
recomputes deterministic scores and canonical costs before projection.

## How the next measurements address CBT

The [CBT process/outcome evidence review](cbt-process-outcome-evidence.md) links
memory fidelity to goals, practice review, guided discovery, realistic behavioural
learning and adaptation to corrections. Each link needs its own evidence. A
retrieved quotation does not prove a helpful response; a helpful response does not
prove that a skill was learned; skill-task success does not prove symptom benefit.

The nonclinical study compares managed memory against competent raw-history RAG
as the primary contrast, with a short-term condition as a secondary comparator.
One participant stays in one condition across four visits to limit carryover.
All arms must receive the same admitted source information, model, CBT resources,
interface opportunity and access/deletion policy. Task success and participant
effort are measured together; failing quickly cannot become a positive overall
finding. Independent CBT-trained human reviewers assess the supplied history and
responses without condition labels or model scores. Missing, uncertain and not
applicable judgments remain distinguishable.

The study is not launched. Allocation concealment, actual arm fidelity, qualified
facilitation, consent/data governance and institutional review where applicable
remain gates before involving people. No recruitment, clinical intervention or
participant contact has been performed by the assistant. A clinical study needs
an appropriate target population, one prespecified primary symptom endpoint,
functioning and deterioration measures, prospective sample-size justification,
and follow-up under qualified oversight. Model-generated ratings cannot replace
those observations.

Three fictional task variants now supply 36 draft tasks and 48 dated source
turns with exact quotation boundaries. They are researcher-authored materials
with **zero independent human source validators**, not expert gold. The current
collector records operational events and final history/query/response artifacts;
the production source-only generation adapter, full extraction/update/retrieval/
context trace connection and independent check-opportunity annotations remain
unimplemented. Full causal error attribution is therefore unresolved. Do not feed
the combined scoring ledger, which retains deleted/forbidden references, to a model.

## Execution and preserved identity

Software work and archive processing have run in the independent Codex cloud
Linux container. Full inputs, model/API calls and source-based scoring ran in
genuine hosted Colab; post-run report construction outside Colab used derived
metadata only. The user's Mac, Chrome, terminal and files were not used.

The [authorized notebook](../evaluation/objective-memory/Objective_Memory_Authorized_RunAll_Colab.ipynb)
finished after six finite passes, preserving executable
`1511706b3a4b0ac71416998b5b4d752a13c4e313` and suite plan
`940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568`.
Its new run directory did not overwrite earlier runs. A temporary frontend/runtime
connection interruption recovered without restarting or duplicating paid requests.
No environment guard was bypassed and no alternative model or unbounded paid loop
was introduced. Secrets remained in Colab.

Final source and cost identities were validated inside Colab with existing
read-only locks. The 70,083,349-byte controlled final audit archive has SHA-256
`608c76386b70312540397e14951b3fd5f47f5f783d8230b79f246d35b37ebde0`;
the 3,565,003-byte derived export has SHA-256
`21fc3f1879c3033946d78348b1fea5dc9c634e804d4c3f7af67064fc27189ff2`.
Both were transferred cloud-to-cloud and verified. File hashes establish
integrity and identity, not independent clinical validation.

This checkpoint establishes the measured public-task results and exposes the
remaining evidence gaps. It does not establish overall dialogue benefit or
clinical efficacy.
