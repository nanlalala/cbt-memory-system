# Evidence checkpoint: CBT continuity, process and outcomes

PR [#23](https://github.com/nanlalala/cbt-memory-system/pull/23) was merged at
`63911d66003f3446f7d720b183d7cc3e2df0c584` after the user explicitly authorized
merging. The exact reviewed head passed GitHub CI. This checkpoint separates
that implementation result from experimental findings. The machine-readable
[evidence registry](../evaluation/dialogue-benefit/evidence_registry.json) records
the corresponding counts and identities.

## Findings available now

| Question | Completed evidence | Conclusion supported |
| --- | --- | --- |
| Did the quotation-control memory arms improve? | 16 questions, 32 canonical calls, 48 logical rows; short 0/16, raw 2/16, managed 2/16 by the frozen official substring rule | Raw and managed prompts were identical. Both credited replies were UNKNOWN/refusal text listing candidate answers; no memory-management benefit demonstrated. |
| Did the semantic follow-up improve? | Four already exposed questions, 18 extraction + 11 answer calls; short 0/4, raw 1/4, semantic 0/4 officially credited | Negative development result. Most extraction requests returned no candidates; source-path observations identify missing support but do not prove a unique causal stage. |
| Did the full new objective comparison finish? | Full source census verified in hosted Colab: 800 MAB and 589 PersonaMem questions | No. The patched v3.1 full run has no new provider calls or full result. MAB is one conservative family and PersonaMem has 20 connected groups, not 1,389 independent people. |
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

## Execution gap

Software work and archive processing have run in the independent Codex cloud
Linux container. Earlier full-data inventories and actual model calls ran in
genuine hosted Colab. The user's Mac, Chrome, terminal and files are excluded.

The current hosted Colab control transport is unavailable, and the independent
Codex environment has no authorized `CBT_EVAL_API_KEY`. Consequently, **zero new
v3.1 model calls** have started. The
[full notebook](../evaluation/objective-memory/Objective_Memory_Full_V3_Colab.ipynb)
pins executable `1511706b3a4b0ac71416998b5b4d752a13c4e313`, uses a new run root and
keeps paid execution disabled by default. It preserves old run identities;
reconnection must not resume an old directory under changed code. No environment
guard was bypassed and no alternative model or unbounded paid loop was started.

This checkpoint supports implementation progress and exposes the remaining
evidence gaps. It does not establish overall dialogue benefit or clinical efficacy.
