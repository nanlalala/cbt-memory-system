# Four-visit dialogue-benefit research protocol

Protocol `dialogue-benefit-four-visit-v1`, version 1.0, 5 November 2026.
Machine-readable contract: [protocol.json](../evaluation/dialogue-benefit/protocol.json).

**Status: study preparation; no human study has started.** There are 60 planned
allocation slots, **zero enrolled participants and zero independent professional
ratings** in this protocol. A generated allocation file is not enrollment.
Collectors, scoring code, synthetic verification and a review packet are research
infrastructure, not evidence of benefit. The current memory demonstration is not
yet a deployed, validated three-arm participant-study service.

## Questions and limits of the inference

The primary question is whether managed memory, compared with a competent raw
history baseline, supports more correct cross-session follow-up with less task
burden under the same response model, admitted history and privacy policy.
The two co-primary outcomes are objective task completion and restricted task
time. Natural checks, repeated questions, correction effort, costs and independent
CBT process ratings explain these outcomes; they are not hidden components of a
single overall score.

Three conclusions require different evidence:

| Claim | Required observation | Current status |
| --- | --- | --- |
| Memory supports fictional four-visit tasks | Randomized participant task outcomes and burden | Not collected |
| Responses support an appropriate CBT process | Independent qualified, blinded professional ratings with evidence | Not collected |
| People experience better symptoms, functioning or sustained practice | Prospectively registered patient trial, suitable comparator, qualified clinical oversight and follow-up | Not conducted |

Correctly recalling a homework statement is not proof that the homework was
helpful, performed well, or changed symptoms. Professional approval of a response
is not a patient outcome. See [CBT process and outcome evidence](cbt-process-outcome-evidence.md)
for the separation between task fidelity, process quality and actual benefit.

## Design, population and allocation

Use a three-arm, parallel, randomized **nonclinical feasibility study**, with 60
adult volunteers, 20 planned slots per arm. Each person completes four visits in
one assigned arm. Use fictional cases about ordinary study/work routines; do not
ask participants to disclose diagnoses, trauma, medication or real treatment
histories. Participants must be able to consent and read the single frozen study
language. Record language, relevant accessibility needs and prior system exposure.
Do not exclude a participant after randomization because a task went poorly.

Parallel allocation avoids carryover from experiencing another memory system or
seeing its treatment of the same history. The former suggestion to counterbalance
arms within a person is superseded **for this protocol**. Counterbalance equivalent
fictional task-set order across arms, not the memory condition. Each participant
has an isolated identity, store, index and session history. No cross-participant
information enters retrieval.

An investigator who does not deliver sessions generates a concealed 1:1:1
allocation, using permuted blocks and a seed committed by hash before enrollment.
The restricted allocation map is unavailable to scorers and professional raters.
Within each arm, assign the three task-set variants as evenly as 20 slots allow;
retain the allocation/order table. Enrollment follows the prespecified slot order;
replacements for withdrawals are prohibited. Participants and facilitators cannot
be fully blinded to visible memory behavior. Collect their arm guesses at close-out
and disclose this limitation. Randomize opaque response IDs and presentation order
for assessors; ask assessors whether an arm was recognizable before unblinding.

| Arm | Information supplied to the common response model |
| --- | --- |
| `short_term` | Identical recent-session window, no older-session retrieval |
| `raw_history_rag` | Recent window plus permitted original older turns with source/time/version metadata and a minimal visibility/update event ledger |
| `managed_memory_rag` | Recent window plus source-supported records produced by the actual memory manager and retrieval/context pipeline |

The primary contrast is **managed minus raw**. Comparisons with short-term are
secondary and measure access to older history as well as representation. Basic
correction, expiry, access, deletion and don't-save controls apply in all arms;
do not give managed memory extra participant answers or make raw history retain
forbidden material. Use the same automatic quotation policy and relevant natural
checking policy. Candidates remain dated historical evidence unless explicitly
confirmed; silence is not confirmation. If raw and managed rendered prompts are
identical, label that exposure as control equivalence and share a canonical
response where applicable; do not manufacture a representation effect.

## Four visits and fixed scoring opportunities

Target visit days are 0, 2, 7 and 14, with respective windows of day 0, days 1–3,
days 5–9 and days 11–17. Keep the same four-visit schedule in every arm. Each visit
has a 900-second elapsed task limit, including model waiting and checking time;
documented breaks are excluded. Expiry tests use declared real timestamps, not
unannounced changes to the application clock.

The participant sees only the current task card. A separate scoring ledger gives
the intended fictional facts, their source spans, effective versions and permitted
visibility. It is fixed before generation, not produced by the system under test,
and never enters an AI prompt. Two independent annotators validate the task
references before any participant outputs are scored. Freeze exact task wording,
equivalent variants, source ledger and eligible-check opportunities in the run
manifest before enrollment; these materials are an outstanding launch gate.

| Visit | Tasks | Objective and CBT-process opportunity |
| --- | --- | --- |
| 1 | Setup; no primary follow-up tasks | Agree a fictional goal/agenda and feasible between-session activity; establish two unrelated facts; add one explicitly time-limited statement. Establish source dates without requiring item-by-item review. |
| 2 | `T01` current goal; `T02` partial homework report; `T03` historical versus current report; `T04` unanswered check | Revisit goals collaboratively, distinguish completion from missing information, link reported situation/thought/emotion/action only where supplied, and leave unanswered current-state checks unresolved. |
| 3 | `T05` explicit goal correction; `T06` activity quantity revision; `T07` historical-version query; `T08` ambiguous update | Incorporate an explicit change, adapt difficulty to the reported barrier, retain history for dated questions, and seek clarification before an ambiguous change rather than inventing a formulation. |
| 4 | `T09` deleted-source follow-up; `T10` don't-save follow-up; `T11` retained unrelated record; `T12` new-session no-return check | Review change and a next step, perform scoped deletion/opt-out, retain permitted unrelated history, and verify that deleted/excluded sources do not return after a new session or import. |

Tasks `T01`–`T12` each count once per randomized participant. Each task has a
predeclared completion rule, permissible historical claims, forbidden claims,
required source evidence, and whether a clarification is eligible. Asking an
appropriate question can satisfy an ambiguity task; claiming a definite answer
cannot. The same task card supplies any response to a clarification in all arms.
Spontaneous extra participant information is retained and annotated as additional
exposure, not silently provided to other conditions. Facilitator assistance is
logged; a rescued task is unsuccessful under the primary definition.

## Co-primary endpoints and denominators

1. **Task success:** per-person successful tasks divided by **12 planned tasks**.
   Success requires the intended task action/answer, no contradictory current-state
   assertion, correct source/person/time/version, and compliance with the applicable
   deletion/opt-out rule. Two arm-blinded annotators independently review source
   evidence and outputs; a third adjudicates disagreement. Missing, timed-out,
   failed, adjudicated unscorable because of missing system evidence, and assisted tasks score zero
   in the conservative primary operational analysis. Publish these categories
   separately; annotation uncertainty is not proven semantic failure. A completed
   task awaiting independent ratings is `unresolved`, not incorrect: it blocks
   comparative confidence intervals until assessment is complete and the study
   is sealed. Do not fill pending professional ratings with zeros.
2. **Restricted elapsed task time:** sum across four visits. For a visit completed
   without rescue, use observed elapsed seconds up to 900, including model latency
   and checks, excluding logged breaks. An incomplete, missing, timed-out or rescued
   visit contributes 900 seconds. Range is 0–3600; lower is better. This is a
   prespecified failure-sensitive operational burden measure, not a validated
   psychological burden scale. Also report actual observed time and failure rates
   so the penalty cannot hide a tradeoff.

The independent unit is the participant, never the turn, task or rating. Analyze
according to allocated arm for every consent-retained randomized participant,
including nonusers and dropouts. Planned but never enrolled slots are not people
and do not enter effect denominators. A participant withdrawing permission to keep
data is shown in the flow table as data withdrawn: do not recreate erased outcomes
or quietly call a retained-data analysis full ITT. Report arm counts and best/worst
outcome bounds for missing/erased randomized data wherever lawful aggregate counts
remain; if this is impossible, report why.

Give arm means and managed-minus-raw differences. Use person-level resampling
within randomized arms, fixed seed, 10,000 bootstrap replicates; report 95% intervals
for descriptive outcomes and **97.5% intervals for each co-primary endpoint**
(Bonferroni family coverage at least 95%, subject to bootstrap approximation).
Report small-sample limitations and a randomization-based sensitivity analysis if
the allocation procedure supports it. Preserve the direction of effects: higher
success and lower time are favorable. A feasibility sample is not a confirmatory
superiority trial; no absence-of-harm or equivalence claim follows from a nonsignificant
result. Do not pick one favorable endpoint and label overall benefit demonstrated.
If a later confirmatory protocol claims joint success, require improvement in both
co-primary domains using its frozen multiplicity rule and clinically/user-justified
effect thresholds.

## Burden, fidelity, safety and cost alongside the primary outcomes

Report automatic captures, manual reviews, edits, correction actions, deletion
actions, check prompts, check replies, unanswered checks and facilitator rescues.
A repeated check is a **subset** of check prompts: same source/record version and
current issue, no new supporting information, within the frozen cooldown. A check
may be both repeated and unnecessary but contributes once to total checks.
Independently classify relevance from the task ledger and visible history. Report
all checks per 12 planned follow-ups, per completed follow-up and per independently
eligible check opportunity; undefined denominators are `not_applicable`, not 100%.
Freeze the present 24-hour pending-check expiry and seven-day repeat-check cooldown
as engineering settings; neither is an established clinical threshold.

Report supported personal-claim precision and eligible-fact recall together;
silence cannot be perfect fidelity. For every correction/deletion/opt-out,
record eligible events, requested actions, completed actions, contexts assembled,
answers and inspected stores/indexes/buffers/caches/new exports. Report any forbidden
exposure as a separate violation, even if the task otherwise succeeds. Absence of
a marker is not proof of absence of a paraphrase. Scoring archives are separately
controlled research copies with disclosed retention; deleting application memory
does not retract previously seen chat text or pretend to erase every archive.

For each attempt retain provider/model revision, stage, request identity, timestamp,
status, retries, input/output/cached tokens where reported, latency, and billed cost
with currency and price/date provenance. Unknown cost stays unknown. Include
extraction, updates, indexing/embedding, retrieval, generation, failed attempts and
participant/facilitator/reviewer time. Preparation and follow-up cost are separate.
Freeze per-stage call ceilings, maximum retries and currency cap before any live
run; stop at the cap, keep the failures, and never silently change model. These
caps and provider identity are mandatory run-manifest gates, not defaults that
authorize unbounded calls. Full data/model/API execution remains hosted Colab;
the cloud capture/scorer can be verified with synthetic fixtures only.

## Evidence-based failure diagnosis

Retain trace identifiers and hashes for source history → extracted representation
→ state/version update → retrieved candidates → rendered context after trimming
→ final answer. Record source version/text hashes and span bounds, not IDs alone.
Identify the first **observed** failing transition only when independent annotations
establish required source content and the corresponding stage evidence exists.
Log all observed policy violations in addition to the first failure. Missing traces,
unclear references and answer mistakes alone remain `unresolved`; a substring hit
does not prove causal attribution. A fixed-evidence replay is a separate oracle
diagnostic, not the deployed arm's outcome. Annotation uncertainty, system failure
and participant task deviations have distinct fields.

## Independent CBT process review

Use at least two independent CBT-trained professionals per sampled response, plus
a third qualified adjudicator. Before access, record relevant professional
registration/credential, CBT training, current practice/supervision experience,
language competence, conflicts and relationship to the developers. Raters must
not be project developers or assistant agents. A professional's identity/credential
is verified under the approved study process, not asserted by a model.

Use a **research-specific, unvalidated** 0/1/2/`NA`/`UNRESOLVED` rubric; never call its sum a
validated therapy competency score. The seven executable dimensions are
`source_fidelity`, `current_goal_alignment`, `homework_review`, `guided_discovery`,
`collaborative_action`, `progress_integration`, and `scope_and_safety`, with anchors in
[the CBT evidence note](cbt-process-outcome-evidence.md). Zero means inappropriate
or materially missing where required; one means partial/mixed; two means appropriate
for the visible task; `NA` means not applicable with reason and `UNRESOLVED` means
insufficient evidence with reason. Neither is zero or success. Source fidelity is
assessed against dated evidence and is never merged into an overall CBT sum. Review whether the response respects
the user's agenda, reviews actual homework rather than inventing adherence, treats
formulation as a hypothesis, offers collaborative discovery rather than labels,
proposes a testable feasible activity when appropriate, adjusts workload, and
supports review/generalization without claiming recovery.

Provide the same deidentified eligible source history/task card and response to
each rater, preserving content but removing arm names/system-specific headers.
Randomize response IDs/order using a restricted map. Score a separate calibration
set first, freeze anchors, then lock independent ratings before adjudication and
arm unblinding. Report rater coverage, every dimension, not-applicable counts,
agreement before adjudication, and disagreement reasons. Weighted agreement for
ordinal dimensions is descriptive; repeated responses are clustered by participant.
Never let untrained crowd judgments or model scores replace this requirement.
Development/exposed historical responses may be packaged for independent review
now but stay a separate, post hoc exploratory packet, not this randomized study.

## Sample size: feasibility and an explicit future calculation

The fixed 60-person target is for feasibility, variance and missingness estimates;
it is not justified as sufficient to prove benefit. With 20 per arm and an assumed
participant-score SD of 0.25, a normal approximation gives a 97.5% difference CI
half-width of approximately **0.177**. Twelve tasks per person do not make 720
independent participants. Do not recruit until a favorable p-value appears.

For a separate two-arm confirmatory contrast, an illustrative equal-size normal
approximation is `n = ceil(2 * (z_(1-alpha/2) + z_power)^2 * sigma^2 / delta^2)`
per arm, where `alpha=0.025` per co-primary endpoint. With 90% power per endpoint
(at least 80% joint power by the union bound), SD 0.25 and a 0.10 absolute success
difference, this yields **156 evaluable people per arm**, or **195 per arm** after
a hypothetical 20% attrition inflation. A burden effect of two checks with SD four
would require 100 evaluable per arm, but checks are **not the present primary time
endpoint**; its SD/effect must be estimated and justified separately. These are
illustrations, not pilot results, clinical thresholds or a final sample-size promise.
An independent statistician must freeze a new confirmatory protocol using justified
effects, time variance, missingness, clustering and multiplicity before recruitment.
External benchmark question counts cannot supply that human-study power.

## Launch gates, ethics, retention and clinical follow-up

Before enrolling anyone, obtain institutional ethics review/determination and approved
information/consent materials, identify responsible investigators and qualified
facilitators, register the protocol/analysis plan, freeze source/task manifests and
code/model/prompt/retrieval identities, demonstrate three-arm fidelity end to end,
verify isolation/deletion/opt-out across all arms, set call/cost caps, and test the
restricted data/export path. Record approval/registration identifiers; do not infer
an exemption or invent one. A code commit is not approval. This work does not
initiate recruitment, contact participants or expose a patient-facing service.

Consent must explain fictional tasks, randomization, AI fallibility, all retained
data and who sees it, independent professional review, provider transfer/region,
compensation not contingent on success, withdrawal and application-versus-research
deletion. The host institution must approve exact retention duration, access roles
and response to unexpected personal disclosure. Until specified these are launch
blockers. Stop an individual's session on request/distress or accidental real
sensitive disclosure; a trained human follows the approved escalation process.
Any cross-user leak, forbidden-source reimport or unsafe instruction pauses the
affected deployment for review. Preserve authorized audit evidence, document the
incident, and resume only under an approved amendment; do not fix mid-study and pool
versions silently.

A future clinical protocol must identify a real target population/indication and
one validated primary symptom outcome with a prespecified meaningful difference,
function and actual practice quality/adherence as separate outcomes, baseline,
post-intervention and longer follow-up, concealed allocation, assessment blinding
where feasible, harms/deterioration monitoring, and qualified clinical oversight.
Use an active comparator with the same CBT content/model/contact and only the
memory component differing to attribute an effect to memory; a waitlist comparison
answers another question. Choose treatment duration and follow-up with clinicians
and participants; four fictional visits are not a treatment dose. Power and ethics
are separate, new gates. Do not auto-fill human symptom or alliance scales with AI.

## Protocol identities and reporting sources

New human runs need a new immutable directory. Preserve prior public QA pilots and
their negative/null results unchanged. Before enrollment create a manifest binding
this protocol JSON, task/ledger hashes, code commit and file hashes, model/provider
revision, prompts, budget units (characters unless actual token counting is used),
retrieval/embedding configuration, access/check policy, allocation commitment,
annotation/rater manual, exclusions, analysis and approved data policy. Reject
resume on a mismatch; use a documented new version/amendment instead.

These sources guide transparent reporting; following a checklist does not by
itself validate the design, establish regulatory compliance or demonstrate benefit:

- [SPIRIT 2025](https://www.bmj.com/content/389/bmj-2024-081477) and
  [CONSORT 2025](https://www.bmj.com/content/389/bmj-2024-081123): current core
  protocol/results reporting, open science and participant flow.
- Cruz Rivera et al., [SPIRIT-AI](https://doi.org/10.1038/s41591-020-1037-7)
  (2020; [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7598944/)):
  describe AI version, data eligibility, use setting and human–AI interactions.
- Liu et al., [CONSORT-AI](https://doi.org/10.1038/s41591-020-1034-x) (2020):
  disclose system/input/output details and performance-error analysis. Use its
  AI-specific items alongside the current core, noting its original 2010-core context.
- Vasey et al., [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9) (2022):
  human factors, workflow and errors in early live clinical evaluation. The present
  fictional study is outside that clinical setting; use relevant considerations,
  not a claim of completed DECIDE-AI clinical evaluation.
- Eysenbach and CONSORT-EHEALTH Group,
  [CONSORT-EHEALTH](https://www.jmir.org/2011/4/e126/) (2011): reproducible
  web-intervention delivery, use/attrition and support details.
- [UCL CBT competence framework](https://www.ucl.ac.uk/brain-sciences/pals/research/clinical-educational-and-health-psychology/research-groups/core/competence-frameworks/cognitive-and-behavioural-therapy):
  collaborative work and guided discovery inform reviewer questions; this does
  not validate the new text-response rubric or establish therapist competence.
