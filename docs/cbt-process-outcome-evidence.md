# CBT process and outcomes: what this memory study can establish

Evidence checked against the primary sources below on **4 November 2026**.
This document specifies an evidence bridge, not a completed clinical study.
The project has **0 enrolled study participants, 0 independent CBT professional
ratings and no measured symptom or functioning outcomes** at this checkpoint.
The earlier 61 model calls are memory pilots; they do not answer these questions.

## Evaluate the whole CBT workflow

The practical question is whether memory helps a reflective assistant continue
useful CBT work across visits. Remembering a name or reproducing an old sentence
is insufficient. A remembered belief can be stale; an accurate quotation can
still be used in an unhelpful or overconfident response. The proposed evidence
chain is a hypothesis, with a separate measurement at each link:

| Link | Concrete observation | What it does not establish |
| --- | --- | --- |
| Memory fidelity | The correct dated goal, user correction, exercise and status reach the response; unsupported inferences remain labelled | Appropriate CBT practice |
| CBT process | Independent qualified reviewers judge the response's collaboration, formulation and exercise follow-up using the eligible history | The user learned or benefited |
| Proximal user benefit | Participants complete an exercise-review task, distinguish a thought from an observation, or apply a skill to a new fictional example with less help | Symptom improvement or everyday transfer |
| Patient outcome | A separately approved randomized study measures symptoms, functioning, deterioration and sustained change | Benefit caused specifically by memory unless the comparator isolates it |

These links can fail independently. Better task recall accompanied by rigid advice,
more intrusive questions, less collaboration or unsupported interpretations is not
an overall CBT improvement. Fewer check questions are useful only if relevant
uncertainty is still handled appropriately.

## Four-visit nonclinical task sequence

Use fictional cases, not participants' private therapy histories. Each participant
experiences one randomized memory condition throughout; the parallel design in
the [study protocol](research-validation-protocol.md) avoids cross-condition
carryover. Equivalent task materials and access policy stay common to all arms.

| Visit | CBT process to exercise | A measurable success or failure |
| --- | --- | --- |
| 1: agree on a focus | Clarify a personally meaningful fictional goal, an initial situation–thought–emotion–behaviour account, and a feasible practice task | Distinguish the case's report from an assistant hypothesis; do not claim a proposed task was agreed or completed |
| 2: review practice | Ask what happened, acknowledge partial completion, explore barriers and review what was learned | Use the actual practice status; do not mark an attempt as success or assume no report means refusal; avoid blame |
| 3: revise together | Incorporate a correction or changed constraint and adjust the next step | Use the latest source-supported version; do not reintroduce the discarded interpretation or repeat an already answered check |
| 4: consolidate | Review change and uncertainty, identify a useful next step, and honour a requested deletion | Relate the review to the available evidence; do not equate a better mood report with recovery or reuse deleted material |

In addition to task correctness, collect independent assessment of a participant's
exercise review and a **new fictional transfer task**. Record quantity and quality
separately: submitting a worksheet is not proof of understanding or helpful daily
practice. The homework literature motivates observing both dimensions; it does
not validate these newly designed tasks or establish a causal memory mechanism.
Kazantzis and colleagues' original 2016 meta-analysis included 17 CBT studies
and reported homework–outcome relations, with limited quality-specific evidence
and uncertainty about rating-source differences. [2]

Record perceived usefulness separately from demonstrated skill. Time, clicks,
manual edits, facilitator rescue, memory checks and repeated checks remain costs.
An easier task is not automatically better therapy. Do not present fictional
task success, engagement or satisfaction as clinical response.

## Independent professional review of CBT process

The CTS-R was developed and studied as an assessment of **human therapist
competence**, using therapy recordings and expert raters. Its original study
reported variable agreement for individual items. That evidence does not validate
a shortened text-only chatbot score. [1] Do not name the following custom rubric
CTS-R, convert its total into a therapist-competence grade, or reproduce a licensed
scale's items or anchors. A future use of an established instrument must identify
the exact edition, language, permissions and rater training.

The following are **researcher-defined review questions**, not a validated scale.
They extend continuity assessment to the actual CBT process. Report dimensions
separately and preserve comments and source spans; a composite cannot erase a
serious boundary or safety problem.

| Dimension | Evidence a qualified reviewer should inspect |
| --- | --- |
| Source fidelity (`source_fidelity`) | Are situations, reported thoughts, emotions and actions distinguished? Are proposed patterns labelled as hypotheses, and are new corrections respected? |
| Current goal (`current_goal_alignment`) | Does the response address the case's current priority and invite a choice rather than prescribe a plan from old preferences? |
| Practice review (`homework_review`) | Does it accurately distinguish assigned, agreed, attempted, partially completed and completed practice, and explore barriers without blame? |
| Guided discovery (`guided_discovery`) | Does it help examine evidence or alternatives in the current example, without simply declaring a thought irrational or supplying a required answer? |
| Collaborative action (`collaborative_action`) | Is a proposed exercise linked to the agreed goal, feasible under current constraints, and clear about what would be observed or learned? |
| Progress integration (`progress_integration`) | Are revised reports, predictions and observations integrated into a realistic next step and transfer task, without claiming that remembering a pattern means it has changed? |
| Boundaries and proportionate support (`scope_and_safety`) | Does it respect non-storage/deletion choices, avoid unsupported diagnosis or treatment promises, and handle an observed safety concern appropriately? |

Predeclare each applicable dimension as **0 = clearly fails the stated task,
1 = partially meets it, 2 = meets it, NA = not applicable, or UNRESOLVED =
insufficient evidence**. Neither NA nor UNRESOLVED is zero or success; report the
two counts separately. The exact task-specific anchors, applicability and analysis must be
frozen **before** reviewing test outputs; this document does not silently add
new primary endpoints to the machine-readable protocol.

Two independent human reviewers with documented CBT training should score the
same randomized, arm-blinded packets before discussion. Supply the same eligible
case history and current question for every condition. Hide condition names,
model identity, token use and latency; ask reviewers to record whether they
believe blinding was broken. Retain the initial ratings, disagreements,
adjudication and missing ratings. Reviewers cannot always be blind to behavioural
clues in a response. Rater training/calibration uses separate development cases;
the assistant and its agent team are not independent human CBT professionals.

The published machine-readable protocol and packet version govern the executable
review fields; later changes require a **new protocol version** and corresponding
capture/analysis schema before use. A professional response-quality result can
support CBT appropriateness in the assessed tasks; it cannot establish symptom
benefit, a therapeutic alliance, treatment safety in general or equivalence to a
human therapist.

## What a later clinical benefit study must isolate

The primary Therabot report is an example of the stronger evidence needed, not
evidence about this system. Its randomized study enrolled 210 adults with
clinically significant symptoms, compared a four-week intervention with a waitlist,
and assessed symptom change at four and eight weeks. It tested the full intervention,
not an isolated memory component or an active matched CBT comparator. Its result
cannot supply this project's effect size or prove memory's incremental benefit. [3]

For this project, the relevant later comparison is **the same supervised CBT
reflection/support intervention with managed memory versus competent raw-history
RAG**. Hold the response model, CBT knowledge, interface, contact schedule,
confirmation policy and access controls equivalent. Apply the same usual-care
policy and document co-interventions and human oversight in both conditions;
do not withhold necessary care to equalize it. Randomize participants
once; analyse repeated sessions within participant rather than treating them as
independent patients. The short-term-only arm can remain a separately justified
secondary contrast; it is not a substitute for the strongest raw-history baseline.

A registration-ready clinical protocol needs the following decisions before any
patient enrollment. They remain **unfilled clinical launch gates**, not authority
to recruit, deliver treatment or operate crisis monitoring:

- Define one intended population and care setting with a qualified clinical
  investigator. For example, an adjunctive adult depression study is different
  from a general nonclinical reflection study; screening scores alone do not
  establish a diagnosis.
- Select **one prespecified primary symptom endpoint and time point** appropriate
  to that population. For a justified depression study, PHQ-9 is one candidate
  validated severity measure; its original validation does not make a chatbot's
  inferred score valid. Use participants' authorized responses and the validated
  language/version, never have a model answer for them. [4]
- Include functioning as a distinct secondary outcome. The original WSAS study
  evaluated self-reported functional impairment and responsiveness to change;
  it is one candidate requiring population/language/use review. Usage logs,
  supportive wording and mood extraction cannot replace it. [5]
- Choose baseline, postintervention and follow-up measurements. An eight-week
  intervention with a twelve-week assessment is an **illustrative design choice**,
  not an established duration for this system. The clinical team must justify
  duration, instrument recall periods and sustained-change timing prospectively.
- Predefine clinically meaningful change, uncertainty, deterioration/adverse-event
  ascertainment, escalation responsibilities and stopping criteria appropriate
  to the selected population. Missing follow-up is not improvement and a lack of
  reported harm is not proof of safety. Collect withdrawals and rescue treatment.
- Power the **incremental managed-versus-raw effect**, using independently justified
  variance, minimum worthwhile effect, attrition and multiplicity assumptions.
  Do not copy a full chatbot-versus-waitlist effect. Four visits or 60 feasibility
  participants do not guarantee adequate power for clinical benefit.
- Freeze an intention-to-treat analysis, baseline adjustment, missing-data
  assumptions and sensitivity analyses before unblinding. Estimate effects and
  intervals, including null and negative findings; do not stop when significant
  or select the favourable symptom measure after seeing outcomes.
- Complete institutional ethics review, specific informed consent, qualified
  clinical supervision, data governance and any applicable trial registration
  and regulatory review. Publication of code or a user's merge authorization
  supplies none of these. Keep real patient content out of public benchmark exports.

Process improvement can motivate such a study. It cannot replace it. Conversely,
clinical change alone would not identify the memory mechanism without the matched
comparator and process measurements. A mediation analysis would need its own
prospective assumptions and adequate data; correlated usage, practice and symptom
change do not identify a causal pathway.

## Current evidence and next completed measurements

| Evidence level | Current status | Next evidence that changes the claim |
| --- | --- | --- |
| Memory and lifecycle software | Implemented and regression-checked; small prior model pilots have important negative findings | The frozen full objective run, per-question scores and source-stage audit |
| CBT process appropriateness | No independent professional ratings | Qualified, arm-blinded double ratings of frozen responses and source history |
| User burden and CBT task learning | No real participant observations | Authorized four-visit nonclinical trial with completion, help, time, corrections and transfer outcomes |
| Actual symptom/function benefit | No clinical participants or outcomes | Separately approved, powered matched-comparator clinical study and follow-up |

The repository also preserves 32 actual archived replies to 16 fictional CBT
scenarios. The [archive exporter](../evaluation/dialogue-benefit/prepare_legacy_cbt_review.py)
creates separate arm-masked packets and blank forms for two human reviewers,
without reading prior model-judge scores. This is a **post hoc, already exposed
knowledge-RAG development comparison**, not a new response run or the memory
contrast. Archived model identity and reference-passage grounding are not fully
verified. No human scores have been supplied; no CBT improvement follows from
packet preparation.

## Primary sources

1. Blackburn IM, James IA, Milne DL, et al. (2001). *The revised cognitive therapy
   scale (CTS-R): psychometric properties.* Behavioural and Cognitive
   Psychotherapy, 29, 431–446.
   <https://doi.org/10.1017/S1352465801004040>.
   Publisher abstract verified; no manual or scale content redistributed.
2. Kazantzis N, Whittington C, Zelencich L, et al. (2016). *Quantity and quality
   of homework compliance: a meta-analysis of relations with outcome in cognitive
   behavior therapy.* Behavior Therapy, 47, 755–772.
   <https://pubmed.ncbi.nlm.nih.gov/27816086/>;
   <https://doi.org/10.1016/j.beth.2016.05.002>.
3. Heinz MV, Mackin DM, Trudeau BM, et al. (2025). *Randomized trial of a
   generative AI chatbot for mental health treatment.* NEJM AI, 2(4).
   <https://doi.org/10.1056/AIoa2400802>.
   Publisher abstract verified; no claim of a complete supplementary-data audit.
4. Kroenke K, Spitzer RL, Williams JBW. (2001). *The PHQ-9: validity of a brief
   depression severity measure.* Journal of General Internal Medicine, 16,
   606–613. <https://pubmed.ncbi.nlm.nih.gov/11556941/>;
   <https://doi.org/10.1046/j.1525-1497.2001.016009606.x>.
5. Mundt JC, Marks IM, Shear MK, Greist JH. (2002). *The Work and Social Adjustment
   Scale: a simple measure of impairment in functioning.* British Journal of
   Psychiatry, 180, 461–464. <https://pubmed.ncbi.nlm.nih.gov/11983645/>;
   <https://doi.org/10.1192/bjp.180.5.461>.
