# Validation protocol: manageable memory and cross-session continuity

**Status — 5 November 2026:** executable public-gold comparison and synthetic
control replay implemented; a completed semantic-memory/user-benefit study is still
missing. The frozen LoCoMo / LongMemEval v2 experiment remains unchanged. Its
historical-QA results and model-assisted grading are supporting evidence; they do
not establish that semantic memory improves CBT follow-up. Regression tests
establish software behavior, not user benefit.

The main research question is:

> Does correctable, expirable and deletable long-term memory support cross-session
> tracking more accurately than retrieval over the same raw conversation history?

The contribution may be better control, fewer stale facts or lower context cost,
even if ordinary factual QA is similar. Do not claim a general dialogue or
treatment improvement from an average memory-QA score.

## What will count as evidence?

| Evidence | Main question | Scoring | What it cannot establish |
| --- | --- | --- | --- |
| External tasks with released answers | Does management help recall current facts and apply changing preferences? | Frozen official deterministic evaluators on selected tasks | Natural dialogue quality, consent, CBT benefit or clinical safety |
| End-to-end event replay | Are explicit updates, expiry, deletion and access rules honored throughout the real pipeline? | Predeclared state/visibility oracles before and after each event | Prevalence of these situations or benefit to real users |
| Independently annotated continuity cases | Are goals, assignments and reported changes followed faithfully? | Linked source spans, task/status labels and independently checked response claims | Treatment efficacy; generated references alone are not independent gold |
| Task-based user study | Can a person review, revise and forget information with little effort? | Task completion, errors, time and review burden; actual participant feedback | Clinical improvement or long-term engagement |

There is **no weighted overall score**. Publish these results separately.
Model-judge ratings may help locate errors, but are not the primary proof of
overall quality. Missing human annotation or a user study stays missing; another
simulated user or model rating does not fill that gap.

This separation is also supported by [CounselBench's expert comparison](https://proceedings.iclr.cc/paper_files/paper/2026/hash/99946cb64d51ead9d3969db0af65ca2e-Abstract-Conference.html),
which found that model judges overrated responses and missed safety concerns
identified by professionals. Its mental-health QA procedure informs reviewer
design; it is not a ready-made longitudinal-memory score for this system.

## A fair comparison

Use a new protocol and output directories. Do not add an arm to the frozen v2
experiment, reuse its scores under a new name or discard negative runs.

| Condition | Information and controls |
| --- | --- |
| Short-term reference | The shared recent-session window, without older-session retrieval |
| Raw-history RAG | The same recent window plus retrieval over original older turns and their source/time/access metadata |
| Managed-memory RAG | The same recent window plus source-supported records written and updated through the real memory manager |

**The primary contrast is managed memory versus raw-history RAG.** The short-term
arm measures the benefit of older history, which is a different question.

The raw-history baseline receives the same correction, confirmation, expiry and
deletion events, and implements basic time, deletion and user/sensitivity filters.
It may maintain a minimal event/visibility log without adopting the project's
semantic schema. Do not win by denying it ordinary privacy controls. Publish how
each baseline interprets updates; an unsupported event is a disclosed capability
limit, not silently an answer error.

Keep eligible source history, current input, response model/revision, language,
recent window, knowledge context, safety rules, retrieval backend and total
context-token limit equal. Tune representation-specific chunking and thresholds
on development data with equal effort, then freeze them. Gold answers, evidence
labels and future sessions never enter extraction, retrieval or generation.
Count prompt overhead against the budget and report actual token use.

Keep confirmation policy equal too. Benchmarks do not provide genuine user
consent: automatic candidates remain unconfirmed under an explicitly labeled
research access policy. Do not relabel them as user-confirmed facts. The separate
event-replay test supplies explicit synthetic confirmation events.

### Low-friction memory use

The intended product default is automatic capture of eligible user quotations,
with an optional natural-language check only when an uncertain or potentially
outdated item is relevant to the current task. Saving a dated observation is not
the same as confirming a summary or its present-day validity. User silence, a
retrieval hit, repeated model output and high self-reported model confidence do
not supply confirmation.

Keep that interaction policy equal in the raw-history and managed-memory arms.
Both may quote verified earlier statements as history; both must observe the
same rules for current-state use and checks. A separate, prespecified interface
comparison may contrast conversational checking with item-by-item review. Do not
confound memory representation with different amounts of user labor or secretly
give one arm extra user answers.

Static public QA has no live participant to answer a check: do not feed a gold
answer back as simulated user confirmation. Score a clarification/non-answer
under the task's frozen official rule and report its frequency separately. In an
interactive study, responses to checks must follow the same participant/task
protocol in both arms; additional information and time are part of the cost.

Add these outcomes to the participant and replay reports:

- explicit review/confirmation actions per successful follow-up;
- relevant versus unnecessary or repeated checks, with a fixed denominator of
  follow-ups and independently eligible check opportunities;
- stale/current-state errors and unsupported personal assertions after checks;
- corrections correctly incorporated, unanswered checks left unresolved and
  opt-out content not stored or re-imported;
- automatic capture, retrieval and checking cost, including any additional model
  calls and failed work.

Evidence strength, freshness and conflict flags are engineering signals, not
validated probability estimates. Freeze thresholds and cooldowns on development
cases. If numeric confidence is later interpreted as probability, assess its
calibration on independent labeled cases before making that claim. Do not measure
lower burden merely by counting fewer clicks while adding many interruptions.
The existing frozen v2 QA run retains its old access/prompt policy; the new mode
requires a new version and cannot inherit those scores.

After the fixed-budget comparison, optionally report a prespecified quality/cost
curve at several budgets. Include extraction, updates, indexing, retrieval,
generation and failed-call cost. Separate preparation cost from per-follow-up
cost, with amortization at stated numbers of follow-ups. One-time summarization
must not disappear from the cost comparison.

## Objective external tasks first

Only two new external task families are needed initially:

| Source and task | Score to preserve | Purpose and limitations |
| --- | --- | --- |
| [MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/main/README.md): `fact_sh` and `fact_mh` | Official `substring_exact_match`, reported separately by task | Tests conflict resolution. A substring hit can coexist with a contradictory old fact; add a **separate** stale-answer diagnostic where old/new labels are unambiguous. The whole benchmark is not judge-free. |
| [PersonaMem-v1](https://github.com/bowen-upenn/PersonaMem/blob/main/README.md): dynamic preferences and their application | Released correct option versus prediction using the [official parser](https://github.com/bowen-upenn/PersonaMem/blob/main/inference.py) | Deterministic multiple-choice correctness. Apply the released `end_index_in_shared_context` history boundary. These are synthetic personas; option accuracy is not free-dialogue helpfulness. |

Before running, pin dataset/evaluator commits, license and redistribution terms,
file sizes/hashes, eligible counts, group/split IDs and adapter changes. The executable [objective comparison](../evaluation/objective-memory/README.md)
now provides the frozen registry, adapters, runner, deterministic scorer and audit
export. Hosted progress/results remain separately labeled; code readiness is not
a completed effect study.
Preserve official scores alongside adaptations. Do not silently replace a
published parser; malformed answers fail under the frozen rule.

The first comparison preserves original quotations. Measure actual prompt identity
before generation: identical raw/managed inputs share one canonical response and
are reported as control equivalence by construction, without an inferential
management-effect interval. This cannot validate semantic extraction. PersonaMem
32k has 589 released questions but 20 connected history groups; MemoryAgentBench
length variants are one conservatively grouped family. These are exploratory
benchmarks, not independent participant trials.

Keep LoCoMo evidence Recall@k as a retrieval diagnostic and LongMemEval
reference-judge results as model-assisted historical-QA evidence. Neither
replaces the raw-history contrast or the lifecycle results below. Do not add
datasets simply to increase the question count.

## Six continuity results, with explicit denominators

Replay ordered, timestamped events through writing, updates, retrieval and final
context assembly. Use public labeled facts where available. Add synthetic
interventions for expiry, deletion, isolation and exact time boundaries; label
these **controlled mechanism tests**, not clinical QA or natural-user evidence.
An expected-state ledger is derived from the event specification, reviewed
separately from the implementation, fixed before execution and never given to
the system under test. Do not generate its answers from the same manager being
evaluated.

| Task | Eligibility and reference | Scores to report |
| --- | --- | --- |
| Goals | An explicit agreed goal and a later query/update, with linked source spans | Current-goal accuracy; abandoned-goal reuse; unknown/ambiguous-goal handling |
| Homework | An agreed task and a later report, including partial or absent reports | Correct task/status linkage; unsupported completion or failure; correct outstanding-task follow-up |
| Reported state | Dated self-reports and a query with an explicit time target | Latest-state accuracy; historical/current confusion; wrong-person/source attribution |
| Corrections | An explicit correction, its target/version and effective point | Latest-version accuracy; stale-fact exposure in retrieval/context; stale-fact reuse in the answer |
| Expiry | A validity interval and queries before, exactly at and after its end | Visibility compliance with `[valid_from, valid_to)`; expired-context and expired-answer rates |
| Deletion and isolation | A deletion or denied user/sensitivity scope, followed by relevant queries | Deleted-content return in storage/index/context/cache/export; prohibited-context exposure and answer leakage, by policy |

Publish eligible histories, events and queries for **each row**, with unknown cases
and exclusion reasons. No later report means `unknown`, not automatically
completed or failed. Preserve historical states for dated questions; every older
fact is not an error. A source substring alone does not prove that a semantic
goal/homework summary is supported.

Freeze deletion scope before replay: current stores, indexes, session buffers,
caches and new exports that the application controls. Scoring-only source/oracle
files are separate test instrumentation. Existing user-visible messages cannot
be retracted by deleting a record, and any retained research archive needs its
own disclosed access/retention policy; do not imply that record deletion erases
every historical copy.

Use exact categorical/value scoring only where the reference determines the
answer, with normalization fixed beforehand. Free-text claims need independent
claim annotation. Show supported-claim precision together with eligible-fact recall
so silence cannot look like perfect fidelity. Undefined denominators are
`not applicable`, never 100%.

For deletion, expiry and access controls, **any observed forbidden exposure fails
that software acceptance check**, regardless of average QA. Also report violations
and exposure opportunities. Zero observations is a finite test result, not proof
of future safety. Exact marker/ID checks are deterministic on controlled fixtures;
paraphrases in real answers need separate review and are not covered by markers.

### Attribute the first failure

Each eligible query keeps a scoring-only trace of:

1. **Source → extraction:** required observation represented; generated personal
   claims supported.
2. **Extraction → current state:** expected version/status after confirmation and
   update rules.
3. **State → retrieval:** permitted, relevant evidence selected.
4. **Retrieval → prompt:** complete evidence survives the context budget.
5. **Prompt → answer:** permitted current fact used correctly, appropriate
   abstention, no invented personal claim.

Publish the first failed stage and all independently observed policy violations.
An included ID proves only that evidence was supplied, not that the answer relied
on it. A fixed-evidence run may diagnose generation failure; label it an oracle
diagnostic, not the deployable system result.

## User value: test the actual four-visit workflow

Use fictional reflection tasks with real participants: **set a goal → report
progress → revise the goal → review or delete it**. The separate
[four-visit nonclinical protocol](research-validation-protocol.md) now specifies
parallel randomized arms, keeping each participant in one memory condition to
limit carryover. Balance equivalent task variants across arms. Its 60 planned
slots are not enrolled participants; collection and analysis tools do not satisfy
the remaining ethics, exact independent task-reference and deployment gates.
The [CBT process/outcome note](cbt-process-outcome-evidence.md) separately specifies
homework learning, appropriate responses and actual clinical outcomes.

Measure completion without facilitator rescue, incorrect saved facts, correction
success, deletion understanding, task time, clicks, candidates reviewed and edits.
Keep dropouts and assistance visible. Ask whether the response addressed the
intended follow-up and whether the memory explanation was understandable. These
operational measures are informed by the [ISO 9241-11
framework](https://www.iso.org/standard/63500.html); ISO does not prescribe this
study or validate our questions.

The interface retains optional candidate revision/review and readable references,
and explains deletion's chat-reset effect before the action. Its conversational
mode should remove manual extraction and item-by-item confirmation from the normal
path; automatic quotations, check questions and confirmed facts must stay visibly
distinct. These are implementation changes with mocked/software checks;
participant usability has not been measured. Count review effort, interruptions
and correction recovery as costs, not free participant labor.

Open-ended reflection quality needs independent arm-blinded reviewers, anchored
criteria, double scoring and disagreement/adjudication reporting. CBT-trained
reviewers are needed for claims about appropriate CBT guidance; actual users for
perceived usefulness. Do not let a model fill WAI, PANAS or another human scale and
call it validated experience or treatment change. Without these resources, report
objective system findings and leave helpfulness/clinical claims unestablished.

## Sample size, analysis and execution order

The independent unit is a history/persona/shared case, not a question or response.
Connect reused histories and related seeds before splitting. Separate development
and test groups before inspecting outputs; record prior exposure. Use paired
group-level differences and intervals, not an IID question bootstrap. Report micro
and group-equal results plus every task above, without choosing the favorable
summary after seeing scores.

Use development data to justify the smallest useful effect and required precision,
estimate group-level variation, and freeze a paired cluster precision/power
calculation before test generation. No question count guarantees enough independent
data. If the available groups cannot meet the target, report an exploratory
estimate and interval; do not sample until significant. Prespecify multiple-test
handling if claiming separate task improvements. Freeze the primary endpoint or
endpoint family before generation; do not promote the best-performing task to
primary after reading results.

| Order | Work | Readiness gate |
| --- | --- | --- |
| 1 | Implement the competent raw-history comparator and stage traces | Same admitted history/events/budgets; deterministic fixtures pass for both arms |
| 2 | Freeze the two external adapters and event ledger | Input/evaluator identities, terms, exclusions and gold-free inputs checked |
| 3 | Freeze test groups, parsing, effect/precision target, retries and call budget | Development-only choices; independent-group audit; no blank protocol decisions |
| 4 | Run paired comparisons; deterministically score and audit errors | Models, embeddings, full datasets and generation **only in hosted Colab**; small local mocks allowed |
| 5 | Publish task results, traces and costs; conduct the user/reviewer study | Permitted raw-artifact retention, independent rescoring, explicit study limitations |

Empty/failed/truncated responses and exhausted retries remain in the denominator.
Report availability separately from correctness and freeze failure-to-score rules.
Preserve identities, attempt logs, source/manifest hashes and raw artifacts.
Real-user content never follows the public benchmark export path.

The public raw-history contrast and fictional source-candidate comparison have
now been executed in hosted Colab. The [full public run](../evaluation/objective-memory/results/full-20261105/README.md),
[bounded v4 repair](../evaluation/objective-memory/results/v4-repair-20261105/README.md)
and [CBT process development report](cbt-process-followup-results-20261105.md)
retain separate protocols, attempts and limitations. Independent natural-CBT
source annotation, the participant study and clinical outcomes remain incomplete.
More model judging cannot replace those gates.

## Executed results and remaining evidence gaps

| Endpoint / task | Independent groups / eligible cases | Raw-history RAG | Managed memory | Paired difference and interval | Failures / total cost | Status |
| --- | --- | --- | --- | --- | --- | --- |
| MemoryAgentBench conflicts | 800 questions; one conservative family; task-specific rows in linked reports | 419/800 | 483/800 in bounded v4 reconstruction | +8 pp; family-level interval not estimable | Repair: 206 new calls / 463,447 tokens / 0 provider failures; original full-run costs separate | 206 changed-input answers newly generated; 2,194 unchanged logical rows reused; not a full contemporaneous rerun |
| PersonaMem-v1 | 589 questions / 20 connected history groups | 416/589 | 416/589 | All 589 raw/managed prompts and canonical replies shared | Costs retained in original full-run report | Executed quotation control; no management-effect evidence |
| Fictional CBT goals / homework / state | 36 exposed development cases; 3 variants of 12 task types; 0 people | 20 met / 36; 9 unresolved | 21 met / 36; 9 unresolved | 2 higher, 1 lower, 24 same, 9 unresolved; same-model draft grades | 200 calls / 203,428 tokens / 0 provider failures; 12 invalid unique extraction outputs | No stable CBT benefit; task-specific results and structural rejection evidence in linked report |
| Corrections / expiry / deletion-isolation | Source replay and separate software controls | Original runs retained | Original runs retained | Structural observations do not establish answer-error causality | Invalid extraction remains counted in costs | Privacy-reference tasks unresolved; no expiry event in the 36-task model batch; independent semantic assessment pending |
| Participant completion / time / correction burden | — | — | — | — | — | Not run |

A useful outcome may be similar factual accuracy with fewer stale facts or easier
correction at a stated cost. Null and negative results stay in the report.
Reliability, continuity accuracy, usability and clinical efficacy are distinct
conclusions.

## CBT-specific sources: available does not mean gold

| Resource | Verified boundary and use |
| --- | --- |
| [MusPsy release](https://github.com/Rener2005/MusPsy/tree/eaae4df12808f4cbda97d8bc8b243b34cc69f613), [paper](https://aclanthology.org/2026.findings-acl.818/) | Case-seeded, LLM-generated multi-visit conversations; release declares MIT. May provide **exploratory** continuations after inventory, seed-overlap and provenance review. Memories/usercards are generated silver references, not independent gold or patient outcomes. |
| [CBT-Bench](https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench/tree/bbd37514ba2e4db46bd9633d9207a57828c2ec9d), [paper](https://arxiv.org/abs/2410.13218v2) | CC-BY-NC-4.0; expert-labeled classification may be a separate capability check. CBT-QA test answers are withheld; therapeutic references are not unique gold responses. It is not a longitudinal-memory test. |
| [HOPE / SPARTA](https://github.com/LCS2-IIITD/SPARTA_WSDM2022), [paper](https://arxiv.org/abs/2111.06647v1) | Access agreement/email required. Same-client visit ordering and longitudinal task labels not verified; blocked for this test. |
| [DiaCBT paper v1](https://arxiv.org/html/2509.02999v1) | Author-endorsed release, dataset license and stable case/session files not verified; blocked. A paper license is not a dataset-use license. |

A future natural goal/homework subset needs two independent annotators linking
agreement, follow-up and revision spans, with adjudication unaware of arm outputs.
Audit eligible counts before claiming coverage. If examples are sparse, report
the gap rather than manufacture a gold set of clinical questions and answers.
Generated memory or model-assisted labels never silently become expert gold.
