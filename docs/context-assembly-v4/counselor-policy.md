# Shared CBT counselor policy

The executable policy is
[`cbt_counselor_policy.py`](../../modules/agent-integration/cbt_counselor_policy.py),
version `cbt-counselor-policy-v1`. It is an original synthesis for this prototype,
not a verbatim prompt supplied or clinically validated by any cited paper.

The role defines what the counselor does across a conversation. The dynamic
reference block supplies the particular evidence available this turn. Clinical
strategy, a model's working formulation and user-reported facts retain separate
identities. The system does not infer a complete treatment history from a role
instruction or invent patient details to fill a CBT form.

## Design basis

| Policy component | Primary source and relevant mechanism | Application in this project |
| --- | --- | --- |
| Collaborative focus and action plans | [Beck Institute, What Are CBT Sessions Like?](https://cares.beckinstitute.org/about-cbt/what-are-sessions-like/), agenda, progress and therapeutic relationship sections | Work toward the user's priorities; reconnect to available prior work, invite feedback and support manageable practice |
| Guided discovery and homework review | [Beck Institute Cognitive Therapy Scale Rating Manual](https://beckinstitute.org/wp-content/uploads/2021/06/CTRS-Manual-2020.pdf), guided discovery and homework items | Explore the person's evidence and conclusions; review attempts and obstacles without requiring a new assignment every turn |
| Accurate or partly accurate thoughts | [Beck Institute Thought Record](https://beckinstitute.org/wp-content/uploads/2021/08/Thought-Record-Worksheet.pdf), introduction and worksheet | Distinguish situation, thought, emotion, behavior and outcome; avoid automatically labeling distress as distorted thinking |
| Prediction, observation and learning | [CCI Behavioural Experiment Monitoring worksheet](https://www.cci.health.wa.gov.au/~/media/CCI/Mental-Health-Professionals/Anxiety/Anxiety---Worksheets/Anxiety-Worksheet---10---Behavioural-experiment.pdf) | Keep an intended test, actual observations and the person's learning distinct; completion alone does not establish benefit |
| Flexible technique and stage | [CoCoA v1](https://arxiv.org/html/2402.17546v1), sections 2.3.1–2.3.2 | Keep fixed counseling rules separate from turn-specific technique and progress information; avoid restarting a completed step |
| Evolving conceptualization | [CCD-CBT, AACL 2026](https://aclanthology.org/2026.aacl-long.79.pdf), section 3.1 and Appendix B.3, Figure 8 | Use inferred formulations and observable history; hidden simulated-client profiles remain outside generation |
| User-generated alternatives | [HealMe, ACL 2024](https://aclanthology.org/2024.acl-long.93.pdf), Figure 1 and section 1 | Invite the user's alternative interpretations instead of imposing a positive rewrite |
| Feedback and repetition | [Socrates 2.0, JMIR 2025](https://mental.jmir.org/2025/1/e80461/), Introduction and development description | Use answers already given; change course after feedback about repetitive questions, burden or lack of usefulness |
| Counselor and supervisor roles | [AutoCBT v1](https://arxiv.org/html/2501.09426v1), sections 3.2 and 4.3.2 | Generate the actual counselor response; use empathy, thought exploration, reflection and strategy as flexible checks rather than mandatory paragraphs |
| Present safety needs | [NICE NG225](https://www.nice.org.uk/guidance/ng225/chapter/recommendations), sections 1.5–1.6 | Attend to present safety concerns; an old note or a scalar score does not establish present safety |

The one-focus conversational default, lightweight checking at the point of use,
and explicit proposal/acceptance/pause distinctions are implementation choices
for this project. They are not presented as a prescribed universal CBT script.
The cited studies have different designs and outcomes; their results do not
establish the effectiveness of this implementation.

## Three-condition evaluation contract

The current user instruction fixes exactly three generation conditions:

| Condition | Recent conversation | Older memory supplied |
| --- | --- | --- |
| `no_rag` | Same admitted recent context | None |
| `raw_history_rag` | Same admitted recent context | Retrieved original source statements |
| `managed_memory_rag` | Same admitted recent context | Selected supported processed memory with necessary evidence and source dependencies |

All three import the same complete counselor policy. Answer model, sampling,
current question, ordinary recent context and full request ceiling match within
each study. The reference block changes with the memory condition. No fourth
short-window or representation-ablation arm is part of the requested rerun.

Factual benchmark and quotation instructions are identical task-format additions
in every condition. They require the relevant short answer or JSON rather than
turning a recall question into a counseling monologue. Counselor identity is not
used to grant access to hidden labels, references or future dialogue.

Each frozen protocol records executable policy code, actual messages and request
hashes. Whole-request budget checks include this policy and metadata. Any budget
change required by the expanded policy is declared before generation and applied
to every condition. Previous protocols and results remain separately identified.

## Scope of evidence

The role prompt is a behavioral specification. It does not itself verify source
support, prevent prompt injection, provide a deployed crisis service, or prove
counseling quality. Source admission, version checks and request journaling are
separate modules. New effectiveness claims require the authenticated reruns.
Automatic R1–R5 scores remain model assessments; the published-label calibration
does not validate custom R1/R3 or reliable safety-risk detection.
