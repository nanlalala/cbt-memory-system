# Next validation: CBT continuity and semantic memory

Status: design and source check, 4 November 2026. No new benchmark results are
reported here. The frozen LoCoMo / LongMemEval three-condition v2 comparison stays
unchanged. It tests historical QA with sourced candidate records; its default
verbatim representation does not establish that semantic CBT memory works.

## Data we can actually use

| Resource | What the primary source establishes | Decision for this project |
| --- | --- | --- |
| [HOPE / SPARTA](https://github.com/LCS2-IIITD/SPARTA_WSDM2022), [paper](https://arxiv.org/abs/2111.06647v1) | Counseling-video transcripts with twelve dialogue-act labels. The official repository requires an agreement form and an access-request email. A session collection is not proof of repeated visits by the same client. | **Blocked for longitudinal evaluation** until access terms and a reliable client/session-order map are obtained. Do not describe HOPE as ten sessions from one person. Dialogue-act labels are not gold goals, homework completion or longitudinal memory. |
| [MusPsy](https://github.com/Rener2005/MusPsy/tree/eaae4df12808f4cbda97d8bc8b243b34cc69f613), [paper](https://aclanthology.org/2026.findings-acl.818/) | The paper describes case-report-seeded, LLM-generated conversations across visits by the same simulated client. The official release includes dialogues, memories and usercards, and declares [MIT](https://github.com/Rener2005/MusPsy/blob/eaae4df12808f4cbda97d8bc8b243b34cc69f613/LICENSE). A small ID-1000 sample has six separated dialogue blocks and corresponding numbered memories. | **Preferred next source**, pinned to the linked commit. First inventory all case/session IDs and record license plus seed-source provenance. Released memories/goals are generated references, **not independently verified gold**. Do not transfer the paper's client count to the usable release without a census. |
| [DiaCBT paper v1](https://arxiv.org/html/2509.02999v1) | Describes 108 simulated cases / 540 sessions, CCDs and fourteen CBT strategy labels. Its paper publication license is not a dataset-use license. | **Blocked**: no author-endorsed downloadable release, dataset license and stable case/session files were verified in this check. Seek these before ingestion; do not substitute an unofficial mirror. Strategy labels would not by themselves validate memory correctness. |
| [CBT-Bench official card](https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench), [paper](https://arxiv.org/abs/2410.13218v2) | Dataset card declares CC-BY-NC-4.0. Cognitive-distortion/core-belief labels and deliberate-practice human references are available. CBT-QA test answers are withheld; the authors provide a scoring Space. | **Available for a separate noncommercial CBT capability check**, with attribution and pinned files. It is not a same-client longitudinal-memory benchmark. Do not invent CBT-QA answers or treat a reference therapeutic reply as the only valid response. |

MusPsy release identity: `eaae4df12808f4cbda97d8bc8b243b34cc69f613`.
CBT-Bench card revision checked:
[`bbd37514ba2e4db46bd9633d9207a57828c2ec9d`](https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench/tree/bbd37514ba2e4db46bd9633d9207a57828c2ec9d).
Before execution, save actual file byte sizes/SHA-256, terms, split identities and
retrieval permissions in the dataset manifest. A public repository/license badge
does not establish the rights of every underlying case source. Keep a source
disabled where that review is unresolved. This pass inspected papers, metadata
and small examples only; it did not download a corpus or load models.

## Four memory fields and their evidence

Every item needs client, session and turn IDs, exact supporting spans, timestamp
or session order, author (client/counselor/model), and confirmation status. Never
present a model inference as a confirmed client statement.

| Field | Store | Reference and verification |
| --- | --- | --- |
| Profile | Explicit background, preferences and constraints | Original client spans; measure supported-claim precision and eligible-fact recall. Published profiles may seed simulation but must not expose future sessions to the system. |
| Session summary | Events, agreed actions and unresolved issues from one completed session | Independent atomic-claim annotation of that session; score factual coverage, unsupported claims and attribution errors. Released summaries provide a secondary silver-reference comparison only. |
| Goals and homework | Goal/task, who agreed, timing, status and latest update | Original agreement and later client reports, independently linked across sessions. Separate proposed, accepted, attempted, completed, revised and unknown; absence of a report is not completion or failure. |
| Patterns and state changes | Repeated thoughts/behavior, reported emotion and changes over time | Independently linked evidence spans and temporal labels. Mark pattern interpretations as hypotheses. A newer state supersedes an older current-state claim without erasing its historical provenance; this is not a clinical diagnosis. |

The following are proposed metrics, **not already validated scores**:

| Task | Metric | Required annotation source |
| --- | --- | --- |
| Find earlier relevant information | Evidence Recall@k / precision, split by the four fields | Independent turn-level relevance labels from released history, not similarity to generated memory text |
| Follow goals/homework | Correct task/status linkage; unsupported completion rate; next-session acknowledgment of the correct outstanding task | Two annotators link the actual assignment and follow-up spans; assess only cases with adequate evidence |
| Apply corrections and state changes | Latest-supported-version accuracy, stale-fact reuse rate, wrong-person/source attribution rate | Explicit original correction/update turns and version links; inferred contradictions require adjudication |
| Preserve evidence in a response | Supported personal-claim precision; correct citation/turn linkage | Annotated history plus the generated claims; report omissions separately from invented facts |

If naturally occurring homework or corrections are too scarce, report their
eligible denominators and missing coverage. Do not pad the benchmark with
self-written questions and answers. Synthetic regression fixtures remain
software checks, separately labeled.

## Execution and annotation gates

1. In Colab, enumerate the admitted frozen release, validate separators and
   client/session links, detect duplicated/shared case seeds, and report usable
   counts and exclusions. Hold out entire clients and related seeds. No future
   session, usercard goal or reference memory may enter an earlier-session input.
2. Use released session continuations/client turns as response inputs. Build a
   small annotation-development split first. Two independent annotators mark
   source spans and task/version links without seeing experiment conditions;
   adjudicate disagreements and publish the guideline plus agreement. Model
   suggestions may assist, but are not independent gold. If independent
   annotation is unavailable, label this stage **exploratory/model-assisted**.
3. Freeze the final client split, task denominators, scoring rules, model identity,
   prompt/context budgets and analysis before generation. Use all eligible held-out
   cases; set the sample size from a client-clustered paired precision/power
   calculation using development data only. Many turns from one client are not
   many independent subjects. Do not claim adequate power from raw row count.
4. Run extraction, embeddings, response generation and grading only in Colab.
   Preserve hashed prompts, responses, references and per-stage failures for
   offline rescoring within the dataset's allowed access/redistribution scope.

## Later ablation: does semantic structure add value?

Register this as a **new experiment/version after v2**, not an added arm or a
replacement result inside the current frozen comparison:

| Condition | Older-session information |
| --- | --- |
| Short-term reference | None; the same recent-session window in every condition |
| Raw-history retrieval | Original older-session turns, with provenance preserved |
| Semantic structured retrieval | Source-supported items in the four fields above, extracted only from the same older sessions |

Keep response model, knowledge context, current question, retrieval backend and
total context-token budget fixed. Predeclare chunking, item lengths, retrieval
thresholds, tie-breaking and truncation; report tokens, latency, extraction cost
and failures. Semantic items stay unconfirmed unless actual confirmation evidence
exists. Retain the raw-history condition even if it performs better.

The primary paired contrast is semantic structured versus raw-history retrieval
on independently annotated continuity tasks. Report each field and task separately,
with client/shared-seed cluster intervals; semantic extraction fidelity is a
separate prerequisite. Blind model-assisted response ratings are supplementary,
and reference-token overlap is diagnostic. Neither synthetic state improvement,
simulated PANAS/WAI scores, nor these memory metrics establish treatment efficacy
or clinical safety. Until the data/annotation gates pass, the CBT-continuity
experiment is **not ready to report an objective effect**.
