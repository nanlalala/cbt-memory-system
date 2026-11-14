# CBT evaluation audit and context v5 draft (revision 2)

Status: proposal. Nothing here changes a frozen runner, protocol or published
result. Scope: the CBT continuation evaluation (professional cases, CACTUS) and
the context assembly that feeds it. PersonaMem is out of scope by decision.

Evidence labels used below:

- **[code]** read directly in this repository.
- **[lit]** from an external source listed in section 8. Where the source is a
  blog, preprint or secondary report, section 8 says so.
- **[data]** needs per-case outputs that are not in the repository, so it is a
  check to run, not a finding.

## 1. What is already solid

Targets are frozen before any answer is generated [code]. Arms share the policy
text, recent window and size ceilings [code]. The judge family must differ from
the answer family [code]. Source families are the resampling unit [code].
Literal quotes are required for every judged claim [code]. Receipts, fixed
denominators and kept failures follow the spirit of TRIPOD-LLM's reporting items
on model identification, prompts and human involvement [lit 6]. Historical
results are never overwritten [code].

## 2. Findings from reading the code more closely

| # | Finding | Where | Consequence |
| --- | --- | --- | --- |
| C1 | The first cutoff is a user turn with at least 4 prior messages, while every arm gets the 6 most recent messages directly. With 6 or fewer prior messages there is no "older" source at all, so all three arms receive identical requests. | `cactus_materials.choose_cutoffs` (`i >= 4`), `runner.SETTINGS["recent_messages"] = 6`, `older = sources[:-6]` | Some cases cannot show a memory effect by construction. Share is [data]. |
| C2 | The target-proposal prompt reads the whole visible prefix, including the 6 recent messages that every arm already sees. | `memory_assessment.TARGETS` | Required facts from the recent window are free for every arm and dilute the older-memory effect. |
| C3 | The same Qwen-family assessor proposes the targets, binds the answer's assertions to them, and the R1-R5 judge is also Qwen-family. Targets are labelled `model_proposed`. | `TARGETS`, `ADJUDICATE`, `judge.py` | Shared blind spots; coverage and error rates are not independent of the judge. |
| C4 | A supported claim only counts toward a required fact if its literal quote lies in that fact's `allowed_source_ids`. Otherwise it is downgraded to `unsupported`. | `evidence.score_memory` (`binding_mismatch`) | A correct answer that cites an equivalent later message loses recall. The output already carries `allowed_fact_binding_mismatch` per claim, so the rate per arm is [data] and costs nothing to compute. |
| C5 | Precision is `None` when an answer makes no assertions, and such pairs are dropped from the conditional comparison. | `evidence.score_memory`, `paired_report` | A more cautious arm drops out of precision exactly when it says little. Report precision with the completion-adjusted grid too. |
| C6 | `judge.summarize` compares only pairs where both reviews completed. The evidence report has a completion-adjusted grid; the quality report does not. | `judge.summarize` vs `evidence.paired_report` | Invalid or unresolved reviews can bias R1-R5 differences. |
| C7 | The judge request disables model reasoning (`thinking: disabled`) and scores five dimensions in one call. | `judge.prepare` payload, `quality.GUIDE` | Joint scoring invites halo between dimensions. Reasoning-off may raise style bias (see L2). |
| C8 | 10% of packets are judged twice at temperature 0. | `judge.prepare`, `quality.plan_quality` | At temperature 0 this mostly measures provider nondeterminism, not judge instability. Swap order, rubric paraphrase or a second judge instead. |
| C9 | Bootstrap uses 1000 draws in both `judge.summarize` and `memory_assessment`. | `judge.py`, `memory_assessment.py` | Interval ends move between seeds. Teaching material recommends 10,000 to 100,000 draws [lit 8]. |
| C10 | AB/BA pairwise planning exists but is outside the default call cap. | `quality.plan_pairwise` | The only bias control for position effects is not in the default run. |
| C11 | `auto_eval_v2.retrieval.select_evidence` (facts as retrieval keys, original messages as content) exists, but the three evaluated arms do not use it. | grep: only `canonical` is imported elsewhere | The strongest literature-backed design (L5) has never been run. |

## 3. What the wider literature adds

**L1. LLM judges have known biases.** Position, verbosity and self-preference
effects are documented across surveys and bias studies [lit 1-3]. Self-preference
tracks familiarity of the text [lit 1]. A 2026 study of mitigation strategies
finds chain-of-thought gave the largest average reduction in style bias, position
swapping reduced style bias but raised verbosity bias, and judges showed a strong
preference for Markdown formatting [lit 2]. For this project: response length and
Markdown use per arm should be measured, and the judge should not run with
reasoning off without a check (C7).

**L2. Judges in health settings need human anchoring.** A 2026 scoping review of
LLM-as-judge in healthcare reports judge-expert agreement from 0.66 to 0.96
(median 0.83), notes that multi-dimensional rubrics tend to be less reliable, and
shows a case of 0.80 raw agreement with kappa 0.59 when one category dominates
[lit 4]. Weighted kappa with raw agreement reported together is the standard for
ordinal scales [lit 4]. Your professional cases (62 of 84 scores of 5) are exactly
the one-dominant-category situation.

**L3. Mental-health evaluations are often thin.** A review of 135 ACL papers
found about half rely only on automated NLP metrics, over half have no human
evaluation, and of those with human evaluation 29% involve no mental-health
expert [lit 5]. It recommends depth that rises with intended role and harm:
technical validation, then expert-involved evaluation, then ecological study
[lit 5]. This project is honestly at stage one; the report should say so.
CTRS, the common CBT fidelity scale, was designed for audio or video and a
chatbot CBT study (Aarhus, reported by PsyPost) described difficulty rating text transcripts with it [lit 7]. Your
R1-R5 rubric is project-specific, so it needs its own human validation.

**L4. LoCoMo has known quality problems.** A third-party audit (blog plus public
repository, not peer reviewed) reports 99 wrong answer-key entries in the 1,540
questions (6.4%), a ceiling near 93.6%, and that a lenient judge accepted a large
share of deliberately vague wrong answers; its smallest category has 96
questions [lit 9]. Your LoCoMo run uses exactly these 1,540 questions and a
smallest category of 96. Treat the audit as a caveat to verify, not a settled
fact. It matters for the +13.9 pp claim only through noise and judge leniency; a
paired comparison on identical questions is less exposed than an absolute score.

**L5. Facts as keys, original text as content.** LongMemEval found that round-level
storage works better than session-level, that fact-augmented key expansion
improves both recall and answer accuracy, and that temporal queries need
time-aware handling [lit 10]. LongMemEval also names knowledge updates and
abstention as core abilities, which are the abilities your lifecycle manager is
built for [lit 10]. This supports running the arm described in C11.

**L6. Statistics.** Paired bootstrap is the usual tool for comparing systems on
any metric [lit 8]; a guide for NLP significance testing stresses effect size and
matching the test to the metric, and reports frequent misuse [lit 8]. For many
small comparisons, correct for multiplicity [lit 8]. A cross-judge check with a
second judge model is used in other memory work [lit 11].

## 4. Evaluation: required changes (ordered by impact)

| # | Change | Fixes | New runs? |
| --- | --- | --- | --- |
| E1 | Make older-memory coverage the primary endpoint: split targets into recent-window and older, report both. | C2 | No (re-score from stored targets) |
| E2 | Report the share of identical-request cases and the effect on the memory-admitted subset. | C1 | No |
| E3 | Per arm, count `allowed_fact_binding_mismatch` claims; if asymmetric, re-score with alternative allowed sources. | C4 | Re-adjudication only if asymmetric |
| E4 | Add completion-adjusted analysis to quality and to precision. | C5, C6 | No |
| E5 | Pre-declare R1 and R3 as the only primary quality endpoints; Holm correction across them; report others as descriptive. | multiplicity | No |
| E6 | Use at least 10,000 bootstrap draws and publish the seed. | C9 | No |
| E7 | Measure response length and Markdown use per arm; report length-matched pairs or include length as a covariate. | L1 | No |
| E8 | Add a second judge family on a random subset (about 100 cases) and report raw agreement plus weighted kappa. | C3, L2 | Yes, judge calls only |
| E9 | Replace the temperature-0 repeats with order-swapped or paraphrased-rubric repeats, and run AB/BA pairwise on the subset. | C8, C10 | Yes, judge calls only |
| E10 | Human-rate 20 to 30 cases on R1 and R3 with two raters; report weighted kappa and raw agreement with the judge. | L2, L3 | Human time |
| E11 | Human-check a random 100 model-proposed targets and report target precision. | C3 | Human time |
| E12 | Treat the 7 professional cases as descriptive only. | ceiling | No |
| E13 | State the maturity stage (technical validation) and attach a TRIPOD-LLM checklist as an appendix. | L3, section 1 | No |

Zero-cost diagnostics, in order: E2, E3, E1, E7. They use stored outputs and tell
you whether any further spending is justified.

## 5. Proposed measurement addition (no new data collection)

From existing CACTUS transcripts, flag turns where the user revises a task,
pauses, corrects a fact, or voices discouragement, using the same literal-quote
discipline as `bind_candidates`. Score those turns with assertion checks (old
value reused, paused task pushed, correction ignored). Freeze the flagging rules
before looking at arm outputs. Report them as a separately named cohort, never
pooled with the main result.

## 6. Architecture

### 6.1 Context packing (draft v5, implemented)

`cbt_context.py` ranks candidates by a fixed priority, then retrieval rank, and
drops anything that does not fit [code]. `turn_guidance` (proposed, lowest
epistemic status) gets priority 30 while source-supported user facts get 10, so
under a tight budget guidance wins. A user correction is not distinguished from
other facts, and a dropped item has the same reason as any ordinary budget drop.

`modules/agent-integration/cbt_context_v5_draft.py` (new; v4 untouched):

| Tier | Priority | Content |
| --- | --- | --- |
| pinned | 1000 | `origin == user_correction` |
| fact | 10 | other source-supported items |
| guidance | 5 | `turn_guidance` |
| hypothesis | 3 | `agent_inference` working formulations |

A pinned item that cannot fit is reported as `pinned_item_exceeds_outbound_budget`,
and selected pinned ids are written to `trace.pinned_item_ids`.
Tests: `tests/test_cbt_context_v5_draft.py` (4 tests; v4 tests still pass).

### 6.2 Proposed, not implemented (each needs a new evaluation identity)

| # | Change | Why | Cost |
| --- | --- | --- | --- |
| A1 | Fourth arm: facts as keys, original messages as content, using the existing `select_evidence`. | L5, C11. Best candidate to recover the 6 missing targets: raw text is kept, management still routes. | New generation for one arm |
| A2 | Tag pause and deletion events so they are pinned too (today only `user_correction` is). | The pinned tier is only as good as the tags. Verify first how the extractor labels them [data]. | Extractor check, then maybe new contract version |
| A3 | Replace "prefer rejection" with a low-priority "unverified cue" slot for rejected-but-relevant candidates, clearly labelled. | Possible over-strict verifier; first measure how many rejected candidates were relevant [data]. | Touches hash-pinned `cbt_projection_contract.py`: new contract version |
| A4 | Wire `cbt_wording_cues.py` into `controlled_dialogue.py` behind a default-off flag. | Emotion wording cues; weak signal, so keep it tentative [lit: Tackman 2019 in earlier notes]. | Small code change, new run to evaluate |
| A5 | Time-aware query handling for temporal turns. | LongMemEval temporal finding [lit 10]. CACTUS has only ordinal order, so this only pays off in a dated cohort. | Larger |

Hash pinning: `architecture_retest/runner.py` hashes `cbt_context.py`,
`cbt_projection*.py`, `cbt_retrieval.py` and the retest package. Using v5 or A1-A5
in any evaluation needs a new runner version and new result directories; it
cannot be mixed into the published runs.

## 7. Suggested order of work

1. Zero-cost diagnostics (E2, E3, E1, E7) on the existing outputs.
2. Decide from the results: if older-memory coverage and binding mismatch are
   the story, do A1; if the verifier rejects many relevant items, do A3.
3. Reporting fixes that need no runs (E4, E5, E6, E12, E13).
4. Judge validation (E8 to E11) before any new headline claim.

## 8. Sources

1. LLMs-as-Judges survey: https://arxiv.org/pdf/2412.05579 ; A Survey on LLM-as-a-Judge: https://arxiv.org/pdf/2411.15594 ; Self-preference bias: https://arxiv.org/pdf/2410.21819
2. Judging the Judges (bias mitigation strategies, preprint): https://arxiv.org/pdf/2604.23178
3. Justice or Prejudice? Quantifying biases in LLM-as-a-Judge: https://arxiv.org/html/2410.02736v2
4. LLM-as-a-Judge in Healthcare: A Scoping Analysis (arXiv, May 2026): https://arxiv.org/abs/2605.25273 ; weighted kappa guidance: https://arxiv.org/pdf/2512.15783
5. Responsible Evaluation of AI for Mental Health (review of 135 ACL papers, preprint): https://arxiv.org/html/2602.00065v1
6. TRIPOD-LLM reporting guideline, Nat Med 2025: https://pubmed.ncbi.nlm.nih.gov/39779929/
7. LLM chatbot CBT competence study (press summary; CTRS designed for audio/video): https://www.psypost.org/study-finds-ai-chatbots-can-conduct-basic-therapy-sessions-but-struggle-to-adapt-to-individual-needs/ ; CTRS in LLM-vs-human CBT study: https://arxiv.org/pdf/2409.02244
8. Berg-Kirkpatrick et al. 2012: https://nlp.cs.berkeley.edu/pubs/BergKirkpatrick-Burkett-Klein_2012_Significance_paper.pdf ; Dror et al. 2018 appendix: https://arxiv.org/pdf/1809.01448v1.pdf ; Bestgen on bootstrap CIs: https://ar5iv.labs.arxiv.org/html/2205.11134 ; lecture notes (10,000 to 100,000 draws): https://people.cs.umass.edu/~brenocon/anlp2018/lectures/11-sigtest.pdf
9. LoCoMo audit (blog and repository, not peer reviewed): https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer ; https://github.com/dial481/locomo-audit
10. LongMemEval (ICLR 2025): https://arxiv.org/pdf/2410.10813
11. EvalMem (cross-judge robustness check): https://arxiv.org/pdf/2609.22231
