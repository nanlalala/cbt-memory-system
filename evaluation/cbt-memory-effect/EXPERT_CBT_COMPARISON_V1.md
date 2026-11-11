# Expert-grounded CBT memory comparison — protocol v1

This study asks whether managed memory helps a fixed answering model understand
the client and choose a more appropriate CBT response than capable retrieval of
the same original history. The earlier four-word quotation task remains a
separate experiment. Its three development probes are not this study's results.

## What counts as better

The primary outcome is independent CBT-trained reviewers' blinded preference
for managed versus raw retrieval at the same conversation cutoff. Reviewers may
choose either response or a tie; the recorded human therapist response is a
reference, not the only acceptable wording or action. Preference must concern
the actual response's clinical/contextual suitability, not its length or style.

Report R1–R5 separately. R1 is factual/contextual fidelity, including correct
use of earlier necessary facts and corrections. R2 is appropriate CBT method;
R3 is the next step's fit to the client's stated goal, agreement, current
readiness and earlier attempt/barrier; R4 is empathy and collaboration; R5 is
safety. The existing R1–R4 1–5 scales and R5 pass/fail remain the project's
scales, not official CTRS scores. Published CBT-specific skills can guide
case-level anchors, with explicit applicability rather than forcing every CBT
technique into every turn.

Also report source-supported fact precision and necessary-fact recall, omitted
or contradicted information, unneeded repeat questions, current-state errors,
and failures to distinguish a therapist suggestion from client agreement.
Independent evidence spans and allowed alternatives are needed to score these
semantically; an LLM proposal is not a released or human-approved gold label.

The prospective improvement claim requires positive net clinician preference
(wins minus losses divided by all scored pairs), with a family-cluster 95%
bootstrap interval lower bound above zero; improved necessary-fact use without
worse fact precision; no worsening in R1 or R3; and no newly introduced severe
safety or permission error. This is a research decision rule, not a clinical
effect-size threshold. Report all component estimates and uncertainty; failure
to meet a gate means continue development, not change the test or hide ties.
Missing or blocked cases remain in the planned denominator with null ratings.
Any comparative claim is withheld while required reviews are incomplete.

## Materials and their exact roles

1. **CBT-Bench / CBT-DP** is the immediately accessible professional reference
   anchor: professional textbook responses and released human pairwise ratings.
   Its short client statements have no longitudinal history. Use it to examine
   the quality rubric and, if authorized later, calibrate an automatic reviewer
   against existing human judgments. Do not score it as a memory-effect test.
   Dataset revision: `bbd37514ba2e4db46bd9633d9207a57828c2ec9d`;
   declared licence: CC-BY-NC-4.0. The paper's reviewer qualification is a
   relevant graduate degree plus CBT training; do not rename all reviewers
   licensed doctors. References are assessor-only, never user memory or RAG.
2. **RealCBT** is a candidate for real multi-turn dialogue. The official
   repository, checked at `bf5b7914831d439d00a10e54c8abb1dad1a9bde4`, says raw
   transcripts were removed from public distribution on 4 August 2026 because
   of third-party rights, privacy and attribution. Only documentation and
   metadata are currently public. Acquire appropriate author access before
   using transcripts; do not restore removed text from history or mirrors.
   Individually verify CBT approach and professional provenance. The paper
   alone does not establish every video's therapist qualification or quality.
3. **Bedi et al., LREC 2026** describes 17 sessions by licensed professionals
   with actors as clients, plus 26 guideline sessions. This is an appropriate
   professional demonstration type, not real-patient outcome evidence. No
   official reusable transcript package was verified during this review.
4. **CareBench-CBT** describes professionally reviewed multi-turn material and
   a hierarchical therapy-memory method. Official supplementary artifacts and
   reuse terms have not yet been verified as obtainable. Do not substitute a
   third-party derivative or reuse its projected numbers as our measurements.
5. **CACTUS** remains a synthetic development/stress-test source. Keep its
   original identity. It cannot stand in for human professional reference data.

## Conversation task and leakage boundary

For an eligible, licensed dialogue, retain its original speaker/content/order.
The generation input ends at an actual client message; the next recorded
therapist response and all later turns are hidden from all memory, extraction,
retrieval and response-generation code. Whole-session summaries, cognitive
formulations, future homework and treatment plans must not be supplied unless
they were already explicitly known at that cutoff. Session boundaries, dates,
patient identity and week numbers are never inferred from row numbers.

Use teacher-forced prefix replay for the primary comparison: all conditions see
the same real historical messages. Prior generated answers are not fed into
later primary prefixes, because that would create different histories and
confound memory with conversation drift. Separately label any later interactive
simulation and never call a simulated client an actual participant.

Keep a fixed six-message recent buffer, the same current input, answering
model/prompt/output limit, complete serialized request ceiling, permission and
confirmation policies, and professional-knowledge channel in all conditions.
Both primary groups may retrieve across the entire eligible original history.
The raw baseline receives competent relevance retrieval and the common output
validator; managed adds only the implemented memory/index/state policy. The
short-term condition is auxiliary. Record identical payloads as ties/shared
requests; do not induce artificial differences through labels.

Give both primary groups the same maximum of nine distinct historical original
sources and the same opportunity to fill the request budget. In particular, do
not compare the product's six raw hits plus three managed supplements against
a raw arm artificially stopped at six hits with room left. A new comparison
adapter must allow raw top-nine relevance candidates (or the same declared
budget-filling policy), record all packing exclusions, and hold direct recent
messages constant. The old fixed-top-six replay is not this new protocol.

Current ordinal-only code does not produce dated task/weekly state. An undated
dialogue may test history use and explicit source-linked corrections, but cannot
test dates, elapsed weeks or due-date semantics. Put engineered corrections,
pauses, deletions and refusals in a separately labelled mechanism suite.

## Development, reviewers and analysis

Group by actual case/client and linked sessions, then by known shared source
video/material; connect duplicate/near-duplicate variants before splitting.
Do not assume public session labels identify independent people. Freeze the
grouping, exposure register and development/validation/test assignment before
optimization. If identities cannot support a valid independent split, do not
declare one. The previously inspected 200 CACTUS families are exposed.

Optimize only on development material. Freeze the chosen code, protocol,
sampling rule, models, request caps and scorer before a held-out comparison.
Use at least two independent CBT-trained human reviewers, record relevant
training/clinical experience and conflicts, calibrate on separate materials,
mask condition and randomize response position. Keep original scores and
disagreements; report weighted kappa/ICC as appropriate, and separately identify
any adjudication. Do not let software or an AI invent a reviewer identity,
credential, agreement record or human score.

If a model-assisted reviewer is used, first assess its agreement against held-out
published expert judgments. Split by underlying client statement rather than
individual comparison row. Freeze its calibration and do not optimize against
the final target test labels. Model-assisted results remain separately labelled
and cannot be counted as new independent human review of our system outputs.

For clinician preference, aggregate multiple prefix judgments within the
predeclared source family so longer cases do not create artificial independent
samples; use 10,000 family bootstrap draws, seed 20261112. Report the family
count, pair count, ties, rater-specific results and paired estimates. Single
observed reference responses do not define treatment success. Text overlap and
embedding similarity are secondary diagnostics, not the primary decision rule.

## Implementation and current execution gate

The implemented improvement under review is distinct-source candidate
selection: several summaries pointing to one original message no longer consume
the entire supplemental source allowance. Resolve candidates back to complete
permitted user originals, check all source dependencies, retain original-source
relevance priority under budget, and recheck withdrawal before display. It does
not upgrade summaries to user-confirmed facts or manufacture current task state.

The standalone reference collector performs zero model calls and creates a new
hosted-Colab directory with exact source bytes/hashes and assessor-only outputs.
It must label CBT-DP as ineligible for the main memory effect. Software fixtures
are engineering validation, not an observed quality advantage.

**Current main-study gate: blocked.** We have not acquired authorized, verified
human-professional multi-turn transcripts or independent new reviewer ratings.
There is no new main-study effect score. The previous nine-call allowance is
already exhausted; a future paid batch needs a concrete separately recorded cap,
but no paid batch should start while its data/reference gates are unresolved.
Preserve old journals; never retry unknown sends or modify frozen hashes.

## Sources

1. Zhang et al. CBT-Bench. NAACL 2025, §§3.3, 5.2, 8.
   https://aclanthology.org/2025.naacl-long.196.pdf
   https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench
2. Wang et al. RealCBT. Findings of EMNLP 2025.
   https://aclanthology.org/2025.findings-emnlp.1089/
   https://gitlab.com/xiaoyi.wang/realcbt-dataset
3. Bedi et al. Assessing the Effectiveness of LLMs in Delivering Cognitive
   Behavioral Therapy. LREC 2026, §2.1.
   https://aclanthology.org/2026.lrec-1.372/
4. Yuan et al. Can LLMs Move Beyond Short Exchanges to Realistic Therapy
   Conversations? CareBench-CBT. ICLR 2026.
   https://openreview.net/pdf?id=3Bdl1wL1S3
