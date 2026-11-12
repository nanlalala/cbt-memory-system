# CareBench-CBT and HTM: access and evaluation route audit

Audit date: 2026-11-12. This is a public-source metadata review. No therapy corpus,
model, tokenizer, inference, external access application, or browser verification
was executed by this audit. Existing experiments are unchanged.

## Decision

CareBench-CBT is a particularly relevant candidate for the professional CBT
evaluation requested for this project. **The original release has not been
obtained or cleared for a new experiment in this session.** The immediate usable
output is a verified related-work lead and an explicit access checklist, not a new
CBT result.

## Primary identity and bounded paper findings

Paper: Zhengqing Yuan and colleagues, *Can LLMs Move Beyond Short Exchanges to
Realistic Therapy Conversations?*, ICLR 2026.

- [Official OpenReview record](https://openreview.net/forum?id=3Bdl1wL1S3)
- [Official paper](https://openreview.net/pdf?id=3Bdl1wL1S3)
- [Indexed accepted PDF identity](https://openreview.net/pdf/e3b1c076cae290e831566cfd0d9b5189bdc634c4.pdf)
- [Indexed review-version PDF for the ethics paragraph](https://openreview.net/pdf/1cb527bf2e5d8b845e2d78ece189e7d1f06f58bd.pdf)
- [Coauthor's publication list](https://weixiang-sun.github.io/publications/)

The indexed primary paper reports 256 de-identified sessions and 7,442 turns,
provided and cross-reviewed by 21 licensed mental-health counselors. It describes
turn/session assessments of therapeutic process, not source-span memory answers.
Counselors are not thereby verified as medical doctors; patient-level consent,
visit linkage, and release annotation fields were not independently inspected.
Its HTM combines persistent therapy state, retrieved episode summaries, and recent
messages under a bounded input. Algorithm 1 builds a query from the new message
and prior state, retrieves episodes, compresses evidence, updates state, and packs
context. The paper reports better dialogue scores and latency than its baseline;
the accessible text does not establish superiority to our capable RAG control.
The indexed review-version ethics statement permits academic/noncommercial research
but requires written permission for redistribution. A current release LICENSE file
or accepted-version change to these terms has not been inspected.
Supplementary code/data are claimed; their bytes and terms were not acquired.

The summary above is deliberately bounded to the primary text that could be
retrieved through public search indexing. It is not an audit of the supplement.

## What was actually accessible

| Item | Observed outcome | Consequence |
| --- | --- | --- |
| Coauthor publication page | Lists the exact title and 2026; no code or data link is exposed for this entry | Confirms identity; supplies no corpus artifact |
| Original OpenReview forum | Direct access returned “Verifying your browser” | No attachment listing, supplement file, checksum, or current license inspected |
| Original OpenReview PDF | Direct access returned the same verification page; selected primary PDF passages are publicly search-indexed | Paper claims can be attributed, but full-paper/file verification remains incomplete |
| Original author code/data repository | None verified in this bounded public search | Do not describe the original dataset as downloaded, available in the VM, or ready to run |

No UI verification, alternate API endpoint, attachment guessing, fingerprint
change, mirror retrieval, form submission, or author contact was attempted.
This audit does not claim that no legitimate public artifact exists elsewhere.

The original supplement is still needed for its exact file inventory, immutable
identity, license/release conditions, history and therapist-response boundaries,
patient/session grouping, annotation schema, prompts, HTM settings, and complete
cost-accounting protocol. The access failure is specific to the attachment route
in this environment, not evidence that the scientific resource is fictitious.

## Namesakes and derived publications

CareBench-CBT above must not be confused with emotion-appraisal CAREBench,
simulated-client CARE-Bench, clinical decision-agent CareBench, or child-safety
CAREBench. A similar name supplies no provenance or license for this resource.

A later, separate research release at
[Cognitive Atrophy Benchmark](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark)
and its
[author code repository](https://github.com/abeerbadawi/cognitive-atrophy-benchmark)
describes CareBench-derived material. This is a different release with different
authors, data processing and labels. Its publication or broad card license does
not alone establish authorization to redistribute the original CareBench-CBT
text. Do not fetch a derived copy as a workaround for the blocked original
supplement or assume its ratings transfer to our future model answers.

The derivative's public [file tree](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark/tree/main)
was inspected at observed revision `c2dbbadaa150746f5bf666182c4f397164d0028f`.
Its card's 360 CareBench annotation rows are not 360 independently rated turns;
the row identity includes reviewer, conversation and turn. Clinician-trained
reviewers are not thereby verified as licensed doctors. No CSV contents were
downloaded by this audit.

The inspected [annotated-response directory](https://huggingface.co/datasets/abadawi/Cognitive_Atrophy_Benchmark/tree/main/data/annotated_responses)
visibly lists `data/annotated_responses/carebench_annotated.csv`, with displayed
size **5.06 MB**. This proves a separately published artifact is listed; it does
not establish its exact byte count, SHA256, original-source permission, or
readiness for this experiment. The displayed MB value is not a verified length.

## Implementable next step after legitimate artifact access

The following are this project's proposed controls, **not claims that they were
implemented or established by the HTM paper**:

1. Inspect metadata first and freeze the exact release bytes/terms. Establish
   whether a session is an actual clinical encounter, professional roleplay, or
   edited reconstruction. Retain that source classification in reports. Do not
   invent patient identities or link unrelated sessions into treatment courses.
2. Use a fixed original-history prefix ending at the current client turn for the
   main paired comparison. Supply identical recent-message concatenation to all
   arms. Hide the next therapist answer and all later turns from extraction,
   index construction, state update, retrieval, prompts, and answer generation.
3. Compare the existing managed pipeline against capable original-text retrieval
   with the same reader, history, budget, query, permissions, and confirmation
   policy. Tune retrieval on separate development families. Keep a recent-only
   arm auxiliary. Add HTM as a separately identified related-work/ablation arm
   only after its actual settings and implementation are inspectable.
4. Build memory targets from allowed earlier source spans before opening held-out
   outputs. Separate client reports, therapist suggestions, agreed plans and
   later corrections. Professional response references are useful anchors; they
   do not automatically supply independent memory-answer gold or one uniquely
   correct continuation. Record proposed versus human-adjudicated labels.
5. Report paired memory precision, necessary-fact coverage, unsupported assertions
   and update errors first. Score the requested five CBT quality dimensions
   separately, using blind qualified review for new answers. Existing published
   clinician ratings describe their original responses, not our newly generated
   outputs. Keep automated scoring explicitly identified.
6. Split by the strongest available source/patient/session family before tuning;
   report unresolved grouping information. Count preprocessing, extraction,
   indexing, state updates, reading and failed attempts in resource totals.

The currently frozen CACTUS quotation-cue entry and completed controlled state
checks keep their existing identities. This candidate neither replaces their
archives nor turns their results into professional therapy evidence.
