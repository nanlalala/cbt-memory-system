# Source-retrieval optimization decisions

This development study asks whether memory organization helps recover a complete
Client quotation and its exact source ID from synthetic CACTUS CBT dialogue,
given a four-word locator. The question and evidence unit remain unchanged.

## Design decisions

1. Append the same six recent messages directly in every arm, preserving role,
   order and source identities. This is a common context improvement, not a
   managed-memory treatment difference.
2. Give the raw-history baseline access to the full eligible source pool with
   the same encoder, source controls and complete-request budget. Do not restrict
   its retrieval depth to manufacture a managed advantage.
3. Keep a source-backed value: the original text is what the reader sees.
   Extracted candidate descriptions may affect an index, but never become
   user-confirmed facts or replace the original quotation.
4. Compare raw source keys, the frozen candidate/source rank-fusion version,
   and a separately versioned source-key augmentation. Freeze augmentation
   caps and ordering before opening assessor targets. Preserve the original
   three-probe answer proposal without silently substituting new inputs.

LongMemEval separates indexing, retrieval and reading, and reports better
performance for fact-augmented key merging than its post-retrieval rank merging
alternative [1, section 5.3 and appendix E.3]. This motivates an ablation; its
result does not predict the winner on four-word quotation recall. LoCoMo retains
turn-level evidence for extracted observations and shows that retrieving more
items need not improve answers [2, section 6.1 and appendix A.2]. RAPTOR provides
a useful precedent for controlling retrieval and reader budgets when comparing
memory abstractions [3, sections 3–4]. ALCE motivates reporting answer correctness
and citation support separately [4, section 3].

## Measurements and the question each answers

| Measurement | What it verifies |
| --- | --- |
| Target source rank | Whether an index helps locate the required original evidence |
| Complete quotation and bound ID in the final request | Whether selection and budget packing preserve usable evidence |
| Quotation, ID, and joint answer success | Whether the reader correctly uses that evidence |
| Wrong quotation, wrong ID, unknown, and malformed output | Which observable memory error occurs |
| Source sets and ordering separately | Whether a changed request actually contains different evidence |
| All extraction, embedding and answer attempts | The resource cost of each treatment, including failures and reused development work |

No generated answer means `answer_accuracy = null`, never zero and never an
estimate from target-in-context. No human assessment means no human score.
The frozen literal scoring rule remains the answer comparator for this task.

## Development and held-out boundaries

The current three probes share one source family and one historical prefix.
They are a development case, not three independent samples or a dataset-level
effect estimate. The source family identified in the frozen development
protocol is exposed; source-ID case fragments are not used as family IDs.

Before a new held-out answer run:

- Reconcile all earlier v1/v2 exposure inventories. A family with uncertain
  prior exposure is not silently declared held out.
- Assign entire source families to development or evaluation; keep all three
  probes of a family in the same partition.
- Freeze code, encoder, recent-message policy, source/candidate admission,
  key representation, budget, score rule, sample membership and stopping rule.
- Run the fixed evaluation once. Report every planned arm, loss, failure and
  unknown; do not add families or change retrieval parameters in response to
  the observed winner.
- Report paired wins/losses and uncertainty at the family level if there are
  enough independent families. Keep this source-quotation experiment separate
  from update, deletion, homework-tracking and CBT response-quality studies.

## Paid-call state

The previously frozen proposal contains three development probes and nine
logical arm requests, with at most nine new provider attempts. It was not
executed: automatic approval review rejected creation of a new paid-call
authorization from the general instruction to continue optimizing. No
authorization record or provider request is inferred from that instruction.
The source gateway remains the one in the frozen protocol. A separate explicit
approval for its data destination and attempt ceiling is needed for this pilot.
Offline source and index diagnostics continue without Secret access or API use.

## References

[1] Wu D, et al. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf

[2] Maharana A, et al. Evaluating Very Long-Term Conversational Memory of LLM Agents. ACL, 2024. https://aclanthology.org/2024.acl-long.747/

[3] Sarthi P, et al. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR, 2024. https://arxiv.org/abs/2401.18059

[4] Gao T, et al. Enabling Large Language Models to Generate Text with Citations. EMNLP, 2023. https://aclanthology.org/2023.emnlp-main.398/
