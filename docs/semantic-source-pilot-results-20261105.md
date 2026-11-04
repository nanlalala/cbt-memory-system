# Semantic source-index follow-up — actual results, 5 November 2026

**No benefit demonstrated.** This is a four-question development follow-up using
all four 6k cases from the already reviewed 16-question pilot, not a new independent
held-out sample. One conservative fact-consolidation family remains the only
independence unit. These are synthetic fact-conflict tasks, not CBT participants.

All source preparation, extraction and answer API calls ran in genuine hosted
Google Colab. The old v1 run and executable were preserved. New executable:
`10eba29bef51b94b9c105ecd2f58505138201af2`.

## Actually different conditions

The raw baseline searches eligible original history. The semantic arm searches
unconfirmed typed summary hints produced by a separate model extraction stage;
it renders only their eligible original source quotations. Both use the same
answer model, current question, recent window, lexical ranker, character budgets,
confirmation policy and access/lifecycle checks. Model-produced summaries do not
become confirmed facts, user consent or destructive version updates.

The source-only extractor received neither questions, scoring answers nor prior
model answers. The preparer reads compact case metadata (which contains question
fields) for source identity and scope verification, but those fields are not copied
into extraction requests. Exact quote matching authenticates provenance, not
summary entailment. This selection does not measure semantic version management.

Extraction processed 912 source documents in two separate released contexts,
through 18 frozen source-only requests, all successful. Only two of the 18 API responses contain any candidates (55 each); the other
16 valid JSON responses return an empty candidate list. Thus 18/18 technical
success must not be confused with useful source coverage. It produced 110 unique
source-bound candidates (55 per context); the same context's candidates are reused
across its two questions. This is an observed selection count, not evidence recall
or semantic precision against independent labels.

All four raw/semantic input pairs differ in source content. No heading changes
manufacture the contrast. There are 12 logical arm rows and 11 actual canonical
answer calls; one semantic prompt equals its recent-only prompt because no older
source was retrieved. That exact pair shares one response.

## Official per-question results

| Released question suffix | Recent | Raw RAG | Semantic source index |
| --- | ---: | ---: | ---: |
| mh_6k_no8 | 0 | 0 | 0 |
| mh_6k_no99 | 0 | 0 | 0 |
| sh_6k_no62 | 0 | 1 | 0 |
| sh_6k_no80 | 0 | 0 | 0 |
| Total | 0 / 4 | 1 / 4 | 0 / 4 |

The official semantic-minus-raw difference is -25 percentage points (one loss,
three ties). There is no estimable independent-cluster interval. The sole raw
positive again says it cannot determine Twitter's CEO and lists conflicting
entities, including the accepted string. The official substring metric is kept,
but that point is not reliable conflict-resolution success. These data cannot
establish that semantic memory generally harms or helps users.

## Evidence-grounded stage observations

The semantic arm retrieved 0, 1, 1 and 1 older source documents respectively;
the raw arm retrieved 12 for each question. No selected semantic document was
recorded as dropped by the character budget. Selection is thus visibly sparse
before context assembly, without proving which complete evidence set was needed.

For Lisa Leslie (`sh_6k_no80`), the raw answer context contains:

- Source order 170, document
  `522546114f4b89a963cae355ba5600db98e0b1ec063803030b00c596ee9ac223`:
  the later position value `goaltender`. This document is absent from the compiled
  semantic index and therefore absent from semantic retrieval/context.
- Source order 15, document
  `cd55c07c4865efbd77049e3314e2c73922564500ad37490b8292cd018172ccd0`:
  the earlier value `center`. This document survives index selection, retrieval
  and context assembly; the semantic answer is `Center.`.

For Twitter (`sh_6k_no62`), the raw context includes both the later source-order
276 document (`80a3a2a4f78a6a15362c5f35330a42da9d430db0bbf99c8058ae6424049a3b8f`)
and earlier source-order 110 document
(`80ad0f996ee7177a641c2528ef244d42d3605b0f3657c1fa884a5a6311467129`).
Both are absent from the compiled semantic index; the semantic answer says the
provided history does not state the CEO.

These are mechanically verifiable source-path losses and model-assisted relation
interpretations. They identify observable index-selection behavior, not complete
causal blame. Independent sufficient-evidence annotation, alternatives, and
counterfactual interventions are still absent. We do not label an entire wrong
answer “caused by extraction” solely because a familiar answer string is missing.
The first multihop semantic arm has no older retrieval and is identical to recent
history; the second retrieves one source and still abstains. Neither is evidence
that a complete multi-hop evidence chain was available.

## Full pipeline cost and failures

| Stage | Actual calls | Prompt tokens | Completion tokens | Total |
| --- | ---: | ---: | ---: | ---: |
| Source-only extraction | 18 | 75,307 | 9,914 | 85,221 |
| Three-arm answer generation (deduplicated) | 11 | 18,096 | 220 | 18,316 |
| This follow-up pipeline | 29 | 93,403 | 10,134 | 103,537 |

All 29 calls succeeded, with no inflight or exhausted failures and no missing token
usage. Semantic preparation cost must not be hidden behind its shorter answer
contexts. This shared pipeline ledger is not a marginal per-user price or an
amortized deployment estimate. Monetary billing was not observed; no currency
amount is inferred. Across both actual pilots, there were 61 calls and 161,445
reported tokens, with overlapping questions rather than 20 independent subjects.
Participant effort, unnecessary checks, repeated questions, professional review
and clinical outcomes remain unmeasured.

An export-launcher SyntaxError caused by a nested newline literal was corrected
without changing frozen code/protocols or repeating any model request. It was an
operator packaging error, not a provider or benchmark failure.

## Frozen evidence

- Extraction protocol:
  `f7b215425214584f54fc860d8306e75e61a0e02b16614ced96b4f02dbf5c5d4a`.
- Compiled index SHA-256:
  `2b0991fdfaa755c3b459311630d969d5761ec75491cd1c8431242c1c4c5d79c4`.
- New QA protocol:
  `8ffde096e3301569211a12203852c9f2d2f78fb4c71027cfc0a7fa4dc3485ef3`.
- Controlled full-pipeline archive: 399,428 bytes, SHA-256
  `db963856d95030a72ac57d852200a6e14d7800ebbe4e46a7ba6ce853629339f6`.

The archive retains source-only prompts, exact source mappings, extraction outputs,
all attempts/outcomes, compiled index, code identity, nested independently scored
QA audit and total-cost ledger. Both outer and nested member checksums were checked.
Full benchmark-derived text stays in controlled project artifacts, not this public
repository report. The 800-question benchmark, PersonaMem cohort, independent
semantic-evidence labels, actual CBT goal/homework version study and four-visit
participant/professional review have not been completed by these pilots.
