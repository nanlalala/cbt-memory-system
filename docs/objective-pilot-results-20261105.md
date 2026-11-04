# Objective memory pilot — actual results, 5 November 2026

This completes the predeclared **16-question exploratory pilot**, not the full
800-question benchmark, semantic-management evaluation, user study or clinical study.
All full-data preparation and paid calls ran in genuine hosted Google Colab.

| Condition | Official substring-match points | Score |
| --- | ---: | ---: |
| Recent 16 turns | 0 / 16 | 0% |
| Recent turns + raw-history RAG | 2 / 16 | 12.5% |
| Recent turns + managed quotations | 2 / 16 | 12.5% |

Raw and managed prompts are identical in all 16 cases and share one canonical
response per pair. Their zero difference is imposed by the design. It estimates
neither semantic-indexing nor version-management benefit. The difference from
recent-only context is descriptive (+12.5 percentage points), with one conservative
independent family and **no estimable cluster interval** or population inference.

## Per-question official scores

`mh` and `sh` retain the released multihop/single-hop task and context-length labels.
A 1 is an upstream metric match, not independent verification of a reliable answer.

| Released task / question | Recent | Raw RAG | Managed quotations |
| --- | ---: | ---: | ---: |
| mh_262k_no82 | 0 | 0 | 0 |
| mh_262k_no9 | 0 | 0 | 0 |
| mh_32k_no93 | 0 | 0 | 0 |
| mh_32k_no33 | 0 | 0 | 0 |
| mh_64k_no58 | 0 | 0 | 0 |
| mh_64k_no87 | 0 | 0 | 0 |
| mh_6k_no8 | 0 | 0 | 0 |
| mh_6k_no99 | 0 | 0 | 0 |
| sh_262k_no86 | 0 | 0 | 0 |
| sh_262k_no10 | 0 | 0 | 0 |
| sh_32k_no69 | 0 | 0 | 0 |
| sh_32k_no38 | 0 | 0 | 0 |
| sh_64k_no51 | 0 | 0 | 0 |
| sh_64k_no14 | 0 | 1 | 1 |
| sh_6k_no62 | 0 | 1 | 1 |
| sh_6k_no80 | 0 | 0 | 0 |

## Error diagnosis and metric limitations

The two metric-positive raw/managed answers are the CERN director and Twitter CEO
questions. Both replies list conflicting candidate entities and say they cannot
confidently determine the answer. The official substring matcher accepts these
because an accepted answer string occurs in the reply. **Neither positive point
should be described as independently established conflict-resolution success.**
This observational review is model-assisted, not human annotation or CBT review.
The original metric remains unchanged; no post-hoc replacement success metric is
introduced.

There are 44 incorrect logical arm rows (30 incorrect unique responses), zero
provider failures and no incomplete requests. All incorrect rows retain
`answer_incorrect_stage_unresolved`. Extraction and semantic update did not run in
this v1 quotation control. Retrieval includes 12 older source documents per RAG
answer, with no recorded character-budget drops among those selected documents.
That does not establish that the retrieved documents include all necessary evidence.
There are no independent sufficient-source labels for these questions. In
particular, answer substrings are not used to manufacture retrieval or generation
failure attribution. A source document in context also would not prove its correct
interpretation. The detailed controlled ledger retains actual replies, gold labels,
source IDs and stage traces for review.

Model-assisted review identified eight wrong raw-RAG responses that match an
older relation value even though a later value matching the released label is
already present in the actual answer context. For example, the Sebald language
question contains both source-order 1632 and 2112, and the Lisa Leslie 6k position
question contains both source-order 15 and 170. This rules out describing those
observations simply as “no updated evidence was retrieved,” but does not prove
whether priors, presentation order, instructions or another mechanism caused them.
Five multihop refusals lack a complete visible chain; another wrong answer follows
a visible chain inconsistent with gold. Independent sufficient-evidence annotations
and full-source review are still needed to allocate stage responsibility.

Several replies abstain, list conflicting facts, or supply an answer that does not
match the released label. These are observed response patterns; their source-stage
cause, use of external priors and correctness under the synthetic task require
separate evidence review. No diagnosis, treatment benefit, or participant experience
is inferred.

## Cost, failure and effort ledger

| Recorded measure | Value |
| --- | ---: |
| Logical arm rows completed | 48 / 48 |
| Unique paid provider attempts | 32 |
| Unique successful provider calls | 32 |
| Failed provider attempts | 0 |
| Prompt tokens | 57,193 |
| Completion tokens | 715 |
| Total tokens | 57,908 |
| Median unique-call API latency | 1.613 seconds |
| Sum unique-call API latencies | 51.040 seconds |
| Notebook batch wall time (UI receipt) | 78.557 seconds |
| Monetary bill | Unobserved; not inferred |
| Participant effort / unnecessary checks / repeated questions | Not measured |

Tokens and latency are counted once per canonical call, not once per derived arm.
API timing excludes data preparation, retrieval and operator actions. The two
Secret-read preflights timed out before provider invocation; the user then explicitly
approved granting access, the permission was granted, and the bounded batch
completed. These preflight events and the earlier corrected notebook-editing
SyntaxError are operator/execution records, not benchmark answer errors. No Secret
value was printed, committed, exported or placed in this report. Zero interactive
checks in static QA do not demonstrate reduced user effort.

## Immutable identities and recovery

- Remote main checked: `80551b7edc0199b8870b174e3f144155b0d60c85`.
- Executed v1 code: `7832f38a639eb5b5e3c27d6c564db5db25f3999c`.
- Protocol: `9b798544eaf0f7b4355eb7b2772e199b45f80420dc648154942db54f0751d4fd`.
- Complete audit ZIP SHA-256: `b6591495178e0e902307d2cc9346a5f7ff122d0c96e838050115eab2f14369dd`.
- Archive: 4,354,627 bytes, 31 members, validated by the hosted exporter.
- Reported provider model: `deepseek-v4-1-flash-260910`; name allowlist checked,
  underlying provider weights not independently authenticated.

The full archive is controlled project material, with frozen prompts, scoring-only
references, responses, all request intents/outcomes, scoring code, checksums and
source provenance. Neither the archive nor the original benchmark texts are
published to GitHub. The older LongMemEval checkpoint and the preflight snapshot
remain separate and unchanged.

## Remaining work

The v2 source-bound semantic-index path has genuine source-selection differences
in fixtures. A real extraction/indexing journal and paired model run remain pending;
software fixture differences are not response-effectiveness results. It does not
implement or establish semantic version-management benefit. Independent evidence
annotation, four-visit participant testing, and CBT professional review remain
outstanding. This pilot cannot close those gaps.
