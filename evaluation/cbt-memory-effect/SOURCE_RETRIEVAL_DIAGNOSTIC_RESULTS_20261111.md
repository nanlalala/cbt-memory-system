# Hosted-Colab source-retrieval optimization results

The frozen inputs and old experiment were restored, not regenerated. All work
reported here ran in genuine hosted Colab. No new paid answer call or Secret
access occurred. The three development probes share one source family.

## Original evidence in the final context

| Probe position | Recent six messages | Full-pool original retrieval | Candidate/source RRF | Source-key augmentation |
| --- | --- | --- | --- | --- |
| Early | Absent | Present, target rank 1 | Present, target rank 1 | Present, target rank 1 |
| Middle | Absent | Present, target rank 1 | Present, target rank 2 | Present, target rank 1 |
| Recent | Present | Present, target rank 1 | Present, target rank 4 | Present, target rank 1 |

Presence requires the complete Client quotation and its correctly bound source
ID in the actual structured request. Every retrieval condition retained 3/3
targets; recent context retained 1/3. Key augmentation corrected the two rank
demotions in the RRF condition, but did not improve target coverage over the
original-text baseline. These are input diagnostics, not generated-answer
accuracy. No answer was generated for any of these nine proposed arm rows.

## Index reuse, with unchanged complete requests

| Quantity | Uncached v1 | Per-run exact cache v2 |
| --- | ---: | ---: |
| Logical encode calls | 18 | 18 |
| Requested encode texts | 225 | 225 |
| Physical encode calls | 18 | 6 |
| Physically encoded texts | 225 | 57 |
| Cache hits | 0 | 168 |
| Observed encoder computation | 10.5589 s | 2.0609 s |
| Complete request hashes and source order equal | Reference | 9/9 |

Physical text encoding fell by 74.7%. The two observed runs used the same pinned
E5 model and same three conditions. The wall-time comparison is a single
development observation, not a repeated timing benchmark. Model loading was
13.3958 s and 9.9673 s respectively and is reported separately; its difference
is not attributed to memoization. The cache is new in memory per invocation,
keyed by exact text, query/passage role and model identity. It is not a persistent
user-memory cache. Every arm receives the same optimization.

## The original locator benchmark has a shortcut

The whole frozen set was audited without generation or embeddings. The matching
phase read only generation history and the explicit JSON locator, wrote its
results, then a separate assessor phase opened the checksum-verified targets.

| Stratum | Eligible probes | Unique prefix matches verifying the original target |
| --- | ---: | ---: |
| Early | 197 | 197 |
| Middle | 198 | 198 |
| Recent | 195 | 195 |
| All 200 source families | 590 | 590 |

Every retained four-word locator uniquely identifies its complete original
Client message. This confirms a deterministic lookup shortcut in this task;
it is not a score for a raw-history LLM, source-ID joint answering, semantic
memory, or management gain. All 200 families opened in this assessor are
registered as assessed and cannot later be described as untouched holdout.

## Consequence for the next experiment

The four-word quotation task remains archived under its original identity.
Repeatedly tuning on it is unlikely to establish the intended semantic memory
benefit. A separate natural follow-up study should use source-linked factual
targets and independently reviewed questions, with all probes of one source
family kept in one split. The CACTUS and MusPsy materials inspected in the
official releases do not provide a ready independent memory-QA/span/answer gold
set; generated plans or memories are not silently promoted into that role.

The existing three-probe answer proposal is unchanged. It can check reader
copying and source-binding behavior, but its nine paid attempts have not been
authorized by automatic approval review. The review rejected deriving that
specific payment and external-data authorization from a general instruction to
continue optimizing. No authorization file was created and no retry occurred.

## Reproduction identities

- Original frozen base: `34aec110811a61eda50431c555b3a5a70620e8bd`.
- Uncached preparation code: `2ec207647f4169a299a5ac556189e91f8216c6c2`.
- Read-only assessors: `778fee7a1643d9545022f21eeab6530aa1130d67`.
- Cached v2 preparation code: `ede92a1700e345bc9f56f6b592e791897104df09`.
- v1 key protocol: `a359b1ddd356b2326cf0534b2d6645f4b7c3e5481a877c6b510e480166e45852`.
- Cached v2 protocol: `2d72d30e6de929eb74090d00161f01f04ee37ce3c91609b38e36e6a235357b6a`.
- Original answer proposal: `a600c93183d23247824cb30fe077ef7291996e4edae32865297f440e22c965c6`.
- Source/family exposure register SHA256: `711d590a5c763fa5ddb8217268181f1e86cc46f5c019a1a151d0cccd4a2a43d4`.

The hosted notebook retains printed summaries. Detailed private JSON/CSV and
traces remain under the new Colab directories; the earlier browser download
restriction was not bypassed. No full-CACTUS re-download replaced the original
verified source. The original paid journal and checkpoint remain unchanged.

See [research decisions](RESEARCH_DECISIONS_20261111.md) for numbered primary
references and the role of each metric. Engineering validation is separate:
25 targeted fixture tests pass; they are not memory-effect scores.
