# Professional dialogue access audit — 12 November 2026

Purpose: identify an accessible professionally grounded CBT source for a future
memory comparison and reference-response evaluation. This audit fetched only
public metadata, documentation and papers. It downloaded no transcript corpus
or model to the local workspace and admitted no generation requests.

## RealCBT: public transcripts withdrawn; not generation eligible

Primary publication: Xiaoyi Wang, Jiwei Zhang, Guangtao Zhang and Honglei Guo.
2025. *Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and
LLM-Generated CBT Sessions*. Findings of EMNLP 2025, pp. 19999–20017.

* Publisher: https://aclanthology.org/2025.findings-emnlp.1089/
* Paper: https://aclanthology.org/2025.findings-emnlp.1089.pdf
* Author-linked release: https://gitlab.com/xiaoyi.wang/realcbt-dataset
* Official checklist:
  https://aclanthology.org/attachments/2025.findings-emnlp.1089.checklist.pdf

The paper describes 76 CBT-labeled public-video conversations, transcribed and
manually corrected. Search terms include role play, so “authentic” does not
establish that every conversation involves real patients or credentialed
therapists. Each video is treated as a session; no verified repeated-client
visit grouping or source-span memory gold is supplied by this audit. Published
metadata concern problem, gender and attitude, rather than memory correctness
or independent clinician response-quality scores.

The original web retrieval could read only the GitLab project header. A
subsequent genuine hosted-Colab metadata check resolved that retrieval gap:
the official project and tree API calls both returned HTTP 200. Project
74581779 uses default branch `main`; the project API reports `license: null`.
The observed tree contains only `.gitignore`, `README.md` and
`RealCBT_Dataset_Metadata.xlsx`, with no publicly distributed raw transcripts.

The README's access update dated **4 August 2026** explicitly withdraws public
raw-transcript distribution while retaining project documentation, metadata
and aggregate findings. It identifies third-party rights, privacy, attribution
and downstream reuse conditions as the reasons. This is an observed access
change, rather than an unresolved page-loading error. The same README contains
duplicated older paragraphs describing an open release; those stale lower
paragraphs do not override the dated access update at its top.

The immutable README export was independently checked locally against both
its Git blob identity and its recorded SHA256. It is documentation, not a
transcript corpus:

| Evidence | Observed identity |
|---|---|
| Official project URL | `https://gitlab.com/xiaoyi.wang/realcbt-dataset` |
| Project metadata API | `https://gitlab.com/api/v4/projects/74581779?license=true` |
| Current tree API | `https://gitlab.com/api/v4/projects/74581779/repository/tree?per_page=100` |
| README Git blob | `8f772dffb03170a4311f46aa5eb3be2aec22405c` |
| README bytes / SHA256 | 11,589 / `f583f55ec208fb8e8ec549cbfbc40fab7d8af9c57e4a6f573e8677b2c7a2f7ac` |
| Project metadata SHA256 | `b65b9357ddbe420dbacc4e7208d1ceadea4b2b5e85b5a36fa4ce63b519b7a08a` |
| Tree metadata SHA256 | `2b64125080a513aea62a8ba576103a67c4cdc4ad960b10938d14fdfd3022779c` |
| Metadata workbook Git blob | `6a2fc8e93701bdb2dbd6ff69da9db36a3b4999c7` |

The private evidence bundle retains `source_metadata/RealCBT_README_immutable.txt`,
`source_metadata/setup_metadata_receipt.json`, the project/tree responses and
their HTTP-status receipts. Only documentation and metadata were obtained; the
metadata workbook itself was not downloaded by this audit.

**Current admission: `generation_eligible=false`.** Paper citation and aggregate
findings remain usable as research background, but this project has no verified
transcript access grant. It will not retrieve removed transcripts from prior
commits, archives or mirrors to bypass the current access restriction. No
author emails, access requests or forms were sent. The absence of a currently
usable corpus does not alter the published paper's aggregate findings.

## CBT-DP: accessible professional CBT response references, not memory history

* Author repository: https://github.com/mianzhang/CBT-Bench
* Current author-linked dataset:
  https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench
* CBT-DP files:
  https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench/tree/main/CBT-DP
* Paper: https://aclanthology.org/2025.naacl-long.196.pdf
* Observed current repository revision:
  `bbd37514ba2e4db46bd9633d9207a57828c2ec9d`.
* Reference-file introduction commit:
  `436fac0ac178362a5d4c48aa18a29af08800dd4b`.

The official card declares CC BY-NC 4.0. CBT-DP contains 156 deliberate-practice
exercises across ten skills, released human reference responses in
`CBT-DP/dp-reference-exe-1.json` through `-10.json`, and pairwise-comparison
material. The publication reports expert participation and explicitly rejects
treating a reference response as the sole acceptable answer; it uses expert
pairwise criteria. These are response exercises, not a longitudinal patient
corpus or independent memory targets. Suitable immediate role: professional
CBT response/rubric anchor. New generated answers still require new review;
published preferences for other models cannot be transferred to this system.

## ALIA clinic roleplay: accessible professional multi-turn supplement

* Official university research-group card and files:
  https://huggingface.co/datasets/SINAI/ALIA-es-clinical-psychology-dialogues
* Fixed observed revision:
  `2f49440c82969e45104d6f6532a0bd94d0d36bd3`.
* File: `ALIA-es-clinical-psychology-dialogues.jsonl`, approximately 657 kB.

The SINAI card describes 66 Spanish roleplay dialogues developed and refined
by mental-health professionals, approximately 30 turns per session. No real
patients are involved. It includes cognitive restructuring but is not verified
as exclusively CBT. Its structured fields preserve within-session speaker
order; no manual memory labels or repeated-client visit identifiers are
documented. The top-level license declares CC BY-NC-SA 4.0 whereas the card's
clinic-subset table says CC BY-SA 4.0. Academic reuse should conservatively
retain the noncommercial and share-alike conditions until that discrepancy is
resolved. The separately named TerapIA subset involves an AI assistant and
must not be relabeled as therapist-authored dialogue.

Potential role: a professionally authored multi-turn counseling supplement
after hosted identity/terms checks and source-span annotation, not a ready
CBT-specific or clinician-scored memory benchmark. Any memory targets created
by this project must remain project annotations until independently reviewed.

## Other candidate checked

https://github.com/Columbia-ICSL/CaiTI_dataset describes task-specific material
prepared and labeled by licensed psychotherapists. A CBT CSV is present, but
the README still says availability is forthcoming and no reuse license was
verified. It is not admitted as an immediately licensed source.

## Separate reader-setup metadata, not CBT evidence

The same hosted metadata session obtained an HTTP 200 official model API
response from
`https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-GGUF?blobs=true`.
The preserved 9,781-byte JSON has SHA256
`4beda630a9a3b5c4f3bf0a0c4fdda517639d4dc359ed1c6c947165802290d8aa`.
It identifies revision `bb5d59e06d9551d752d08b292a50eb208b07ab1f` and
license `apache-2.0`, and fixes the two Q4_K_M shard sizes and LFS digests:

| Official filename | Bytes | SHA256 |
|---|---:|---|
| `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` | 3,993,201,344 | `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` |
| `qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf` | 689,872,288 | `539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a` |

This section verifies downloaded metadata only. Successful weight transport,
model loading, reading-gate completion and a memory benefit require their own
actual receipts and results; none is inferred from this metadata response.

## Decision

No source checked here simultaneously supplies verified immediately usable
therapist CBT multi-session transcripts, independent source-span memory gold
and ratings of this system's generated responses. RealCBT is excluded from
generation under its current public-access state. CBT-DP remains a separate
professional-response anchor; ALIA is a possible broader professional roleplay
supplement after its documented limitations and terms are checked. Any new
memory targets require actual pre-cutoff source spans and separately identified
review. Dataset identity, memory correctness and CBT response quality remain
separate evidence roles. This finding does not alter the completed constructed
v3 comparison or manufacture a clinician score.
