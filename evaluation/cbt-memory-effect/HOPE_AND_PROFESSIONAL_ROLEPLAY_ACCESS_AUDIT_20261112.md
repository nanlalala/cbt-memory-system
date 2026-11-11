# HOPE and professional role-play source audit — 12 November 2026

This audit identifies a lawful professional-material route for a small controlled
validation study. It does not claim that additional model responses, clinical
ratings or a new memory-effect result have been produced.

## HOPE: what is and is not verified

The original SPARTA/HOPE repository is
`https://github.com/LCS2-IIITD/SPARTA_WSDM2022`, current commit
`040856906172e32a809c2ffbe3a454ecbeb93d29`.

Its README says the `HOPE_data/` directory contains the dataset, but also
explicitly requires an agreement form and forwarding its confirmation to the
authors to request access. Public file visibility is not evidence that the
required research access has already been granted to this project. Reuse any
documented, still-valid existing access; otherwise complete the stated process
before acquisition. Do not silently substitute a mirror.

The original paper describes YouTube-derived counseling transcripts with
dialogue-act annotations; the creator-associated tutorial reports about 12.9K
utterances from 212 dialogues. No primary evidence examined here establishes
that the dataset is ten visits from one client, exclusively CBT, or uniformly
delivered by credential-verified high-quality CBT clinicians. The earlier
project reference to ten sessions therefore needs a specific source manifest;
it cannot establish these properties by itself.

An indexed original-publisher CSV path is
`HOPE_data/HOPE_therapy_session_transcripts/67.csv`. Its visible search excerpt
shows utterance IDs, T/P speaker labels and dialogue acts, not a CBT-quality or
memory-question gold standard. Case 67 is now design-exposed and must be treated
as development if later used. No additional HOPE transcript text was retrieved
or dataset copied during this audit.

If authorized HOPE material is later available, a lawful within-dialogue prefix
comparison is possible: stop at a real patient turn, hide the next therapist
response and all future content, and group linked source videos before splitting.
That is a within-dialogue memory evaluation unless true client/visit links are
independently established. A single recorded therapist response is a reference,
not the only appropriate answer. CBT qualification and rating anchors must be
verified separately; dialogue-act agreement is not sufficient.

## Immediately accessible professional source: WHO/UNICEF EQUIP

WHO identifies EQUIP as its joint initiative with UNICEF. The official resource
page provides standardized role-play scripts and competency tools. Its terms
declare **CC BY-NC-SA 3.0 IGO**, allowing attributed noncommercial copying and
adaptation under the same or an equivalent licence. Do not imply WHO endorsement,
use its logo, or label a translation an official WHO translation. Keep adapted
case materials separately licensed; the code licence does not replace the
source-material licence.

Official overview:
https://www.who.int/teams/mental-health-and-substance-use/treatment-care/equip-ensuring-quality-in-psychological-support

Official role-play resource index:
https://equipcompetency.org/en-gb/resources/5051

Official reuse terms:
https://equipcompetency.org/en-gb/policies/terms-of-use

### Concrete files for a source freeze

Acquire these original-publisher files in genuine hosted Colab. Record retrieval
time, exact URLs, bytes and SHA256 before parsing. URLs alone do not establish
an immutable content identity. No byte sizes or hashes are fabricated here.

| File | Verified content and appropriate role |
| --- | --- |
| ENACT Role Play Instructions & Scripts, 23-page PDF | Four standardized example contexts; helper instructions are separate from hidden actor information and prompts. Useful for professional foundational-support validation. Not all ENACT skills are CBT-specific. |
| Cognitive competency descriptions and single-item role-play prompts, 4-page PDF | Professional CBT-related expectations for relating thoughts, emotions and behavior and other cognitive skills. Useful as prespecified response criteria, not an independent longitudinal answer set. |
| Behavioural Activation competency descriptions / assessment tool, 4-page PDF | Prespecified helpful/unhelpful behavior, including reviewing practice and exploring barriers. Uses its own four-level competency categories, not project R1–R5 or CTRS. |
| Individual PM+ role-play script, 8-page PDF | Professional actor/helper assessment scenario; useful only after addressing the internal source inconsistency below. PM+ is a transdiagnostic intervention with CBT-related components, not a full pure-CBT transcript corpus. |

Exact observed original-publisher URLs:

```text
https://equipcompetency.org/sites/default/files/downloads/2022-08/ENACT%20Role%20Play%20Instructions%20&%20Scripts_platform_220830.pdf
https://equipcompetency.org/sites/default/files/downloads/2022-08/Cog%20competency%20descrip_single%20role%20plays_v1.0_220320.pdf
https://whoequipremote.org/sites/default/files/downloads/2026-04/BA_EN_260417.pdf
https://equipcompetency.org/sites/default/files/downloads/2022-08/EQUIP%20NS%20NYC%20PM%2B%20Role%20Play%20Scripts_v21Jan_Clean220307.pdf
```

The resource index also links a Behavioural Activation single-item role-play
prompt PDF and a Problem Solving prompt PDF:

```text
https://equipcompetency.org/sites/default/files/downloads/2022-08/BA%20competency%20descrip_single%20role%20plays_v1.0_220320.pdf
https://whoequipremote.org/sites/default/files/downloads/2022-08/Problem%20Solving%20competency%20descrip_single%20role%20plays_v1.0__220320.pdf
```

The versioned collector `freeze_equip_sources.py` acquires only these six PDFs
and the official terms HTML. Run it in a genuine hosted Colab checkout, using
a fresh output directory:

```python
!python evaluation/cbt-memory-effect/freeze_equip_sources.py --output /content/equip-source-freeze-20261112-v1
```

It records original bytes, SHA256, byte sizes, retrieval time and exact HTML
licence-evidence byte ranges. Existing outputs are never replaced. Any failure
preserves its archive and journal without an automatic retry. All complete
source files remain assessor-only; the collector creates no benchmark inputs,
adapted conversations, gold labels or effect scores. A successful source-bank
freeze therefore precedes the span audit and input/assessor separation below.

### Observed source inconsistency and exposure

The examined Individual PM+ script's helper instructions and actor prompts
describe Sam, including shelter/job-search circumstances, while the adjacent
hidden actor-background sections describe Reese, an employed parent working
remotely. These contexts must not be merged into a supposedly coherent client.
Preserve the original file, flag the mismatch, and exclude that combined unit
unless an authoritative clarification or a separately labelled adaptation
resolves it. The Sam/Reese example and the cognitive-prompt examples viewed
during this audit are development-exposed.

Only the ENACT file's metadata and instruction structure were inspected here;
individual case details were not manually examined. This is not certification
that its cases are untouched in earlier project work. Record prior exposure
before assigning a validation role.

## Small, runnable validation proposal

Use the four published ENACT contexts as an externally authored source bank,
then prespecify applicable cognitive/behavioral-activation skill criteria. After
Colab acquisition, reject any contradictory or incomplete source unit before
seeing system answers. Freeze source spans, exclusions and adaptation rules.

This should be labelled **professionally grounded controlled validation**, not
a naturally recorded patient study or a released independent memory benchmark.
The scripts are conditional assessment scenarios, not a timestamped transcript.
Any transformation into a sequence of messages is an explicit adaptation, with
an adaptation map to exact source spans and a corresponding material licence.
Do not invent dates, sessions or unseen client agreement.

The generation view may contain only helper-visible instructions and actor
information actually disclosed before the cutoff. Hidden actor background,
future prompts, competency-specific answer guidance and the rating rubric are
assessor-only. All arms receive the same disclosed history, direct recent buffer,
knowledge channel, source cap and complete request budget. Competent raw RAG
must have equal retrieval/packing opportunities.

Prespecify source-supported historical facts and current task-state questions
where the disclosed evidence actually supports them; leave unsupported status
unknown. Separately assess response appropriateness using the applicable
professional competency criteria. Automated criterion matching and researcher
adaptations remain distinct from new independent clinical reviewer ratings.
The published materials do not supply an independently rated answer to every
new memory question.

Because only four original contexts are available, report per-context results
and paired cases rather than implying a broad clinical population effect.
Do not call paraphrases or variations independent patients. This limited
professional-source check can complement a larger synthetic development suite
and CBT-Bench's existing human-judgment calibration; it cannot replace a future
authorized real-therapist multi-turn comparison.

## Other checked candidate

DiaCBT's original paper describes LLM-generated, expert-filtered multi-session
CBT material, but an official downloadable data package and reuse licence were
not verified. Its hidden cognitive conceptualization diagram is intended for
the simulated client, not the counselor. It is not an immediately runnable
replacement for the frozen MusPsy candidate.

Primary paper: https://arxiv.org/html/2509.02999v1

Beck's annotated Abe role-play transcripts are a genuine professional
demonstration candidate, but the official permission page requires requesting
research permission. They are not an unrestricted dataset substitute:

- https://beckinstitute.org/cbt-resources/resources-for-professionals-and-students/cbtresources/
- https://beckinstitute.org/permission-to-use-beck-institute-materials/
