# MusPsy source and boundary audit — 12 November 2026

## Frozen candidate and evidence role

- Original publisher: `Rener2005/MusPsy`.
- Verified current repository commit:
  `eaae4df12808f4cbda97d8bc8b243b34cc69f613`.
- Official repository README declares MIT and contains the directory names
  `datasets/dialogues`, `datasets/memories`, `datasets/usercards`, and
  `userProfiles`. The LICENSE identifies Wang Junzhe, 2026. Preserve the actual
  pinned licence bytes and checksum when acquiring the source in hosted Colab.
- The ACL 2026 paper explicitly describes CBT-grounded **synthetic** multi-session
  dialogues generated from published case-report profiles. It reports 1,400
  training profiles and 100 testing profiles, with 6.17 sessions per client on
  average. These paper totals are not a substitute for counting the actual
  released files, checking missing IDs, or proving an official split is present.
- The released memories are generated session summaries. They are not an
  independently annotated memory-question/answer gold standard. Source profiles
  and counseling goals can contain states from later sessions. They must not be
  installed as already-known user facts at an earlier conversation cutoff.

This resource can support explicitly labelled synthetic longitudinal development
after boundary validation. Neither its dialogues nor its next therapist turn
are newly recorded professional clinicians' responses. Do not describe it as
real-patient clinical records or use its simulated changes as clinical outcomes.

## Bounded inspection and manual exposure register

The following source-case IDs were exposed during primary-source search or
bounded format inspection and must be assigned to development, including linked
or duplicate variants:

```json
[241, 308, 318, 364, 409, 626, 634, 664, 679, 687, 1073, 1076, 1110, 1300]
```

Search snippets of `datasets/memories/memories_{id}.txt` exposed IDs 241, 308,
318, 364, 634, 664, 679 and 687. Search snippets from the dialogue directory
exposed IDs 409, 626, 1073, 1076, 1110 and 1300. Only the already exposed
`datasets/dialogues/dialogues_1076.txt` was subsequently inspected as bounded
source text, lines 1–90 and 90–220, through the original publisher's file view.
No additional case contents are needed for this audit. Unlisted IDs are not
automatically certified independent held-out cases.

The inspected file's Git blob SHA is
`8715a3c10124b7b0605cf45f25a8f8aa1515501d`. This is a Git object identity,
**not** a claimed SHA256 checksum of downloaded dataset bytes. A hosted-Colab
acquisition must additionally record the exact byte length and SHA256.

## Actual boundary problem observed

The bounded example uses explicit line-leading `Consultant:` and `User:`
labels. Other exposed search snippets show a Markdown role variant,
`**Consultant:**`. Literal `#######` markers occur between blocks.

In case 1076, the block before the first marker already contains multiple
greetings and a coarse progression through several follow-ups and a concluding
review. The next block restarts one of those earlier follow-up openings and
then expands its discussion. A later marker is followed by another earlier
opening and expansion. Thus the release has overlapping narrative material in
this example. The hypothesis that the initial block is a seed or outline is an
**inference from the observed text**, not an established field in the README.

Consequences:

1. `#######` must not automatically be renamed a visit boundary. In particular,
   do not call the first block session 1 merely because it is first in the file.
2. Ingesting the initial coarse series before testing later expanded material
   could reveal future client changes and later therapist plans. Ordinary
   chronological line slicing would not prevent that leakage.
3. Multiple opening greetings or consecutive consultant turns are not by
   themselves independently verified session IDs. They can flag ambiguous
   source structure but cannot supply missing dates or appointments.
4. Do not silently repair the corpus by deleting or reordering content. Keep
   exact source bytes, report exclusions, and version any derived parsing rule.

## Safe acquisition and preparation gates

Perform real acquisition and dataset execution only in genuine hosted Colab.
An appropriate first step is a no-checkout, blob-filtered clone of the original
repository at the pinned commit, followed by `git ls-tree` metadata inspection.
Count the actual released dialogue paths and IDs before selecting cases. Archive
the exact selected source bytes, original paths, Git blob IDs and SHA256 values.
Do not use a third-party transcript mirror.

For a prospective continuation experiment:

- Build the generation view from validated original dialogue spans only.
  `datasets/memories`, `datasets/usercards`, and `userProfiles` remain outside
  generation unless a separately audited field can be shown to be available at
  the exact cutoff. The simplest initial policy is to exclude all of them.
- Preserve speaker, original order and character offsets. Recognize only
  predeclared exact role grammars; preserve body text and retain a mapping to
  the unmodified source. Unknown or ambiguous role/boundary cases are explicit
  exclusions, not guessed repairs.
- First determine whether a clean prefix can be formed without the overlapping
  whole-series material. If not, quarantine that case from longitudinal testing.
  A possible rule that excludes a proven outline is a new derived protocol,
  requiring validation; it must not be assumed from this single example.
- End a generation prefix at an actual client message. Keep its next released
  consultant reply and every later span assessor-only. A released synthetic
  reply can be a contextual reference, not a unique correct clinical answer.
- Group by source case and any connected source-report/duplicate variants
  before splitting. Filename identity is an initial grouping key, not proof of
  independent patients. If original report identity is not available, report
  the remaining grouping uncertainty.
- Keep calendar dates null unless explicitly supplied. Relative phrases in
  dialogue remain original text; a block ordinal cannot establish a week,
  elapsed interval, homework deadline, or the expiry of a refusal.

The source remains **not ready for a natural cross-session effect claim** until
these boundary and grouping checks succeed. No new response accuracy, professional
quality, or memory-effect score is produced by this audit.

## Professional CBT grounding for the memory schema

Beck Institute's official session guidance and session checklist support keeping
track of previous-session work, intervening events, present mood/functioning,
the client's selected goals, and the response to prior action plans. They also
require collaboration and attention to whether the client understood and agreed
with the proposed plan. Judith Beck's homework guidance emphasizes reviewing
earlier work and examining practical or thought-related barriers.

This motivates the following source-bound relationships, rather than a generic
unattributed summary:

| Memory relationship | Source-bound distinction to preserve |
| --- | --- |
| Earlier action plan → client agreement | Therapist proposal is separate from explicit client acceptance; absent agreement stays unknown. |
| Action plan → later reported attempt | The latest actual client report determines observed progress, without inventing completion. |
| Attempt → barrier or learning | Preserve the reported obstacle and what was learned before suggesting a repeat or escalation. |
| Previous goal → current goal | Keep historical goals distinct from the user's current priority or pause. |
| Observation/formulation → evidence | Therapist or AI hypotheses remain attributed and unconfirmed unless explicitly supported/confirmed. |

Retrieving this evidence dependency chain is an implementable hypothesis for
improving contextual accuracy and next-step fit. It still requires a fair
same-budget comparison against capable original-history retrieval. The clinical
sources justify the task; they do not establish that our algorithm has won.

### Rating-scale identity

The official **2022 CTRS-R rubric used as a source here has a 0–3 scale**. It is
not the older CTRS 0–6 scale and not the project's R1–R4 1–5 / R5 pass-fail rubric.
Do not mix their scores or call a project-specific score official CTRS-R.
Whole-session checklist items such as pacing and agenda completion also cannot
all be scored from one short reply. State applicability and report project
R1/R3 as their own measures.

## Primary sources

1. Wang et al. *Psychological Counseling Cannot Be Achieved Overnight:
   Automated Psychological Counseling Through Multi-Session Conversations*.
   Findings of ACL 2026, §§3–4 and Appendix D.
   https://aclanthology.org/2026.findings-acl.818.pdf
2. Original publisher's repository, README and licence:
   https://github.com/Rener2005/MusPsy
   https://github.com/Rener2005/MusPsy/blob/main/LICENSE
3. Bounded development-case source view:
   https://github.com/Rener2005/MusPsy/blob/eaae4df12808f4cbda97d8bc8b243b34cc69f613/datasets/dialogues/dialogues_1076.txt
4. Beck Institute. *Cognitive Therapy Rating Scale-Revised: Scoring Rubric and
   CBT Session Checklist* (2022).
   https://beckinstitute.org/wp-content/uploads/2022/09/Beck-Institute-CTRS-R-Scoring-Rubric-and-CBT-Checklist.pdf
5. Beck Institute Cares. *What Are CBT Sessions Like?*
   https://cares.beckinstitute.org/about-cbt/what-are-sessions-like/
6. Judith S. Beck / Beck Institute. *Homework* (2021).
   https://beckinstitute.org/blog/homework/
