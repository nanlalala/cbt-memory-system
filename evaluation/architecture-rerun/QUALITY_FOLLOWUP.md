# Separate quality-format follow-up

The new `quality_span_judge_v3.py` is the reviewed format-only candidate. Only
`VERSION` and `GUIDE` differ from v2: the complete outer response example
contains all five scalar ratings and all five evidence rows. The original
rubric, validators, source segments, answering model and generated answers
are unchanged. No effectiveness or better formatting is assumed.

`cloud_quality_followup.schedule(prior_state, repo=..., expected_commit=...)`
can be scheduled in the genuine parent Colab kernel while v4 runs. It waits
for that exact controller thread and worker to finish and for its relay to
close. It never patches the active dispatcher, old prompt or old files. A
fresh credential-free subprocess prepares each v3 protocol; a new relay
reuses the existing initialized Qwen transport object. There is no new Secret
read and no provider credential enters a worker or manifest.

The only allowed output root is
`/content/capstone-quality-v3-<original invocation UUID>`. Atomic creation
allows one scheduler for that original invocation; repeated clicks or
alternative output paths cannot spend the same remaining budget twice.

After the original workflow drains, the controller reconciles its frozen
finite plan and actual parent relay intents. Every CBT Qwen admission counts;
LoCoMo's separate allowance is not borrowed. The fixed original ceiling is
5,032 plus the separately retained 24-call P7 allowance, less historical and
current actual paid intents. No new allowance is introduced. P7's entire
21-primary plus 3-repeat grid is requested only if its 24 calls fit. CACTUS's
entire available response grid and ten-percent repeats require at most 1,980
calls and start only if their actual plan fits the remaining original cap.
Otherwise that cohort is recorded as not started with the exact required and
remaining counts. A new unknown stops further admissions without retry.

Results go to new `quality_v3` directories. Original v2 and post-hoc results
remain unchanged. V3 packet/repeat identities can differ because their
version and payload hashes differ. Report v3 separately, including planned
denominators, valid-score coverage and missing outputs. Processing completion
does not mean every rating is valid or that either arm is better.

The existing post-hoc CLI loads v2 and must not be pointed at a v3 directory:
its frozen code hash will correctly reject that mix. An explicitly versioned
v3 readout adapter would need to supply the v3 API and retain the same parser
rules; this entry reports v3's original strict summaries only.

Validation is zero-provider: run
`python -m unittest discover -s tests -p test_quality_followup.py`.
