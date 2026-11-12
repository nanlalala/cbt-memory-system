# Completed CPU reader gate — 12 November 2026

This gate checks whether the answer reader can distinguish a corrected target,
a repeated cumulative count, a reported outcome and an unagreed proposal before
a full memory comparison. It uses four previously exposed invented fixtures,
each with two evidence views. All eight answers completed and passed the fixed
whole-JSON and exact typed-value checks in genuine hosted Colab.

| Reading fixture | Fixed answer | Source only | Source plus cited state |
|---|---|---|---|
| Corrected target | `5` | `{"answer":5}` | `{"answer":5}` |
| Repeated cumulative total | `3` | `{"answer":3}` | `{"answer":3}` |
| Reported outcome | `"mixed"` | `{"answer":"mixed"}` | `{"answer":"mixed"}` |
| Proposal without agreement | `null` | `{"answer":null}` | `{"answer":null}` |

The gate result is **8/8**, with **4/4 in each evidence view**. These are basic
reader checks, not a measured memory-management effect or professional CBT
quality score. The entry authorizes no automatic main run. Eight original
responses, generated token IDs, input vectors, journals and the prior actual
native-authentication receipt remain in the checkpoint.

## Actual condition and measured resources

The reader is the official `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M release,
revision `bb5d59e06d9551d752d08b292a50eb208b07ab1f`, running on hosted CPU with
two threads. Native context capacity is explicitly 4,352; the logical limits
remain 4,096 input tokens and 64 generated tokens. Output uses the same
question-only complete-JSON grammar for both evidence views, without a fixed
assistant-answer prefix or target values supplied to generation.

| Item | Observed |
|---|---:|
| Admitted / published answer generations | 8 / 8 |
| Prompt / completion / total tokens | 1,393 / 48 / 1,441 |
| Eight answer paths, combined | 252.810225 seconds |
| Native generation time, included in answer paths | 252.776505 seconds |
| Model preparation and load, separate | 12.987104 seconds |
| Freeze preparation, separate | 5.589532 seconds |
| All-eight input preflight, separate | 0.027088 seconds |
| Official shard transport and verification, separate | 196.474920 seconds; 4,683,073,632 bytes |
| Transport requests, including the range preflight | 282 |
| Isolated Python dependency preparation, separate | 2.750959 seconds |
| New paid inference calls / Secret access | 0 / false |

Setup subcomponent times are already included in the setup total. Ingestion and
indexing were not timed, and monetary cost is unmeasured. The transport receipt
and package preparation describe shared setup, rather than per-answer cost.

## Earlier outcomes retained

* The prior 1.5B V4 reading gate remains **6/8**, with its original two incorrect
  abstentions. Its answer-path time is 86.285901 seconds and its separate model
  setup is 36.175989 seconds; no old score or response was rewritten.
* The first 7B V5 entry stopped during native setup with `ValueError`, before
  any generation admission. Its failed setup attempt took **7.732603 seconds**;
  it has no answer score and no response ledger.
* A separate one-load, zero-answer diagnostic took **13.986137 seconds**. It
  observed native allocation 4,352 for a requested context of 4,160. Its record
  was retained, and V6 explicitly declares the observed native capacity.

V6 is separately versioned and does not reuse the failed V5 allowance. The
preserved records support each stage's actual completion, failure and cost.
Seven tiny restore-safety fixture tests and an independent archive byte audit
passed; these software checks generate no model-effect score.

## Frozen source and archive identities

| Artifact | SHA256 |
|---|---|
| V6 gate protocol | `b144f85a82f4a03fa72c69d8f70354a0547361c1fe8e6b0add24ff5b901a30a4` |
| V6 completed report | `322a1bd623bf8e34a7c2272af489c31f14b327a1c7775b5917024ea1f58e81bc` |
| Prior all-eight native authentication receipt | `9c3429fb79631bb7159db5656eaa9310f0568132067d787dd5722a4d4a5f1443` |
| V6 native backend | `4bf93ee4f6f549ce8ddfe393082bee459f4aa2fe9168c1d0174d7f3841beefac` |
| V6 gate runner | `72e264ae9622de38ab81b41e74a42a58b1a71701ac370fdc7af30b6743404b02` |
| CPU closure ZIP | `5356f82a0330af8c467bbc2de546b27d889858f1a7a4abc4931b8a8c66e8b9dc` |

The private archive `CBT_CPU_V6_Closure_20261112_PRIVATE.zip` is 1,374,854 bytes,
with **225 payload files plus one manifest**. Every payload size and SHA256 was
independently verified after export. Native detokenization was not repeated
by that byte audit; the original actual authentication receipt is preserved.
Large model files and the 27 MB CPU wheel are excluded with restoration
dependencies explicitly recorded. No archive entry executes on restoration.

Existing private Library identity, version 0:
`libfile_fde3_f-e6573639cb008191a5bd6ab4a8316a51_7e9d829b317c8191a098e8015cc46514`.
The successful upload was confirmed before any new save was attempted, so the
checkpoint was not duplicated.
