# CACTUS checkpoint recovery audit — 12 November 2026

Purpose: locate existing frozen CBT experiment material before considering any
new transfer or model request. This is a read-only checkpoint recovery audit,
not an experimental rerun. No CACTUS source corpus was downloaded again, no
model or real-data experiment ran in the workspace, and no Secret was accessed.

## Located persistent artifacts

| File | Library identity | Observed size/version |
|---|---|---|
| `CBT_CACTUS_V2_One_Case_Result_and_Audit_PRIVATE.zip` | `libfile_fde3_f-155b3a9cf284819199fa579b949c7bd8_7e9d829b317c8191a098e8015cc46514` | 838,903 bytes / version 0 |
| `Capstone_Experiment_Source_Index_20261109.txt` | `libfile_fde3_f-13296994f8ac8191b9550bcb8d8db6eb_7e9d829b317c8191a098e8015cc46514` | 7,478 bytes / version 0 |

The source index was read in full. The named small checkpoint was materialized
with its existing Library identity only after permission to inspect archive
metadata. The current searches located no separately named full CACTUS source
artifact; this limited search is not evidence that no other archive exists.
The checkpoint was not duplicated in Library.

The outer ZIP SHA256 is
`a40ca76f6e694aa732f49e67ae937c51ae8b52e70bc36295ce2424b94f617514`.
It contains the original nested checkpoint
`CBT_CACTUS_V2_AUTHORIZED_ONE_CASE_20261109_PRIVATE.zip`, 741,139 bytes,
SHA256 `740bbaa6691af7421374e56aa7c41c40d66791463ee99044b21f8cff575fe6e1`.
All seven outer manifest entries and all eighteen nested manifest entries
matched their recorded byte hashes. Hashing and newline counts did not parse
the generation conversations or target quotations.

## What can actually be recovered

The nested archive has nineteen entries totaling **6,042,365 uncompressed
bytes**. Its complete central directory and the decompressed code-overlay
inventory establish that the **270,309,432-byte full source JSON is absent**.
This statement concerns this exact checkpoint, not all user storage or an
unobserved runtime.

It does preserve the frozen selected-prefix inputs for the existing quotation
experiment:

| Nested path | Uncompressed bytes | Metadata count / role |
|---|---:|---|
| `run/generation/cases.jsonl` | 4,547,649 | 590 frozen generation rows |
| `run/assessor_only/targets.jsonl` | 488,522 | 590 assessor-only target rows |
| `run/inventory.jsonl` | 230,898 | 600 planned probes / 200 source families |
| `run/exposure_exclusions.json` | 34,844 | Original family exclusion/selection metadata |
| `run/protocol.json` | 13,206 | Actual v2 protocol |
| `run/private_code_overlay.json.gz` | 80,627 | Twelve frozen text files over the base commit |
| `run/call_intents.jsonl` | 83,211 | Sixteen admitted attempts |
| `run/call_outcomes.jsonl` | 97,363 | Sixteen recorded outcomes |
| `run/answers.jsonl` | 2,723 | Three logical arm rows for one completed probe |
| `run/cases/quote-9c41c28d8cb3adbadab409fe827296c2/memory.sqlite` | 356,352 | Existing one-case store, with accompanying control records |
| `run/cases/quote-9c41c28d8cb3adbadab409fe827296c2/prepared_contexts.json` | 93,322 | Existing prepared one-case evidence contexts |

Inventory counts are 590 ready, eight target-length exclusions and two
ambiguous-cue exclusions. Saved prefixes may support a separately versioned
hosted replay of these exact quotation questions without obtaining the full
raw corpus again. They are not a reconstruction of the full released source
or independent annotations of natural CBT memory accuracy. Existing database
control identities must not be relocated or reset to authorize a new send.

## Frozen identity verified

* Dataset: `LangAGI-Lab/cactus`, revision
  `237cb781cb6cbab609bc11f02a24ea6bc408ea4b`.
* Source identity recorded by the old protocol: 270,309,432 bytes; SHA256
  `be3421495f9dd76dd47d5fd4abd9fdabed9c97fcb7f7bba34ecfc78fe07d3d18`.
  This audit verifies the recorded identity, not unavailable source bytes.
* Base executable commit:
  `34aec110811a61eda50431c555b3a5a70620e8bd`.
* Actual private overlay gzip SHA256:
  `9f7568116bf3f98f2c76d19b7e20ee2ce890bc2276588c9e0b74813140f87efb`.
  Its twelve text-file hashes were checked; the six executable overlay files
  also match their entries in the protocol dependency manifest.
* All **65** executable dependency hashes matched either the overlay or exact
  base-commit Git object, without executing the archived code.
* Protocol version: `cactus-source-quotation-recall-v2`.
* Canonical protocol SHA256:
  `80c7cd60ea8099defc42e92edd97c6d750560be96e1226fa5e45141d9ce27198`.
* Physical `protocol.json` file SHA256, distinct from its canonical identity:
  `77b69ac9710170d08e93dde7724f6ea1f581d51b74980c104f09a1775f6beab9`.
* Generation file SHA256:
  `6075b00d85e00f899f031cca788acb7f910b550598d6b21515164b618183ed9b`.
* Targets file SHA256:
  `c9d56dafe43ff025bd6f1337f54a8b815c84f57cd3e7228b16e1aa8c1e51bbf0`.

The actual overlay differs materially from the base README. Its eligible
family slice is **[200:400)**, excluding all first 200 v1-ordered eligible
families regardless of whether they were answered. The selected-family-list
hash is `1db5293a091b8663fa11c64550984faafdef7ebab09b4c705ae06c69b06632c6`;
the excluded-family-list hash is
`9bf02a01152824da34eeca6a3a52ba5d42fb62c5c6aa2092840ef12bca476815`.
The v2 code's global ceiling is **4,800**, rather than the base v1's 5,000.
Neither ceiling is a new spending authorization.

The archived invocation authorizes one case and at most 100 combined extraction
and answer attempts. It completed sixteen attempts: fourteen extraction and
two canonical response attempts, with no recorded unmatched intent. The old
report records 18,908 tokens. Raw and managed shared an identical request and
answer for the one completed probe. These are historical findings, not a new
effect estimate, and unused invocation allowance is not transferred to a new
model protocol.

## Continuing without changing the experiment's meaning

The existing entry asks for an entire original Client quotation and exact
source ID after a four-word locator cue. All three distance probes use the
same latest released prefix. The actual v2 retrieval query is the shared
locator cue alone, excluding output-format instruction text. Source order is
`ordinal-only-v1`: no invented date, week, due date or calendar update is added.

The base `run_cactus_effect.prepare` verifies the full source identity before
compilation. Its existing-source path checks the local bytes rather than
downloading the source again, while separately fetching the pinned dataset
card. By contrast, `run`/`analyze` consume the frozen preparation files and do
not reopen the full source JSON. For an unchanged historical replay they
require the actual private-overlay code and original provider identity; the
current base file cannot silently stand in for that executable.

A stronger local hosted reader therefore needs a **new** frozen model/provider
protocol, finite attempt cap, actual request binding and cost journal around
the recovered inputs. The old protocol, sixteen receipts and scores stay
intact. Preserve all 600 planned inventory rows, the 590 ready questions,
family grouping, history boundaries, four-word cues and literal scoring. Do
not call count/status reading-gate success proof that the reader can copy a
quotation envelope: that task requires its own fixed small quote/source-ID
reading gate. Direct recent messages, if added, must be identical across arms
under an explicitly new context condition.

The cue is deliberately unambiguous among the eligible Client messages. A
capable original-source retriever can exploit an exact-prefix lookup; this
creates a strong ceiling and a possible shortcut by construction. It is not
appropriate to weaken that baseline or redesign the frozen questions to force
a managed-memory win. This entry measures cue-assisted literal recall and
source binding, even if both groups perform well.

Natural CBT memory accuracy and coverage require a separate source-span
annotation contract over pre-cutoff facts and necessary information. The
existing `cactus_materials.compile_rows` already separates natural continuation
inputs from pending assessor references, marks independent memory targets
unavailable, excludes future counselor answers from gold and keeps intake,
plans and patterns out of generation. Reuse that preparation boundary in a
separately versioned evaluation rather than renaming these 600 quotation
probes. Professional CBT answer review remains separate from either literal
recall or constructed state-mechanism tests.
