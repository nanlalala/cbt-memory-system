# Memory demonstrations

## Default CBT dialogue: context v4

The default `app/memory_demo.py` / `app/dialogue_demo.py` entry opens
`ControlledDialogueSession` (`controlled-memory-v4.0`) with
`processed_with_evidence` memory. Its [context-v4 renderer](../docs/context-assembly-v4/README.md)
supplies supported processed memory and evidence to the actual response request.
The [message-path specification](../docs/context-assembly-v4/IMPLEMENTATION_PATHS.md)
distinguishes this product contract from benchmark reference-message adapters.

## Controlled launcher and recorded replay

Open [Controlled Memory Colab](Controlled_Memory_V2_Colab.ipynb) in an actual
hosted Google Colab runtime. This recorded notebook keeps its explicit
`CODE_REVISION` (`2d48a7917e17aa8e02fa13f419ebe35961e11d0f`); it does not
automatically select current main. Model calls start disabled. Configure the approved
absolute cloud directory, SQLite file, stable user ID and new run ID explicitly;
there is no in-memory fallback. Four per-input purposes distinguish saving,
long-term use, answer transmission and organization transmission. Enabling a
provider does not itself grant either transmission permission.

The new UI separately displays source-commit, answer and organization receipts,
plus acceptance, progress, participation, outcome and due-review. A no-save
message is displayed only for the current operation and cleared from the private
widget at the next operation; it is not copied into session state or extraction
background. Normal processing stays paused across connections after the bounded
demonstration risk rule triggers; explicit local withdrawal remains available.
This rule is not a production clinical risk assessment.

For a finite engineering replay in your own approved cloud environment, create
an approved cloud directory, then provide a fresh file and explicit run ID:

```bash
python app/controlled_memory_demo.py --scripted \
  --database /YOUR/APPROVED/CLOUD/ROOT/fresh.sqlite \
  --approved-root /YOUR/APPROVED/CLOUD/ROOT \
  --run-id YOUR_NEW_RUN_ID
```

This uses fictional fixtures, the real store/manager/projector/retrieval path,
and explicitly marked local response mocks. Real API calls are zero. An old run
is refused rather than overwritten. Read the resulting assertions and frozen
protocol identities as software verification only. Approved storage must retain
the newest control anchor independently of old DB backups; the runtime's process
lifetime alone does not provide cross-runtime persistence.

The [recorded v2.1 replay](https://github.com/nanlalala/cbt-memory-system/blob/737c93236cb2c23b20ee44eaa4dd3aa4840ab532/app/controlled_memory_demo.py)
contains five fictional scenarios, 22 assertions and six local
mock calls. It includes historical-source-only retrieval, zero-hit/top-k controls
and exact final outbound body/offset checks. The earlier four-scenario records
are preserved at their original executable identities. Review repairs and their
regressions are listed in the [remediation record](../docs/review-c757-remediation-20261109.md).

The [control contract](../docs/cbt-memory-implementation-contract-v2.md) documents
the underlying lifecycle and recovery rules. The current default response path
is described by the context-v4 specification above; replay assertions are
software evidence rather than model-response or clinical-outcome scores.

## Retained legacy dialogue and memory controls


Open [Structured Memory Colab](Structured_Memory_Colab.ipynb) in Colab. The
notebook runs the dialogue demonstration with automatic historical observations
and an optional memory panel. Model loading, vector encoding and
large evaluations belong in Colab, not on the operator's computer.

The notebook UI uses the real extraction contract, memory manager, SQLite
store, retriever and context builder. It does not start a web server or expose
a public port. The default store is ephemeral and uses fictional inputs.

1. Check **我只输入虚构演示资料** and chat normally about a fictional goal,
   assignment or obstacle. **自动记录对话中的历史信息** is on by default. Suitable
   user statements are saved as exact, source-bound historical observations after
   a successful reply. No extraction or confirmation click is needed for each turn.
   This does not add an automatic model-extraction request.
2. Start another session and revisit the task. The system distinguishes “you said
   this earlier” from “this is still true.” Relevant uncertainty or changing goals
   may prompt one natural question, such as asking whether thought records still
   feel burdensome. Silence and an unrelated “yes” do not confirm a memory. Model
   confidence alone cannot activate a claim or override deletion/access filters.
3. Open **本次回答收到的记忆参考** if you want to inspect what was supplied to the
   response and what was asked only to check the current situation. The trace
   does not prove that an answer used the evidence correctly. Historical quotes
   and AI inferences are labeled separately, without presenting a confidence
   number as a calibrated probability of truth.
4. Turn automatic recording off at any time. This stops new automatic records;
   existing records remain available for inspection or deletion under
   **查看、修改或删除记忆 · 可选**. Deletion first shows its scope, then asks whether
   to remove the selected record and reset the current chat/last answer. Other
   long-term records remain. Cancel leaves the current state unchanged.

Automatic recording currently preserves selected original statements. It is not
complete semantic extraction of CBT goals, homework outcomes or clinical patterns.
Records remain candidates unless a distinct confirmation occurs; permitted
historical use is a separate policy and is not disguised as confirmation.

### Optional explicit review

The management panel retains the stricter tools for inspecting extracted summaries
and deliberately confirming current information. A separate
`DialogueSession(memory_mode="review")` keeps the old manual-review behavior for
controlled tests. In that workflow:

1. Check the extraction range, then click **整理当前窗口的记忆候选**. The service
   retains at most 24 conversation messages by default; the page shows how many
   user turns remain, how many have moved out of the window and how many have
   not yet been processed. This button does not cover an entire long session.
   Only pending turns can supply new candidates; already processed turns remain
   available as context but are not re-imported, so revising a candidate does not
   make the next extraction recreate its old wording.
   Read each candidate and its exact source quote. If it is inaccurate, edit it
   and click **保存修订，保持待确认** before deciding whether to confirm. Revision
   never accepts the old wording, adds it to ordinary chat or activates a memory.
   The original extraction quote remains available beside the revision evidence
   only after the service verifies the matching operator revision event; model
   payload fields cannot create a displayed revision history. Goal and assignment
   slots keep their type and slot when revised, with status reset to `unknown`.
   Emotion, cognitive-pattern, assignment-outcome and safety slots need structured
   field editing, which this text-only demo does not provide. Their cards explain
   this limit and require removing the candidate and supplying explicit information
   again rather than offering an edit that will silently change its meaning.
   **确认这条记忆** creates a distinct, operator-triggered confirmation event.
2. Start the next session and ask about the earlier information. Expand
   **本次回答收到的记忆参考** to read the memory text actually supplied in the final
   response context, with source IDs available for inspection and budget-dropped
   IDs listed separately. Inclusion is not proof that the generated answer
   relied on the information correctly. Restricted and dropped source bodies
   are not displayed in this trace. If unprocessed turns remain, switching
   sessions first offers a cancelable warning; they will not be saved automatically.
3. Submit a correction to an active memory. It remains a candidate until a
   separate confirmation replaces the current version.
4. Click delete to review its scope before changing anything: it removes the
   selected memory and resets the current short-term context and last response.
   Other long-term records remain. Cancel to keep the current state, or confirm
   the deletion explicitly. Clearing the chat prevents the deleted text returning
   through that old context; deleting one version does not erase every related
   memory, a provider's logs or database backups.

Correction text is management evidence attached to the correction record; it is
not added to ordinary chat or re-extracted as a standard fact. Failed sends and
failed correction writes leave the successful conversation untouched. Execution
status separately reports requests attempted and valid responses, so a timeout or
rejected empty answer does not appear as “not called.” A failed external request
may still be billed; errors shown in the interface omit provider response bodies.

The response backend is separate from the memory-interaction mode:

- **Offline demonstration** copies supported source text into candidates and
  uses template responses. It makes no model or network calls and does not
  establish extraction or dialogue quality.
- **Colab model mode** uses the separately configured Colab provider. A
  send or optional manual-extract click can make an API request; automatic quote
  recording adds no extraction call and the page renders without a model call.
  Only fictional inputs are supported. Credentials are handled outside the
  UI and must never be pasted into the conversation.

The notebook imports the UI from the checkout:

```python
import sys
sys.path.insert(0, "/content/cbt-memory-system/app")
from dialogue_demo import launch_notebook_demo
# New default: explicit approved cloud storage, configured in previous cells.
demo = launch_notebook_demo(database=DATABASE_FILE, approved_root=APPROVED_STORAGE_ROOT,
                            user_id=USER_ID, run_id=RUN_ID)
```

`launch_notebook_demo(session=...)` also accepts a `DialogueSession` with a
Colab-configured provider. Use
`DialogueSession(provider=provider, memory_mode="conversational")` for the low-friction
interface; the lower-level class still defaults to `review` for compatibility.
The notebook pins the executable code containing these options. The UI imports `ipywidgets` lazily and refuses launch
outside Colab. Importing its rendering helpers does not load models or widgets.
It is a software demonstration, not a benchmark or clinical result; the small
research safety gate is not a production crisis-detection/referral service.

## Complete-cohort evaluation in the same notebook

After the demo, the notebook offers a separate `verify → prepare → reader → judge
→ status/export` workflow. Select one complete dataset: 1,540 eligible LoCoMo
questions or all 500 LongMemEval-S questions. Preparation uses hybrid retrieval
on the Colab GPU. Reader and judge cells each require explicit opt-in and attempt
at most 60 calls per execution; rerunning resumes the frozen intent journals.
No automatic loop, model change, case removal or Drive mount is used.

Download a checkpoint after each batch and before ending the runtime. The normal
export contains raw scoring inputs/outputs, code snapshots and SHA-256 checksums;
partial runs remain `final: false`. If validation stops an export, a separately
acknowledged `invalid_unscored` backup preserves only allowlisted original
artifacts, without derived scores or any result claim. Neither export includes
keys, environment variables, SQLite files or the full original dataset downloads.
Keep benchmark-derived archives under controlled access.

These are historical-QA comparisons with unconfirmed source records, not evidence
of improved CBT care. The [next validation plan](../docs/validation-next.md)
separately covers CBT continuity and semantic memory, including source and
annotation gates.

## Fixed-fixture CLI

This is a small CLI for showing the memory lifecycle with fictional data. It is
not a clinical chatbot, a public service, or an outcome experiment. It makes no
model/API calls and does not implement a crisis detection or referral router.

Run from the repository root:

```bash
python app/memory_demo.py
```

The menu walks through these steps:

1. Parse a fixed extraction JSON fixture into an **unconfirmed candidate**.
2. Display the candidate and its exact fictional source quote. The operator must
   type `CONFIRM <memory-id>` to activate it. That input creates a consent event,
   and the event/confirmation-turn IDs are linked in the audit trail.
3. Retrieve the confirmed goal and show the actual context and included IDs.
   No Agent response is generated.
4. Stage a correction pointing to the current active version. The old version
   remains current until the operator separately confirms the correction.
5. Type `DELETE ALL <run-id>` to hard-delete **all versions from this run**,
   including superseded versions. Show that retrieval then returns no memory.

The supplied fixture changes a weekly walking goal from two to three occasions.
It is software test input, not a therapeutic recommendation or study example.
Only the fixed fictional fixtures are accepted; the CLI does not collect journals
or clinical conversations.

## Repeatable software smoke

```bash
python app/memory_demo.py --scripted
python -m unittest discover -s tests -p test_demo.py -v
```

`--scripted` prints JSON labelled `synthetic_software_smoke`. Its confirmation
events explicitly say `scripted_fixture`, **not real user consent**. A passing
run proves the demo's state transitions and module wiring only. It does not
measure extraction quality, retrieval benchmark accuracy or dialogue benefit.

## Storage and privacy boundary

The default database is ephemeral SQLite `:memory:` and disappears on exit.
An explicit optional path enables a persistent **plaintext** database:

```bash
python app/memory_demo.py --database /path/to/synthetic-demo.sqlite3
```

Use fictional data only. This reference CLI has no authentication, encryption,
or production retention controls. Hard deletion removes memory rows; it leaves
content-free audit metadata and does not guarantee forensic erasure of database
files, SQLite journals or backups. Each run uses a separate synthetic user ID.

A user-facing application with genuine identity/consent binding, safe clinical
workflow and production privacy controls remains future work.

