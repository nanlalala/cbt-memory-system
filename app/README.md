# Memory demonstrations

## Dialogue and memory controls in Colab

Open [Structured Memory Colab](Structured_Memory_Colab.ipynb) in Colab. The
notebook runs the dialogue demonstration, with candidate memories and their
source quotes beside the conversation. Model loading, vector encoding and
large evaluations belong in Colab, not on the operator's computer.

The notebook UI uses the real extraction contract, memory manager, SQLite
store, retriever and context builder. It does not start a web server or expose
a public port. The default store is ephemeral and uses fictional inputs.

1. Check **我只输入虚构演示资料**, enter a fictional fact or goal and send it.
2. Check the extraction range, then click **整理当前窗口的记忆候选**. The service
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
3. Start the next session and ask about the earlier information. Expand
   **本次回答收到的记忆参考** to read the memory text actually supplied in the final
   response context, with source IDs available for inspection and budget-dropped
   IDs listed separately. Inclusion is not proof that the generated answer
   relied on the information correctly. Restricted and dropped source bodies
   are not displayed in this trace. If unprocessed turns remain, switching
   sessions first offers a cancelable warning; they will not be saved automatically.
4. Submit a correction to an active memory. It remains a candidate until a
   separate confirmation replaces the current version.
5. Click delete to review its scope before changing anything: it removes the
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

Two modes are kept explicit:

- **Offline demonstration** copies supported source text into candidates and
  uses template responses. It makes no model or network calls and does not
  establish extraction or dialogue quality.
- **Colab model mode** uses the separately configured Colab provider. A
  send/extract click can make an API request; the page renders without one.
  Only fictional inputs are supported. Credentials are handled outside the
  UI and must never be pasted into the conversation.

The notebook imports the UI from the checkout:

```python
import sys
sys.path.insert(0, "/content/cbt-memory-system/app")
from dialogue_demo import launch_notebook_demo
demo = launch_notebook_demo()
```

`launch_notebook_demo(session=...)` also accepts a `DialogueSession` with a
Colab-configured provider. The UI imports `ipywidgets` lazily and refuses launch
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

