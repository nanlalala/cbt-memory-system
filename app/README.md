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
2. Click **整理本次记忆候选**. Read each candidate and its exact source quote.
   **确认这条记忆** creates a distinct, operator-triggered confirmation event;
   extraction alone never activates a memory.
3. Start the next session and ask about the earlier information. Expand
   **本次回答用到了什么** to see only the memory IDs included in the final
   response context, with budget-dropped IDs shown separately.
4. Submit a correction to an active memory. It remains a candidate until a
   separate confirmation replaces the current version.
5. Delete a memory. The service also clears the current short-term context
   and last response so a forgotten fact cannot return from the old chat
   context. Deleting one version does not claim to erase a provider's logs,
   database backups or every related memory.

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

