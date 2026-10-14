# Local memory demo

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

