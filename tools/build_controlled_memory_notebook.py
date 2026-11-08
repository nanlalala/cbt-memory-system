"""Build the separate controlled-memory Colab entry; never execute its cells."""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
UNPINNED = "PIN_AFTER_IMPLEMENTATION_COMMIT"


def build(revision: str = UNPINNED) -> dict:
    if revision != UNPINNED and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be the complete executable SHA or explicit unpinned placeholder")
    cells = []
    def markdown(text):
        cells.append({"cell_type": "markdown", "id": f"controlled-v2-{len(cells):02d}",
                      "metadata": {}, "source": dedent(text).strip() + "\n"})
    def code(text):
        source = dedent(text).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "id": f"controlled-v2-{len(cells):02d}",
                      "metadata": {}, "execution_count": None, "outputs": [], "source": source})

    markdown("""
        # Controlled CBT memory prototype v2

        **Run only in an actual hosted Google Colab runtime. Never connect a local runtime or
        execute this notebook on the user's computer.** This separate entry preserves the old
        notebooks and frozen experiment identities. It is an engineering prototype for fictional
        input, not evidence of improved CBT dialogue, reduced burden, or clinical effectiveness.

        Model calls start disabled. Opening the notebook does not read a Secret, download a
        dataset, start a model, or create a database. An executable SHA and explicit approved
        cloud storage configuration are required before opening the session.
    """)
    code("""
        import os, sys, json, re, subprocess, importlib.util
        from pathlib import Path
        if not (Path('/content').is_dir() and os.environ.get('COLAB_RELEASE_TAG')
                and importlib.util.find_spec('google.colab') is not None):
            raise RuntimeError('Actual hosted Colab required; do not use a local runtime or personal computer.')
        import google.colab
        print('Execution location: hosted Google Colab')
    """)
    code(f"""
        CODE_REVISION = '{revision}'
        if not re.fullmatch(r'[0-9a-f]{{40}}', CODE_REVISION):
            raise RuntimeError('This entry is not released yet: replace the explicit placeholder with the reviewed executable SHA.')
        REPO = Path('/content/cbt-controlled-v2-' + CODE_REVISION[:12])
        if not (REPO / '.git').exists():
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout',
                            'https://github.com/nanlalala/cbt-memory-system.git', str(REPO)], check=True)
        subprocess.run(['git', '-C', str(REPO), 'checkout', '--quiet', '--detach', CODE_REVISION], check=True)
        assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
        for relative in ('app', 'modules/agent-integration', 'modules/long-term-memory/schema',
                         'modules/long-term-memory/storage', 'modules/long-term-memory/lifecycle',
                         'modules/long-term-memory/extraction', 'modules/long-term-memory/retrieval'):
            sys.path.insert(0, str(REPO / relative))
        os.chdir(REPO)
        print('Executable identity:', CODE_REVISION)
        print('No datasets or embedding models are loaded by this entry.')
    """)
    markdown("""
        ## Explicit storage and frozen limits

        Fill all four identifiers below with an approved **cloud** location and stable research
        IDs. No in-memory database or temporary directory is silently substituted. The approved
        root must retain the SQLite file, its sidecar, and the newest `.cbt-memory-control` ledger.
        A Colab VM's process lifetime is not cross-runtime persistence; `/content` itself is
        ephemeral. This notebook does not mount Drive or recover old backups automatically.
        Restoring a database without the latest control ledger must be refused.

        The run freezes one worker, queue/batch/call/output limits, code hashes and policies.
        Use a new explicit RUN_ID after a protocol or provider change. Do not reset a run ID or
        enlarge a budget to bypass an exhausted/unknown attempt. Secret **name** `Capstone`
        refers to the already configured Colab Secret; never place or print its value here.
    """)
    code("""
        APPROVED_STORAGE_ROOT = ''  # Explicit approved absolute CLOUD directory
        DATABASE_FILE = ''          # Explicit absolute .sqlite file beneath that root
        USER_ID = ''                # Stable fictional/research user identity
        RUN_ID = ''                 # Explicit new protocol/run identity

        ENABLE_PAID_MODEL = False
        COLAB_SECRET_NAME = 'Capstone'  # Name only, never a key value
        MAX_RUN_CALLS = 12
        MAX_SESSION_CALLS = 12
        MAX_QUEUE_LENGTH = 32
        MAX_BATCH_SIZE = 1
        MAX_WORKER_JOBS_PER_TURN = 1
        MAX_OUTBOUND_CHARS = 18000  # Complete escaped request JSON, not a token budget
        MAX_RESPONSE_TOKENS = 4096
        MAX_REQUESTED_OUTPUT_TOKENS = MAX_RUN_CALLS * MAX_RESPONSE_TOKENS
        SOURCE_PERMISSIONS = {
            'save': True, 'long_term_use': True,
            'response_external': False, 'extraction_external': False,
        }
        # Saving, long-term use, answer transmission and organization transmission
        # are separate. The optional UI controls apply explicitly to each input.
    """)
    code("""
        from colab_provider import require_colab, ColabChatProvider
        from controlled_dialogue import ControlledDialogueSession, PrototypeLimits
        from durable_memory import DurableLimits
        from dialogue_demo import launch_notebook_demo
        require_colab()
        if not all(isinstance(value, str) and value.strip() for value in
                   (APPROVED_STORAGE_ROOT, DATABASE_FILE, USER_ID, RUN_ID)):
            raise ValueError('Set the explicit approved cloud directory, file, USER_ID and RUN_ID first.')
        approved = Path(APPROVED_STORAGE_ROOT)
        database = Path(DATABASE_FILE)
        if not approved.is_absolute() or not database.is_absolute() or not database.is_relative_to(approved):
            raise ValueError('Use an explicit absolute cloud database below the approved root.')
        approved.mkdir(parents=True, exist_ok=True)
        if 'controlled_demo' in globals():
            controlled_demo.close()
        if 'controlled_session' in globals():
            controlled_session.close()
        provider = None
        if ENABLE_PAID_MODEL:
            provider = ColabChatProvider.from_colab_secret(
                secret_name=COLAB_SECRET_NAME, model='deepseek-v4.1-flash',
                accepted_provider_models=('deepseek-v4-1-flash-260910',),
                max_calls=MAX_RUN_CALLS, max_prompt_chars=MAX_OUTBOUND_CHARS,
                max_response_tokens=MAX_RESPONSE_TOKENS,
                max_requested_output_tokens=MAX_REQUESTED_OUTPUT_TOKENS,
                thinking_mode='disabled', extraction_schema_version='2.0')
        controlled_session = ControlledDialogueSession.open(
            database, approved_root=approved, user_id=USER_ID, run_id=RUN_ID,
            provider=provider, permissions=SOURCE_PERMISSIONS, timezone_name='Asia/Shanghai',
            limits=PrototypeLimits(max_outbound_chars=MAX_OUTBOUND_CHARS,
                                   max_response_tokens=MAX_RESPONSE_TOKENS,
                                   max_worker_jobs_per_turn=MAX_WORKER_JOBS_PER_TURN),
            durable_limits=DurableLimits(max_run_calls=MAX_RUN_CALLS,
                                         max_session_calls=MAX_SESSION_CALLS,
                                         max_queue_length=MAX_QUEUE_LENGTH,
                                         max_batch_size=MAX_BATCH_SIZE),
            background=True)
        controlled_demo = launch_notebook_demo(session=controlled_session)
        print('Model backend enabled:', ENABLE_PAID_MODEL)
        print('Protocol hash:', controlled_session.protocol['protocol_hash'])
        # No model request occurs until an explicitly permitted UI action.
    """)
    markdown("""
        ## Read the actual receipts

        After a fictional input, read **source commit**, **answer**, and **organization** separately.
        A completed answer does not imply a saved source or completed extraction. A queue limit
        retains a permitted source as unprocessed rather than silently growing the queue. Missing
        token usage stays unknown. Each send may cause an answer plus a bounded background
        organization request when those transmissions were explicitly allowed; both count toward
        the same frozen run/session caps. No request is retried automatically.

        Try acceptance, a cumulative partial report, and a pause. Acceptance, progress,
        participation, outcome and due-review remain separate. “This message must not be saved”
        may affect the current reply but is not copied into later context, stored task state,
        extraction background, summary or last-reply text. The UI displays that turn only in a
        temporary widget cleared on the next operation. Do not export that private widget output.

        Local withdrawal remains available during a safety pause. Representation deletion is
        not forgetting content; source/span/object/stop-use scopes are distinct. A deletion
        receipt reports logical blocking and controlled primary cleanup separately from any
        external copies. The seven old knowledge sources remain disabled; an empty K0 channel
        must never be described as professional textbook support.
    """)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "controlled_memory_v2": {"executable_revision": revision,
                                                  "paid_calls_default": False,
                                                  "requires_explicit_storage": True,
                                                  "execution": "hosted_colab_only"}},
            "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default=UNPINNED)
    parser.add_argument('--output', type=Path, default=ROOT / 'app/Controlled_Memory_V2_Colab.ipynb')
    args = parser.parse_args()
    args.output.write_text(json.dumps(build(args.revision), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
