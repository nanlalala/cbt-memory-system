"""Build the Colab entry notebook without executing any cell or model."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]


def build(revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use the full immutable commit containing the executable code")
    cells = []

    def markdown(text: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"})

    def code(text: str) -> None:
        source = dedent(text).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source})

    markdown("""
        # Structured memory: dialogue demo and benchmark preparation

        Run in **hosted Google Colab only**. Select **T4 GPU → runtime version 2026.07**.
        Nothing in this notebook should be executed on a personal computer or a local runtime.

        The interface demonstrates fictional dialogue, candidate extraction, explicit confirmation,
        correction and deletion. The benchmark has no user-consent labels: its separately named
        **unconfirmed-candidate experiment** never invents confirmation. Software demo output is
        not an effectiveness result. Paid model calls are disabled by default.
    """)
    code("""
        import os, sys, json, subprocess, importlib.util, importlib.metadata as metadata
        from pathlib import Path
        if not (Path('/content').is_dir() and os.environ.get('COLAB_RELEASE_TAG')
                and importlib.util.find_spec('google.colab') is not None):
            raise RuntimeError('Hosted Google Colab required. Do not run on your personal computer.')
        import google.colab
        print('Execution location: hosted Google Colab')
        print('Python:', sys.version.split()[0])
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True)
        print('GPU:', gpu.stdout.strip() or 'none; select T4 for neural retrieval')
        print({name: metadata.version(name) for name in
               ('torch', 'sentence-transformers', 'transformers', 'huggingface-hub', 'ipywidgets')})
    """)
    code(f"""
        CODE_REVISION = '{revision}'
        REPO = Path('/content/cbt-structured-' + CODE_REVISION[:12])
        if not (REPO / '.git').exists():
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout',
                            'https://github.com/nanlalala/cbt-memory-system.git', str(REPO)], check=True)
        subprocess.run(['git', '-C', str(REPO), 'checkout', '--quiet', '--detach', CODE_REVISION], check=True)
        assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
        for relative in ('app', 'modules/agent-integration', 'evaluation/end-to-end-dialogue'):
            sys.path.insert(0, str(REPO / relative))
        os.chdir(REPO)
        print('Immutable code:', CODE_REVISION)
        print('No package upgrades, local runtime connections or public web-server ports are used.')
    """)
    markdown("""
        ## 1. Try the memory interface

        Acknowledge fictional data in the interface, send a fictional goal, and extract candidates.
        Confirm one candidate, start a new session and ask about it. Try correcting and confirming
        its replacement, then deleting it. The trace shows only memories actually included.
        The default responder displays selected context deterministically; it is **not an LLM**.
    """)
    code("""
        from IPython.display import clear_output
        from dialogue_demo import launch_notebook_demo
        if 'demo' in globals():
            demo.close()
        clear_output(wait=True)
        demo = launch_notebook_demo()
    """)
    markdown("""
        ### Optional: a capped model-backed fictional demo

        Set the name of an existing Colab Secret, then explicitly enable this cell. Never put the
        API key in notebook source, output, chat or GitHub. It sends fictional demonstration turns
        to the frozen TokenDance endpoint. The adapter caps attempts and requested output tokens;
        failed requests also consume the cap. It does not silently retry or change the model.
    """)
    code("""
        ENABLE_PAID_DEMO = False
        COLAB_SECRET_NAME = 'Capstone'  # Existing Secret NAME, never its value
        if ENABLE_PAID_DEMO:
            from colab_provider import ColabChatProvider
            from dialogue_service import DialogueSession
            provider = ColabChatProvider.from_colab_secret(
                secret_name=COLAB_SECRET_NAME, max_calls=12,
                model='deepseek-v4.1-flash',
                accepted_provider_models=('deepseek-v4-1-flash-260910',),
                max_response_tokens=4096, thinking_mode='disabled')
            live_session = DialogueSession(provider=provider)
            live_demo = launch_notebook_demo(session=live_session)
        else:
            print('Paid demo disabled. No Secret read and no model request.')
    """)
    markdown("""
        ## 2. Download the fixed external benchmarks in Colab

        LongMemEval-S is approximately 277 MB and includes distractor history. The small Oracle
        file is not a substitute. Dataset bytes and SHA-256 must match the frozen plan. Files remain
        in the Colab runtime; no full benchmark inputs or arbitrary raw outputs are committed.
    """)
    code("""
        import hashlib
        from urllib.request import urlopen
        from prepare_formal_evaluation import load_plan
        PLAN_PATH = REPO / 'evaluation/end-to-end-dialogue/formal_evaluation_plan.json'
        plan = load_plan(PLAN_PATH)
        DATA = Path('/content/cbt-formal-data'); DATA.mkdir(exist_ok=True)
        for dataset_key, spec in plan['datasets'].items():
            filename = 'locomo10.json' if dataset_key == 'locomo10' else 'longmemeval_s_cleaned.json'
            path = DATA / filename
            if not path.exists():
                temporary = path.with_suffix('.download')
                with urlopen(spec['source'], timeout=180) as source, temporary.open('wb') as target:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                temporary.replace(path)
            with path.open('rb') as handle:
                actual_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual_sha != spec['sha256'] or path.stat().st_size != spec['bytes']:
                raise RuntimeError('Dataset identity mismatch; stop before preparation.')
            print(dataset_key, path.stat().st_size, actual_sha)
    """)
    markdown("""
        ## 3. Freeze the full cohorts before generation

        LoCoMo has 1,540 eligible questions but only 10 conversation groups, nine already exposed
        during development. Its full result is exploratory. LongMemEval-S has 500 questions; prior
        exposure must be audited before declaring a locked independent test. Both below therefore
        use the explicit exploratory role. They are reported separately. No score is computed here.
    """)
    code("""
        from prepare_formal_evaluation import prepare_dataset
        WORK = Path('/content/cbt-structured-evaluation-20261021'); WORK.mkdir(exist_ok=True)
        COHORTS = {}
        for dataset_key, filename in (('locomo10', 'locomo10.json'),
                                      ('longmemeval_s_cleaned', 'longmemeval_s_cleaned.json')):
            directory = WORK / ('cohort-' + dataset_key)
            manifest = prepare_dataset(DATA / filename, dataset_key, PLAN_PATH, directory, role='exploratory')
            COHORTS[dataset_key] = directory
            print(json.dumps({key: manifest.get(key) for key in
                  ('dataset_key', 'role', 'counts',
                   'confirmatory_preparation_blockers', 'confirmatory_claim_allowed')}, ensure_ascii=False))
        print('Full cohorts prepared. Preparation is not an effectiveness result.')
    """)
    markdown("""
        ## 4. Optional: prepare structured-candidate response requests

        Default preprocessing copies older user turns into source-checked candidate records.
        It is a **verbatim-record baseline**, not semantic LLM extraction. Model-based extraction
        must be generated source-session-first using `--prepare-extraction`, then supplied as a
        frozen `--extraction-artifact` in a new run directory. Neither path invents user consent.
        All three conditions share the frozen model, short window, knowledge condition and budgets.
    """)
    code("""
        PREPARE_RESPONSE_REQUESTS = False
        if PREPARE_RESPONSE_REQUESTS:
            from run_three_arm import Settings
            from run_structured_memory import prepare_structured_run
            settings = Settings(model='deepseek-v4.1-flash',
                accepted_provider_models=('deepseek-v4-1-flash-260910',),
                backend='hybrid', max_response_tokens=4096, thinking_mode='disabled')
            for dataset_key, cohort_dir in COHORTS.items():
                summary = prepare_structured_run(cohort_dir, WORK / ('run-' + dataset_key), settings,
                    'https://tokendance.space/gateway/v1', device='cuda')
                print(json.dumps(summary, ensure_ascii=False))
        else:
            print('Response preparation disabled; set True here in Colab when ready.')
    """)
    markdown("""
        ## 5. Generation and reference-based evaluation

        Use the commands on the experiment README after reviewing the frozen cohort, extraction
        representation and requested-call totals. `--generate --max-calls N` is explicit opt-in;
        checkpoints preserve the cohort and fixed lifetime retry limits. References are scoring-only.
        `reference_judge.py` preserves official LongMemEval per-type prompts and its pinned source;
        using DeepSeek as judge is an **adapted, model-assisted evaluator**, not the paper's official
        score or a human rating. Lexical overlap remains diagnostic only.

        The primary contrast is structured candidates versus the short window. Report separate
        datasets, group-equal paired confidence intervals, technical failures and per-category scores.
        If exposure, independent groups or interval precision are insufficient, report that limitation.
        Never remove failures, change grouping or add convenient cases after seeing the results.
    """)
    return {
        "nbformat": 4, "nbformat_minor": 5, "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": "Structured_Memory_Colab.ipynb", "provenance": []},
            "accelerator": "GPU", "code_revision": revision,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-revision', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'app/Structured_Memory_Colab.ipynb')
    args = parser.parse_args()
    notebook = build(args.code_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(notebook["cells"])} cells; no cell executed.')


if __name__ == '__main__':
    main()
