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
        # Structured memory: dialogue demo and complete-cohort evaluation

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
        ## 3. Select one full cohort and verify its identity

        LoCoMo has 1,540 eligible questions but only 10 conversation groups, nine already exposed
        during development. Its full result is exploratory. LongMemEval-S has 500 questions; prior
        exposure must be audited before declaring a locked independent test. Shared background
        sessions connect the full LongMemEval-S release into one conservative history group;
        500 questions are not 500 independent subjects. Both datasets therefore use the explicit
        exploratory role and are reported separately. No subset or early-stopping score is offered.

        Select one dataset below. Finish its frozen three-condition grid before switching to the
        other directory. Source-preserving v2 uses a fresh root; do not relabel v1 artifacts.
    """)
    code("""
        from run_formal_pipeline import run_stage
        DATASET_KEY = 'locomo10'  # Or 'longmemeval_s_cleaned'; never a sampled subset.
        FILENAMES = {'locomo10': 'locomo10.json',
                     'longmemeval_s_cleaned': 'longmemeval_s_cleaned.json'}
        if DATASET_KEY not in FILENAMES:
            raise ValueError('Choose a frozen full benchmark.')
        WORK = Path('/content/cbt-formal-v2-' + CODE_REVISION[:12])
        RUN_ROOT = WORK / DATASET_KEY
        INPUT_PATH = DATA / FILENAMES[DATASET_KEY]
        print(json.dumps(run_stage('verify', RUN_ROOT, dataset_key=DATASET_KEY,
                                  input_path=INPUT_PATH, plan_path=PLAN_PATH), ensure_ascii=False))
    """)
    markdown("""
        ## 4. Prepare all requests once on the Colab GPU

        Default preprocessing copies older user turns into source-checked candidate records.
        It is a **verbatim-record baseline**, not semantic LLM extraction. Model-based extraction
        must be generated source-session-first using `--prepare-extraction`, then supplied as a
        frozen `--extraction-artifact` in a new run directory. Neither path invents user consent.
        All three conditions share the frozen model, short window, empty knowledge context and
        budgets. Set the switch below only when ready. Preparation uses hybrid retrieval and CUDA;
        it makes **no paid generation call**. A resumed prepared root is validated without rebuilding
        embeddings. Interrupted or inconsistent preparation stops rather than overwriting artifacts.
    """)
    code("""
        PREPARE_FULL_COHORT = False
        if PREPARE_FULL_COHORT:
            result = run_stage('prepare', RUN_ROOT, dataset_key=DATASET_KEY,
                               input_path=INPUT_PATH, plan_path=PLAN_PATH,
                               backend='hybrid', device='cuda', role='exploratory')
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print('Preparation disabled. Enable here to prepare the entire eligible cohort in Colab.')
    """)
    markdown("""
        ## 5. Bounded paid batches and explicit checkpoint downloads

        Each enabled cell attempts at most **60 calls**, including failed attempts, then stops.
        Rerun a batch deliberately to resume. Durable intent logs retain the frozen three-attempt
        lifetime cap per request. There is no background loop, automatic model replacement,
        setting change or removal of failed cases. A model identity, integrity or authentication
        error must be inspected before continuing; do not bypass it with a new directory.

        Only these opted-in cells read the existing Colab Secret named `Capstone`. Its value never
        enters notebook source or output. The temporary benchmark environment variable is restored
        after each batch. The notebook neither mounts Drive nor uploads artifacts automatically.

        Click **Download checkpoint** after a batch to save its raw run artifacts and checksums.
        Colab files can disappear when a runtime ends; a file left in `/content` is not durable.
        A partial archive is only a recovery checkpoint. An invalid run can instead be preserved
        using the separately acknowledged **invalid_unscored** backup button: it copies only the
        fixed artifact allowlist unchanged, excludes derived scores, and never declares a result.
        Keep these benchmark-derived archives in controlled project storage, not a public upload.
    """)
    code("""
        from contextlib import contextmanager
        from google.colab import userdata, files
        import ipywidgets as widgets
        from IPython.display import display
        from run_formal_pipeline import COHORT_FILES, READER_FILES, JUDGE_FILES
        from execution_lock import exclusive_directory, exclusive_directories
        import zipfile, tempfile
        from datetime import datetime, timezone

        BENCHMARK_SECRET_NAME = 'Capstone'  # Secret NAME only.

        @contextmanager
        def benchmark_secret():
            previous = os.environ.get('CBT_EVAL_API_KEY')
            secret = userdata.get(BENCHMARK_SECRET_NAME)
            if not isinstance(secret, str) or not secret.strip():
                raise RuntimeError('The configured Colab Secret is unavailable.')
            os.environ['CBT_EVAL_API_KEY'] = secret
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop('CBT_EVAL_API_KEY', None)
                else:
                    os.environ['CBT_EVAL_API_KEY'] = previous

        def backup_invalid_unscored():
            # Forensic backup only: no validation relaxation, scoring or model call.
            target = RUN_ROOT / 'invalid_unscored.zip'
            if target.is_symlink():
                raise RuntimeError('Backup output cannot be a symlink.')
            checksums, skipped = [], []
            with exclusive_directory(RUN_ROOT / '.pipeline'):
                with exclusive_directories(RUN_ROOT / 'cohort', RUN_ROOT / 'reader', RUN_ROOT / 'judge'):
                    descriptor, filename = tempfile.mkstemp(prefix='.invalid-', suffix='.zip', dir=RUN_ROOT)
                    os.close(descriptor)
                    temporary = Path(filename)
                    try:
                        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                            for directory, names in (
                                ('', ('pipeline.json', 'frozen_plan.json')),
                                ('cohort', COHORT_FILES), ('reader', READER_FILES), ('judge', JUDGE_FILES)):
                                for name in names:
                                    if name in ('reference_judge_scores.jsonl', 'reference_judge_summary.json'):
                                        continue
                                    relative = f'{directory}/{name}' if directory else name
                                    source = RUN_ROOT / relative
                                    if source.is_symlink():
                                        skipped.append(relative)
                                        continue
                                    if not source.is_file():
                                        continue
                                    digest = hashlib.sha256()
                                    with source.open('rb') as original, archive.open(relative, 'w', force_zip64=True) as copy:
                                        while block := original.read(1024 * 1024):
                                            copy.write(block)
                                            digest.update(block)
                                    checksums.append(f'{digest.hexdigest()}  {relative}\\n')
                            declaration = (json.dumps({'status': 'invalid_unscored', 'final': False,
                                'code_revision': CODE_REVISION, 'created_utc': datetime.now(timezone.utc).isoformat(),
                                'skipped_symlinks': skipped,
                                'boundary': 'Unvalidated original artifact backup only. Do not score, publish as results or resume until the recorded failure has been investigated. No keys, environment, raw benchmark download or SQLite database included.'},
                                ensure_ascii=False, indent=2) + '\\n').encode()
                            archive.writestr('INVALID_UNSCORED.json', declaration)
                            checksums.append(f'{hashlib.sha256(declaration).hexdigest()}  INVALID_UNSCORED.json\\n')
                            archive.writestr('SHA256SUMS', ''.join(checksums))
                        temporary.replace(target)
                    finally:
                        temporary.unlink(missing_ok=True)
            return target

        def checkpoint_controls():
            normal = widgets.Button(description='Download checkpoint', icon='download')
            acknowledge = widgets.Checkbox(value=False, description='I need an invalid_unscored backup')
            invalid = widgets.Button(description='Backup invalid run', icon='download')
            output = widgets.Output()
            def save_normal(_):
                normal.disabled = True
                with output:
                    output.clear_output()
                    try:
                        result = run_stage('export', RUN_ROOT)
                        print(result['status'], '| final:', result['final'])
                        files.download(result['archive'])
                    except Exception:
                        print('Validated export stopped. Preserve the runtime; inspect the failure or explicitly choose invalid_unscored backup.')
                    finally:
                        normal.disabled = False
            def save_invalid(_):
                if not acknowledge.value:
                    with output:
                        print('Acknowledge invalid_unscored backup first. It will not be an experiment result.')
                    return
                invalid.disabled = True
                with output:
                    output.clear_output()
                    try:
                        files.download(str(backup_invalid_unscored()))
                        print('invalid_unscored: original allowlisted artifacts only; no result declared.')
                    except Exception:
                        print('Backup failed. Keep the runtime and inspect its local files; no result was declared.')
                    finally:
                        invalid.disabled = False
            normal.on_click(save_normal)
            invalid.on_click(save_invalid)
            display(widgets.VBox([normal, acknowledge, invalid, output]))

        def run_paid_batch(stage, max_calls, **kwargs):
            try:
                with benchmark_secret():
                    result = run_stage(stage, RUN_ROOT, max_calls=max_calls, **kwargs)
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
            except Exception as error:
                print('Stopped:', type(error).__name__, '| checkpoints retained; no automatic retry or model change.')
                raise RuntimeError('Inspect frozen configuration and journals before another batch. Failed requests may have been billed.') from None
            finally:
                checkpoint_controls()
        print('Batch helpers ready. No Secret read, API request, Drive mount or download occurred.')
    """)
    markdown("""
        ### Reader: at most 60 calls this time

        Complete eligible cohorts mean 4,620 LoCoMo reader requests or 1,500 LongMemEval-S requests,
        plus recorded retries. The gold answers are withheld from reader prompts. Leave the switch
        off to inspect the notebook without spending calls.
    """)
    code("""
        RUN_READER_BATCH = False
        READER_MAX_CALLS = 60
        if RUN_READER_BATCH:
            reader_result = run_paid_batch('reader', READER_MAX_CALLS)
        else:
            print('Reader disabled. Set True for one explicitly capped batch, then download its checkpoint.')
    """)
    markdown("""
        ### Judge: at most 60 calls this time

        Begin only after reader completion. `FINALIZE_READER_FAILURES` is a separate explicit decision:
        it permits the existing frozen policy to retain exhausted/nonretryable technical failures
        as zero scores in the full denominator. It cannot skip unattempted or retryable requests.
        Identity/integrity failures are never converted into ordinary zero scores.

        The evaluator preserves pinned LongMemEval per-type reference prompts. DeepSeek judging is
        **adapted, model-assisted evaluation**, not the paper's official score or a human rating;
        the reader and judge share a model family. LoCoMo use is a transferred auxiliary evaluation.
    """)
    code("""
        RUN_JUDGE_BATCH = False
        JUDGE_MAX_CALLS = 60
        FINALIZE_READER_FAILURES = False
        if RUN_JUDGE_BATCH:
            judge_result = run_paid_batch('judge', JUDGE_MAX_CALLS,
                                         finalize_reader_failures=FINALIZE_READER_FAILURES)
        else:
            print('Judge disabled. Enable only for one capped grading batch after reader readiness.')
    """)
    markdown("""
        ## 6. Inspect status and export before ending the runtime

        The primary contrast is structured candidates versus the short window. Report separate
        datasets, group-equal paired confidence intervals, technical failures and per-category scores.
        If exposure, independent groups or interval precision are insufficient, report that limitation.
        Never remove failures, change grouping or add convenient cases after seeing the results.

        Status and export make no API calls. Only complete frozen grids with the registered failure
        policy can produce a model-assisted endpoint. Partial checkpoints are marked `final: false`;
        even complete results do not automatically establish clinical benefit or adequate precision.
        Download explicitly below. Restoring a checkpoint requires the same immutable code, files,
        hashes and directory layout; do not merge different versions or reset attempt journals.
    """)
    code("""
        if (RUN_ROOT / 'pipeline.json').exists():
            print(json.dumps(run_stage('status', RUN_ROOT), ensure_ascii=False, indent=2))
            checkpoint_controls()
        else:
            print('No prepared pipeline yet. Verify and prepare the selected full cohort first.')
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
