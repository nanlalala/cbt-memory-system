"""Build the hosted-Colab objective-memory entry without executing cells."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation/objective-memory/Objective_Memory_Comparison_Colab.ipynb"


def build(revision: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision == "0" * 40:
        raise ValueError("Use the full immutable commit containing the objective experiment code")
    cells: list[dict] = []

    def markdown(text: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"})

    def code(text: str) -> None:
        source = dedent(text).strip() + "\n"
        ast.parse(source)
        cells.append({"cell_type": "code", "metadata": {}, "source": source,
                      "execution_count": None, "outputs": []})

    markdown("""
        # Objective memory comparison

        Run this version in a **fresh hosted Google Colab runtime**. Do not connect a local
        runtime or run full datasets/models on a personal computer. The lexical retrieval
        comparison does not require a GPU or local model download; response generation uses
        the existing remote provider only after an explicitly enabled, bounded batch.

        The three conditions share source history, response settings and context budgets:
        short-term history, raw-history RAG and managed quotation memory. The managed arm
        uses source-preserving candidate records, **not automatic semantic goal extraction**.
        Exact duplicate prompts can share one provider response; derived arm rows are not
        additional independent calls. Equal answers caused by equal inputs do not prove a
        general memory benefit. Public-answer scoring, mechanism checks and user experience
        are different findings.

        This is a new experiment. It neither resumes nor overwrites the older 1,500-response
        LongMemEval checkpoint. No paid call, Secret read or full-run loop is enabled by default.
    """)
    code("""
        import os, sys, json, hashlib, subprocess, importlib.util
        import importlib.metadata as metadata
        from pathlib import Path
        if not (Path('/content').is_dir() and os.environ.get('COLAB_RELEASE_TAG')
                and importlib.util.find_spec('google.colab') is not None):
            raise RuntimeError('Hosted Google Colab required; do not use a personal-computer runtime.')
        import google.colab
        def package_version(name):
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                return None
        RUNTIME_IDENTITY = {'location': 'hosted Google Colab',
                            'python': sys.version.split()[0],
                            'packages': {name: package_version(name) for name in
                                         ('pyarrow', 'numpy', 'ipywidgets')}}
        print(json.dumps(RUNTIME_IDENTITY, indent=2))
        print('No model weights or package upgrades are requested by this notebook.')
    """)
    code(f"""
        CODE_REVISION = '{revision}'
        REPO = Path('/content/cbt-objective-code-' + CODE_REVISION[:12])
        if not (REPO / '.git').exists():
            subprocess.run(['git', 'clone', '--quiet', '--no-checkout',
                            'https://github.com/nanlalala/cbt-memory-system.git', str(REPO)], check=True)
        subprocess.run(['git', '-C', str(REPO), 'checkout', '--quiet', '--detach', CODE_REVISION], check=True)
        assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == CODE_REVISION
        if subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD']).returncode:
            raise RuntimeError('Pinned checkout has modifications; use a clean hosted runtime.')
        for name in ('objective_datasets', 'run_objective_comparison', 'score_objective_comparison', 'objective_audit'):
            loaded = sys.modules.get(name)
            origin = getattr(loaded, '__file__', '') if loaded is not None else ''
            if origin and not Path(origin).resolve().is_relative_to(REPO.resolve()):
                raise RuntimeError('An older objective module is already loaded; use a fresh hosted runtime.')
        for relative in ('evaluation/objective-memory', 'evaluation/end-to-end-dialogue'):
            sys.path.insert(0, str(REPO / relative))
        os.chdir(REPO)
        WORK = Path('/content/cbt-objective-runs-' + CODE_REVISION[:12])
        DATA = WORK / 'registered-data'
        WORK.mkdir(exist_ok=True)
        print('Immutable code:', CODE_REVISION)
        print('Separate experiment root:', WORK)
    """)
    markdown("""
        ## 1. Download and inventory one admitted release

        Downloads verify the registered immutable source identity. PersonaMem files that have
        a published Git-blob identity are additionally hashed to local SHA-256; Git SHA-1 is
        not mislabeled as a byte SHA-256. The source registry and preparation protocol retain
        the distinction. Do not replace a failed source with an unverified mirror.

        Select one dataset key below. MemoryAgentBench uses the released conflict-resolution
        file; PersonaMem-v1 uses the 32k release and its exclusive context-end boundary. The
        registry must admit the chosen source before it can be downloaded. Full data stays in
        Colab. The inventory prints counts and identities, not conversations or gold answers.
    """)
    code("""
        from objective_datasets import download_registered_dataset, load_dataset_cases, inventory_cases
        DATASET_KEY = 'memoryagentbench_conflict'
        # Or 'personamem_v1', the registered fixed 32k release.
        if DATASET_KEY not in {'memoryagentbench_conflict', 'personamem_v1'}:
            raise ValueError('Select one of the admitted frozen dataset families.')
        paths = download_registered_dataset(DATASET_KEY, DATA)
        INPUT_PATH = paths.get('data') or paths.get('questions')
        CONTEXT_PATH = paths.get('contexts')
        if INPUT_PATH is None:
            raise RuntimeError('Registered download did not provide the required input file.')
        cases, scoring_references = load_dataset_cases(DATASET_KEY, INPUT_PATH, context_path=CONTEXT_PATH)
        inventory = inventory_cases(cases, scoring_references)
        inventory['input_files'] = {name: {'name': Path(path).name, 'bytes': Path(path).stat().st_size,
                                         'sha256': hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()}
                                   for name, path in paths.items()}
        inventory['claim'] = 'inventory only; no outcome, power or independence claim'
        print(json.dumps(inventory, indent=2))
        # References are never passed to the response model; preparation keeps them separate.
        del cases, scoring_references
    """)
    markdown("""
        ## 2. Freeze a new comparison before generation

        The same model, recent window, older-history budget, retrieval backend and knowledge
        condition apply to all arms. Preparation is a Colab operation without paid model calls.
        It freezes gold-free cases/prompts and a separate scoring-only reference file. An
        existing directory must validate; do not edit a protocol or make a new directory to
        hide failed attempts. Keep dataset families in separate directories.

        A question count is not a participant count. Review the inventory's independent groups
        and development exposure before setting a formal sample-size/precision claim. The
        initial full-release comparison is exploratory unless a separately frozen independent
        test design meets its declared criteria. Static QA has nobody to answer clarification
        questions: never feed the released answer back as a simulated confirmation.
    """)
    code("""
        OBJECTIVE = REPO / 'evaluation/objective-memory'
        RUN_DIR = WORK / DATASET_KEY
        RESPONSE_MODEL = 'deepseek-v4.1-flash'
        ACCEPTED_PROVIDER_MODEL = 'deepseek-v4-1-flash-260910'
        MAX_RESPONSE_TOKENS = 256

        def objective_cli(script, *arguments):
            result = subprocess.run([sys.executable, str(OBJECTIVE / script), *map(str, arguments)],
                                    cwd=REPO, capture_output=True, text=True)
            if result.returncode:
                # Never echo raw transport errors, endpoint bodies or credential values.
                raise RuntimeError('Objective stage failed; inspect its sanitized journal/checkpoint. '
                                   + script + ' exit=' + str(result.returncode))
            output = result.stdout.strip()
            if output:
                try:
                    value = json.loads(output)
                except json.JSONDecodeError:
                    raise RuntimeError('Expected a JSON stage receipt; keep artifacts for inspection.') from None
                print(json.dumps(value, ensure_ascii=False, indent=2))
                return value
            return {}

        PREPARE_RUN = False
        if PREPARE_RUN:
            arguments = ['prepare', '--dataset', DATASET_KEY, '--input', INPUT_PATH,
                         '--output-dir', RUN_DIR, '--model', RESPONSE_MODEL,
                         '--accepted-provider-model', ACCEPTED_PROVIDER_MODEL,
                         '--api-base', 'https://tokendance.space/gateway/v1',
                         '--thinking-mode', 'disabled', '--backend', 'lexical',
                         '--max-response-tokens', MAX_RESPONSE_TOKENS,
                         '--short-turns', 16, '--top-k', 12,
                         '--short-chars', 6000, '--long-chars', 12000,
                         '--max-context-chars', 24000]
            if CONTEXT_PATH is not None:
                arguments.extend(['--context', CONTEXT_PATH])
            receipt = objective_cli('run_objective_comparison.py', *arguments)
        else:
            print('Preparation disabled. Review inventory/settings, then enable this cell deliberately.')
    """)
    markdown("""
        ## 3. Run one explicitly bounded batch

        This cell is the only Secret-reading/model-generation path. It uses the existing Colab
        Secret named **Capstone**, not a value pasted into source. The temporary environment
        variable is restored after the batch. Attempts, including failed or interrupted calls,
        remain journaled. No retry loop, model replacement, Drive mount or public upload is
        automatic.

        A batch attempts at most the selected number of unique canonical prompts, not that many
        independent arm answers. Exact duplicate prompts may derive several arm rows from one
        response; the protocol records this and the scorer counts actual calls/costs once. Rerun
        a batch deliberately to continue the same frozen run. Keep checkpoints after each batch.

        If status reports an interrupted `inflight` intent, it may already have incurred a
        charge. Do not create a fresh run or retry it automatically. The runner's explicit
        `recover` command can use an already recorded validated response, or close the intent
        as `terminal_unknown_failure`; that latter outcome remains in the failure denominator.
    """)
    code("""
        from contextlib import contextmanager
        COLAB_SECRET_NAME = 'Capstone'  # Existing Secret NAME only.

        @contextmanager
        def objective_secret():
            from google.colab import userdata
            previous = os.environ.get('CBT_EVAL_API_KEY')
            secret = userdata.get(COLAB_SECRET_NAME)
            if not isinstance(secret, str) or not secret.strip():
                raise RuntimeError('The existing Colab Secret is unavailable; no request started.')
            os.environ['CBT_EVAL_API_KEY'] = secret
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop('CBT_EVAL_API_KEY', None)
                else:
                    os.environ['CBT_EVAL_API_KEY'] = previous
                secret = None

        RUN_MODEL_BATCH = False
        MAX_CALLS_THIS_BATCH = 30
        if RUN_MODEL_BATCH:
            if type(MAX_CALLS_THIS_BATCH) is not int or not 1 <= MAX_CALLS_THIS_BATCH <= 60:
                raise ValueError('Choose one finite batch of 1–60 attempted calls.')
            with objective_secret():
                receipt = objective_cli('run_objective_comparison.py', 'generate',
                                        '--run-dir', RUN_DIR, '--max-calls', MAX_CALLS_THIS_BATCH)
        else:
            print('Paid batch disabled. No Secret read and no model request.')
    """)
    markdown("""
        ## 4. Inspect progress, then deterministically score

        The official matching/choice rules score released answers without another model judge.
        Incomplete runs produce progress only. A final table requires the complete frozen
        cohort, with terminal failures retained in the denominator and availability shown
        separately. Small numbers of independent histories cannot support a precise general
        effect claim. Task scores, shared-prompt equivalence and preparation/response costs
        remain separate rather than becoming an invented overall score.
    """)
    code("""
        if (RUN_DIR / 'protocol.json').exists():
            objective_cli('run_objective_comparison.py', 'status', '--run-dir', RUN_DIR)
            objective_cli('score_objective_comparison.py', '--run-dir', RUN_DIR)
        else:
            print('No frozen run prepared. No scoring performed.')
    """)
    markdown("""
        ## 5. Export a checked recovery/audit package

        An archive contains the frozen protocol, actual prompts, references, responses,
        attempt journals, recomputed scores/progress, source-code snapshots and SHA-256
        checksums. It omits credentials and the original full dataset files. These artifacts
        can still contain benchmark-derived text: retain them in controlled project storage,
        respecting the source terms, rather than publishing arbitrary raw files to GitHub.

        Colab storage is temporary. Download the checkpoint before ending a runtime. A partial
        archive is a recovery checkpoint, never a completed effectiveness result. Hashes verify
        byte consistency; they do not authenticate the provider or establish clinical benefit.
        No actual-user data belongs in this benchmark export path.
    """)
    code("""
        EXPORT_CHECKPOINT = False
        if EXPORT_CHECKPOINT:
            from datetime import datetime, timezone
            from google.colab import files
            from objective_audit import export_audit_zip
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            archive = WORK / ('objective-' + DATASET_KEY + '-' + stamp + '.zip')
            receipt = export_audit_zip(RUN_DIR, archive)
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            files.download(str(archive))
        else:
            print('Export disabled. Enable after a batch to retain its checked raw artifacts.')
    """)
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "colab": {"name": DEFAULT_OUTPUT.name, "provenance": []},
                         "code_revision": revision, "experiment": "objective-memory-comparison"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    notebook = build(args.code_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f'Wrote {len(notebook["cells"])} cells; no cell executed.')


if __name__ == "__main__":
    main()
