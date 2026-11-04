"""Export a validated objective-run checkpoint using a fixed artifact allowlist.

This module does not call a model, read credentials or copy arbitrary run files.
Benchmark-derived text remains controlled audit material, not a public release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_REQUIRED = (
    "protocol.json", "cases.jsonl", "references.jsonl", "prompts.jsonl",
    "inventory.jsonl", "preparation_summary.json",
)
RUN_OPTIONAL = (
    "response_attempts.jsonl", "response_outcomes.jsonl", "failures.jsonl", "responses.jsonl",
)
SCORE_FILES = ("objective_scores.jsonl", "objective_summary.json")
PROGRESS_FILE = "objective_progress.json"
CODE_FILES = (
    "evaluation/objective-memory/objective_datasets.py",
    "evaluation/objective-memory/dataset_registry.json",
    "evaluation/objective-memory/run_objective_comparison.py",
    "evaluation/objective-memory/score_objective_comparison.py",
    "evaluation/objective-memory/objective_audit.py",
    "evaluation/end-to-end-dialogue/run_three_arm.py",
    "evaluation/end-to-end-dialogue/execution_lock.py",
    "evaluation/memory-retrieval/benchmark_adapters.py",
    "evaluation/memory-retrieval/validate_external_benchmarks.py",
    "evaluation/memory-retrieval/benchmark_manifest.json",
    "modules/model_snapshots.py",
    "modules/model_snapshots.json",
    "modules/long-term-memory/schema/memory_models.py",
    "modules/long-term-memory/lifecycle/memory_manager.py",
    "modules/long-term-memory/storage/sqlite_store.py",
    "modules/long-term-memory/retrieval/memory_retriever.py",
    "modules/long-term-memory/retrieval/memory_use_policy.py",
    "modules/agent-integration/context_builder.py",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _runtime_identity() -> dict:
    packages = {}
    for name in ("pyarrow", "numpy", "sentence-transformers", "transformers", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages,
            "scope": "export-time runtime metadata; no credentials or environment dump"}


def _source_snapshot(root: Path = ROOT) -> tuple[dict[str, bytes], str]:
    """Bind actual source bytes to their immutable clean-checkout commit."""
    root = root.resolve()
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.PIPE).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("A pinned Git checkout is required for code-snapshot export") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid code revision")
    members = {}
    for relative in CODE_FILES:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or unsafe code-snapshot file: {relative}")
        actual = path.read_bytes()
        try:
            committed = subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{relative}"],
                                                stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"Code file is absent from the pinned commit: {relative}") from exc
        if actual != committed:
            raise ValueError(f"Code checkout differs from its reported revision: {relative}")
        members["code/" + relative] = actual
    return members, revision


def _dependencies():
    directory = str(ROOT / "evaluation/end-to-end-dialogue")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from execution_lock import exclusive_directory
    from run_objective_comparison import _load_run_locked
    from score_objective_comparison import _score_run_locked
    return exclusive_directory, _load_run_locked, _score_run_locked


def export_audit_zip(run_dir: Path | str, output_path: Path | str) -> dict:
    """Validate/recompute under the writer lock, then atomically save a new ZIP.

    Partial runs include only a progress receipt, never a stale final score.
    Output files are never placed inside the live run or overwritten in place.
    Only the allowlist is copied; original datasets, SQLite files, Secrets and
    arbitrary user files are not discovered or traversed.
    """
    raw_run = Path(run_dir)
    if raw_run.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    run = raw_run.resolve()
    raw_output = Path(output_path)
    if raw_output.is_symlink():
        raise ValueError("Output archive must not be a symlink")
    output = raw_output.resolve()
    if not run.is_dir():
        raise ValueError("Run directory does not exist")
    if output.is_relative_to(run) or output.suffix != ".zip":
        raise ValueError("Choose a .zip output outside the live run directory")
    if output.exists():
        raise FileExistsError("Choose a new checkpoint name; existing archives are not overwritten")
    exclusive_directory, load_locked, score_locked = _dependencies()
    with exclusive_directory(run):
        # Check before loading or scoring as those operations may read/write
        # the same fixed paths. Never follow a pre-existing artifact symlink.
        for name in (*RUN_REQUIRED, *RUN_OPTIONAL, *SCORE_FILES, PROGRESS_FILE):
            path = run / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Unsafe run artifact: {name}")
        load_locked(run)
        summary = score_locked(run, write_outputs=True)
        status = summary.get("status")
        if status not in {"complete", "partial"}:
            raise ValueError("Unrecognized scoring completion state; checkpoint not exported")
        final = status == "complete"
        source_members, code_revision = _source_snapshot()
        members: dict[str, bytes] = dict(source_members)

        for name in (*RUN_REQUIRED, *RUN_OPTIONAL):
            path = run / name
            if not path.exists():
                if name in RUN_REQUIRED:
                    raise ValueError(f"Required run artifact is missing: {name}")
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unsafe run artifact: {name}")
            members["run/" + name] = path.read_bytes()
        derived = SCORE_FILES if final else (PROGRESS_FILE,)
        for name in derived:
            path = run / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Fresh scoring artifact is missing or unsafe: {name}")
            members["run/" + name] = path.read_bytes()

        protocol = json.loads(members["run/protocol.json"])
        # New runner dependencies must not silently disappear from a future
        # archive when its frozen code-hash set grows. A self-contained export
        # includes every exact byte sequence the loader will demand offline.
        for relative, expected in protocol.get("code_sha256", {}).items():
            value = source_members.get("code/" + relative)
            if value is None or _sha(value) != expected:
                raise ValueError(f"Frozen code dependency is absent or mismatched in export: {relative}")
        declaration = {
            "schema_version": "objective-memory-audit-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete_deterministic_run" if final else "partial_recovery_checkpoint",
            "final": final,
            "dataset_key": protocol.get("dataset_manifest", {}).get("dataset_key", protocol.get("dataset_key")),
            "protocol_sha256": _sha(members["run/protocol.json"]),
            "code_revision": code_revision,
            "runtime": _runtime_identity(),
            "files": {name: {"bytes": len(value), "sha256": _sha(value)}
                      for name, value in sorted(members.items())},
            "contains_benchmark_derived_text": True,
            "redistribution": "Controlled audit only; follow the frozen registry's source terms. No real-user data.",
            "boundary": "Byte consistency and deterministic scoring are not provider authentication, user benefit or clinical efficacy.",
            "recompute": "From the archive root: python code/evaluation/objective-memory/score_objective_comparison.py --run-dir run . No generation or Secret is required.",
        }
        members["AUDIT_DECLARATION.json"] = _json_bytes(declaration)
        members["SHA256SUMS"] = "".join(
            f"{_sha(value)}  {name}\n" for name, value in sorted(members.items())
        ).encode("utf-8")

        output.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(prefix=".objective-audit-", suffix=".zip.part",
                                             dir=output.parent, delete=False)
        temporary = Path(handle.name)
        handle.close()
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, value in sorted(members.items()):
                    archive.writestr(name, value)
            # Hard-link publication is atomic and refuses an existing name,
            # including a name created concurrently after the check above.
            # Both paths are in one directory/filesystem. The temporary path
            # is removed below, leaving the complete archive at its new name.
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"archive": str(output), "bytes": output.stat().st_size,
            "sha256": _sha(output.read_bytes()), "members": len(members),
            "status": declaration["status"], "final": final,
            "code_revision": code_revision, "clinical_claim": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_audit_zip(args.run_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
