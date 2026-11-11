"""Save a private audit checkpoint without interrupting hosted inference.

Only the named study artifacts are copied. Weights, credentials and repository
configuration are excluded. A live collection is explicitly non-atomic and must
not be treated as permission to resume or resend an in-progress generation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import zipfile


VERSION = "cbt-open-state-audit-checkpoint-v1"
CONTENT = Path("/content")
MODEL_ROOT = CONTENT / "cbt-open-reader-20261112-v1"
ROOTS = {
    "prepared_before": CONTENT / "cbt-open-state-prepared-before-20261112-v2",
    "prepared_after": CONTENT / "cbt-open-state-prepared-after-20261112-v2",
    "answers_before": CONTENT / "cbt-open-state-answers-before-20261112-v2",
    "answers_after": CONTENT / "cbt-open-state-answers-after-20261112-v2",
    "equip": CONTENT / "cbt-equip-professional-sourcebank-20261112-v1",
    "therapyjudge": CONTENT / "therapyjudgebench-v2-source-freeze-20261112",
    "calibration_prepared": CONTENT / "therapyjudge-five-calibration-prepared-v1",
    "calibration_answers": CONTENT / "therapyjudge-five-calibration-results-v1",
}
MODEL_METADATA = ("load_protocol.json", "model_files.json", "model_load_receipt.json",
                  "model_load_events.jsonl", "generation_ledger.jsonl")
EXTENSIONS = {".json", ".jsonl", ".sqlite", ".pdf", ".html", ".md", ".log"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(output, archive, *, final=False):
    if not (os.environ.get("COLAB_RELEASE_TAG") and CONTENT.is_dir()
            and importlib.util.find_spec("google.colab")):
        raise RuntimeError("genuine_hosted_colab_required_before_any_io")
    output, archive = Path(output).resolve(), Path(archive).resolve()
    if any(not p.is_relative_to(CONTENT) or p == CONTENT for p in (output, archive)):
        raise ValueError("named_hosted_content_paths_required")
    if output.exists() or archive.exists():
        raise FileExistsError("preserve_previous_checkpoint_use_new_paths")
    if final:
        for name in ("answers_before", "answers_after"):
            complete = json.loads((ROOTS[name] / "completion_summary.json").read_text())
            if complete.get("complete") is not True or complete.get("logical_completed") != 48:
                raise ValueError("both_complete_memory_versions_required_for_final_checkpoint")
        ledger = [json.loads(line) for line in (MODEL_ROOT / "generation_ledger.jsonl").read_text().splitlines()]
        intents = {r["canonical_key"] for r in ledger if r["event"] == "generation_intent"}
        outcomes = {r["canonical_key"]: r for r in ledger if r["event"] == "generation_outcome"}
        if set(outcomes) != intents or any(r.get("status") != "published" for r in outcomes.values()):
            raise ValueError("unsettled_or_unsuccessful_generation_requires_live_audit_status")
    output.mkdir(mode=0o700, parents=True)
    started = stamp()
    files, total = [], 0

    def copy(source, relative):
        nonlocal total
        if source.is_symlink() or not source.is_file():
            raise ValueError("only_regular_study_artifacts_may_be_archived")
        destination = output / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        method = "exact_bytes_read_during_capture"
        if source.suffix == ".sqlite":
            method = "consistent_sqlite_backup_from_read_only_connection"
            with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=10) as original:
                with sqlite3.connect(destination) as backup:
                    original.backup(backup)
        else:
            data = source.read_bytes()
            if len(data) > 16_000_000:
                raise ValueError("unexpected_large_artifact_no_weights_allowed")
            with destination.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(destination, 0o600)
        size = destination.stat().st_size
        total += size
        if total > 128_000_000:
            raise ValueError("bounded_checkpoint_size_exceeded")
        files.append({"path": str(relative), "source_path": str(source), "bytes": size,
                      "sha256": digest(destination), "method": method})

    for name, source_root in ROOTS.items():
        if source_root.exists():
            for source in sorted(source_root.rglob("*")):
                if source.is_file() and (source.suffix in EXTENSIONS or source.name == "LICENSE"):
                    copy(source, Path(name) / source.relative_to(source_root))
    for filename in MODEL_METADATA:
        source = MODEL_ROOT / filename
        if source.exists():
            copy(source, Path("model_metadata") / filename)
    for pattern in ("cbt-open-state-*-20261112-v*.json", "cbt-open-state-reader-*-20261112-v2.log",
                    "therapyjudge-five-calibration-token-preflight-v*.json"):
        for source in sorted(CONTENT.glob(pattern)):
            copy(source, Path("reports") / source.name)
    revisions = {}
    for name, root in {
        "before": CONTENT / "cbt-open-state-before-20261112-v2",
        "after": CONTENT / "cbt-open-state-after-20261112-v2",
        "audit_helpers": Path(__file__).resolve().parents[2],
    }.items():
        revisions[name] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"version": VERSION, "stage": "completed_memory_audit" if final else "live_partial_audit",
                "started_at_utc": started, "finished_at_utc": stamp(), "files": files,
                "uncompressed_bytes": total, "code_revisions": revisions,
                "checkpoint_builder_sha256": digest(__file__),
                "collection_is_not_atomic_across_files": True,
                "sqlite_backups_individually_consistent": True,
                "original_runtime_may_still_be_active": not final,
                "automatic_resume_or_resend_authorized": False,
                "weights_credentials_repository_configuration_included": False,
                "model_calls_from_checkpoint_builder": 0,
                "source_files_modified": False}
    (output / "checkpoint_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                zipped.write(source, source.relative_to(output))
    os.chmod(partial, 0o600)
    partial.rename(archive)
    return {"stage": manifest["stage"], "archive": str(archive), "bytes": archive.stat().st_size,
            "sha256": digest(archive), "artifact_files": len(files),
            "new_model_calls": 0, "automatic_resume_authorized": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(checkpoint(args.output, args.archive, final=args.final), sort_keys=True))
