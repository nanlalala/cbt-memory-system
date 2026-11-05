"""Authorized finite launcher in a fresh genuine-Colab subprocess.

Credential arrives only through the process environment. This file never reads a
browser credential, prints an error body, changes an old run or mounts Drive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

import prepare_generation_materials as compiler
import run_followup_pipeline as runner

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = (
    "protocol.json", "requests.jsonl", "attempts.jsonl", "outcomes.jsonl",
    "extraction_event_rows.jsonl", "extraction_event_rows_partial.jsonl",
    "prepared_cases.jsonl", "stage_failures.jsonl", "generation_rows.jsonl",
    "generation_rows_partial.jsonl", "generation_receipt.json", "generation_progress.json",
    "judge_requests.jsonl", "reviewed_rows.jsonl", "reviewed_rows_partial.jsonl",
    "followup_summary.json", "review_progress.json",
)


def archive(run, target):
    with runner.locked(run):
        protocol = runner.load_protocol(run, include_scoring=True)
        if target.exists() or target.is_symlink():
            raise ValueError("fresh audit archive required")
        temporary = target.with_suffix(".tmp")
        sums = {}
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            files = [(run / name, "run/" + name) for name in ARTIFACTS]
            files += [(run / "source_package" / name, "run/source_package/" + name) for name in runner.INPUT_FILES]
            files += [(ROOT / name, "code/" + name) for name in protocol["code_sha256"]]
            ledger = ROOT / "evaluation/dialogue-benefit/task_materials_draft.json"
            expected_ledger = runner.read_json(run / "source_package/manifest.json")["original_ledger_file_sha256"]
            if runner.digest(ledger) != expected_ledger:
                raise ValueError("original materials changed during audit")
            files.append((ledger, "materials/task_materials_draft.json"))
            for path, member in files:
                if not path.exists():
                    continue
                if path.is_symlink() or not path.is_file():
                    raise ValueError("audit accepts only regular allowlisted files")
                body = path.read_bytes()
                if member.startswith("code/") and hashlib.sha256(body).hexdigest() != protocol["code_sha256"][member[5:]]:
                    raise ValueError("frozen code changed during audit")
                stream.writestr(member, body)
                sums[member] = hashlib.sha256(body).hexdigest()
            stream.writestr("SHA256SUMS.json", json.dumps(sums, sort_keys=True, indent=2) + "\n")
            stream.writestr("CONTROLLED_RESEARCH_AUDIT.txt", "Fictional development replay only. Contains controlled historical event/scoring material. Do not publish or send the whole archive to a model. Runtime deletion isolates future context; research audit/provider erasure is not claimed. Zero human participants or clinical outcomes.\n")
        temporary.replace(target)
    return {"cloud_archive": str(target), "bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def launch(work):
    runner.require_colab()
    if not os.environ.get("CBT_EVAL_API_KEY"):
        raise RuntimeError("Authorized Colab credential unavailable; no model calls")
    work = Path(work).expanduser()
    if work.is_symlink() or not work.resolve().is_relative_to(Path("/content")):
        raise ValueError("Use an independent genuine Colab /content directory")
    work.mkdir(parents=True, exist_ok=True)
    source, run = work / "source-package", work / "run"
    if not source.exists():
        compiler.prepare_materials(ROOT / "evaluation/dialogue-benefit/task_materials_draft.json", source)
    compiler.read_prepared_generation(source)
    if not run.exists():
        prepared = runner.prepare(source, run)
        print(json.dumps({"phase": "prepared", **prepared}), flush=True)
    else:
        runner.load_protocol(run)
    generated = runner.generate(run, max_new_calls=160, workers=4)
    first = archive(run, work / "generation-checkpoint.zip") if not (work / "generation-checkpoint.zip").exists() else None
    print(json.dumps({"phase": "generation", "status": generated["status"],
                      "terminal_arm_rows": generated.get("terminal_arm_rows", 0), "cost": generated["cost"], "checkpoint": first}), flush=True)
    if generated["status"] != "complete":
        return {"status": "partial", "new_model_result_claim": False}
    reviewed = runner.review(run, max_new_calls=160)
    final = archive(run, work / "final-audit.zip") if not (work / "final-audit.zip").exists() else None
    result = {"status": reviewed["status"], "source_candidate_pipeline_development": True,
              "planned_cases": reviewed["planned_cases"], "planned_arm_rows": reviewed["planned_arm_rows"],
              "conditions": reviewed.get("conditions", {}), "cost": reviewed["cost"],
              "clinical_claim": False, "human_participants": 0, "independent_human_ratings": 0,
              "checkpoint": final}
    print(json.dumps({"phase": "completed", **result}, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    try:
        launch(args.work_dir)
    except Exception as error:
        # Preserve raw artifacts; never echo exception/provider/credential text.
        print(json.dumps({"status": "safe_stop", "error_type": type(error).__name__,
                          "raw_error_or_secret_displayed": False, "automatic_retry": False}), flush=True)
        raise SystemExit(1) from None
