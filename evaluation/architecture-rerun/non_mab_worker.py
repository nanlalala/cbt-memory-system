"""Credential-free one-phase worker for cloud_continue_non_mab.py."""
from hashlib import sha256
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from hosted_relay import RelayTransport

ANSWER_IDENTITY = {"provider": "https://tokendance.space/gateway/v1", "model": "deepseek-v4.1-flash",
                   "returned_model": "deepseek-v4-1-flash-260910", "model_family": "deepseek"}
JUDGE_IDENTITY = {"provider": "https://tokendance.space/gateway/v1", "model": "qwen3.5-plus",
                  "returned_model": "qwen3.5-plus-2026-04-20", "model_family": "qwen"}
SOURCE_ACTIONS = {"cbt-source", "cbt-quality-source", "locomo-source", "locomo-judge-source", "lme-source-only"}
PAID_ACTIONS = {"cbt-run", "cbt-quality-run", "locomo-run", "locomo-judge-run"}


def _guard(manifest_path):
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required; zero model calls")
    manifest_path = Path(manifest_path)
    if manifest_path.is_symlink():
        raise ValueError("symlinked execution manifest")
    root = manifest_path.resolve().parent
    manifest = json.loads(manifest_path.read_text())
    repo, output = Path(manifest["selected_repo"]), Path(manifest["run_root"])
    if any(p.is_symlink() or not p.resolve().is_relative_to("/content") for p in (root, repo, output)):
        raise ValueError("worker, output and checkout must remain under /content")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != manifest["selected_commit"]
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()):
        raise ValueError("selected clean checkout changed")
    for name, expected in manifest["helper_sha256"].items():
        path = root / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("isolated helper bytes changed")
    action, cohort = manifest["action"], manifest["cohort"]
    if action not in SOURCE_ACTIONS | PAID_ACTIONS:
        raise ValueError("unknown fixed scientific action")
    if (action.startswith("cbt-") and cohort not in {"professional7", "cactus600"}
            or action.startswith("locomo-") and cohort != "locomo1540"
            or action == "lme-source-only" and cohort != "longmemeval500"):
        raise ValueError("action/cohort mismatch; MAB is excluded")
    caps = manifest["relay_limits"]
    if (type(caps) is not dict or not caps or any(k not in {ANSWER_IDENTITY["model"], JUDGE_IDENTITY["model"]}
            or type(v) is not int or v < 0 for k, v in caps.items())
            or (action in SOURCE_ACTIONS and any(caps.values()))
            or (action in PAID_ACTIONS and not any(caps.values()))):
        raise ValueError("finite explicit phase allowances required")
    for name, identity in (("answer_max_new_calls", ANSWER_IDENTITY), ("judge_max_new_calls", JUDGE_IDENTITY)):
        cap = manifest[name]
        if type(cap) is not int or cap < 0 or cap > caps.get(identity["model"], 0):
            raise ValueError("worker call cap exceeds parent relay allowance")
    return manifest, root, repo, output


def execute(manifest_path):
    manifest, root, repo, output = _guard(manifest_path)
    sys.path.insert(0, str(repo / "modules/agent-integration"))
    sys.path.insert(0, str(repo / "evaluation/cbt-memory-effect"))
    sys.path.insert(0, str(repo / "evaluation/architecture-rerun"))
    from cbt_runtime_guard import assert_repository_runtime
    (root / "initial_module_origins.json").write_text(json.dumps(assert_repository_runtime(repo), indent=2) + "\n")
    action, source = manifest["action"], manifest["source"]
    answer_cap, judge_cap = manifest["answer_max_new_calls"], manifest["judge_max_new_calls"]
    def transport(identity):
        return RelayTransport(identity, manifest["socket"], manifest["job_id"])
    if action in {"cbt-source", "cbt-run"}:
        from architecture_retest import continuation
        if action == "cbt-source":
            result = continuation.prepare(source["reader"], source["memory"], output)
        else:
            result = continuation.run(output, transport(ANSWER_IDENTITY), transport(JUDGE_IDENTITY),
                max_new_calls=answer_cap, assessment_max_new_calls=judge_cap, workers=4)
    elif action in {"cbt-quality-source", "cbt-quality-run"}:
        api = runpy.run_path(str(repo / "evaluation/architecture-rerun/quality_span_judge.py"))
        if action == "cbt-quality-source":
            result = api["prepare"](output / "reader", output / "quality_v2", JUDGE_IDENTITY)
        else:
            result = api["run"](output / "quality_v2", transport(JUDGE_IDENTITY), max_new_calls=judge_cap, workers=4)
    elif action in {"locomo-source", "locomo-run", "lme-source-only"}:
        api = runpy.run_path(str(repo / "evaluation/architecture-rerun/formal_continuation.py"))
        if action == "locomo-source":
            result = api["prepare"](source["reader"], output)
        elif action == "locomo-run":
            result = api["run"](output, transport(ANSWER_IDENTITY), max_new_calls=answer_cap)
        else:
            result = api["inspect_source_only"](source["reader"])
    else:
        # Formal judging preserves its original reference-based scoring method.
        sys.path.insert(0, str(repo / "evaluation/architecture-rerun"))
        api = runpy.run_path(str(repo / "evaluation/architecture-rerun/formal_judge.py"))
        if action == "locomo-judge-source":
            result = api["prepare"](output, output / "formal_judge", JUDGE_IDENTITY)
        else:
            result = api["run"](output / "formal_judge", transport(JUDGE_IDENTITY), max_new_calls=judge_cap, workers=4)
    (root / "final_module_origins.json").write_text(json.dumps(assert_repository_runtime(repo), indent=2) + "\n")
    (root / "complete.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        execute(args.manifest)
        print(json.dumps({"status": "complete", "result_file": str(args.manifest.parent / "complete.json")}))
        return 0
    except Exception as exc:
        # Clinical text and credential-bearing exceptions are never serialized.
        error = {"status": "failed_without_retry", "error_type": type(exc).__name__}
        (args.manifest.parent / "error.json").write_text(json.dumps(error) + "\n")
        print(json.dumps(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
