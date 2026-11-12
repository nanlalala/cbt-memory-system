"""Credential-free isolated controller worker in the genuine Colab VM.

Launched from a byte-verified external helper directory. The selected checkout
supplies all scientific code; only local RPC transport lives in this helper.
"""
from hashlib import sha256
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import traceback

from hosted_relay import RelayTransport


def _hosted():
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required; no runtime flags are fabricated")


def execute(manifest_path):
    _hosted()
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    repo = Path(manifest["selected_repo"]).resolve()
    if not root.is_relative_to("/content") or not repo.is_relative_to("/content"):
        raise ValueError("isolated worker and selected checkout must remain in /content")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != manifest["selected_commit"]:
        raise ValueError("selected checkout changed")
    for name, expected in manifest["helper_sha256"].items():
        path = root / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("worker/relay helper bytes changed")
    # No provider credentials are accessed, loaded or deserialized in this process.
    sys.path.insert(0, str(repo / "modules/agent-integration"))
    sys.path.insert(0, str(repo / "evaluation/cbt-memory-effect"))
    from cbt_runtime_guard import assert_repository_runtime
    audit = assert_repository_runtime(repo)
    (root / "initial_module_origins.json").write_text(json.dumps(audit, indent=2) + "\n")
    action = manifest["action"]
    run_root = Path(manifest["run_root"])
    socket, job_id = manifest["socket"], manifest["job_id"]
    if action in {"cbt-prepare", "cbt-run"}:
        controller = runpy.run_path(str(repo / "evaluation/architecture-rerun/cbt_hosted_controller.py"))
        inherited = ({"professional7": manifest["inherited_memory_root"]}
                     if manifest.get("inherited_memory_root") else {})
        if action == "cbt-prepare":
            result = controller["prepare_all"](repo, run_root, inherited_memory_roots=inherited)
        else:
            from architecture_retest import runner
            answer = RelayTransport(runner.IDENTITY, socket, job_id)
            judge = RelayTransport(controller["JUDGE_IDENTITY"], socket, job_id)
            result = controller["run_all"](repo, answer, judge, run_root,
                stage=manifest["stage"], workers=4, inherited_memory_roots=inherited)
    elif action == "formal-judge":
        controller = runpy.run_path(str(repo / "evaluation/architecture-rerun/hosted_controller.py"))
        from architecture_retest.transport import require_hosted
        require_hosted()
        # Reference judging imports no encoder and does not initialize E5/Torch.
        identity = {"provider": "https://tokendance.space/gateway/v1", "model": "qwen3.5-plus",
                    "returned_model": "qwen3.5-plus-2026-04-20", "model_family": "qwen"}
        judge = RelayTransport(identity, socket, job_id)
        result = controller["judge_batch"](repo, manifest["cohort"], judge,
            max_new_calls=manifest["max_new_calls"], root=run_root)
    else:
        raise ValueError("unknown isolated worker action")
    final_audit = assert_repository_runtime(repo)
    (root / "final_module_origins.json").write_text(json.dumps(final_audit, indent=2) + "\n")
    (root / "complete.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute(args.manifest)
        print(json.dumps({"status": "complete", "manifest": str(args.manifest),
                          "result_file": str(args.manifest.parent / "complete.json")}))
    except Exception as exc:
        error = {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
        (args.manifest.parent / "error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error_file": str(args.manifest.parent / "error.json")}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
