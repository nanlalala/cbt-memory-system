"""One credential-free phase for the separately versioned quality follow-up."""
from hashlib import sha256
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def execute(manifest_path):
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required")
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    repo = Path(manifest["repo"]).resolve()
    reader, grade = Path(manifest["reader"]).resolve(), Path(manifest["grade"]).resolve()
    if (any(not p.is_relative_to("/content") or p == Path("/content")
            for p in (path, repo, reader, grade))
            or manifest["cohort"] not in {"professional7", "cactus600"}
            or manifest["stage"] not in {"prepare", "run"}):
        raise ValueError("fixed original CBT cohort, stage and hosted paths required")
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != manifest["commit"]
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()):
        raise ValueError("pinned clean follow-up checkout required")
    for relative, expected in manifest["code_sha256"].items():
        source = repo / relative
        if (source.is_symlink() or not source.resolve().is_relative_to(repo)
                or sha256(source.read_bytes()).hexdigest() != expected):
            raise ValueError("follow-up helper source differs")
    if sha256(Path(__file__).read_bytes()).hexdigest() != manifest["worker_sha256"]:
        raise ValueError("copied worker source differs")
    sys.path[:0] = [str(repo / p) for p in ("modules/agent-integration", "evaluation/cbt-memory-effect",
        "evaluation/architecture-rerun")]
    from cbt_runtime_guard import assert_repository_runtime
    assert_repository_runtime(repo)
    api = runpy.run_path(str(repo / "evaluation/architecture-rerun/quality_span_judge_v3.py"))
    if manifest["stage"] == "prepare":
        if manifest["max_new_calls"] != 0:
            raise ValueError("source preparation has no provider allowance")
        result = api["prepare"](reader, grade, api["JUDGE_IDENTITY"])
    else:
        from hosted_relay import RelayTransport
        if type(manifest["max_new_calls"]) is not int or manifest["max_new_calls"] < 1:
            raise ValueError("positive finite follow-up call allowance required")
        transport = RelayTransport(api["JUDGE_IDENTITY"], manifest["socket"], manifest["job_id"])
        result = api["run"](grade, transport, manifest["max_new_calls"], workers=4)
    assert_repository_runtime(repo)
    (path.parent / "complete.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        execute(args.manifest)
        return 0
    except Exception as exc:
        # Never expose provider messages, dialogue or credential-bearing locals.
        (Path(args.manifest).parent / "error.json").write_text(json.dumps({
            "status": "failed_without_retry", "error_type": type(exc).__name__}) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
