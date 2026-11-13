"""Credential-free source, generation or grading phase for the new C6 run."""
from hashlib import sha256
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def execute(path):
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required")
    path=Path(path).resolve();m=json.loads(path.read_text());repo=Path(m["repo"]).resolve()
    if (m["action"] not in {"source","generation","quality-source","quality-run"}
            or m["cohort"] not in {"cactus600","professional7"}
            or any(not Path(m[name]).resolve().is_relative_to("/content") for name in ("repo","study_root"))
            or subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()!=m["commit"]
            or subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()):
        raise ValueError("fixed source/action and clean pinned hosted checkout required")
    for name,expected in m["code_sha256"].items():
        source=repo/name
        if source.is_symlink() or sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError("new recovery helper source differs")
    if sha256(Path(__file__).read_bytes()).hexdigest()!=m["worker_sha256"]:
        raise ValueError("copied recovery worker differs")
    sys.path[:0]=[str(repo/p) for p in ("modules/agent-integration","evaluation/cbt-memory-effect","evaluation/architecture-rerun")]
    from cbt_runtime_guard import assert_repository_runtime
    from architecture_retest import continuation,runner
    from hosted_relay import RelayTransport
    (path.parent/"initial_module_origins.json").write_text(json.dumps(assert_repository_runtime(repo),sort_keys=True,indent=2)+"\n")
    study=Path(m["study_root"])
    if m["action"]=="source":
        if any(m["caps"].values()):raise ValueError("source phase cannot admit model calls")
        result=continuation.prepare(m["source"]["reader"],m["source"]["memory"],study)
    elif m["action"]=="generation":
        answer=RelayTransport(runner.IDENTITY,m["socket"],m["job_id"])
        identity=json.loads((study/"memory/protocol.json").read_text())["provider"]
        assessor=RelayTransport(identity,m["socket"],m["job_id"])
        if m["cohort"]=="professional7" and any(m["caps"].values()):
            raise ValueError("P7 migration may only replay exact cached receipts with zero new calls")
        result=continuation.run(study,answer,assessor,max_new_calls=m["caps"][runner.IDENTITY["model"]],
            assessment_max_new_calls=m["caps"][identity["model"]],workers=4)
    else:
        api=runpy.run_path(str(repo/"evaluation/architecture-rerun/quality_span_judge_v3.py"))
        if m["action"]=="quality-source":
            if any(m["caps"].values()):raise ValueError("quality-source phase cannot admit model calls")
            result=api["prepare"](study/"reader",study/"quality_v3",api["JUDGE_IDENTITY"])
        else:
            transport=RelayTransport(api["JUDGE_IDENTITY"],m["socket"],m["job_id"])
            result=api["run"](study/"quality_v3",transport,m["caps"][api["JUDGE_IDENTITY"]["model"]],workers=4)
    (path.parent/"final_module_origins.json").write_text(json.dumps(assert_repository_runtime(repo),sort_keys=True,indent=2)+"\n")
    (path.parent/"complete.json").write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--manifest",required=True)
    args=parser.parse_args()
    try:execute(args.manifest)
    except Exception as exc:
        (Path(args.manifest).parent/"error.json").write_text(json.dumps({"status":"failed_without_retry","error_type":type(exc).__name__})+"\n")
        raise SystemExit(1)
