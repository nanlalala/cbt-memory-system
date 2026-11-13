"""Resume only retained professional-seven quality grading in its live Colab.

Call ``run(globals())`` from the original notebook. This module never reads a
Secret, creates a VM, regenerates an answer, resumes another cohort or retries
an existing invocation. A repeated notebook execution reports its saved trace.
"""
from datetime import datetime, timezone
from hashlib import sha256
import base64
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import threading
import time
import uuid
import zipfile

COMMIT = "e7dadbc2ca0a50bcb0a7adcff4589fb3f302c85c"
REMOTE = "https://github.com/nanlalala/cbt-memory-system.git"
ROOT = Path("/content/capstone-professional7-regrade-20261113-v1")
REPO = Path("/content/cbt-professional7-quality-e7dadbc2")
RUN_ROOT = Path("/content/cbt-v4-isolated-cbt-full")
MODEL = "qwen3.5-plus"
PREPARE_JOB = "professional7-quality-v2-20261113-prepare"
GRADE_JOB = "professional7-quality-v2-20261113-grade"
CAP = 24
ARMS = ("no_rag", "raw_history_rag", "managed_memory_rag")
READER_HASHES = {
    "cases.json": "291feb7a237143c7c4c13a17842783682624a3b6addeaef0c50c17bff455d6f1",
    "intents.jsonl": "d0f95ea54ce3f024b663d5368806b975f47211494b23d2cae7f966767286a93d",
    "outcomes.jsonl": "9f9901350afbbe6818202fe02d937f32f1433da3e87430084d9fc0bd6f1378ef",
    "protocol.json": "04bc0d01a4642de90ea6c2a95ad566f4098d6eb7ee40a38b4ecae1fb16a2c452",
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _json(path):
    if Path(path).is_symlink():
        raise ValueError("symlinked retained artifact")
    return json.loads(Path(path).read_text())


def _rows(path):
    body = Path(path).read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError("incomplete retained journal; do not retry")
    return [json.loads(line) for line in body.split(b"\n") if line]


def _atomic(path, value):
    target = Path(path)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(target)


def _hosted():
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG")
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("the original genuine hosted Colab kernel is required")


def _reader_status(folder):
    for name, expected in READER_HASHES.items():
        path = folder / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("the original completed professional-seven snapshot differs")
    cases = _json(folder / "cases.json")
    protocol = _json(folder / "protocol.json")
    if (len(cases) != 7 or len({r["case_id"] for r in cases}) != 7
            or protocol.get("case_count") != 7 or protocol.get("fixture_only") is not False
            or protocol.get("arms") != list(ARMS)):
        raise ValueError("the complete original seven-case reader is required")
    intents, outcomes = _rows(folder / "intents.jsonl"), _rows(folder / "outcomes.jsonl")
    left, right = {r["key"] for r in intents}, {r["key"] for r in outcomes}
    if (len(left) != len(intents) or len(right) != len(outcomes) or left != right
            or any(r.get("status") != "received" for r in outcomes)):
        raise ValueError("outstanding, failed or unknown professional reader sends")
    expected = {folder / "cases" / r["case_id"] / "completed.json" for r in cases}
    if set((folder / "cases").glob("*/completed.json")) != expected:
        raise ValueError("all original cases must have terminal completion records")
    available = dict.fromkeys(ARMS, 0)
    for marker in expected:
        complete = _json(marker)
        if complete.get("case_id") != marker.parent.name:
            raise ValueError("foreign completion record")
        states = complete.get("rows", [])
        if "all_arms_preparation_failure" not in complete:
            if len(states) != 3 or {r["arm"] for r in states} != set(ARMS):
                raise ValueError("completion must retain all three arms")
            for row in states:
                available[row["arm"]] += row.get("status") == "received"
    if available != {"no_rag": 7, "raw_history_rag": 7, "managed_memory_rag": 3} or len(outcomes) != 27:
        raise ValueError("the retained original response coverage differs")
    return {"planned_cases": 7, "available_answers": available,
            "reader_received_calls": len(outcomes), "reader_unknown_sends": 0}


def _parent(namespace):
    if any(name not in namespace for name in ("v5_answer", "v5_judge", "v6_broker")):
        raise RuntimeError("original live parent transports are missing; no Secret was read")
    broker = namespace["v6_broker"]
    if broker.fixture_only is not False or not Path(broker.endpoint).is_socket():
        raise RuntimeError("original hosted parent relay is unavailable")
    if any(t.is_alive() and (t.name.startswith(("cbt-v5-", "cbt-v6-")))
           for t in threading.enumerate()):
        raise RuntimeError("an older study thread is still active; inspect before restoring")
    process = namespace.get("v6_cbt_process")
    if process is not None and process.poll() is None:
        raise RuntimeError("older full-CBT child is still active")
    if PREPARE_JOB in broker.snapshot()["jobs"] or GRADE_JOB in broker.snapshot()["jobs"]:
        raise RuntimeError("this grading job was already registered; inspect its retained trace")
    transport_class = type(namespace["v5_answer"])
    if transport_class is not type(namespace["v5_judge"]):
        raise RuntimeError("parent transport classes differ")
    original_call = getattr(transport_class, "_legacy_selection_original_call", None)
    current_call = transport_class.__call__
    original_register = getattr(broker, "_legacy_selection_original_register", None)
    current_register = broker.register_job
    if (not callable(original_call) or not callable(original_register)
            or getattr(original_register, "__self__", None) is not broker
            or getattr(original_register, "__name__", None) != "register_job"):
        raise RuntimeError("the exact retained legacy-pause originals are unavailable")
    if current_call is not original_call and current_call.__name__ != "legacy_cancel_future_call":
        raise RuntimeError("unexpected parent transport wrapper; leave it untouched")
    if current_register != original_register and current_register.__name__ != "legacy_selection_block_registration":
        raise RuntimeError("unexpected job-registration wrapper; leave it untouched")
    return broker, transport_class, original_call, original_register


def _checkout():
    if REPO.is_symlink():
        raise ValueError("symlinked selected checkout")
    if not REPO.exists():
        subprocess.run(["git", "clone", "--quiet", "--no-checkout", REMOTE, str(REPO)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run(["git", "-C", str(REPO), "checkout", "--quiet", "--detach", COMMIT],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True)
    origin = subprocess.check_output(["git", "-C", str(REPO), "remote", "get-url", "origin"], text=True).strip()
    if head != COMMIT or clean.strip() or origin != REMOTE:
        raise ValueError("selected public checkout is not the pinned clean revision")


def _wait(process, folder):
    deadline = time.monotonic() + 1800
    while process.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError("child remains running; inspect this invocation, never relaunch blindly")
        time.sleep(1)
    if process.returncode != 0 or not (folder / "complete.json").is_file():
        raise RuntimeError("isolated worker failed; retained attempt is not automatically retried")
    return _json(folder / "complete.json")


def _emit_export(folder):
    archive = ROOT / "professional7_quality_v2_PRIVATE.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.name != ".execution.lock":
                output.write(path, "professional7_quality_v2/" + str(path.relative_to(folder)))
        output.write(ROOT / "entry.json", "resume_entry.json")
    info = {"path": str(archive), "bytes": archive.stat().st_size,
            "sha256": sha256(archive.read_bytes()).hexdigest()}
    if info["bytes"] <= 2 * 1024 * 1024:
        from IPython.display import HTML, display
        body = base64.b64encode(archive.read_bytes()).decode()
        display(HTML('<a id="professional7-quality-v2-export" download="professional7_quality_v2_PRIVATE.zip" '
                     'href="data:application/zip;base64,' + body + '">Download professional-seven quality audit</a>'))
        info["small_notebook_export"] = True
    else:
        info["small_notebook_export"] = False
    return info


def run(namespace):
    """Run once in the original notebook namespace; repeated runs only report."""
    _hosted()
    if ROOT.is_symlink():
        raise ValueError("symlinked recovery directory")
    marker = ROOT / "entry.json"
    if marker.exists():
        retained = _json(marker)
        print("CAPSTONE_PROFESSIONAL7_RECOVERY", json.dumps(retained, ensure_ascii=False))
        return retained
    status = _reader_status(RUN_ROOT / "professional7")
    broker, transport_class, original_call, original_register = _parent(namespace)
    print("CAPSTONE_PROFESSIONAL7_PREFLIGHT", json.dumps({
        "at_utc": _utc(), "reader": status, "source_hashes_verified": True,
        "new_provider_calls": 0, "old_unknown_cohorts_resumed": False}, ensure_ascii=False))
    ROOT.mkdir(mode=0o700, exist_ok=True)
    trace = {"version": "professional7-colab-recovery-v1", "nonce": str(uuid.uuid4()),
             "started_at_utc": _utc(), "state": "claimed", "scientific_commit": COMMIT,
             "entry_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
             "scope": "professional7 quality-v2 only; no MAB or unresolved-cohort reruns",
             "maximum_new_grading_calls": CAP, "reader": status}
    try:
        with marker.open("x") as stream:
            json.dump(trace, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return _json(marker)
    try:
        _checkout()
        helper = runpy.run_path(str(REPO / "evaluation/architecture-rerun/hosted_relay.py"))
        checked = _parent(namespace)
        if checked[0] is not broker or checked[1] is not transport_class:
            raise RuntimeError("the original parent changed during checkout preparation")
        # Restore only our recorded pause. The exact old-quality admission gate
        # and all original call caps, journals and unknown-send records remain.
        transport_class.__call__ = original_call
        broker.register_job = original_register
        trace.update(state="preparing", restored_pause_at_utc=_utc())
        _atomic(marker, trace)
        broker.register_job(PREPARE_JOB, {MODEL: 0})
        preparation = helper["launch_worker"](broker, repo=REPO, job_id=PREPARE_JOB,
            action="quality-v2", stage="prepare", cohort="professional7", run_root=RUN_ROOT,
            worker_root=ROOT / "prepare-worker", max_new_calls=0)
        trace["active_worker"] = {"stage": "prepare", "pid": preparation.pid,
                                  "folder": str(ROOT / "prepare-worker")}
        _atomic(marker, trace)
        preflight = _wait(preparation, ROOT / "prepare-worker")
        if (preflight.get("full_grid_primary") != 21 or preflight.get("maximum_attempts", CAP+1) > CAP
                or preflight.get("ready") is not True):
            raise ValueError("source-only grading grid or budget is not ready")
        trace.update(state="grading", preflight=preflight)
        _atomic(marker, trace)
        broker.register_job(GRADE_JOB, {MODEL: CAP})
        grading = helper["launch_worker"](broker, repo=REPO, job_id=GRADE_JOB,
            action="quality-v2", stage="run", cohort="professional7", run_root=RUN_ROOT,
            worker_root=ROOT / "grade-worker", max_new_calls=CAP)
        trace["active_worker"] = {"stage": "run", "pid": grading.pid,
                                  "folder": str(ROOT / "grade-worker")}
        _atomic(marker, trace)
        summary = _wait(grading, ROOT / "grade-worker")
        output = RUN_ROOT / "professional7_quality_v2"
        receipts = _rows(output / "outcomes.jsonl")
        if any(row.get("status") != "received" for row in receipts):
            raise RuntimeError("unknown grading send retained; no automatic retry")
        trace.update(state="complete", completed_at_utc=_utc(), summary=summary,
                     received_grading_calls=len(receipts), relay_job=broker.snapshot()["jobs"][GRADE_JOB])
        _atomic(marker, trace)
        trace["export"] = _emit_export(output)
        _atomic(marker, trace)
    except Exception as error:
        trace.update(state="child_may_still_be_running" if isinstance(error, TimeoutError) else "stopped",
                     inspected_at_utc=_utc(), error_type=type(error).__name__,
                     automatic_retry=False, inspect_before_any_further_attempt=True)
        _atomic(marker, trace)
    print("CAPSTONE_PROFESSIONAL7_RECOVERY", json.dumps(trace, ensure_ascii=False))
    return trace
