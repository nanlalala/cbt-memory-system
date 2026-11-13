"""Separate format-only quality run after the original hosted workflow drains.

Does not interrupt, monkeypatch or alter the running v4 invocation. It waits for
that invocation to finish, prepares new full-grid protocols, and spends only
the actual remaining original CBT Qwen allowance. No Secret is read. Original
v2 results and the post-hoc readout remain separate, unchanged artifacts.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import uuid

VERSION = "cbt-quality-format-followup-v1"
MODEL = "qwen3.5-plus"
COHORTS = ("professional7", "cactus600")
ORIGINAL_QWEN_CAP, SEPARATE_P7_ALLOWANCE = 5032, 24
FILES = ("evaluation/architecture-rerun/cloud_quality_followup.py",
         "evaluation/architecture-rerun/quality_followup_worker.py",
         "evaluation/architecture-rerun/quality_span_judge_v3.py",
         "evaluation/architecture-rerun/hosted_relay.py")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def followup_root(prior_nonce):
    if type(prior_nonce) is not str or str(uuid.UUID(prior_nonce)) != prior_nonce:
        raise ValueError("canonical original invocation UUID required")
    return Path("/content") / ("capstone-quality-v3-" + prior_nonce)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_hash(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            result.update(block)
    return result.hexdigest()


def _event_rows(path):
    # Physical LF records, not str.splitlines(): source text may legally contain
    # Unicode paragraph separators. Keep the large transcript payloads streaming.
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("incomplete prior relay row")
            yield json.loads(line)


def reconciled_budget(plan, events, nonce):
    """Count every admitted v4 CBT Qwen intent, including failed sends.

    Old sources and prior attempts are already represented in the frozen v4
    finite plan. LoCoMo's separate allowance is never borrowed for CBT.
    """
    if (plan["original_cbt_envelope"][MODEL] != ORIGINAL_QWEN_CAP
            or plan["separate_professional7_quality_repair_allowance"] != SEPARATE_P7_ALLOWANCE):
        raise ValueError("original CBT Qwen allowance changed")
    old = plan["cbt_actual_local_paid_intents"][MODEL]
    prior = plan["previous_authorized_recovery_attempts_counted"][MODEL]
    base = plan["cbt_remaining_after_paid_intents"][MODEL]
    if (any(type(x) is not int or x < 0 for x in (old, prior, base))
            or base != ORIGINAL_QWEN_CAP + SEPARATE_P7_ALLOWANCE - old - prior):
        raise ValueError("original frozen paid census and allowance differ")
    intents, outcomes = {}, {}
    for row in events:
        if row.get("event") == "register":
            continue
        request = row["request_id"]
        if row["event"] == "intent":
            if request in intents:
                raise ValueError("duplicate prior relay intent")
            intents[request] = {k: row[k] for k in ("job_id", "model", "wire_sha256")}
        elif row["event"] == "outcome":
            if request in outcomes:
                raise ValueError("duplicate prior relay outcome")
            outcomes[request] = {k: row[k] for k in ("job_id", "model", "wire_sha256", "status")}
        else:
            raise ValueError("unrecognized prior relay event")
    if set(intents) != set(outcomes):
        raise RuntimeError("prior relay still has in-flight or unmatched requests")
    counts = Counter()
    for key, intent in intents.items():
        outcome = outcomes[key]
        if (outcome.get("status") != "received" or any(outcome[k] != intent[k]
                for k in ("job_id", "model", "wire_sha256"))):
            raise RuntimeError("prior invocation has an ambiguous or mismatched provider outcome")
        job = intent["job_id"]
        if not job.startswith(nonce + ":"):
            raise ValueError("foreign invocation in prior relay")
        if intent["model"] == MODEL and any(job.startswith(nonce + ":" + c + "-") for c in COHORTS):
            counts[MODEL] += 1
    remaining = base - counts[MODEL]
    if remaining < 0:
        raise ValueError("prior CBT requests already exceed the original allowance")
    return {"original_qwen_cap": ORIGINAL_QWEN_CAP, "separate_p7_allowance": SEPARATE_P7_ALLOWANCE,
        "old_local_paid_intents": old, "prior_paid_attempts": prior,
        "v4_cbt_qwen_admitted": counts[MODEL], "remaining_qwen": remaining,
        "locomo_allowance_borrowed": False}


def _terminal(prior):
    thread, process = prior.get("thread"), prior.get("process")
    if thread is not None and thread.is_alive():
        raise RuntimeError("original workflow is still active")
    if process is not None and process.poll() is None:
        raise RuntimeError("original worker is still active")
    if prior["status"]["status"] != "finished":
        raise RuntimeError("original invocation did not reach a terminal completed controller state")
    if _read(Path(prior["root"]) / "entry.json")["nonce"] != prior["nonce"]:
        raise ValueError("original invocation entry and parent state differ")
    if _read(Path(prior["root"]) / "status.json")["status"] != "finished":
        raise RuntimeError("original terminal status file differs")
    if Path(prior["relay"].endpoint).exists():
        raise RuntimeError("original relay has not closed and drained")


def _phase(state, cohort, stage, cap=0):
    repo, root = state["repo"], state["root"]
    worker = root / "workers" / (cohort + "-" + stage)
    worker.mkdir(parents=True, exist_ok=False)
    source = repo / "evaluation/architecture-rerun/quality_followup_worker.py"
    copied = worker / "quality_followup_worker.py"
    copied.write_bytes(source.read_bytes())
    job = state["nonce"] + ":" + cohort + "-quality-v3"
    manifest = {"repo": str(repo), "commit": state["commit"], "cohort": cohort, "stage": stage,
        "reader": str(state["prior_root"] / "studies" / cohort / "reader"),
        "grade": str(root / "studies" / cohort / "quality_v3"),
        "max_new_calls": cap, "job_id": job,
        "code_sha256": state["code_sha256"], "worker_sha256": sha256(copied.read_bytes()).hexdigest()}
    if stage == "run":
        state["relay"].register_job(job, {MODEL: cap})
        manifest["socket"] = state["relay"].endpoint
    path = worker / "manifest.json"
    _write(path, manifest)
    with (worker / "worker.log").open("ab", buffering=0) as stream:
        process = subprocess.Popen([sys.executable, "-B", str(copied), "--manifest", str(path)],
            cwd=worker, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            env=state["relay_api"]["sanitized_child_environment"](), start_new_session=True)
        state["process"] = process
        while process.poll() is None:
            time.sleep(1)
    state["process"] = None
    if process.returncode != 0 or not (worker / "complete.json").is_file():
        raise RuntimeError("quality follow-up phase failed; no automatic retry")
    result = _read(worker / "complete.json")
    if stage == "run":
        invocation = _read(Path(manifest["grade"]) / "last_invocation.json")
        if invocation.get("errors") or result.get("complete") is not True:
            raise RuntimeError("new quality phase has missing requests; no automatic retry")
    return result


def _status(state, status, **details):
    state["status"] = {"version": VERSION, "nonce": state["nonce"], "status": status,
        "at_utc": _utc(), **details}
    _write(state["root"] / "status.json", state["status"])
    print(json.dumps(state["status"]), flush=True)


def _controller(state, prior):
    try:
        thread = prior.get("thread")
        while thread is not None and thread.is_alive():
            thread.join(timeout=30)
        _terminal(prior)
        plan_path = state["prior_root"] / "finite_budget_plan.json"
        events_path = state["prior_root"] / "relay/events.jsonl"
        budget = reconciled_budget(_read(plan_path), _event_rows(events_path), prior["nonce"])
        budget["source_plan_sha256"] = _file_hash(plan_path)
        budget["source_events_sha256"] = _file_hash(events_path)
        _write(state["root"] / "original_budget_reconciliation.json", budget)
        plans = {}
        for cohort in COHORTS:
            _status(state, "preparing_separate_quality_protocol", cohort=cohort)
            plans[cohort] = _phase(state, cohort, "prepare")
        _write(state["root"] / "source_plans.json", plans)
        transport = prior["relay"].transports[MODEL]
        if (budget["v4_cbt_qwen_admitted"] < 1 or not callable(transport)
                or transport.identity.get("model") != MODEL
                or transport.identity.get("returned_model") != "qwen3.5-plus-2026-04-20"):
            raise RuntimeError("already initialized matching Qwen parent transport required")
        api = state["relay_api"]
        Base = api["Relay"]
        class FollowupRelay(Base):
            halted = False
            def _append(self, row):
                # Base relay invokes this while holding its admission mutex.
                if row.get("event") == "intent" and self.halted:
                    raise RuntimeError("new unknown halted further admission")
                if row.get("event") == "outcome" and row.get("status") == "unknown":
                    self.halted = True
                return super()._append(row)
        state["relay"] = FollowupRelay(state["root"] / "relay", {MODEL: transport})
        available, results = budget["remaining_qwen"], {}
        for cohort in COHORTS:
            needed = plans[cohort]["planned_judge_requests"]
            if (plans[cohort].get("ready") is not True or needed > available):
                results[cohort] = {"status": "not_started", "reason": "request_preflight_or_original_budget",
                    "new_calls": 0, "needed": needed, "remaining": available}
                continue
            _status(state, "grading_separate_full_grid", cohort=cohort, maximum_new_calls=needed)
            result = _phase(state, cohort, "run", needed)
            results[cohort] = result
            if state["relay"].halted:
                raise RuntimeError("new ambiguous provider result retained; no further cohort started")
            admitted = sum(v[MODEL]["admitted"] for v in state["relay"].snapshot()["jobs"].values())
            available = budget["remaining_qwen"] - admitted
            _write(state["root"] / "results.json", results)
        _write(state["root"] / "results.json", results)
        _status(state, "processing_finished", cohorts=results, remaining_original_qwen_allowance=available,
            interpretation="separate format-version ratings; retain coverage, never merge preferred grades into v2")
    except Exception as exc:
        _status(state, "stopped_without_retry", error_type=type(exc).__name__)
    finally:
        if state.get("relay") is not None and state.get("process") is None:
            state["relay"].close()


def schedule(prior_state, *, repo, expected_commit, root=None):
    """Schedule one new, bounded quality protocol after the unchanged old run.

    This call performs no paid requests itself. Returned state owns a background
    controller; re-evaluation starts only after original quiescence and a fresh
    actual-intent budget census. A pre-existing root is never automatically retried.
    """
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required")
    # One immutable root per original invocation prevents a second scheduler
    # from spending the same remaining allowance under an alternative path.
    expected_root = followup_root(prior_state["nonce"])
    repo, root = Path(repo).resolve(), expected_root if root is None else Path(root).resolve()
    if root != expected_root:
        raise ValueError("only the fixed follow-up root for this original invocation is allowed")
    if (any(not p.is_relative_to("/content") or p == Path("/content") for p in (repo, root))
            or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != expected_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip()):
        raise ValueError("new named hosted root and clean pinned checkout required")
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    state = {"repo": repo, "root": root, "commit": expected_commit, "nonce": str(uuid.uuid4()),
        "prior_root": Path(prior_state["root"]), "relay": None, "process": None,
        "code_sha256": {name: sha256((repo / name).read_bytes()).hexdigest() for name in FILES}}
    state["relay_api"] = runpy.run_path(str(repo / "evaluation/architecture-rerun/hosted_relay.py"))
    _write(root / "entry.json", {"version": VERSION, "nonce": state["nonce"], "at_utc": _utc(),
        "prior_root": str(state["prior_root"]), "prior_nonce": prior_state["nonce"],
        "repo": str(repo), "commit": expected_commit, "code_sha256": state["code_sha256"],
        "new_secret_reads": 0, "old_protocols_changed": False, "mab_accessed": False})
    _status(state, "waiting_for_original_workflow_to_drain")
    state["thread"] = threading.Thread(target=_controller, args=(state, prior_state),
        name="cbt-separate-quality-followup", daemon=False)
    state["thread"].start()
    return state
