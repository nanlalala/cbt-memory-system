"""Continue the exact drained C6 grid independently while old LoCoMo runs.

No active old worker, dispatcher or protocol is changed. Existing initialized
parent transports retain the shared four-call semaphore and own credentials.
The original finite CBT budget is debited for all prior and new paid attempts.
"""
from collections import Counter
from datetime import datetime,timezone
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

VERSION="cbt-terminal-length-case-recovery-v1"
ANSWER="deepseek-v4.1-flash";JUDGE="qwen3.5-plus"
COHORTS=("professional7","cactus600")
FILES=("evaluation/architecture-rerun/cloud_c6_terminal_recovery.py",
    "evaluation/architecture-rerun/c6_terminal_recovery_worker.py",
    "evaluation/architecture-rerun/quality_span_judge_v3.py",
    "evaluation/architecture-rerun/hosted_relay.py")


def _utc():return datetime.now(timezone.utc).isoformat()
def _read(path):return json.loads(Path(path).read_text())
def _write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n");temp.replace(path)
def _hash(path):
    result=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1_048_576),b""):result.update(block)
    return result.hexdigest()
def _events(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):break  # Only unrelated LoCoMo may still append.
            yield json.loads(line)


def remaining_cbt_budget(plan,events,nonce):
    """Only the already drained CBT jobs are paired; LoCoMo can be in flight."""
    expected={ANSWER:4805,JUDGE:5032}
    if plan["original_cbt_envelope"]!=expected or plan["separate_professional7_quality_repair_allowance"]!=24:
        raise ValueError("original finite CBT allowance differs")
    if any(type(plan[field][m]) is not int or plan[field][m]<0 for m in expected
            for field in ("cbt_actual_local_paid_intents","previous_authorized_recovery_attempts_counted","cbt_remaining_after_paid_intents")):
        raise ValueError("nonnegative integer original paid-attempt counters required")
    before={m:expected[m]+(24 if m==JUDGE else 0)
        -plan["cbt_actual_local_paid_intents"][m]-plan["previous_authorized_recovery_attempts_counted"][m]
        for m in expected}
    if before!=plan["cbt_remaining_after_paid_intents"]:
        raise ValueError("original paid-attempt budget arithmetic differs")
    intents,outcomes={},{}
    for row in events:
        if row.get("event")=="register":continue
        job=row.get("job_id","")
        if not any(job.startswith(nonce+":"+c+"-") for c in COHORTS):continue
        target=intents if row["event"]=="intent" else outcomes if row["event"]=="outcome" else None
        if target is None or row["request_id"] in target:raise ValueError("duplicate or foreign CBT parent row")
        target[row["request_id"]]={k:row[k] for k in ("job_id","model","wire_sha256")}
        if row["event"]=="outcome":target[row["request_id"]]["status"]=row["status"]
    if set(intents)!=set(outcomes):raise RuntimeError("CBT still has outstanding provider sends")
    counts=Counter()
    for key,intent in intents.items():
        outcome=outcomes[key]
        if outcome["status"]!="received" or any(outcome[k]!=intent[k] for k in intent):
            raise RuntimeError("CBT has ambiguous or mismatched parent outcomes")
        if intent["model"] not in expected:raise ValueError("unexpected CBT provider")
        counts[intent["model"]]+=1
    remaining={m:before[m]-counts[m] for m in expected}
    if any(type(v)is not int or v<0 for v in remaining.values()):raise ValueError("old CBT cost exceeds its allowance")
    return {"original_caps":expected,"separate_p7_quality_allowance":24,
        "old_local_paid_intents":plan["cbt_actual_local_paid_intents"],
        "prior_recovery_attempts_already_debited":plan["previous_authorized_recovery_attempts_counted"],
        "v4_cbt_parent_admissions":dict(counts),"available":remaining,
        "locomo_allowance_borrowed":False}


def _source_files(prior_root):
    study=Path(prior_root)/"studies/cactus600"
    return [study/part/name for part in ("reader","memory") for name in ("protocol.json","cases.json","intents.jsonl","outcomes.jsonl")]


def drained_c6(prior):
    root=Path(prior["root"]);nonce=prior["nonce"]
    if str(uuid.UUID(nonce))!=nonce or _read(root/"entry.json")["nonce"]!=nonce:
        raise ValueError("exact original parent invocation required")
    if prior["status"].get("cohorts",{}).get("cactus600",{}).get("status")!="stopped_without_retry":
        raise RuntimeError("old C6 controller has not recorded its drained stop")
    worker=root/"workers/cactus600-continuation"
    manifest_path=worker/"execution_manifest.json";manifest=_read(manifest_path)
    process=_read(worker/"process.json")
    if (manifest["nonce"]!=nonce or manifest["phase"]!="cactus600-continuation"
            or manifest.get("cohort")!="cactus600" or manifest.get("action")!="cbt-run"
            or manifest.get("run_root")!=str(root/"studies/cactus600")
            or process["manifest_sha256"]!=_hash(manifest_path)):
        raise ValueError("original C6 worker boundary differs")
    cmd=Path("/proc")/str(process["pid"])/"cmdline"
    if cmd.exists() and str(manifest_path).encode() in cmd.read_bytes():
        raise RuntimeError("old C6 worker remains active")
    invocation=_read(root/"studies/cactus600/reader/last_invocation.json")
    if (invocation.get("new_calls")!=485 or len(invocation.get("errors",[]))!=327
            or any(r.get("status")!="UnknownAttempt" for r in invocation["errors"])):
        raise ValueError("observed 485-call terminal C6 stop differs")
    complete=_read(worker/"complete.json")
    if complete.get("planned_cases")!=600 or complete.get("completed_cases")!=273:
        raise ValueError("observed drained 600-case source differs")
    return {"prior_nonce":nonce,"worker_manifest_sha256":_hash(manifest_path),
        "last_invocation_sha256":_hash(root/"studies/cactus600/reader/last_invocation.json"),
        "source_file_sha256":{str(p):_hash(p) for p in _source_files(root)},
        "old_source_complete_cases":273,"planned_cases":600,"old_active_worker_untouched":True}


def _source_unchanged(state):
    for path,expected in state["source_proof"]["source_file_sha256"].items():
        if _hash(path)!=expected:raise ValueError("drained C6 source changed")


def _status(state,status,**extra):
    value={"version":VERSION,"nonce":state["nonce"],"status":status,"at_utc":_utc(),**extra}
    state["status"]=value;_write(state["root"]/"status.json",value)
    print(json.dumps(value,allow_nan=False),flush=True)


def _phase(state,cohort,action,caps=None):
    caps=caps or {ANSWER:0,JUDGE:0};job=state["nonce"]+":"+cohort+"-"+action
    root=state["root"]/"workers"/(cohort+"-"+action);root.mkdir(parents=True,exist_ok=False)
    helper=state["repo"]/"evaluation/architecture-rerun/c6_terminal_recovery_worker.py"
    copied=root/"c6_terminal_recovery_worker.py";copied.write_bytes(helper.read_bytes())
    manifest={"repo":str(state["repo"]),"commit":state["commit"],"action":action,"cohort":cohort,
        "study_root":str(state["root"]/"studies"/cohort),
        "source":{name:str(state["prior_root"]/"studies"/cohort/name) for name in ("reader","memory")},
        "caps":caps,"job_id":job,"code_sha256":state["code_sha256"],
        "worker_sha256":sha256(copied.read_bytes()).hexdigest()}
    if action in {"generation","quality-run"}:
        state["relay"].register_job(job,caps);manifest["socket"]=state["relay"].endpoint
    path=root/"execution_manifest.json";_write(path,manifest)
    _status(state,"running",cohort=cohort,phase=action,phase_caps=caps)
    with (root/"worker.log").open("ab",buffering=0) as stream:
        process=subprocess.Popen([sys.executable,"-B",str(copied),"--manifest",str(path)],cwd=root,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            env=state["relay_api"]["sanitized_child_environment"](),start_new_session=True)
        state["process"]=process
        _write(root/"process.json",{"pid":process.pid,"manifest_sha256":_hash(path)})
        while process.poll() is None:time.sleep(1)
    state["process"]=None
    if process.returncode!=0 or not (root/"complete.json").is_file():
        raise RuntimeError("new C6 phase failed; no automatic retry")
    result=_read(root/"complete.json")
    if action in {"generation","quality-run"}:
        folder=state["root"]/"studies"/cohort/("reader" if action=="generation" else "quality_v3")
        if _read(folder/"last_invocation.json").get("errors"):
            raise RuntimeError("new phase retained errors; no automatic retry")
        if action=="generation":
            n=600 if cohort=="cactus600" else 7
            if result.get("completed_cases")!=n or result.get("unknown_receipts") or result.get("unbound_answer_files"):
                raise RuntimeError("fixed generation grid incomplete")
            if cohort=="professional7" and any(result["arms"][a]["received"]!=7 for a in result["arms"]):
                raise RuntimeError("zero-call P7 migration did not retain all 21 original answers")
        elif result.get("complete")is not True:raise RuntimeError("quality request grid incomplete")
    _source_unchanged(state)
    return result


def _spent(state):
    return {m:sum(job.get(m,{}).get("admitted",0) for job in state["relay"].snapshot()["jobs"].values())
        for m in (ANSWER,JUDGE)}


def _controller(state,prior):
    try:
        source_plan=_phase(state,"cactus600","source")
        budget_plan_path=state["prior_root"]/"finite_budget_plan.json"
        budget=remaining_cbt_budget(_read(budget_plan_path),_events(state["prior_root"]/"relay/events.jsonl"),prior["nonce"])
        budget["source_plan_sha256"]=_hash(budget_plan_path)
        _write(state["root"]/"original_budget_reconciliation.json",budget)
        need={ANSWER:source_plan["maximum_new_generation_requests"],
            JUDGE:source_plan["maximum_new_assertion_requests"]+source_plan["quality_after_complete_grid"]["with_independent_10_percent_repeats"]}
        if any(need[m]>budget["available"][m] for m in need):
            _status(state,"original_budget_insufficient",needed=need,available=budget["available"],new_calls=0);return
        api=state["relay_api"];Base=api["Relay"]
        class FiniteC6Relay(Base):
            admission_stopped=False
            def _append(self,row):
                if row.get("event")=="intent":
                    old_halted=prior["relay"].snapshot().get("admission_stopped",False)
                    if self.admission_stopped or old_halted:raise RuntimeError("new unknown stopped admission")
                    model=row["model"];admitted=sum(count for (job,m),count in self._admitted.items() if m==model)
                    if admitted>=budget["available"][model]:raise RuntimeError("original CBT model allowance exhausted")
                elif row.get("event")=="outcome" and row.get("status")=="unknown":self.admission_stopped=True
                return super()._append(row)
        transports={m:prior["relay"].transports[m] for m in (ANSWER,JUDGE)}
        if any(not callable(t) or t.identity["model"]!=m for m,t in transports.items()):
            raise RuntimeError("existing initialized matching parent transports required")
        state["relay"]=FiniteC6Relay(state["root"]/"relay",transports)
        generation=_phase(state,"cactus600","generation",{ANSWER:source_plan["maximum_new_generation_requests"],JUDGE:source_plan["maximum_new_assertion_requests"]})
        quality_plan=_phase(state,"cactus600","quality-source")
        if quality_plan.get("ready")is not True or quality_plan["planned_judge_requests"]>1980:
            raise RuntimeError("C6 quality source is not the original full-grid allowance")
        quality=_phase(state,"cactus600","quality-run",{ANSWER:0,JUDGE:quality_plan["planned_judge_requests"]})
        results={"cactus600":{"generation":generation,"quality_v3":quality}}
        _write(state["root"]/"results.json",results)
        # Migrate P7 only through exact cached receipts under the new code hash.
        _phase(state,"professional7","source")
        migrated=_phase(state,"professional7","generation",{ANSWER:0,JUDGE:0})
        p7_plan=_phase(state,"professional7","quality-source")
        remaining=budget["available"][JUDGE]-_spent(state)[JUDGE]
        if p7_plan.get("ready")is True and p7_plan["planned_judge_requests"]==24 and remaining>=24:
            p7_quality=_phase(state,"professional7","quality-run",{ANSWER:0,JUDGE:24})
            results["professional7"]={"zero_call_migration":migrated,"quality_v3":p7_quality}
        else:
            results["professional7"]={"status":"quality_not_started","zero_call_migration":migrated,
                "needed":p7_plan["planned_judge_requests"],"remaining_original_qwen_allowance":remaining}
        _write(state["root"]/"results.json",results)
        _status(state,"processing_finished",cohorts=results,model_admissions=_spent(state),mab_changed=False,
            interpretation="new protocol, original answers/roster, automatic scores with missingness; no improvement assumed")
    except Exception as exc:_status(state,"stopped_without_retry",error_type=type(exc).__name__)
    finally:
        if state.get("relay")is not None and state.get("process")is None:state["relay"].close()


def start(prior_state,*,repo,expected_commit):
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab")is not None):raise RuntimeError("genuine hosted Colab required")
    repo=Path(repo).resolve()
    if (not repo.is_relative_to("/content") or subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()!=expected_commit
            or subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip()):
        raise ValueError("clean independently pinned hosted checkout required")
    proof=drained_c6(prior_state)
    root=Path("/content")/("capstone-c6-terminal-v1-"+prior_state["nonce"])
    if root.exists():return _read(root/"status.json")
    root.mkdir(mode=0o700,exist_ok=False)
    state={"root":root,"repo":repo,"commit":expected_commit,"nonce":str(uuid.uuid4()),
        "prior_root":Path(prior_state["root"]),"source_proof":proof,"relay":None,"process":None,
        "code_sha256":{name:_hash(repo/name) for name in FILES}}
    state["relay_api"]=runpy.run_path(str(repo/"evaluation/architecture-rerun/hosted_relay.py"))
    _write(root/"entry.json",{"version":VERSION,"nonce":state["nonce"],"at_utc":_utc(),"repo":str(repo),
        "commit":expected_commit,"source_proof":proof,"code_sha256":state["code_sha256"],
        "new_secret_reads":0,"old_worker_or_protocol_changed":False,"mab_accessed":False})
    _status(state,"source_preparation_zero_paid_calls")
    state["thread"]=threading.Thread(target=_controller,args=(state,prior_state),name="cbt-c6-terminal-recovery",daemon=False)
    state["thread"].start()
    return state
