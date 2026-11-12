"""Independent R1--R5 judging using the existing rubric and exact alias receipts."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .runner import (Journal, UnknownAttempt, CallLimit, atomic, code_hashes, digest,
                     lock, rows, utc, require_hosted)
from auto_eval_v2 import quality
from auto_eval_v2.evidence import _cluster_interval


def prepare(packets,identity,folder,*,maximum_attempts,fixture_only=False):
    if not fixture_only:
        require_hosted()
    if set(identity)!={"provider","model","returned_model","model_family"}:
        raise ValueError("explicit requested and returned judge identity required")
    narrow={k:identity[k] for k in ("provider","model","model_family")}
    plan=quality.plan_quality(packets,narrow,repeat_fraction=0.1,seed=20261113)
    for row in plan:
        row["payload"]={"model":identity["model"],"messages":row["messages"],"temperature":0,
                        "max_tokens":2048,"thinking":{"type":"disabled"}}
        if len(json.dumps(row["payload"]))>48000:
            raise ValueError("complete judge request exceeds frozen cap")
    folder=Path(folder).resolve()
    if not fixture_only and (not folder.is_relative_to("/content") or folder==Path("/content")):
        raise ValueError("new hosted judge directory required")
    folder.mkdir(exist_ok=False,parents=True)
    protocol={"version":"cbt-v4-independent-quality-v1","provider":identity,"fixture_only":fixture_only,
              "maximum_attempts":maximum_attempts,"request_limits":{"answer":48000,"preparation":48000},"cases_sha256":digest(packets),"plan_sha256":digest(plan),
              "code_sha256":code_hashes(),"created_at_utc":utc(),"repeat_fraction":0.1,
              "authorization_reference":"user_requested_full_architecture_rerun"}
    protocol["protocol_sha256"]=digest(protocol)
    atomic(folder/"cases.json",packets)
    atomic(folder/"plan.json",plan)
    atomic(folder/"protocol.json",protocol)
    return {"planned_primary":len(packets),"planned_including_repeats":len(plan),
            "maximum_attempts":maximum_attempts,"request_limits":{"answer":48000,"preparation":48000},"protocol_sha256":protocol["protocol_sha256"]}


def run(folder,transport,*,max_new_calls,workers=4,fixture_only=False):
    if not fixture_only:
        require_hosted()
    folder=Path(folder)
    protocol=json.loads((folder/"protocol.json").read_text())
    packets=json.loads((folder/"cases.json").read_text())
    plan=json.loads((folder/"plan.json").read_text())
    if (protocol["code_sha256"]!=code_hashes() or digest(packets)!=protocol["cases_sha256"] or
        digest(plan)!=protocol["plan_sha256"] or protocol["fixture_only"]!=fixture_only):
        raise ValueError("judge code or frozen inputs differ")
    if fixture_only and not getattr(transport,"fixture_only",False):
        raise ValueError("fixture transport required")
    packet_by_id={p["case_id"]:p for p in packets}
    with lock(folder):
        journal=Journal(folder,protocol,transport,max_new_calls)
        reviewed={r["item_id"] for r in rows(folder/"reviews.jsonl")}
        def one(item):
            purpose="quality" if item["repeat_of"] is None else "quality_repeat:"+item["item_id"]
            receipt=journal.call({"case_id":item["case_id"]},purpose,item["payload"])
            checked=quality.validate_review(receipt["text"],packet_by_id[item["case_id"]])
            row={"item_id":item["item_id"],"case_id":item["case_id"],"source_family":item["source_family"],
                 "repeat_of":item["repeat_of"],"review":checked,"request_key":receipt["key"]}
            with journal.mutex:
                journal.append("reviews.jsonl",row)
        failures=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(one,item):item for item in plan if item["item_id"] not in reviewed}
            for future in as_completed(pending):
                try:
                    future.result()
                except (CallLimit,UnknownAttempt) as exc:
                    journal.stop.set()
                    failures.append(type(exc).__name__)
                except Exception as exc:
                    journal.stop.set()
                    failures.append(type(exc).__name__)
        result=summarize(folder)
        atomic(folder/"summary.json",result)
        atomic(folder/"last_invocation.json",{"errors":failures,"new_calls":journal.new_calls,"at_utc":utc()})
        return result


def summarize(folder):
    folder=Path(folder)
    mapping=json.loads((folder/"controller_arm_mapping.json").read_text())
    reviews=rows(folder/"reviews.jsonl")
    primary={r["case_id"]:r for r in reviews if r["repeat_of"] is None}
    report={"planned_primary":len(mapping),"reviewed_primary":len(primary),"dimensions":{},
            "interpretation":"independent automatic response assessment, not clinical efficacy"}
    arms=sorted({r["arm"] for r in mapping})
    grouped={}
    for row in mapping:
        grouped.setdefault(row["case_id"],{})[row["arm"]]=row
    for dimension in quality.DIMENSIONS:
        report["dimensions"][dimension]={}
        for arm in arms:
            values=[]
            for row in mapping:
                review=primary.get(row["packet_id"],{}).get("review",{})
                if row["arm"]==arm and review.get("status")=="completed":
                    v=review["assessment"][dimension]
                    values.append(int(v=="pass") if dimension=="R5" else v)
            report["dimensions"][dimension][arm]={"scored":len(values),"planned":sum(r["arm"]==arm for r in mapping),
                "mean_or_safety_pass_rate":sum(values)/len(values) if values else None}
    report["comparisons"]={}
    for left,right in (("raw_history_rag","managed_memory_rag"),):
        dimensions={}
        for dimension in quality.DIMENSIONS:
            differences=[]
            for pair in grouped.values():
                a,b=pair.get(left),pair.get(right)
                if not a or not b:
                    continue
                ra=primary.get(a["packet_id"],{}).get("review",{})
                rb=primary.get(b["packet_id"],{}).get("review",{})
                if ra.get("status")!="completed" or rb.get("status")!="completed":
                    continue
                va,vb=ra["assessment"][dimension],rb["assessment"][dimension]
                delta=int(vb=="pass")-int(va=="pass") if dimension=="R5" else vb-va
                differences.append((a["source_family"],delta))
            dimensions[dimension]={"planned_pairs":len(grouped),"scored_pairs":len(differences),
                "managed_wins":sum(d>0 for _,d in differences),"ties":sum(d==0 for _,d in differences),
                "raw_wins":sum(d<0 for _,d in differences),
                "family_cluster_difference":_cluster_interval(differences,draws=1000,seed=20261113)}
        report["comparisons"][left+"__"+right]=dimensions
    by_item={r["item_id"]:r for r in reviews}
    repeats=[]
    for r in reviews:
        if r["repeat_of"] is None:
            continue
        other=by_item.get(r["repeat_of"],{})
        if r["review"].get("status")=="completed" and other.get("review",{}).get("status")=="completed":
            repeats.append({d:abs(r["review"]["assessment"][d]-other["review"]["assessment"][d])
                if d!="R5" else int(r["review"]["assessment"][d]!=other["review"]["assessment"][d]) for d in quality.DIMENSIONS})
    report["repeat_consistency"]={"paired_repeats":len(repeats),"mean_absolute_difference_or_safety_disagreement":{
        d:sum(r[d] for r in repeats)/len(repeats) if repeats else None for d in quality.DIMENSIONS}}
    return report
