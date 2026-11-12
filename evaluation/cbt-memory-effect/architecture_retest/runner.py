"""Hosted, resumable full-roster natural CBT continuation replication.

No import reads a credential or invokes a model. A new run preserves all old
experiments. The notebook supplies the approved attempt cap and live transport.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import threading
import time

from . import VERSION
from . import projections as projection
from .retrieval import DENSE_MODEL, DENSE_REVISION, DenseRetriever, rank

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import cactus_materials as materials
from auto_eval_v2 import quality
from auto_eval_v2.run_colab import require_hosted

ARMS = ("no_rag", "raw_history_rag", "managed_memory_rag")
MODEL = "deepseek-v4.1-flash"
RETURNED_MODEL = "deepseek-v4-1-flash-260910"
IDENTITY = {"provider": "https://tokendance.space/gateway/v1", "model": MODEL,
            "returned_model": RETURNED_MODEL, "model_family": "deepseek"}
SETTINGS = {"recent_messages": 6, "primary_sources": 6, "top_k_sources": 9, "max_answer_chars": 24000,
            "max_preparation_chars": 48000, "answer_output_tokens": 1024,
            "preparation_output_tokens": 4096, "dense_model": DENSE_MODEL,
            "dense_revision": DENSE_REVISION, "clock_mode": "ordinal-only-v1"}


def code_hashes():
    files = list(HERE.glob("*.py")) + [HERE.parent/"cactus_materials.py",
        HERE.parent/"auto_eval_v2/quality.py",HERE.parent/"auto_eval_v2/evidence.py",
        ROOT/"modules/agent-integration/cbt_projection_contract.py",
        ROOT/"modules/agent-integration/cbt_projection.py",
        ROOT/"modules/agent-integration/cbt_retrieval.py",
        ROOT/"modules/agent-integration/cbt_context.py",
        ROOT/"modules/agent-integration/cbt_counselor_policy.py",
        ROOT/"modules/agent-integration/evidence_bundles.py"]
    return {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}


def payload(messages, purpose):
    return {"model": MODEL, "messages": messages, "temperature": 0,
            "max_tokens": SETTINGS["answer_output_tokens" if purpose=="answer" else "preparation_output_tokens"],
            "thinking": {"type":"disabled"}}


from .journal import (Journal, UnknownAttempt, CallLimit, InvalidReceipt, atomic, canonical, digest, lock, rows, utc)

def source_views(case):
    uid=case["source_family_id"]
    return [{"source_id":r.get("source_id",f"{uid}:{r['message_id']}"),"source_order":r["source_order"],
             "role":r["role"],"content":r["content"],"subject_id":uid,
             **({"date":r["date"]} if case.get("calendar_time_available") and r.get("date") is not None else {})}
            for r in case["history"]]


def _api():
    sys.path.insert(0,str(ROOT/"modules/agent-integration"))
    import cbt_context
    if Path(cbt_context.__file__).resolve()!=(ROOT/"modules/agent-integration/cbt_context.py").resolve():
        raise RuntimeError("foreign cached context module; use a fresh process")
    from evidence_bundles import EvidenceSpan
    return cbt_context,EvidenceSpan


def assemble(case, sources, selected_ids, items, representation, *, quotation_evidence=False):
    api,Span=_api()
    uid=case["source_family_id"]
    by_id={s["source_id"]:s for s in sources}
    recent=sources[-SETTINGS["recent_messages"]:]
    registry={s["source_id"]:dict(user_id=uid,source_id=s["source_id"],text=s["content"],role=s["role"],
              source_time=s.get("date"),source_order=s["source_order"],
              permissions={"save":True,"long_term_use":True,"response_external":True}) for s in sources}
    registry["CURRENT"]=dict(user_id=uid,source_id="CURRENT",text=case["input"],role="user",
        source_time=case.get("input_date") if case.get("calendar_time_available") else None,
        source_order=case["history_cutoff_order_exclusive"],
        permissions={"save":True,"long_term_use":True,"response_external":True})
    contextual=[]
    for item in items:
        evidence=tuple(Span(e["source_id"],item["item_id"]+":"+str(i),e["text"],e["start"],e["end"],by_id[e["source_id"]].get("date"),
                            by_id[e["source_id"]]["role"]) for i,e in enumerate(item["evidence"]))
        contextual.append(api.ContextualItem(item_id=item["item_id"],section=item["section"],value=item["value"],
            evidence=evidence,origin=item["origin"],support_status="source_supported" if item["origin"]!="agent_inference" else "hypothesis",
            verifier_revision=item.get("verifier_revision",projection.VERIFIER_VERSION),version=1,
            source_scope={"kind":"dated_historical_reports" if case.get("calendar_time_available") else "historical_report_at_ordinal_cutoff","cutoff_order":case["history_cutoff_order_exclusive"]-1,
                          "calendar_date":None}))
    options={"model":MODEL,"temperature":0,"thinking":{"type":"disabled"}}
    bundles=[]
    if quotation_evidence:
        from evidence_bundles import EvidenceBundle, EvidenceClaim
        for sid in selected_ids:
            source=by_id[sid]
            span=Span(sid,"quote:"+sid,source["content"],0,len(source["content"]),None,source["role"])
            claim=EvidenceClaim("quote:"+sid,"history_quote",source["content"],(span,),
                support_status="source_quote",origin="assistant_proposal" if source["role"]=="assistant" else "explicit_user_statement")
            bundles.append(EvidenceBundle("quote:"+sid,(claim,)))
    return api.build_cbt_context([{"role":"system","content":api.CBT_POLICY},{"role":"user","content":case["input"]}],
        bundles=bundles,
        source_registry=registry,subject_id=uid,max_outbound_chars=SETTINGS["max_answer_chars"],
        max_response_tokens=SETTINGS["answer_output_tokens"],request_metadata=options,
        representation=representation,contextual_items=contextual,
        recent_messages=[{"role":s["role"],"content":s["content"]} for s in recent],
        recent_source_ids=[s["source_id"] for s in recent],selected_source_ids=selected_ids,current_source_id="CURRENT")


def _package(package):
    return {"payload":package.request_payload,"trace":package.trace}


def source_preflight(cases):
    """Size actual known source requests without loading embeddings or a model."""
    _api()
    from cbt_projection import extraction_messages
    preflight=[]
    for case in cases:
        sources=source_views(case)
        originals=[dict(source_id=s["source_id"],text=s["content"],role=s["role"],source_time=s.get("date")) for s in sources]
        extraction=payload(extraction_messages(originals),"extraction")
        extraction_chars=len(json.dumps(extraction))
        mandatory=None
        try:
            mandatory=assemble(case,sources,[],[],"raw").char_count
        except ValueError:
            pass
        preflight.append({"case_id":case["case_id"],"extraction_request_chars":extraction_chars,
            "extraction_within_cap":extraction_chars<=SETTINGS["max_preparation_chars"],
            "mandatory_answer_request_chars":mandatory,"mandatory_answer_within_cap":mandatory is not None,
            "source_messages":len(sources),"scope":"known full source-only and mandatory-recent requests; later payloads checked when constructed"})
    return preflight


def prepare_case_requests(case,sources,items,dense,dense_mutex):
    older=sources[:-SETTINGS["recent_messages"]]
    query=case["input"]
    with dense_mutex:
        ranking=rank(query,older,dense)
    selected=[]
    raw_rejections=[]
    for index in ranking:
        candidate=[*selected,older[index]["source_id"]]
        if len(candidate)>SETTINGS["primary_sources"]:
            raw_rejections.append({"source_id":older[index]["source_id"],"phase":"primary","reason":"primary_source_limit"})
            continue
        try:
            assemble(case,sources,candidate,[],"raw")
            selected=candidate
        except ValueError as exc:
            raw_rejections.append({"source_id":older[index]["source_id"],"phase":"primary","reason":type(exc).__name__})
            continue
    # Neighbor expansion protects immediately adjacent questions/negation context.
    for index in ranking:
        for near in (index-1,index+1):
            if 0<=near<len(older) and older[near]["source_id"] not in selected and len(selected)<SETTINGS["top_k_sources"]:
                candidate=[*selected,older[near]["source_id"]]
                try:
                    assemble(case,sources,candidate,[],"raw")
                    selected=candidate
                except ValueError as exc:
                    raw_rejections.append({"source_id":older[near]["source_id"],"phase":"neighbor","reason":type(exc).__name__})
                    continue
    results={"raw_history_rag":_package(assemble(case,sources,selected,[],"raw")),
             "no_rag":_package(assemble(case,sources,[],[],"raw"))}
    recent_ids={s["source_id"] for s in sources[-SETTINGS["recent_messages"]:]}
    with dense_mutex:
        item_rank=rank(query,[{"content":projection.canonical(i["value"])} for i in items],dense)
    managed=[]
    managed_rejections=[]
    for index in item_rank:
        trial=[*managed,items[index]]
        dependencies=list(dict.fromkeys(e["source_id"] for item in trial for e in item["evidence"] if e["source_id"] not in recent_ids))
        try:
            assemble(case,sources,dependencies,trial,"processed_with_evidence")
            managed=trial
        except ValueError as exc:
            managed_rejections.append({"item_id":items[index]["item_id"],"reason":type(exc).__name__,
                                       "dependency_source_ids":dependencies})
            continue
    dependencies=list(dict.fromkeys(e["source_id"] for item in managed for e in item["evidence"] if e["source_id"] not in recent_ids))
    results["managed_memory_rag"]=_package(assemble(case,sources,dependencies,managed,"processed_with_evidence"))
    for name,row in results.items():
        row.update(arm=name,case_id=case["case_id"],source_family=case["source_family_id"])
        row["selection_audit"]={"query_sha256":sha256(query.encode()).hexdigest(),
            "raw_ranked_source_ids":[older[i]["source_id"] for i in ranking],
            "raw_selected_source_ids":selected,"raw_selection_attempt_rejections":raw_rejections,
            "raw_omitted_source_ids":sorted({s["source_id"] for s in older}-set(selected)),
            "managed_ranked_item_ids":[items[i]["item_id"] for i in item_rank],
            "managed_selected_item_ids":[item["item_id"] for item in managed],
            "managed_selection_rejections":managed_rejections,
            "condition":"recent_only" if name=="no_rag" else "managed_selection" if name=="managed_memory_rag" else "raw_history_selection"}
    return results


def prepare(source,folder,*,family_limit=200,fixture_only=False,family_ids=None):
    if not fixture_only:
        require_hosted()
    source,folder=Path(source).resolve(),Path(folder).resolve()
    if not fixture_only and (not folder.is_relative_to("/content") or folder==Path("/content")):
        raise ValueError("hosted /content experiment directory required")
    data=source.read_bytes()
    if not fixture_only and (len(data)!=materials.SOURCE_BYTES or sha256(data).hexdigest()!=materials.SOURCE_SHA256):
        raise ValueError("released source identity mismatch")
    released_rows=json.loads(data)
    original_indices=None
    if family_ids is not None:
        if (type(family_ids) is not list or len(family_ids)!=family_limit or
            len(set(family_ids))!=family_limit or any(type(v) is not str for v in family_ids)):
            raise ValueError("exact distinct frozen source-family roster required")
        groups,rejected=materials.group_rows(released_rows)
        by_id={g["family_id"]:g for g in groups}
        if rejected or set(family_ids)-set(by_id):
            raise ValueError("frozen source families do not bind to the checked released corpus")
        original_indices=sorted(i for fid in family_ids for i in by_id[fid]["members"])
        released_rows=[released_rows[i] for i in original_indices]
    compiled=materials.compile_rows(released_rows,family_limit=family_limit,fixture_only=fixture_only)
    if family_ids is not None and set(family_ids)!={c["source_family_id"] for c in compiled["generation"]}:
        raise ValueError("reconstructed source-family membership differs")
    if original_indices is not None:
        for row in compiled["controller"]:
            row["source_row_index"]=original_indices[row["source_row_index"]]
            row["family_member_rows"]=[original_indices[i] for i in row["family_member_rows"]]
    if len(compiled["generation"])!=family_limit*3:
        raise ValueError("full outcome-blind roster unavailable")
    folder.mkdir(parents=True,exist_ok=False)
    cases=compiled["generation"]
    preflight=source_preflight(cases)
    protocol={"version":VERSION,"fixture_only":fixture_only,"source_sha256":sha256(data).hexdigest(),
        "dataset":materials.DATASET,"dataset_revision":materials.REVISION if not fixture_only else None,
        "case_count":len(cases),"source_families":family_limit,"cases_sha256":digest(cases),
        "provider":IDENTITY,"settings":SETTINGS,"arms":list(ARMS),
        "maximum_attempts":len(cases)*(len(ARMS)+2),"request_limits":{"answer":24000,"preparation":48000},"authorization_reference":"user_requested_full_architecture_rerun",
        "quality_judge":"separate_explicit_non_deepseek_family_required",
        "family_roster_policy":"supplied_historical_family_ids" if family_ids is not None else "public_deterministic_compiler",
        "family_ids":sorted({c["source_family_id"] for c in cases}),
        "support_check":"separate automatic model assessment, not clinical or human confirmation",
        "source_preflight_sha256":digest(preflight),
        "code_sha256":code_hashes(),"created_at_utc":utc()}
    protocol["protocol_sha256"]=digest(protocol)
    atomic(folder/"cases.json",cases)
    atomic(folder/"source_inventory.json",compiled["controller"])
    atomic(folder/"preparation_diagnostics.json",compiled["diagnostics"])
    atomic(folder/"protocol.json",protocol)
    atomic(folder/"source_preflight.json",preflight)
    return protocol


def prepare_case_pool(cases,folder,source_identity,*,fixture_only=False):
    """Freeze a recovered seven/24-case source roster after hosted archive parsing.

    The caller supplies original source messages, not old assembled prompts or
    reference answers. Only the admitted generation fields are retained.
    """
    if not fixture_only:
        require_hosted()
    if type(cases) is not list or not cases or type(source_identity) is not dict or not source_identity:
        raise ValueError("nonempty recovered roster and archive/source identity required")
    clean=[]
    ids=set()
    for case in cases:
        cid=case["case_id"]
        if not isinstance(cid,str) or not cid or cid in ids or any(c in cid for c in ('/','\\')):
            raise ValueError("unique path-safe historical case ID required")
        ids.add(cid)
        uid=case["source_family_id"]
        if not isinstance(uid,str) or not uid or not isinstance(case["input"],str) or not case["input"].strip():
            raise ValueError("historical family and current input required")
        history=[]
        for index,row in enumerate(case["history"]):
            if row["role"] not in {"user","assistant"} or not isinstance(row["content"],str) or not row["content"]:
                raise ValueError("original user/assistant source messages required")
            history.append({"message_id":row.get("message_id",row.get("source_id",f"turn-{index:03d}")),
                "source_id":row.get("source_id",f"{uid}:turn-{index:03d}"),"source_order":row.get("source_order",index),
                "role":row["role"],"content":row["content"],"date":row.get("date")})
        if not history or len({r["source_id"] for r in history})!=len(history):
            raise ValueError("nonempty distinct original history required")
        orders=[r["source_order"] for r in history]
        if any(type(o) is not int or o<0 for o in orders) or orders!=sorted(set(orders)):
            raise ValueError("ordered unique nonnegative history positions required")
        available=case.get("calendar_time_available",False)
        if type(available) is not bool or (not available and any(r["date"] is not None for r in history)):
            raise ValueError("explicit source calendar availability required")
        row={"case_id":cid,"source_family_id":uid,"input":case["input"],"history":history,
            "history_cutoff_order_exclusive":max(orders)+1,"calendar_time_available":available,
            "input_date":case.get("input_date") if available else None,
            "source_kind":case.get("source_kind","recovered_frozen_cbt_source_case")}
        row["case_sha256"]=digest(row)
        clean.append(row)
    folder=Path(folder).resolve()
    if not fixture_only and (not folder.is_relative_to("/content") or folder==Path("/content")):
        raise ValueError("new hosted case-pool directory required")
    folder.mkdir(exist_ok=False,parents=True)
    preflight=source_preflight(clean)
    protocol={"version":VERSION,"fixture_only":fixture_only,"source_identity":source_identity,
        "case_count":len(clean),"source_families":len({c["source_family_id"] for c in clean}),
        "cases_sha256":digest(clean),"provider":IDENTITY,"settings":SETTINGS,"arms":list(ARMS),
        "maximum_attempts":len(clean)*(len(ARMS)+2),"request_limits":{"answer":24000,"preparation":48000},
        "authorization_reference":"user_requested_full_architecture_rerun",
        "family_roster_policy":"recovered_historical_case_pool","source_preflight_sha256":digest(preflight),
        "code_sha256":code_hashes(),"created_at_utc":utc()}
    protocol["protocol_sha256"]=digest(protocol)
    atomic(folder/"cases.json",clean)
    atomic(folder/"protocol.json",protocol)
    atomic(folder/"source_preflight.json",preflight)
    return protocol


def _case(case,folder,journal,dense,assessor=None):
    dest=folder/"cases"/case["case_id"]
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"completed.json").exists():
        return
    sources=source_views(case)
    items=[]
    failures=[]
    if assessor is not None:
        try:
            assessor.freeze_targets(case,sources)
        except ValueError as exc:
            atomic(dest/"completed.json",{"case_id":case["case_id"],
                "all_arms_preparation_failure":"independent_target_validation_failed",
                "error_type":type(exc).__name__})
            return
    try:
        _api()
        from cbt_projection import extraction_messages, projection_candidates, verification_messages, supported_projections
        product_sources=[dict(source_id=s["source_id"],text=s["content"],role=s["role"],source_time=s.get("date")) for s in sources]
        extracted=journal.call(case,"extraction",payload(extraction_messages(product_sources),"extraction"))
        candidates=projection_candidates(extracted["text"],product_sources)
        reasons={}
        if candidates:
            checked=journal.call(case,"support_check",payload(verification_messages(candidates,product_sources),"support_check"))
            items=supported_projections(checked["text"],candidates,product_sources,verifier_model=checked["provider_model"])
            _,reasons=projection.checked_items(checked["text"],candidates)
        atomic(dest/"projections.json",{"candidates":candidates,"accepted":items,"verdict_reasons":reasons,
            "verifier_version":projection.VERIFIER_VERSION,"automatic_assessment":True})
    except (ValueError,InvalidReceipt) as exc:
        failures.append({"stage":"memory_preparation","reason":type(exc).__name__})
    try:
        prepared=prepare_case_requests(case,sources,items,dense,journal.dense_mutex)
    except ValueError as exc:
        atomic(dest/"completed.json",{"case_id":case["case_id"],"all_arms_preparation_failure":type(exc).__name__})
        return
    answers=[]
    for arm in ARMS:
        row=prepared[arm]
        if arm=="managed_memory_rag" and any(f["stage"]=="memory_preparation" for f in failures):
            row={**row,"preparation_failure":"memory_preparation_failed"}
        atomic(dest/(arm+".request.json"),row)
        if "preparation_failure" in row:
            answers.append({"arm":arm,"case_id":case["case_id"],"status":"preparation_failed"})
            continue
        receipt=journal.call(case,"answer",row["payload"])
        answer={"arm":arm,"case_id":case["case_id"],"source_family":case["source_family_id"],
                "status":"received","request_key":receipt["key"],"response":receipt["text"],
                "provider_model":receipt["provider_model"],"usage":receipt["usage"]}
        atomic(dest/(arm+".answer.json"),answer)
        answers.append(answer)
        if assessor is not None:
            try:
                assessor.score_answer(case,sources,answer,row["trace"]["source_ids"])
            except ValueError as exc:
                failures.append({"stage":"independent_assertion_scoring","arm":arm,"reason":type(exc).__name__})
    atomic(dest/"completed.json",{"case_id":case["case_id"],"rows":answers,"preparation_failures":failures})


def load(folder,fixture_only=False):
    folder=Path(folder).resolve()
    protocol=json.loads((folder/"protocol.json").read_text())
    cases=json.loads((folder/"cases.json").read_text())
    if not fixture_only:
        require_hosted()
    if protocol["fixture_only"]!=fixture_only or protocol["version"]!=VERSION or protocol["cases_sha256"]!=digest(cases):
        raise ValueError("frozen source/protocol differs")
    if protocol["protocol_sha256"]!=digest({k:v for k,v in protocol.items() if k!="protocol_sha256"}):
        raise ValueError("protocol checksum differs")
    if protocol["code_sha256"]!=code_hashes() or protocol["settings"]!=SETTINGS:
        raise ValueError("frozen code/settings changed; prepare a new independent run")
    if digest(json.loads((folder/"source_preflight.json").read_text()))!=protocol["source_preflight_sha256"]:
        raise ValueError("frozen source-only preflight differs")
    return folder,protocol,cases


def run(folder,transport,*,max_new_calls,workers=4,dense=None,fixture_only=False,
        assessment_folder=None,assessment_transport=None,assessment_max_new_calls=0):
    folder,protocol,cases=load(folder,fixture_only)
    if type(max_new_calls) is not int or max_new_calls<0 or type(workers) is not int or not 1<=workers<=4:
        raise ValueError("explicit bounded invocation budget and workers required")
    if fixture_only and getattr(transport,"fixture_only",False) is not True:
        raise ValueError("fixture execution cannot send to a real provider")
    if not fixture_only and (assessment_folder is None or assessment_transport is None):
        raise ValueError("independent frozen pre-answer targets and assessor transport required")
    with lock(folder):
        journal=Journal(folder,protocol,transport,max_new_calls)
        assessor=None
        if assessment_folder is not None:
            from .memory_assessment import Assessor
            assessor=Assessor(assessment_folder,assessment_transport,max_new_calls=assessment_max_new_calls,
                              fixture_only=fixture_only)
            if assessor.protocol["cases_sha256"]!=protocol["cases_sha256"]:
                raise ValueError("independent assessor roster differs from answering roster")
        dense=dense if dense is not None else DenseRetriever()
        if not fixture_only and (getattr(dense,"model_id",None),getattr(dense,"revision",None))!=(DENSE_MODEL,DENSE_REVISION):
            raise ValueError("dense ranker identity differs from frozen E5 model/revision")
        failures=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(_case,c,folder,journal,dense,assessor):c["case_id"] for c in cases}
            for future in as_completed(futures):
                try:
                    future.result()
                except (UnknownAttempt,CallLimit) as exc:
                    journal.stop.set()
                    failures.append({"case_id":futures[future],"status":type(exc).__name__})
                except Exception as exc:
                    journal.stop.set()
                    failures.append({"case_id":futures[future],"status":"runner_error","error_type":type(exc).__name__})
                with journal.mutex:
                    atomic(folder/"progress.json",summarize(folder,fixture_only=fixture_only))
        summary=summarize(folder,fixture_only=fixture_only)
        atomic(folder/"last_invocation.json",{"new_calls":journal.new_calls,"errors":failures,"at_utc":utc()})
        if assessor is not None:
            atomic(Path(assessment_folder)/"summary.json",assessor.summarize())
        return summary


def summarize(folder,*,fixture_only=False):
    folder=Path(folder)
    protocol=json.loads((folder/"protocol.json").read_text())
    completed=[json.loads(p.read_text()) for p in (folder/"cases").glob("*/completed.json")]
    # Individual completed answers remain evidence even if a later arm stopped.
    arm_rows,unbound=verified_answers(folder)
    outcomes=rows(folder/"outcomes.jsonl")
    usage={key:sum(r.get("usage",{}).get(key,0) for r in outcomes if type(r.get("usage",{}).get(key,0)) is int)
           for key in ("prompt_tokens","completion_tokens","total_tokens")}
    return {"version":VERSION,"planned_cases":protocol["case_count"],"completed_cases":len(completed),
        "planned_source_families":protocol["source_families"],"arms":{arm:{"planned":protocol["case_count"],
         "received":sum(r["arm"]==arm and r["status"]=="received" for r in arm_rows)} for arm in ARMS},
        "attempts":len(rows(folder/"intents.jsonl")),"received_receipts":sum(r["status"]=="received" for r in outcomes),
        "unknown_receipts":sum(r["status"]=="unknown" for r in outcomes),"provider_usage":usage,
        "response_quality_scores_available":False,"currency_cost_verified":False,
        "unbound_answer_files":unbound}


def verified_answers(folder):
    """Count only files bound to a received request and its exact response."""
    folder=Path(folder)
    intents={r["key"]:r for r in rows(folder/"intents.jsonl")}
    outcomes={r["key"]:r for r in rows(folder/"outcomes.jsonl")}
    accepted,unbound=[],[]
    for path in (folder/"cases").glob("*/*.answer.json"):
        answer=json.loads(path.read_text())
        key=answer.get("request_key")
        receipt=outcomes.get(key,{})
        intent=intents.get(key,{})
        request_path=path.with_name(path.name.replace('.answer.json','.request.json'))
        request=json.loads(request_path.read_text()) if request_path.exists() else {}
        if (receipt.get("status")!="received" or receipt.get("text")!=answer.get("response") or
            receipt.get("provider_model")!=answer.get("provider_model") or
            request.get("payload")!=intent.get("payload") or
            answer.get("case_id")!=path.parent.name or answer.get("arm")!=path.name.removesuffix('.answer.json')):
            unbound.append(str(path.relative_to(folder)))
            continue
        accepted.append(answer)
    return accepted,unbound


def export_quality(folder,judge_identity,output,*,maximum_attempts,fixture_only=False):
    from . import judge
    folder,protocol,cases=load(folder,fixture_only)
    packets=[]
    mapping=[]
    received,_=verified_answers(folder)
    answers={(r["case_id"],r["arm"]):r for r in received}
    for case in cases:
        dest=folder/"cases"/case["case_id"]
        history=[{"id":s["source_id"],"role":s["role"],"content":s["content"]} for s in source_views(case)]
        for arm in ARMS:
            answer=answers.get((case["case_id"],arm))
            if answer is None:
                mapping.append({"case_id":case["case_id"],"arm":arm,"packet_id":None,"status":"not_received"})
                continue
            opaque="response-"+digest([case["case_id"],arm])[:32]
            packets.append({"case_id":opaque,"source_family":case["source_family_id"],"answer_model_family":"deepseek",
                            "history":history,"input":case["input"],"response":answer["response"]})
            mapping.append({"case_id":case["case_id"],"source_family":case["source_family_id"],"arm":arm,"packet_id":opaque,"status":"received"})
    result=judge.prepare(packets,judge_identity,output,maximum_attempts=maximum_attempts,
                         fixture_only=fixture_only)
    atomic(Path(output)/"controller_arm_mapping.json",mapping)
    return result
