"""Dated 24-case product-rule replay with the original direct JSON metric.

This is a development mechanism regression, not natural CBT or clinical
effectiveness. Full source handling and all answer calls require hosted Colab.
The old input cases and literal targets are imported rather than reconstructed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for relative in ("modules/agent-integration","modules/long-term-memory/schema",
                 "modules/long-term-memory/storage","modules/long-term-memory/extraction",
                 "modules/long-term-memory/lifecycle","modules/long-term-memory/retrieval",
                 "evaluation/cbt-memory-effect"):
    sys.path.insert(0,str(ROOT/relative))
import controlled_replay_adapter as replay
from cbt_context import build_cbt_context
from architecture_retest.journal import Journal, atomic, digest, lock, rows, utc
from architecture_retest.transport import require_hosted

VERSION="controlled24-product-context-v4"
ARMS=("no_rag","raw_history_rag","managed_memory_rag")
IDENTITY={"provider":"https://tokendance.space/gateway/v1","model":"deepseek-v4.1-flash",
          "returned_model":"deepseek-v4-1-flash-260910","model_family":"deepseek"}
MAX_CHARS,MAX_OUTPUT,RECENT,SOURCE_CAP=24000,64,6,9
LEGACY_MAX_CHARS=6000
PERMISSIONS={k:True for k in ("save","long_term_use","response_external","extraction_external")}
INSTRUCTION=("Answer the current factual question from the explicit dated user reports. "
 "You may reconstruct current reported state from original quotations. "
 "A later explicit correction replaces the matching earlier value. Cumulative totals are not "
 "separate attempts and repeated totals must not be added. A plan/proposal is not agreement. "
 "Reported progress does not establish benefit. Return exactly one JSON object with the single "
 "key answer: an integer for counts, a lowercase status word for outcome, or null if unknown. "
 "Do not include explanations, markdown, source IDs or additional keys.")
OLD_SCORER_SHA256="b4dc236135c421e2451afd2b264adcd24c52fd16a2a305e458e3ade9107d12ed"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    files=[Path(__file__),HERE/"formal_full.py",HERE/"formal_sources.py",
           ROOT/"evaluation/cbt-memory-effect/controlled_replay_adapter.py"]
    files += sorted((ROOT/"modules").rglob("*.py"))
    files += [ROOT/"evaluation/cbt-memory-effect/architecture_retest"/n for n in ("journal.py","transport.py")]
    return {str(path.relative_to(ROOT)):file_hash(path) for path in files}


def strict_answer(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError("duplicate_answer_key")
            result[key]=value
        return result
    value=json.loads(text,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite_answer")))
    if type(value) is not dict or set(value)!={"answer"}:
        raise ValueError("single_answer_key_required")
    answer=value["answer"]
    if answer is not None and type(answer) not in (int,str):
        raise ValueError("invalid_answer_type")
    return answer


def strict_result(text,expected):
    try:
        value=strict_answer(text)
    except (ValueError,TypeError):
        return {"correct":False,"reason":"malformed_output","observed_answer":None}
    correct=type(value) is type(expected) and value==expected
    reason=("correct" if correct else "unsupported_assertion" if expected is None else
            "incorrect_abstention" if value is None else "output_type_error" if type(value) is not type(expected) else "wrong_value")
    return {"correct":correct,"reason":reason,"observed_answer":value}


def prepare_one(case,directory,*,fixture_only=False):
    if not fixture_only:
        require_hosted()
    config=replay.ReplayConfig(max_outbound_chars=MAX_CHARS,max_response_tokens=MAX_OUTPUT,
                              max_history_turns=64,maximum_response_calls=3)
    replay.validate_case(case,config)
    m=replay._modules()
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    database=directory/"state.sqlite"
    uid="case-"+replay.stable(case["case_id"])[:20]
    session=m.ControlledDialogueSession.open(database,approved_root=directory,user_id=uid,
        run_id=VERSION,session_id=uid+":released-session",provider=replay._ContextOnlyProvider(),background=False,
        permissions=PERMISSIONS,limits=m.PrototypeLimits(max_outbound_chars=MAX_CHARS,
            max_response_tokens=MAX_OUTPUT,max_worker_jobs_per_turn=1,max_session_turns=65),
        durable_limits=m.DurableLimits(max_run_calls=3,max_session_calls=3,max_queue_length=1,max_batch_size=1))
    source_orders={}
    receipts=[]
    try:
        for index,turn in enumerate(case["history"]):
            source=replay._source(session,turn,uid,PERMISSIONS,config=config)
            source_orders[source["source_id"]]=turn.get("source_order",index)
            receipt,_,_,_=replay._commit(session,source,enqueue=False,temporal_semantics=True)
            receipts.append({"source_id":source["source_id"],"source_sha256":digest(source),"receipt":receipt})
        current=replay._source(session,{"message_id":case["input_message_id"],"role":"user","content":case["input"],
            "date":case["input_date"],"source_order":len(case["history"])},uid,PERMISSIONS,config=config)
        source_orders[current["source_id"]]=len(case["history"])
        current_receipt,task,events,_=replay._commit(session,current,enqueue=False,temporal_semantics=True)
        if task and events and current_receipt["source_status"]=="committed":
            task={"object_id":task["object_id"],**session.durable.get_object_state(uid,task["object_id"])}
        stamp=case["input_date"]
        paused,_=session._paused_source_ids(stamp)
        pool=[s for s in session.durable.list_sources(uid,as_of=stamp) if s["source_id"] not in paused]
        registry={s["source_id"]:{**s,"source_order":source_orders[s["source_id"]]} for s in pool}
        registry[current["source_id"]]={**current,"source_order":source_orders[current["source_id"]]}
        history=[s for s in pool if s["source_id"]!=current["source_id"] and
                 all(s["permissions"].get(k) is True for k in ("save","long_term_use","response_external"))]
        history.sort(key=lambda s:source_orders[s["source_id"]])
        recent=history[-RECENT:]
        omitted_recent=[]
        base=[{"role":"system","content":m.SYSTEM+"\n"+INSTRUCTION+"\nAs of: "+stamp},
              {"role":"user","content":case["input"]}]
        metadata={"model":IDENTITY["model"],"temperature":0,"thinking":{"type":"disabled"}}
        def pack(bundles=(),*,selected=None,representation="processed_with_evidence",maximum=MAX_CHARS):
            return build_cbt_context(base,bundles,source_registry=registry,subject_id=uid,
                max_outbound_chars=maximum,max_response_tokens=MAX_OUTPUT,request_metadata=metadata,
                representation=representation,recent_messages=[{"role":s["role"],"content":s["text"]} for s in recent],
                recent_source_ids=[s["source_id"] for s in recent],current_source_id=current["source_id"],
                selected_source_ids=selected)
        try:
            pack(selected=[],representation="raw",maximum=LEGACY_MAX_CHARS)
            legacy_mandatory_fits=True
        except ValueError:
            legacy_mandatory_fits=False
        while True:
            try:
                pack(selected=[],representation="raw")
                break
            except ValueError:
                if not recent:
                    raise
                omitted_recent.append(recent.pop(0)["source_id"])
        recent_ids={s["source_id"] for s in recent}
        older=[s for s in history if s["source_id"] not in recent_ids]
        rawsearch=m.StableSourceRetriever(history).search(case["input"],user_id=uid,source_cutoff=stamp,as_of=stamp,
            for_external_response=True,allowed_source_ids={s["source_id"] for s in older},top_k=SOURCE_CAP)
        selected=[]
        rejected={}
        for hit in rawsearch.hits:
            if hit.source_id in selected:
                continue
            trial=[*selected,hit.source_id]
            try:
                pack(selected=trial,representation="raw")
                selected=trial
            except ValueError:
                rejected[hit.source_id]="complete_source_exceeds_shared_budget"
        no_rag_package=pack(selected=[],representation="raw")
        rawpackage=pack(selected=selected,representation="raw")
        managed,managedtrace,state=session._bundles(case["input"],as_of=stamp,current_source=current,task=task)
        state_bundles=[b for b in managed if any(c.field!="history_quote" for c in b.claims)]
        raw_quotes=replay._quote_bundles([h for h in rawsearch.hits if h.source_id in selected])
        managed_bundles=[]
        used=set()
        dropped=[]
        for bundle in sorted([*state_bundles,*raw_quotes],key=lambda b:-max((c.priority for c in b.claims),default=0)):
            ids={s.source_id for c in bundle.claims for s in c.spans}
            if len(used|ids)>SOURCE_CAP:
                dropped.append({"bundle_id":bundle.bundle_id,"reason":"common_distinct_source_cap"})
            else:
                used|=ids
                managed_bundles.append(bundle)
        managedpackage=pack(managed_bundles)
        contexts={}
        for arm,package in zip(ARMS,(no_rag_package,rawpackage,managedpackage)):
            trace=dict(package.trace)
            trace.update(omitted_recent_source_ids=omitted_recent,raw_ranking=rawsearch.trace,
                raw_budget_rejected_source_ids=rejected,managed_source_cap_dropped=dropped if arm=="managed_memory_rag" else [],
                task_projection_present=bool(state_bundles) if arm=="managed_memory_rag" else False,
                projected_task_state=state if arm=="managed_memory_rag" else None,
                clock_mode="dated-sources-v1",current_date=stamp,
                legacy_full_request_chars=LEGACY_MAX_CHARS,new_full_request_chars=MAX_CHARS,
                legacy_mandatory_context_fits=legacy_mandatory_fits,
                budget_change_reason="complete shared CBT counselor policy and source metadata; same enlarged ceiling in all three arms")
            dependencies=sorted(set(package.source_ids)|{current["source_id"]})
            contexts[arm]={"case_id":case["case_id"],"source_family_id":case["source_family_id"],"arm":arm,
                "payload":package.request_payload,"trace":trace,"database":str(database),"approved_root":str(directory),
                "binding":{"user_id":uid,"as_of":stamp,"generation":session.durable.current_generation(uid),
                    "source_hashes":{sid:digest(next(s for s in pool+[current] if s["source_id"]==sid)) for sid in dependencies},
                    "expected_task_versions":{task["object_id"]:task["version"]} if task and state else {}}}
        atomic(directory/"source_ingestion_audit.json",{"history":receipts,"current":current_receipt,
            "semantic_model_calls":0,"actual_product_task_rules":True,"original_calendar_dates":True})
        return contexts
    finally:
        session.close()


def prepare(cases_path,targets_path,output):
    require_hosted()
    output=Path(output).resolve()
    if not output.is_relative_to("/content"):
        raise ValueError("hosted /content output required")
    cases=json.loads(Path(cases_path).read_text())
    if type(cases) is not list or len(cases)!=24 or len({c["case_id"] for c in cases})!=24:
        raise ValueError("the full original 24-case source roster is required")
    output.mkdir(parents=True,exist_ok=False)
    atomic(output/"cases.json",cases)
    packets=[]
    for case in cases:
        contexts=prepare_one(case,output/"product_cases"/digest(case["case_id"])[:32])
        packets.extend(contexts[arm] for arm in ARMS)
    atomic(output/"packets.json",packets)
    # Literal targets are copied only after every model request is frozen. The
    # preparer never decodes them or uses product state to construct a label.
    shutil.copyfile(targets_path,output/"targets.json")
    protocol={"version":VERSION,"case_count":24,"arms":list(ARMS),"planned_answers":72,
        "provider":IDENTITY,"cases_sha256":digest(cases),"maximum_attempts":72,
        "request_limits":{"answer":MAX_CHARS,"preparation":MAX_CHARS},"created_at_utc":utc(),
        "files_sha256":{"packets.json":file_hash(output/"packets.json"),"targets.json":file_hash(output/"targets.json")},
        "code_sha256":code_hashes(),"source_case_file_sha256":file_hash(cases_path),
        "original_recent_messages":RECENT,"common_distinct_source_cap":SOURCE_CAP,"complete_request_chars":MAX_CHARS,
        "legacy_complete_request_chars":LEGACY_MAX_CHARS,
        "budget_change_reason":"complete shared CBT counselor policy; uniform 24000-character ceiling, no policy clipping",
        "max_output_tokens":MAX_OUTPUT,"semantic_extraction_calls":0,"actual_product_task_rules":True,
        "legacy_mandatory_overflow_cases":sorted({p["case_id"] for p in packets if not p["trace"]["legacy_mandatory_context_fits"]}),
        "material":"invented development mechanism variants, not natural CBT or independent clinical observations",
        "scorer":"unchanged original strict single-key JSON value/type comparison; no response repair",
        "historical_scorer_source_sha256":OLD_SCORER_SHA256,
        "historical_scorer_member":"content/cbt-qwen7b-gpu-v7-code/evaluation/cbt-memory-effect/gguf_state_comparison_v9_colab.py",
        "old_model_results":"preserved historical Qwen results; this new DeepSeek run is separately identified",
        "authorization_reference":"user_requested_all_previous_experiments_on_updated_architecture"}
    protocol["protocol_sha256"]=digest(protocol)
    atomic(output/"protocol.json",protocol)
    return protocol


def load(folder):
    require_hosted()
    folder=Path(folder).resolve()
    protocol=json.loads((folder/"protocol.json").read_text())
    cases=json.loads((folder/"cases.json").read_text())
    if (protocol.get("version")!=VERSION or protocol.get("cases_sha256")!=digest(cases) or
        protocol.get("code_sha256")!=code_hashes() or
        protocol.get("protocol_sha256")!=digest({k:v for k,v in protocol.items() if k!="protocol_sha256"})):
        raise ValueError("frozen controlled code, roster or protocol differs")
    for name,expected in protocol["files_sha256"].items():
        if file_hash(folder/name)!=expected:
            raise ValueError("frozen controlled requests or labels differ")
    return folder,protocol,cases,json.loads((folder/"packets.json").read_text())


def preflight(packet):
    m=replay._modules()
    binding=packet["binding"]
    store=m.DurableMemoryStore.open(packet["database"],approved_root=packet["approved_root"])
    try:
        if store.current_generation(binding["user_id"])!=binding["generation"]:
            raise ValueError("product source generation changed after request freeze")
        sources={s["source_id"]:s for s in store.list_sources(binding["user_id"],as_of=binding["as_of"])}
        for sid,expected in binding["source_hashes"].items():
            if sid not in sources or digest(sources[sid])!=expected or sources[sid]["permissions"].get("response_external") is not True:
                raise ValueError("a product source was revoked or changed")
        for oid,version in binding["expected_task_versions"].items():
            state=store.get_object_state(binding["user_id"],oid)
            if state["status"]!="available" or state["version"]!=version:
                raise ValueError("product task version changed after request freeze")
    finally:
        store.close()


def run(folder,transport,*,max_new_calls):
    folder,protocol,cases,packets=load(folder)
    if type(max_new_calls) is not int or max_new_calls<0:
        raise ValueError("finite invocation cap required")
    with lock(folder):
        journal=Journal(folder,protocol,transport,max_new_calls)
        errors=[]
        def answer(packet):
            preflight(packet)
            receipt=journal.call(packet,"answer",packet["payload"])
            atomic(folder/"answers"/(digest([packet["case_id"],packet["arm"]])+".json"),
                {"case_id":packet["case_id"],"arm":packet["arm"],"request_key":receipt["key"],
                 "response":receipt["text"],"provider_model":receipt["provider_model"],"usage":receipt["usage"]})
        (folder/"answers").mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(answer,packet):packet for packet in packets}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    journal.stop.set()
                    packet=futures[future]
                    errors.append({"case_id":packet["case_id"],"arm":packet["arm"],"error_type":type(exc).__name__})
        atomic(folder/"last_invocation.json",{"new_calls":journal.new_calls,"errors":errors,"at_utc":utc()})
        return score(folder)


def score(folder):
    from collections import Counter
    folder,protocol,cases,packets=load(folder)
    # Reuse the read-only exact-wire audit; it performs no sends or credential access.
    sys.path.insert(0,str(HERE))
    from formal_full import ReceiptAudit
    journal=ReceiptAudit(folder,protocol)
    targets=json.loads((folder/"targets.json").read_text())
    by_target={t["case_id"]:t for t in targets}
    if len(targets)!=24 or set(by_target)!={c["case_id"] for c in cases}:
        raise ValueError("original literal target roster differs")
    scored=[]
    for packet in packets:
        path=folder/"answers"/(digest([packet["case_id"],packet["arm"]])+".json")
        received=False
        result={"correct":False,"reason":"not_received","observed_answer":None}
        if path.exists():
            answer=json.loads(path.read_text())
            receipt=journal.call(packet,"answer",packet["payload"])
            expected={"case_id":packet["case_id"],"arm":packet["arm"],"request_key":receipt["key"],
                      "response":receipt["text"],"provider_model":receipt["provider_model"],"usage":receipt["usage"]}
            if answer!=expected:
                raise ValueError("controlled answer differs from its actual provider receipt")
            result=strict_result(answer["response"],by_target[packet["case_id"]]["answer"])
            received=True
        scored.append({"case_id":packet["case_id"],"arm":packet["arm"],"source_family_id":packet["source_family_id"],
                       "received":received,"raw_strict":result})
    by_case={c["case_id"]:{row["arm"]:row for row in scored if row["case_id"]==c["case_id"]} for c in cases}
    summary={"version":VERSION,"planned_answers":72,"received_answers":sum(r["received"] for r in scored),
        "complete":all(r["received"] for r in scored),"all_material_exposed_development_regression_only":True,
        "clinical_quality_claim":False,"raw_metric_is_unchanged_original_parser":True,"new_model_calls_by_scoring":0,
        "arms":{arm:{"planned":24,"received":sum(r["received"] for r in scored if r["arm"]==arm),
            "correct":sum(r["raw_strict"]["correct"] for r in scored if r["arm"]==arm),
            "errors":dict(Counter(r["raw_strict"]["reason"] for r in scored if r["arm"]==arm and not r["raw_strict"]["correct"]))}
            for arm in ARMS},
        "paired":{"managed_wins":sum(p["managed_memory_rag"]["raw_strict"]["correct"] and not p["raw_history_rag"]["raw_strict"]["correct"] for p in by_case.values()),
                  "raw_wins":sum(p["raw_history_rag"]["raw_strict"]["correct"] and not p["managed_memory_rag"]["raw_strict"]["correct"] for p in by_case.values())},
        "per_question":scored,"provider_usage":{k:sum(r.get("usage",{}).get(k,0) for r in journal.outcomes.values())
                                                for k in ("prompt_tokens","completion_tokens","total_tokens")}}
    atomic(folder/"score_progress.json",summary)
    return summary
