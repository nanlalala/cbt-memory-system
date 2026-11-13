"""Re-run the historical quotation endpoint separately from natural CBT quality.

No target quotation enters extraction, support checking, retrieval or generation.
Managed quotation requests expose source originals because the explicit research
question asks for an exact earlier quotation, while natural replies use the
processed readout. Completed natural-prefix projections are reused by hash.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from . import runner
from .journal import Journal, UnknownAttempt, CallLimit, atomic, digest, lock, rows, utc
from .retrieval import DenseRetriever

ARMS=runner.ARMS


def prepare(natural_run,folder,*,fixture_only=False,frozen_generation=None,frozen_gold=None,frozen_inventory=None):
    from run_cactus_effect import choose_probes
    natural,_,cases=runner.load(natural_run,fixture_only)
    if frozen_generation is None:
        if frozen_gold is not None or frozen_inventory is not None:
            raise ValueError("historical quotation generation, targets and inventory must be supplied together")
        generated,gold,inventory=choose_probes(cases)
    else:
        if frozen_gold is None or frozen_inventory is None:
            raise ValueError("exact historical quotation targets and inventory required")
        generated,gold,inventory=frozen_generation,frozen_gold,frozen_inventory
        if (type(generated) is not list or type(gold) is not list or type(inventory) is not list or
            len({r["case_id"] for r in generated})!=len(generated) or
            {r["case_id"] for r in generated}!={r["case_id"] for r in gold}):
            raise ValueError("historical quotation identity grid differs")
    case_by_sha={c["case_sha256"]:c for c in cases}
    latest_by_family={}
    for c in cases:
        family=c["source_family_id"]
        previous=latest_by_family.get(family)
        if previous is None or c["history_cutoff_order_exclusive"]>previous["history_cutoff_order_exclusive"]:
            latest_by_family[family]=c
    prefix_files={}
    for case in generated:
        prefix=case_by_sha.get(case.get("released_continuation_case_sha256")) or latest_by_family.get(case["source_family_id"])
        if prefix is None or case["history_cutoff_order_exclusive"]!=prefix["history_cutoff_order_exclusive"]:
            raise ValueError("historical quotation prefix does not bind to the selected family")
        view=lambda row:[(m["message_id"],m["role"],m["content"],m["source_order"]) for m in row["history"]]
        if view(case)!=view(prefix):
            raise ValueError("historical quotation source prefix changed")
        path=natural/"cases"/prefix["case_id"]/"projections.json"
        prefix_files[case["case_id"]]={"path":str(path),
            "sha256":runner.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "missing_projection_at_freeze":not path.is_file()}
    folder=Path(folder).resolve()
    if not fixture_only and (not folder.is_relative_to("/content") or folder==Path("/content")):
        raise ValueError("new hosted quote directory required")
    folder.mkdir(parents=True,exist_ok=False)
    protocol={"version":"cbt-v4-historical-quote-retest-v1","fixture_only":fixture_only,
        "provider":runner.IDENTITY,"case_count":len(generated),"source_families":len({c["source_family_id"] for c in generated}),
        "maximum_attempts":len(generated)*len(ARMS),"arms":list(ARMS),"request_limits":{"answer":24000,"preparation":48000},
        "cases_sha256":digest(generated),"source_inventory_sha256":digest(prefix_files),
        "gold_sha256":digest(gold),"code_sha256":runner.code_hashes(),
        "natural_run":str(natural),"natural_protocol_sha256":json.loads((natural/"protocol.json").read_text())["protocol_sha256"],
        "authorization_reference":"user_requested_full_architecture_rerun","created_at_utc":utc(),
        "literal_quote_only":True,"clinical_gold":False,"semantic_cbt_quality":False}
    protocol["quotation_roster_policy"]="recovered_historical_generation_and_targets" if frozen_generation is not None else "public_choose_probes"
    protocol["protocol_sha256"]=digest(protocol)
    atomic(folder/"cases.json",generated)
    atomic(folder/"assessor_only_gold.json",gold)
    atomic(folder/"inventory.json",inventory)
    atomic(folder/"prefix_projection_sources.json",prefix_files)
    atomic(folder/"protocol.json",protocol)
    return protocol


def run(folder,transport,*,max_new_calls,workers=4,dense=None,fixture_only=False):
    if not fixture_only:
        runner.require_hosted()
    folder=Path(folder)
    protocol=json.loads((folder/"protocol.json").read_text())
    cases=json.loads((folder/"cases.json").read_text())
    source_files=json.loads((folder/"prefix_projection_sources.json").read_text())
    if (protocol["code_sha256"]!=runner.code_hashes() or protocol["cases_sha256"]!=digest(cases) or
        protocol["source_inventory_sha256"]!=digest(source_files) or protocol["fixture_only"]!=fixture_only):
        raise ValueError("frozen quotation inputs/code differ")
    for source in source_files.values():
        if source["missing_projection_at_freeze"]:
            continue
        if runner.sha256(Path(source["path"]).read_bytes()).hexdigest()!=source["sha256"]:
            raise ValueError("reused extraction/support artifact differs")
    with lock(folder):
        journal=Journal(folder,protocol,transport,max_new_calls)
        dense=dense if dense is not None else DenseRetriever()
        if not fixture_only and (getattr(dense,"model_id",None),getattr(dense,"revision",None))!=(runner.DENSE_MODEL,runner.DENSE_REVISION):
            raise ValueError("dense ranker identity differs from frozen E5 model/revision")
        def one(case):
            dest=folder/"cases"/case["case_id"]
            dest.mkdir(exist_ok=True,parents=True)
            sources=runner.source_views(case)
            reused=source_files[case["case_id"]]
            items=[] if reused["missing_projection_at_freeze"] else json.loads(Path(reused["path"]).read_text())["accepted"]
            prepared=runner.prepare_case_requests(case,sources,items,dense,journal.dense_mutex)
            # Requesting complete quotations changes readout granularity, with
            # the same complete-request ceiling and no access to the gold target.
            managed=prepared["managed_memory_rag"]
            if reused["missing_projection_at_freeze"]:
                prepared["managed_memory_rag"]={"preparation_failure":"natural_projection_unavailable_at_freeze"}
            elif "preparation_failure" not in managed:
                selected=managed["trace"]["selected_source_ids"]
                available=set(selected)|set(managed["trace"]["recent_source_ids"])
                selected_items=[i for i in items if {e["source_id"] for e in i["evidence"]}<=available]
                try:
                    package=runner.assemble(case,sources,selected,selected_items,"processed_with_evidence",quotation_evidence=True)
                    prepared["managed_memory_rag"]={"payload":package.request_payload,"trace":package.trace}
                except ValueError:
                    prepared["managed_memory_rag"]={"preparation_failure":"whole_quote_evidence_budget"}
            for arm in ARMS:
                row=prepared[arm]
                atomic(dest/(arm+".request.json"),row)
                if "preparation_failure" in row:
                    continue
                receipt=journal.call(case,"answer",row["payload"])
                atomic(dest/(arm+".answer.json"),{"case_id":case["case_id"],"arm":arm,
                    "status":"received","response":receipt["text"],"request_key":receipt["key"],
                    "provider_model":receipt["provider_model"],"usage":receipt["usage"]})
        errors=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(one,c):c["case_id"] for c in cases}
            for future in as_completed(pending):
                try:
                    future.result()
                except (UnknownAttempt,CallLimit) as exc:
                    journal.stop.set()
                    errors.append({"case_id":pending[future],"error_type":type(exc).__name__})
                except Exception as exc:
                    journal.stop.set()
                    errors.append({"case_id":pending[future],"error_type":type(exc).__name__})
        result=score(folder)
        atomic(folder/"summary.json",result)
        atomic(folder/"last_invocation.json",{"errors":errors,"new_calls":journal.new_calls,"at_utc":utc()})
        return result


def score(folder):
    from run_cactus_effect import score_quote
    folder=Path(folder)
    cases=json.loads((folder/"cases.json").read_text())
    gold=json.loads((folder/"assessor_only_gold.json").read_text())
    protocol=json.loads((folder/"protocol.json").read_text())
    if digest(gold)!=protocol["gold_sha256"]:
        raise ValueError("frozen quotation scoring targets differ")
    targets={g["case_id"]:g for g in gold}
    received,unbound=runner.verified_answers(folder)
    answers={(a["case_id"],a["arm"]):a for a in received}
    results=[]
    for case in cases:
        target=targets[case["case_id"]]
        sid=case["source_family_id"]+":"+target["target_message_id"]
        for arm in ARMS:
            result={"case_id":case["case_id"],"source_family":case["source_family_id"],"arm":arm,
                    "status":"not_received","literal_success":False}
            answer=answers.get((case["case_id"],arm))
            if answer is not None:
                result.update(score_quote(answer["response"],target,sid),request_key=answer["request_key"])
            results.append(result)
    return {"planned_eligible":len(cases),"by_arm":{arm:{"planned":len(cases),
        "strict_correct":sum(r["literal_success"] for r in results if r["arm"]==arm),
        "not_received":sum(r["status"]=="not_received" for r in results if r["arm"]==arm)} for arm in ARMS},
        "rows":results,"unbound_answer_files":unbound,
        "interpretation":"literal quotation/source identity endpoint only"}
