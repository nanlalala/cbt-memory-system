"""New-architecture MemoryAgentBench replication using the full pinned 800-case roster.

Full source loading and every model request run only in genuine hosted Colab.
The existing source grammar provides deterministic projections. These synthetic
facts retain the `context` role and benchmark provenance, with no user consent.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for part in ("evaluation/objective-memory", "evaluation/end-to-end-dialogue",
             "evaluation/dialogue-benefit", "modules/agent-integration"):
    sys.path.insert(0, str(ROOT / part))

import v3_run_objective_comparison as old
from objective_datasets import dataset_spec, download_registered_dataset, load_dataset_cases
from v3_fact_versions import PARSER_VERSION, consolidate_rows, parse_assertion
from v4_source_retrieval import retrieve_sources, VERSION as RETRIEVER_VERSION
from run_three_arm import HistoryRanker, Settings, fit_turns
from score_objective_comparison import deterministic_score
from v3_response_diagnostics import score_supplemental
from followup_transport import ResponseFailure, generate_response

VERSION = "cbt-v4-general-memoryagentbench-v2"
LEGACY_VERSION = "cbt-v4-general-memoryagentbench-v1"
LEGACY_REVISIONS = ("fe47b4709fd1d0899a2691f20b73ee3c97b6e47b", "b3d561fe9290d49b257a4074c025b27a5358c97a")
MODEL = "deepseek-v4.1-flash"
RETURNED_MODEL = "deepseek-v4-1-flash-260910"
ENDPOINT = "https://tokendance.space/gateway/v1"
ARMS = ("no_rag", "raw_history_rag", "managed_memory_rag")
MAX_CHARS = 24000
OUTPUT_TOKENS = 128
WORKERS = 4
GENERATION_FILES = ("requests.jsonl", "references.jsonl", "source_inventory.jsonl")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(text):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:
                raise ValueError("duplicate persisted JSON key")
            result[key]=value
        return result
    return json.loads(text,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite persisted JSON")))


def request_from_wire(row):
    """Recover insertion order from the retained wire, never canonical metadata."""
    wire=row.get("serialized_request")
    if not isinstance(wire,str):
        raise ValueError("v2 requires an explicit complete serialized request")
    request=strict_json(wire)
    if (request!=row["request"] or json.dumps(request,allow_nan=False)!=wire or
        hashlib.sha256(wire.encode()).hexdigest()!=row["request_sha256"] or
        len(wire)!=row["wire_chars"] or len(wire)>MAX_CHARS):
        raise ValueError("prepared serialized wire or semantic payload identity changed")
    return request


def recover_v1_wire(row):
    """Recover the fixed v1 encoder order and require its ORIGINAL frozen hash."""
    source=row["request"]
    if set(source)!={"model","messages","temperature","max_tokens","thinking"}:
        raise ValueError("unexpected v1 request fields")
    if any(set(m)!={"role","content"} for m in source["messages"]):
        raise ValueError("unexpected v1 message fields")
    body={"model":source["model"],"messages":[{"role":m["role"],"content":m["content"]} for m in source["messages"]],
          "temperature":source["temperature"],"max_tokens":source["max_tokens"],"thinking":{"type":source["thinking"].get("type")}}
    if (body!=source or source["model"]!=MODEL or source["temperature"]!=0 or
        type(source["temperature"]) is not int or type(source["max_tokens"]) is not int or
        source["max_tokens"]!=OUTPUT_TOKENS or source["thinking"]!={"type":"disabled"}):
        raise ValueError("legacy model or decoding settings differ")
    wire=json.dumps(body,allow_nan=False)
    recovered={**row,"serialized_request":wire,"request":body}
    request_from_wire(recovered)  # Checks the old hash/size; never guesses a new body.
    return recovered


def inherited_message_audit(requests):
    from cbt_context import CBT_POLICY
    expected=CBT_POLICY+"\n"+old.MAB_SYSTEM_PROMPT.replace("You answer a synthetic closed-world memory benchmark.",
        "FACTUAL RECALL TASK: Answer this synthetic closed-world benchmark question.")
    current={}
    for row in requests:
        messages=row["request"]["messages"]
        if ([m["role"] for m in messages]!=["system","user","user"] or messages[0]["content"]!=expected):
            raise ValueError("legacy actual messages do not contain the current complete shared role")
        qid=row["case_id"]
        if qid in current and current[qid]!=messages[-1]["content"]:
            raise ValueError("legacy common recent/current context differs across arms")
        current[qid]=messages[-1]["content"]
        if row["arm"]=="no_rag":
            prefix="SOURCED MEMORY DATA ONLY.\n"
            memory=messages[-2]["content"]
            if not memory.startswith(prefix):
                raise ValueError("legacy no_rag reference format differs")
            block=strict_json(memory[len(prefix):])
            if block!={"representation":"raw","source_reports":[]}:
                raise ValueError("legacy no_rag unexpectedly contains older memory")
    return {"all_actual_system_prompts_match":True,"common_recent_current_matches_across_arms":True,
            "no_rag_older_memory_empty":True,"shared_system_sha256":hashlib.sha256(expected.encode()).hexdigest()}


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_new(path, body):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def append(path, row):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    body = path.read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError("incomplete journal retained for reconciliation")
    return [strict_json(line) for line in body.split(b"\n") if line.strip()]


def code_hashes():
    files = set(old._code_hashes())
    files.update({str(Path(__file__).relative_to(ROOT)),
        "modules/agent-integration/cbt_context.py", "modules/agent-integration/cbt_counselor_policy.py", "modules/agent-integration/cbt_runtime_guard.py", "modules/agent-integration/evidence_bundles.py",
        "evaluation/objective-memory/v3_fact_versions.py", "evaluation/objective-memory/v4_source_retrieval.py",
        "evaluation/dialogue-benefit/followup_transport.py",
        "evaluation/cbt-memory-effect/architecture_retest/transport.py"})
    return {name:file_hash(ROOT / name) for name in sorted(files)}


def source_registry(case, admitted):
    return {old._document(case, source).document_id: {
        "source_id":old._document(case, source).document_id,
        "user_id":case["subject_id"], "text":source["text"].strip(), "role":source["role"],
        "source_order":source["order"], "source_time":source.get("observed_at"),
        "permissions":{"save":True, "long_term_use":True, "response_external":True},
    } for source in admitted}


def prepare_case(case, ranker, prepared_history=None):
    from cbt_context import CBT_POLICY, ContextualItem, build_cbt_context
    from evidence_bundles import EvidenceSpan

    if prepared_history is None:
        admitted, excluded = old._eligible(case)
        current, versions = consolidate_rows(admitted)
        managed, storage = old._managed_quotes(case, admitted)
        current_ids = {old._document(case, source).document_id for source in current}
        prepared_history = (admitted, excluded, versions,
                            [d for d in managed if d.document_id in current_ids], storage)
    admitted, excluded, versions, managed, storage = prepared_history
    documents = [old._document(case, source) for source in admitted]
    registry = source_registry(case, admitted)
    short = documents[-16:]
    _, admitted_short, _ = fit_turns(list(reversed(short)), 6000)
    short_ids = set(admitted_short)
    recent = [doc for doc in short if doc.document_id in short_ids]
    current = canonical({"question":case["query"], "answer_format":"one concise value or UNKNOWN",
        "recent_source_context":[{"source_id":doc.document_id, "source_order":registry[doc.document_id]["source_order"],
                                  "role":doc.speaker, "content":doc.text} for doc in recent]})
    common_base = [{"role":"system", "content":CBT_POLICY+"\n"+old.MAB_SYSTEM_PROMPT.replace("You answer a synthetic closed-world memory benchmark.", "FACTUAL RECALL TASK: Answer this synthetic closed-world benchmark question.")}, {"role":"user", "content":current}]
    by_id = {old._document(case, source).document_id:source for source in admitted}
    output = []
    for arm in ARMS:
        corpus = managed if arm == "managed_memory_rag" else documents
        older = [doc for doc in corpus if doc.document_id not in short_ids]
        selected, retrieval_trace = retrieve_sources(case["query"], older, ranker,
            scope_id=case["subject_id"], seed_documents=recent) if arm != "no_rag" else ([], {})
        _, selected_ids, _ = fit_turns(selected, 12000)
        items = []
        for source_id in selected_ids:
            source = by_id[source_id]
            assertion = parse_assertion(source["text"])
            value = ({"subject":assertion.subject, "relation":assertion.relation, "object":assertion.object,
                      "source_order":source["order"], "basis":"synthetic_benchmark_source_grammar"}
                     if assertion and arm == "managed_memory_rag" else
                     {"source_text":source["text"].strip(), "source_order":source["order"], "source_role":"context"})
            text = registry[source_id]["text"]
            items.append(ContextualItem(item_id=source_id, section="case_state", value=value,
                evidence=(EvidenceSpan(source_id, source_id+":full", text, 0, len(text), None, "context"),),
                origin="benchmark_source", support_status="rule_verified", verifier_revision=PARSER_VERSION))
        package = build_cbt_context(common_base, source_registry=registry, subject_id=case["subject_id"],
            max_outbound_chars=MAX_CHARS, max_response_tokens=OUTPUT_TOKENS,
            request_metadata={"model":MODEL, "temperature":0, "thinking":{"type":"disabled"}},
            representation="processed_with_evidence" if arm == "managed_memory_rag" else "raw",
            contextual_items=tuple(items))
        body = {"model":MODEL, "messages":list(package.messages), "temperature":0,
                "max_tokens":OUTPUT_TOKENS, "thinking":{"type":"disabled"}}
        wire = json.dumps(body)
        if len(wire) > MAX_CHARS:
            raise ValueError("actual complete wire exceeds prepared request budget")
        output.append({"case_id":case["case_id"], "group_id":case["group_id"], "task":case["task"],
            "arm":arm, "request":body, "serialized_request":wire,
            "request_sha256":hashlib.sha256(wire.encode()).hexdigest(),
            "wire_chars":len(wire), "trace":{
                "assembly":getattr(package, "trace", None), "recent_source_ids":[d.document_id for d in recent],
                "selected_source_ids":selected_ids, "retrieval":retrieval_trace,
                "version_summary":{k:v for k,v in versions.items() if k != "entries"} if arm=="managed_memory_rag" else None,
                "excluded":excluded, "storage_inventory_sha256":digest(storage) if arm=="managed_memory_rag" else None}})
    return output, prepared_history


def prepare(data_dir, run_dir):
    old.require_colab()
    from cbt_runtime_guard import assert_repository_runtime
    assert_repository_runtime(ROOT)
    paths = download_registered_dataset("memoryagentbench_conflict", data_dir)
    cases, refs = load_dataset_cases("memoryagentbench_conflict", paths["data"])
    tasks = Counter(case["task"] for case in cases)
    if len(cases) != 800 or len(tasks) != 8 or any(n != 100 for n in tasks.values()):
        raise ValueError("full pinned MemoryAgentBench roster differs")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    ranker = HistoryRanker(Settings(model=MODEL, accepted_provider_models=(RETURNED_MODEL,), backend="lexical",
        thinking_mode="disabled", short_turns=16, top_k=12, short_chars=6000, long_chars=12000,
        max_context_chars=MAX_CHARS, max_response_tokens=OUTPUT_TOKENS))
    requests, histories, inventory = [], {}, []
    for case in cases:
        key = case["history_scope_id"]
        batch, cached = prepare_case(case, ranker, histories.get(key))
        requests.extend(batch)
        if key not in histories:
            admitted, excluded, version, managed, storage = cached
            inventory.append({"history_scope_id":key, "sources":admitted, "excluded":excluded,
                              "versions":version, "storage":storage})
            histories[key] = cached
    if len(requests) != 2400:
        raise ValueError("incomplete three-arm grid")
    for filename, records in zip(GENERATION_FILES, (requests, refs, inventory)):
        write_new(run_dir / filename, "".join(canonical(row)+"\n" for row in records))
    protocol = {"version":VERSION, "dataset":dataset_spec("memoryagentbench_conflict"),
        "case_ids":[c["case_id"] for c in cases], "tasks":dict(tasks), "arms":list(ARMS),
        "planned_questions":800, "planned_arm_rows":2400,
        "maximum_provider_attempts":len({r["request_sha256"] for r in requests}),
        "model":MODEL, "required_returned_model":RETURNED_MODEL, "api_base":ENDPOINT,
        "max_output_tokens":OUTPUT_TOKENS, "max_complete_request_chars":MAX_CHARS,
        "maximum_workers":WORKERS,
        "files_sha256":{f:file_hash(run_dir/f) for f in GENERATION_FILES}, "code_sha256":code_hashes(),
        "retriever":RETRIEVER_VERSION, "projection_verifier":PARSER_VERSION,
        "statistical_scope":"one conservative shared factconsolidation family; exploratory",
        "prompt_boundary":"released answers only in references; benchmark fact role remains context",
        "retry":"no automatic retries; unresolved intent blocks further sends in this run"}
    protocol["protocol_sha256"] = digest(protocol)
    write_new(run_dir / "protocol.json", canonical(protocol)+"\n")
    return {k:protocol[k] for k in ("version","planned_questions","planned_arm_rows","maximum_provider_attempts","protocol_sha256")}


def load(run_dir):
    run_dir = Path(run_dir)
    protocol = strict_json((run_dir/"protocol.json").read_text())
    if protocol.get("version")!=VERSION or protocol["protocol_sha256"] != digest({k:v for k,v in protocol.items() if k!="protocol_sha256"}):
        raise ValueError("protocol checksum mismatch")
    if protocol["code_sha256"] != code_hashes():
        raise ValueError("prepared executable changed")
    if any(file_hash(run_dir/f) != expected for f,expected in protocol["files_sha256"].items()):
        raise ValueError("prepared inputs changed")
    requests = rows(run_dir/"requests.jsonl")
    if len(requests) != protocol["planned_arm_rows"]:
        raise ValueError("prepared arm roster changed")
    for row in requests:
        row["request"]=request_from_wire(row)
        if recover_v1_wire(row)["serialized_request"]!=row["serialized_request"]:
            raise ValueError("wire differs from the frozen transport encoder")
    return protocol, requests


def _assert_no_paid_journal(run_dir):
    for name in ("journal.jsonl","intents.jsonl","outcomes.jsonl"):
        path=Path(run_dir)/name
        if path.is_symlink() or (path.exists() and path.stat().st_size):
            raise ValueError("migration is restricted to source-only preparation with zero paid journal records")


def _legacy_code_identity(protocol,revision):
    if revision not in LEGACY_REVISIONS:
        raise ValueError("legacy source revision must be one of the explicitly reviewed commits")
    current=code_hashes()
    expected_names=set(current)
    if revision==LEGACY_REVISIONS[0]:
        expected_names.remove("modules/agent-integration/cbt_runtime_guard.py")
    if set(protocol.get("code_sha256",{}))!=expected_names:
        raise ValueError("legacy executable inventory differs from its reviewed revision")
    allowed={"evaluation/architecture-rerun/mab_full.py","modules/agent-integration/cbt_runtime_guard.py"}
    changed={}
    for name,expected in protocol["code_sha256"].items():
        raw=subprocess.run(["git","-C",str(ROOT),"show",revision+":"+name],
            check=True,capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError("legacy frozen code does not match the reviewed git source")
        if current.get(name)!=expected:
            if name not in allowed:
                raise ValueError("a retrieval, projection, policy or source dependency changed; migration is not allowed")
            changed[name]={"old_sha256":expected,"new_sha256":current[name]}
    for name in set(current)-set(protocol["code_sha256"]):
        if name not in allowed:
            raise ValueError("an unreviewed dependency was introduced")
        changed[name]={"old_sha256":None,"new_sha256":current[name]}
    return changed


def migrate_prepared_v1(source_run_dir,run_dir,*,source_revision,expected_protocol_sha256):
    """Audit and reuse an unsent v1 roster in a distinct v2 run directory.

    This recovers the ORIGINAL intended wire and requires every old wire hash
    to match. It performs no retrieval, extraction, generation or source rewrite.
    Only this script's serialization/admission guard and its pure guard helper
    may differ from the explicitly reviewed legacy source revision.
    """
    old.require_colab()
    from cbt_runtime_guard import assert_repository_runtime
    assert_repository_runtime(ROOT)
    source,run_dir=Path(source_run_dir).resolve(),Path(run_dir).resolve()
    if (source==run_dir or not run_dir.is_relative_to("/content") or
        source.is_relative_to(run_dir) or run_dir.is_relative_to(source)):
        raise ValueError("a distinct new hosted /content migration directory is required")
    _assert_no_paid_journal(source)
    protocol=strict_json((source/"protocol.json").read_text())
    if (protocol.get("version")!=LEGACY_VERSION or protocol.get("protocol_sha256")!=expected_protocol_sha256 or
        protocol.get("protocol_sha256")!=digest({k:v for k,v in protocol.items() if k!="protocol_sha256"})):
        raise ValueError("legacy protocol version, caller binding or checksum differs")
    if set(protocol.get("files_sha256",{}))!=set(GENERATION_FILES):
        raise ValueError("legacy source artifact inventory differs")
    for name,expected in protocol["files_sha256"].items():
        if (source/name).is_symlink() or file_hash(source/name)!=expected:
            raise ValueError("legacy prepared source artifacts changed")
    code_changes=_legacy_code_identity(protocol,source_revision)
    if (protocol.get("dataset")!=dataset_spec("memoryagentbench_conflict") or protocol.get("arms")!=list(ARMS) or
        protocol.get("planned_questions")!=800 or protocol.get("planned_arm_rows")!=2400 or
        protocol.get("model")!=MODEL or protocol.get("required_returned_model")!=RETURNED_MODEL or
        protocol.get("api_base")!=ENDPOINT or protocol.get("max_output_tokens")!=OUTPUT_TOKENS or
        protocol.get("max_complete_request_chars")!=MAX_CHARS or protocol.get("maximum_workers")!=WORKERS or
        protocol.get("retriever")!=RETRIEVER_VERSION or protocol.get("projection_verifier")!=PARSER_VERSION):
        raise ValueError("legacy source cohort, arms, decoding or algorithm settings differ")
    requests=[recover_v1_wire(row) for row in rows(source/"requests.jsonl")]
    message_audit=inherited_message_audit(requests)
    ids=protocol.get("case_ids",[])
    expected_grid={(qid,arm) for qid in ids for arm in ARMS}
    if (len(ids)!=800 or len(set(ids))!=800 or len(requests)!=2400 or
        {(r["case_id"],r["arm"]) for r in requests}!=expected_grid or len(expected_grid)!=2400):
        raise ValueError("legacy full 800-question three-arm grid differs")
    if (len(protocol.get("tasks",{}))!=8 or any(n!=100 for n in protocol["tasks"].values()) or
        Counter(r["task"] for r in requests)!={k:v*3 for k,v in protocol["tasks"].items()}):
        raise ValueError("legacy eight original task strata differ")
    references=rows(source/"references.jsonl")
    if len(references)!=800 or {r["case_id"] for r in references}!=set(ids):
        raise ValueError("legacy independent scoring-reference roster differs")
    unique=len({r["request_sha256"] for r in requests})
    if unique!=protocol.get("maximum_provider_attempts"):
        raise ValueError("legacy unique-wire cap differs")
    migration={"kind":"source_only_v1_exact_wire_recovery","source_run":str(source),"source_revision":source_revision,
        "source_protocol_sha256":protocol["protocol_sha256"],"source_protocol_file_sha256":file_hash(source/"protocol.json"),
        "source_files_sha256":protocol["files_sha256"],"reviewed_code_changes":code_changes,
        "actual_message_audit":message_audit,
        "original_semantic_payloads_retained":True,"original_wire_hashes_recovered_and_retained":True,
        "source_inventory_and_references_copied_byte_exact":True,"retrieval_rerun":False,
        "new_model_calls":0,"paid_journal_records_before_migration":0}
    run_dir.mkdir(parents=True,exist_ok=False)
    write_new(run_dir/"requests.jsonl","".join(canonical(row)+"\n" for row in requests))
    for name in ("references.jsonl","source_inventory.jsonl"):
        shutil.copyfile(source/name,run_dir/name)
    updated={k:v for k,v in protocol.items() if k!="protocol_sha256"}
    updated.update(version=VERSION,code_sha256=code_hashes(),migration=migration,
        files_sha256={name:file_hash(run_dir/name) for name in GENERATION_FILES})
    updated["protocol_sha256"]=digest(updated)
    write_new(run_dir/"protocol.json",canonical(updated)+"\n")
    load(run_dir)  # Require the persisted v2 round trip before reporting readiness.
    _assert_no_paid_journal(source)
    return {"version":VERSION,"planned_questions":800,"planned_arm_rows":2400,
            "maximum_provider_attempts":unique,"protocol_sha256":updated["protocol_sha256"],"migration":migration}


def ledger_state(run_dir, requests):
    allowed = {r["request_sha256"]:r for r in requests}
    protocol = json.loads((Path(run_dir)/"protocol.json").read_text())
    intents, outcomes = {}, {}
    for row in rows(Path(run_dir)/"journal.jsonl"):
        key = row["request_sha256"]
        if key not in allowed or row["event"] not in {"intent","received","rejected","unknown"}:
            raise ValueError("foreign journal row")
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("journal protocol binding changed")
        if row["event"] == "intent" and row.get("request") != allowed[key]["request"]:
            raise ValueError("journal request payload changed")
        if (row["event"]=="intent" and "serialized_request" in allowed[key] and
            row.get("serialized_request")!=allowed[key]["serialized_request"]):
            raise ValueError("journal serialized wire changed")
        if row["event"] == "received" and (row.get("provider_model") != RETURNED_MODEL or
                row.get("finish_reason") != "stop" or not isinstance(row.get("hypothesis"),str) or
                not row["hypothesis"].strip()):
            raise ValueError("invalid completed provider receipt")
        target = intents if row["event"] == "intent" else outcomes
        if key in target:
            raise ValueError("duplicate intent/outcome")
        if target is outcomes and key not in intents:
            raise ValueError("outcome before intent")
        target[key] = row
    return intents, outcomes


def _supplied_transport(request,transport):
    started=time.monotonic()
    result=transport(request)
    if (type(result) is not dict or result.get("provider_model")!=RETURNED_MODEL or
        result.get("finish_reason")!="stop" or type(result.get("text")) is not str or not result["text"].strip() or
        result.get("wire_sha256")!=hashlib.sha256(json.dumps(request,allow_nan=False).encode()).hexdigest()):
        raise ResponseFailure("incomplete_or_mismatched_supplied_transport_receipt",
                              usage=result.get("usage") if type(result) is dict else None,fatal=True)
    return {"hypothesis":result["text"],"provider_model":result["provider_model"],"finish_reason":"stop",
            "usage":result.get("usage",{}),"response_id":result.get("response_id"),"wire_sha256":result["wire_sha256"],
            "latency_seconds":time.monotonic()-started}


def generate(run_dir, max_new_calls, *, key=None, transport=None):
    old.require_colab()
    from cbt_runtime_guard import assert_repository_runtime
    assert_repository_runtime(ROOT)
    run_dir = Path(run_dir)
    with (run_dir/".send.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol, requests = load(run_dir)
        intents, outcomes = ledger_state(run_dir, requests)
        if set(intents) != set(outcomes) or any(r["event"] != "received" for r in outcomes.values()):
            raise RuntimeError("unresolved or rejected prior send requires reconciliation")
        if type(max_new_calls) is not int or max_new_calls < 1:
            raise ValueError("positive finite call batch required")
        if transport is not None and getattr(transport,"identity",None)!={"provider":ENDPOINT,"model":MODEL,
                "returned_model":RETURNED_MODEL,"model_family":"deepseek"}:
            raise ValueError("supplied kernel transport identity differs from the frozen model")
        if transport is None and key is None:
            from google.colab import userdata
            key = userdata.get("Capstone")
        if transport is None and not key:
            raise RuntimeError("Colab Secret is unavailable")
        sent = 0
        unique = {}
        for row in requests:
            unique.setdefault(row["request_sha256"],row)
        remaining = iter(row for identity,row in unique.items() if identity not in outcomes)
        blocked, exhausted, pending = False, False, {}
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            while pending or (not blocked and not exhausted and sent < max_new_calls):
                while not blocked and not exhausted and len(pending)<WORKERS and sent<max_new_calls:
                    row = next(remaining,None)
                    if row is None:
                        exhausted = True
                        break
                    identity = row["request_sha256"]
                    if len(intents) >= protocol["maximum_provider_attempts"]:
                        raise ValueError("frozen unique request ceiling exhausted")
                    common = {"request_sha256":identity,"protocol_sha256":protocol["protocol_sha256"]}
                    append(run_dir/"journal.jsonl", {**common,"event":"intent","request":row["request"],
                        **({"serialized_request":row["serialized_request"]} if "serialized_request" in row else {}),
                        "utc":datetime.now(timezone.utc).isoformat()})
                    intents[identity] = common
                    future = (pool.submit(_supplied_transport,row["request"],transport) if transport is not None else
                        pool.submit(generate_response,row["request"]["messages"],model=MODEL,
                            allowed_model=RETURNED_MODEL,api_base=ENDPOINT,key=key,max_response_tokens=OUTPUT_TOKENS))
                    pending[future] = common
                    sent += 1
                if not pending:
                    break
                completed, _ = wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    common = pending.pop(future)
                    try:
                        result = future.result()
                        outcome = {**common,"event":"received",**result}
                    except ResponseFailure as exc:
                        outcome = {**common,"event":"rejected","failure_kind":exc.failure_kind,
                                   "usage":exc.reported_usage}
                        blocked = True
                    except Exception:
                        outcome = {**common,"event":"unknown"}
                        blocked = True
                    append(run_dir/"journal.jsonl",outcome)
                    outcomes[common["request_sha256"]] = outcome
        # Every already admitted future has an outcome before this error escapes.
        # A rejected or unknown request is never silently scheduled again.
        if blocked:
            raise RuntimeError("provider failure recorded; in-flight requests drained; inspect journal")
        return {"new_calls":sent,"received_unique":len(outcomes),"planned_unique":protocol["maximum_provider_attempts"],
                "complete":len(outcomes)==protocol["maximum_provider_attempts"]}


def score(run_dir):
    old.require_colab()
    protocol, requests = load(run_dir)
    _, outcomes = ledger_state(run_dir, requests)
    references = {r["case_id"]:r for r in rows(Path(run_dir)/"references.jsonl")}
    scores = []
    for row in requests:
        out = outcomes.get(row["request_sha256"], {})
        received = out.get("event") == "received"
        value, parsed = deterministic_score(out["hypothesis"], references[row["case_id"]]) if received else (0,None)
        supplemental = score_supplemental(out["hypothesis"], references[row["case_id"]]) if received else None
        scores.append({"case_id":row["case_id"],"arm":row["arm"],"task":row["task"],"score":value,
                       "parsed":parsed,"received":received,"supplemental":supplemental,
                       "request_sha256":row["request_sha256"]})
    by_arm = {arm:{"planned":800,"received":sum(r["received"] for r in scores if r["arm"]==arm),
                   "correct":sum(r["score"] for r in scores if r["arm"]==arm)} for arm in ARMS}
    report = {"complete":len(outcomes)==protocol["maximum_provider_attempts"] and all(r["received"] for r in scores),
              "by_arm":by_arm,"interpretation":"MemoryAgentBench source-grammar adaptation; one family, exploratory",
              "unique_received":sum(o["event"]=="received" for o in outcomes.values()),
              "usage":{k:sum(o.get("usage",{}).get(k,0) for o in outcomes.values())
                       for k in ("prompt_tokens","completion_tokens","total_tokens")}}
    by_case = defaultdict(dict)
    for row in scores:
        by_case[row["case_id"]][row["arm"]] = row["score"]
    report["managed_vs_raw"] = {"wins":sum(r[ARMS[2]]>r[ARMS[1]] for r in by_case.values()),
        "losses":sum(r[ARMS[2]]<r[ARMS[1]] for r in by_case.values()),
        "ties":sum(r[ARMS[2]]==r[ARMS[1]] for r in by_case.values()),
        "interval":None,"interval_reason":"one conservative source family"}
    report["by_task"] = {task:{arm:{"planned":100,
        "correct":sum(r["score"] for r in scores if r["task"]==task and r["arm"]==arm),
        "strict_correct":sum((r["supplemental"] or {}).get("strict_committed_score",0)
                             for r in scores if r["task"]==task and r["arm"]==arm)}
        for arm in ARMS} for task in sorted(protocol["tasks"])}
    path = Path(run_dir)/"scores.jsonl"
    old._replace_rows(path,scores)
    old._replace_rows(Path(run_dir)/"summary.json",[report])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","migrate-v1","generate","score","status"])
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--max-new-calls",type=int)
    parser.add_argument("--source-run-dir",type=Path)
    parser.add_argument("--source-revision",choices=LEGACY_REVISIONS)
    parser.add_argument("--expected-protocol-sha256")
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.data_dir,args.run_dir)
    elif args.stage == "migrate-v1":
        result = migrate_prepared_v1(args.source_run_dir,args.run_dir,source_revision=args.source_revision,
                                     expected_protocol_sha256=args.expected_protocol_sha256)
    elif args.stage == "generate":
        result = generate(args.run_dir,args.max_new_calls)
    elif args.stage == "score":
        result = score(args.run_dir)
    else:
        protocol, requests = load(args.run_dir)
        intents,outcomes = ledger_state(args.run_dir,requests)
        result = {"planned_rows":len(requests),"planned_unique":protocol["maximum_provider_attempts"],
                  "intents":len(intents),"outcomes":len(outcomes),
                  "unknown":len(set(intents)-set(outcomes))+sum(o['event']=='unknown' for o in outcomes.values()),
                  "rejected":sum(o['event']=='rejected' for o in outcomes.values())}
    print(canonical(result),flush=True)


if __name__ == "__main__":
    main()
