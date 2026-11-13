"""Explicit LoCoMo continuation with immutable, original-protocol receipts.

This module never reconciles an unknown send as a received response. The four
already frozen LoCoMo sends remain technical missing rows; every other inherited
receipt keeps its original protocol and exact-wire identity. Production I/O is
restricted to genuine hosted Colab. Imports have no network or credential work.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast
from hashlib import sha256
import json
from pathlib import Path
import shutil

import formal_full as full
from architecture_retest.journal import (Journal, UnknownAttempt, InvalidReceipt,
    atomic, digest, lock, rows, utc, document_json)

VERSION = "formal-locomo-receipt-continuation-v1"
PARENT_HASH = "7441f21c6ad42572b23cf370bd6576ff3619a2b5bc0a3d3c95655e2d6783c929"
# Only these already observed final sends may be made terminal without a reply.
# New unknowns stop admission and cannot be silently added to this allowlist.
KNOWN_UNKNOWNS = {
    "491c18bea8ae963ba8f5fe59eaaef3b299f427ef651f613ab63c830f7d9f1a66":
        ("conv-47:qa:106", "managed_memory_rag", "0c7edebb9739470171c8985f21596b80052651e98d9292646559213edef4d728"),
    "51e754aef2d0c8a6962da3c7e35db19b3d57d5c17ea709757f73b71812e513bf":
        ("conv-47:qa:107", "managed_memory_rag", "78bea788536d653b88f5d2036a39b42b1f352991473c582c474d494c90943fb7"),
    "b792f4bba8dd80d28166bff7ae2c9fb0991805421b17cc42be3075927d9589db":
        ("conv-47:qa:108", "raw_history_rag", "cba2b4c779b7efa63d534221722203dc482d0b67f059f9fe72fd3f4d5310d409"),
    "3ebc554c689c820eabf02d24c8bf40b7940c1015edabed5664dc6c3e93c1ad27":
        ("conv-47:qa:109", "no_rag", "2596c0bd08730995bc1b94b3ec72ba3c4c25501907173093bd61f615e98e17ab"),
}
GUARD_PATH = "modules/agent-integration/cbt_runtime_guard.py"
OLD_GUARD_HASH = "a1636033a6e12078ef4f6492fefc89cae8455728135514c532d36a761428c85f"
# These two V2 additions serve CBT only. The historical benchmark keeps the V1
# path; its entire original module AST must remain identical after removing
# only the named additive definitions/exports below.
V1_AST = {
    "evaluation/cbt-memory-effect/architecture_retest/projections.py":(
        "006bd44ce333bd2864357db3b6ef1d36d71e9398e1732c59f5576ff5778b777f",
        "e6564e3ce7dddedfdcc02aee878d8d521aec24e63104a48228e24c88fe49a03e"),
    "modules/agent-integration/cbt_projection_contract.py":(
        "a19f9380dc8a0d2f716d999076f1eeb541ba65baaf8be47bd6c7e0bd0fd7a659",
        "2af5dbc2e157a26cbbd2f63802e3d1f843d4c22e7118b0f0628ece4b8b39f59a"),
}
STATIC_FILES = ("cases.json", "histories.json", "chunks.json", "references.json",
                "upstream_source_manifest.json", "mandatory_packet_preflight.json")


def _json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlinked continuation artifact")
    return document_json(path.read_text())


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(folder):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinked inherited artifact")
        if path.is_file() and path.name != ".run.lock":
            if path.suffix not in {".json", ".jsonl"}:
                raise ValueError("unexpected non-data inherited artifact")
            result[str(path.relative_to(folder))] = _hash(path)
    return result


def _guard(folder, fixture_only):
    if not fixture_only:
        full.require_hosted()
        if not Path(folder).resolve().is_relative_to("/content") or Path(folder).resolve() == Path("/content"):
            raise ValueError("continuation requires a hosted /content subdirectory")


def code_hashes():
    return {**full.code_hashes(), str(Path(__file__).relative_to(full.ROOT)):_hash(__file__)}


def _v1_ast_compatibility(name, old):
    if name not in V1_AST or old != V1_AST[name][0]:
        raise ValueError("original rendering code changed; this cannot be inherited as the same scheme")
    tree = ast.parse((full.ROOT/name).read_text())
    retained = []
    for node in tree.body:
        if name.endswith("cbt_projection_contract.py"):
            if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id=="BINDING_VERSION":
                if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                    raise ValueError("nonliteral additive projection constant")
                continue
            if isinstance(node, ast.FunctionDef) and node.name=="bind_candidates_with_status":
                if (node.decorator_list or node.args.defaults or node.args.kw_defaults
                        or node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg
                        or [a.arg for a in node.args.args] != ["text","sources"]
                        or node.returns or any(a.annotation for a in node.args.args)):
                    raise ValueError("additive V2 definition has executable definition-time changes")
                continue
        elif isinstance(node, ast.ImportFrom) and node.module=="cbt_projection_contract":
            if any(a.asname is not None for a in node.names if a.name in {"BINDING_VERSION","bind_candidates_with_status"}):
                raise ValueError("additive V2 export cannot shadow an existing binding")
            node.names = [a for a in node.names if a.name not in {"BINDING_VERSION","bind_candidates_with_status"}]
        retained.append(node)
    tree.body = retained
    actual = sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    if actual != V1_AST[name][1]:
        raise ValueError("frozen V1 projection behavior changed")
    return actual


def _protocol(path):
    value = _json(path)
    if value.get("protocol_sha256") != digest({k:v for k,v in value.items() if k != "protocol_sha256"}):
        raise ValueError("protocol checksum differs")
    return value


def _inventory(record, prepared):
    chunks = [prepared[cid] for cid in record["chunk_ids"]]
    failures = [c["chunk_id"] for c in chunks if c["status"] != "prepared"]
    return {"version_groups":full.historical_version_groups(
                [i for chunk in chunks for i in chunk["accepted"]], record["sources"]),
            "failed_chunks":failures, "automatic_support_assessment":True, "all_versions_preserved":True}


def _packet(case, arm, path):
    value = _json(path)
    if (value.get("case_id") != case["case_id"] or value.get("arm") != arm
            or value.get("source_family_id") != case["source_family_id"]):
        raise ValueError("request case, arm or source family differs")
    serialized = json.dumps(value["payload"], ensure_ascii=True, sort_keys=True, allow_nan=False)
    if (value["trace"].get("request_sha256") != sha256(serialized.encode()).hexdigest()
            or value["trace"].get("request_chars") != len(serialized)):
        raise ValueError("trace differs from the actual request payload")
    return value


def _parent_state(folder, fixture_only):
    """Validate the source-bound original run without loading its blocked Journal."""
    protocol = _protocol(folder/"protocol.json")
    cases, histories, chunks = (_json(folder/name) for name in ("cases.json", "histories.json", "chunks.json"))
    if (protocol.get("version") != full.VERSION or protocol.get("dataset_key") != "locomo10"
            or protocol.get("settings") != full.SETTINGS or protocol.get("provider") != full.IDENTITY
            or protocol.get("arms") != list(full.ARMS) or protocol.get("case_count") != len(cases)
            or protocol.get("planned_answers") != len(cases)*3 or protocol.get("cases_sha256") != digest(cases)
            or len({c["case_id"] for c in cases}) != len(cases)
            or protocol.get("preparation_chunks") != len(chunks)
            or protocol.get("mandatory_packet_preflight_passed") is not True):
        raise ValueError("original full source protocol differs")
    if fixture_only:
        if len(cases) > 10 or len(chunks) > 10:
            raise ValueError("tiny invented fixture limit exceeded")
    elif (len(cases) != 1540 or len(chunks) != 169 or protocol["protocol_sha256"] != PARENT_HASH):
        raise ValueError("only the frozen original LoCoMo cohort is eligible")
    current = full.code_hashes()
    changes = {}
    if set(protocol.get("code_sha256", {})) != set(current):
        raise ValueError("original rendering code roster differs")
    for name, old in protocol["code_sha256"].items():
        if old != current[name]:
            # The runtime-origin guard was repaired after the pause. Its byte
            # change is explicit; extraction, retrieval, rendering and policy
            # must all remain byte-for-byte pinned to the original run.
            if name == GUARD_PATH and old == OLD_GUARD_HASH:
                compatibility = {"reason":"runtime origin check only; no request representation change"}
            else:
                compatibility = {"reason":"additive CBT V2 API only; complete original V1 module AST retained",
                    "original_v1_ast_sha256":_v1_ast_compatibility(name,old)}
            changes[name] = {"parent_sha256":old, "continuation_sha256":current[name], **compatibility}
    for name, expected in protocol["files_sha256"].items():
        if _hash(folder/name) != expected:
            raise ValueError("original frozen source or reference artifact differs")
    references = _json(folder/"references.json")
    if len(references) != len(cases) or {r["case_id"] for r in references} != {c["case_id"] for c in cases}:
        raise ValueError("original reference roster differs")
    audit = full.ReceiptAudit(folder, protocol)
    prepared = {}
    for chunk in chunks:
        if not (folder/"prepared_chunks"/(chunk["chunk_id"]+".json")).is_file():
            raise ValueError("all original preparation chunks must be terminal before continuation")
        prepared[chunk["chunk_id"]] = full._prepare_chunk(chunk, folder, audit, persist=False)
    actual_answer_keys, packet_slots = set(), {}
    for case in cases:
        record = histories[case["history_key"]]
        inventory = _inventory(record, prepared)
        terminal = full._validate_case_outputs(case, record, folder, audit, inventory)
        actual_answer_keys.update(row["request_key"] for row in terminal.values() if row["status"] == "received")
        dest = folder/"cases"/digest(case["case_id"])[:32]
        for arm in full.ARMS:
            path = dest/(arm+".request.json")
            if path.exists():
                packet = _packet(case, arm, path)
                key = digest({"purpose":"answer", "payload":packet["payload"]})
                packet_slots.setdefault(key, []).append((case["case_id"], arm))
    received_answer_keys = {k for k,r in audit.outcomes.items() if r["status"] == "received" and audit.intents[k]["purpose"] == "answer"}
    if received_answer_keys != actual_answer_keys:
        raise ValueError("an inherited received answer has no bound source request/output")
    unknown_keys = set(audit.intents)-set(audit.outcomes)
    unknown_keys.update(k for k,r in audit.outcomes.items() if r["status"] == "unknown")
    if any(r["status"] not in {"received", "unknown"} for r in audit.outcomes.values()):
        raise ValueError("unexpected original rejected receipt")
    quarantines = []
    for key in sorted(unknown_keys):
        intent = audit.intents[key]
        slots = packet_slots.get(key, [])
        if intent["purpose"] != "answer" or len(slots) != 1 or slots[0][0] != intent["case_id"]:
            raise ValueError("only an exact original final answer slot may be quarantined")
        case_id, arm = slots[0]
        if not fixture_only and KNOWN_UNKNOWNS.get(key) != (case_id, arm, intent["wire_sha256"]):
            raise ValueError("unknown send differs from the four frozen observed requests")
        quarantines.append({"case_id":case_id, "arm":arm, "request_key":key,
            "wire_sha256":intent["wire_sha256"], "parent_protocol_sha256":protocol["protocol_sha256"],
            "parent_intent_sha256":digest(intent), "parent_outcome_sha256":digest(audit.outcomes[key]) if key in audit.outcomes else None})
    if not fixture_only and (unknown_keys != set(KNOWN_UNKNOWNS) or len(audit.intents) != 3319
            or sum(r["status"] == "received" for r in audit.outcomes.values()) != 3315
            or len(received_answer_keys) != 2978):
        raise ValueError("original receipt inventory differs from the audited pause")
    return protocol, cases, histories, chunks, prepared, audit, quarantines, changes


def _failure(quarantine):
    return {**quarantine, "status":"terminal_failure", "reason":"original_unknown_answer_not_resent",
            "technical_missing":True, "score_policy":"zero factual-QA task credit in original denominator; no clinical rating"}


def prepare(parent_dir, output, *, fixture_only=False):
    """Freeze a new portable run; no provider call or credential is used."""
    parent, folder = Path(parent_dir).resolve(), Path(output).resolve()
    _guard(parent, fixture_only)
    _guard(folder, fixture_only)
    if folder == parent or folder.is_relative_to(parent) or parent.is_relative_to(folder):
        raise ValueError("new independent directory required")
    state = _parent_state(parent, fixture_only)
    protocol, cases, _, _, _, audit, quarantines, changes = state
    snapshot = _snapshot(parent)
    # Count logical rows, not just receipt keys: an identical wire can already
    # have served multiple rows without an additional provider request.
    answers = sum(name.endswith(".answer.json") for name in snapshot)
    failures = sum(name.endswith(".failure.json") for name in snapshot)
    remaining = protocol["planned_answers"]-answers-failures-len(quarantines)
    original_budget = protocol["maximum_attempts"]-len(audit.intents)
    if remaining < 0 or remaining > original_budget:
        raise ValueError("continuation exceeds the original remaining call envelope")
    folder.mkdir(parents=True, exist_ok=False)
    inherited = folder/"inherited"
    inherited.mkdir()
    for name in snapshot:
        dest = inherited/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/name, dest)
    working = {name:expected for name,expected in snapshot.items()
               if name in STATIC_FILES or name.startswith(("prepared_chunks/", "cases/"))}
    for name in working:
        dest = folder/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inherited/name, dest)
    for quarantine in quarantines:
        path = folder/"cases"/digest(quarantine["case_id"])[:32]/(quarantine["arm"]+".failure.json")
        atomic(path, _failure(quarantine))
    new = {"version":VERSION, "fixture_only":fixture_only, "dataset_key":"locomo10",
        "case_count":len(cases), "arms":list(full.ARMS), "planned_answers":len(cases)*3,
        "preparation_chunks":protocol["preparation_chunks"], "cases_sha256":digest(cases),
        "provider":dict(protocol["provider"]), "settings":dict(protocol["settings"]),
        "maximum_attempts":remaining, "original_remaining_call_envelope":original_budget,
        "request_limits":dict(protocol["request_limits"]), "code_sha256":code_hashes(),
        "parent_protocol_sha256":protocol["protocol_sha256"], "parent_snapshot_sha256":snapshot,
        "inherited_working_sha256":working, "parent_nonrendering_code_changes":changes,
        "quarantined_unknown_answers":quarantines, "inherited_received_receipts":3315 if not fixture_only else sum(r["status"]=="received" for r in audit.outcomes.values()),
        "inherited_answer_rows":answers, "inherited_preparation_receipts":sum(i["purpose"]!="answer" for k,i in audit.intents.items() if audit.outcomes.get(k,{}).get("status")=="received"),
        "preparation_policy":"reuse every original terminal chunk; no new extraction/support calls",
        "receipt_policy":"original receipts retain original protocol/wire/model/source; new sends have separate journal",
        "unknown_policy":"four frozen final sends stay terminal missing; every new unknown stops admission",
        "scoring_policy":"unchanged independent reference judge; original 1540x3 denominator including technical missing rows",
        "maximum_independent_judge_calls":5082, "created_at_utc":utc()}
    new["protocol_sha256"] = digest(new)
    atomic(folder/"protocol.json", new)
    return {k:new[k] for k in ("version", "planned_answers", "maximum_attempts", "inherited_received_receipts", "inherited_answer_rows", "inherited_preparation_receipts", "maximum_independent_judge_calls", "protocol_sha256")}


def load(folder, *, fixture_only=False):
    folder = Path(folder).resolve()
    _guard(folder, fixture_only)
    protocol = _protocol(folder/"protocol.json")
    if (protocol.get("version") != VERSION or protocol.get("fixture_only") is not fixture_only
            or protocol.get("code_sha256") != code_hashes()):
        raise ValueError("frozen continuation code, protocol or mode differs")
    if _snapshot(folder/"inherited") != protocol["parent_snapshot_sha256"]:
        raise ValueError("inherited immutable snapshot changed")
    for name, expected in protocol["inherited_working_sha256"].items():
        if _hash(folder/name) != expected:
            raise ValueError("an inherited source or output was overwritten")
    state = _parent_state(folder/"inherited", fixture_only)
    parent = state[0]
    working = {name:expected for name,expected in protocol["parent_snapshot_sha256"].items()
               if name in STATIC_FILES or name.startswith(("prepared_chunks/", "cases/"))}
    answers = sum(name.endswith(".answer.json") for name in working)
    failures = sum(name.endswith(".failure.json") for name in working)
    remaining = parent["planned_answers"]-answers-failures-len(state[6])
    if (parent["protocol_sha256"] != protocol["parent_protocol_sha256"]
            or state[6] != protocol["quarantined_unknown_answers"]
            or state[7] != protocol["parent_nonrendering_code_changes"]
            or working != protocol["inherited_working_sha256"]
            or protocol.get("maximum_attempts") != remaining
            or protocol.get("original_remaining_call_envelope") != parent["maximum_attempts"]-len(state[5].intents)
            or remaining > protocol["original_remaining_call_envelope"]
            or protocol.get("maximum_independent_judge_calls") != 5082
            or protocol.get("inherited_answer_rows") != answers
            or protocol.get("inherited_received_receipts") != sum(r["status"]=="received" for r in state[5].outcomes.values())
            or any(protocol.get(k) != parent[k] for k in ("dataset_key","case_count","arms","planned_answers",
                    "preparation_chunks","cases_sha256","provider","settings","request_limits"))):
        raise ValueError("continuation provenance or cap differs")
    return folder, protocol, state


class InheritedJournal:
    """Consult original exact receipts first; admit only genuinely new answers."""
    def __init__(self, folder, protocol, parent_audit, *, transport=None, max_new_calls=0):
        self.parent = parent_audit
        if transport is not None and any(r["status"] != "received" for r in rows(folder/"outcomes.jsonl")):
            raise UnknownAttempt("a new incomplete/invalid send blocks further admission; no automatic resend")
        self.current = (full.ReceiptAudit(folder, protocol) if transport is None
                        else Journal(folder, protocol, transport, max_new_calls))
        if set(self.parent.intents) & set(self.current.intents):
            raise ValueError("an original request was resent in the continuation")
        self.intents = {**self.parent.intents, **self.current.intents}
        self.outcomes = {**self.parent.outcomes, **self.current.outcomes}
        if transport is not None:
            self.dense_mutex, self.stop = self.current.dense_mutex, self.current.stop

    def call(self, case, purpose, request):
        key = digest({"purpose":purpose, "payload":request})
        if key in self.parent.intents:
            return self.parent.call(case, purpose, request)
        if purpose != "answer":
            raise ValueError("continuation cannot send or manufacture new preparation")
        return self.current.call(case, purpose, request)


def _terminal(case, record, folder, journal, inventory, quarantines, *, require_complete=False):
    dest = folder/"cases"/digest(case["case_id"])[:32]
    inv = dest/"memory_projection_inventory.json"
    if inv.exists() and _json(inv) != inventory:
        raise ValueError("projection inventory differs from original verified receipts")
    terminal = {}
    for arm in full.ARMS:
        path, failed = dest/(arm+".answer.json"), dest/(arm+".failure.json")
        if path.exists() and failed.exists():
            raise ValueError("answer and terminal failure coexist")
        quarantine = quarantines.get((case["case_id"], arm))
        if quarantine is not None:
            if path.exists() or not failed.exists() or _json(failed) != _failure(quarantine):
                raise ValueError("frozen unknown was overwritten or lost its original intent binding")
            packet = _packet(case, arm, dest/(arm+".request.json"))
            if digest({"purpose":"answer", "payload":packet["payload"]}) != quarantine["request_key"]:
                raise ValueError("unknown failure no longer matches its actual source request")
            terminal[arm] = _failure(quarantine)
        elif path.exists():
            if not inv.exists():
                raise ValueError("answer has no source-bound projection inventory")
            packet = _packet(case, arm, dest/(arm+".request.json"))
            receipt = journal.call(case, "answer", packet["payload"])
            expected = {"case_id":case["case_id"], "arm":arm, "source_family_id":case["source_family_id"],
                "status":"received", "response":receipt["text"], "request_key":receipt["key"],
                "provider_model":receipt["provider_model"], "usage":receipt["usage"]}
            if _json(path) != expected:
                raise ValueError("answer differs from its actual original/new provider receipt")
            terminal[arm] = expected
        elif failed.exists():
            # Preserve only already frozen original failures; no new failure
            # label can turn an unexecuted row into an apparently finished run.
            relative = str(failed.relative_to(folder))
            if relative not in _json(folder/"protocol.json")["inherited_working_sha256"]:
                raise ValueError("unbound new technical terminal failure")
            terminal[arm] = _json(failed)
    marker = dest/"completed.json"
    if require_complete or marker.exists():
        if set(terminal) != set(full.ARMS) or not marker.exists():
            raise ValueError("completed marker has missing or unbound rows")
        complete = _json(marker)
        if complete.get("case_id") != case["case_id"] or ("rows" in complete and complete["rows"] != [terminal[a] for a in full.ARMS]):
            raise ValueError("completion marker differs from the actual terminal rows")
    return terminal


def _answer_case(case, record, folder, journal, dense, prepared, quarantines):
    dest = folder/"cases"/digest(case["case_id"])[:32]
    dest.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(record, prepared)
    full._freeze_or_check(dest/"memory_projection_inventory.json", inventory)
    if (dest/"completed.json").exists():
        _terminal(case, record, folder, journal, inventory, quarantines, require_complete=True)
        return
    packets = full.prepare_case_requests(case, record, inventory["version_groups"], dense, journal.dense_mutex)
    results = []
    for arm in full.ARMS:
        packet = packets[arm]
        failed = inventory["failed_chunks"]
        packet["trace"]["preparation_coverage"] = {"planned_chunks":len(record["chunk_ids"]),
            "failed_chunk_ids":failed, "prepared_chunks":len(record["chunk_ids"])-len(failed),
            "partial_supported_memory":bool(failed)}
        full._freeze_or_check(dest/(arm+".request.json"), packet)
        quarantine = quarantines.get((case["case_id"], arm))
        if quarantine is not None:
            if digest({"purpose":"answer", "payload":packet["payload"]}) != quarantine["request_key"]:
                raise ValueError("frozen unknown rendering differs; paid execution blocked")
            results.append(_failure(quarantine))
            continue
        receipt = journal.call(case, "answer", packet["payload"])
        answer = {"case_id":case["case_id"], "arm":arm, "source_family_id":case["source_family_id"],
            "status":"received", "response":receipt["text"], "request_key":receipt["key"],
            "provider_model":receipt["provider_model"], "usage":receipt["usage"]}
        full._freeze_or_check(dest/(arm+".answer.json"), answer)
        results.append(answer)
    atomic(dest/"completed.json", {"case_id":case["case_id"], "rows":results,
        "failed_preparation_chunks":inventory["failed_chunks"]})


def run(folder, transport, *, max_new_calls, dense=None, fixture_only=False):
    folder, protocol, state = load(folder, fixture_only=fixture_only)
    if type(max_new_calls) is not int or max_new_calls < 0:
        raise ValueError("finite nonnegative invocation cap required")
    if fixture_only and getattr(transport, "fixture_only", False) is not True:
        raise ValueError("software fixtures cannot invoke a real provider")
    _, cases, histories, _, prepared, parent, quarantine_rows, _ = state
    quarantines = {(q["case_id"],q["arm"]):q for q in quarantine_rows}
    with lock(folder):
        # Validate retained answers and quarantine markers before constructing
        # any journal capable of sending. Incomplete case markers do not waive
        # the source/receipt checks for rows already present.
        _, _, completed = _audit_rows(folder, protocol, state)
        if max_new_calls == 0 or len(completed) == len(cases):
            result = summarize(folder, fixture_only=fixture_only)
            atomic(folder/"last_invocation.json", {"new_calls":0,"errors":[],"at_utc":utc()})
            atomic(folder/"progress.json", result)
            return result
        journal = InheritedJournal(folder, protocol, parent, transport=transport, max_new_calls=max_new_calls)
        dense = dense if dense is not None else full.DenseRetriever()
        errors = []
        with ThreadPoolExecutor(max_workers=full.SETTINGS["workers"]) as pool:
            futures = {pool.submit(_answer_case, c, histories[c["history_key"]], folder, journal,
                        dense, prepared, quarantines):c["case_id"] for c in cases}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    journal.stop.set()
                    errors.append({"case_id":futures[future], "error_type":type(exc).__name__})
        atomic(folder/"last_invocation.json", {"new_calls":journal.current.new_calls, "errors":errors, "at_utc":utc()})
        result = summarize(folder, fixture_only=fixture_only)
        atomic(folder/"progress.json", result)
        return result


def _audit_rows(folder, protocol, state):
    _, cases, histories, _, prepared, parent, quarantine_rows, _ = state
    journal = InheritedJournal(folder, protocol, parent)
    quarantines = {(q["case_id"],q["arm"]):q for q in quarantine_rows}
    by_case, completed = {}, []
    for case in cases:
        record = histories[case["history_key"]]
        terminal = _terminal(case, record, folder, journal, _inventory(record, prepared), quarantines)
        by_case[case["case_id"]] = terminal
        if (folder/"cases"/digest(case["case_id"])[:32]/"completed.json").exists():
            completed.append(case["case_id"])
    expected = {folder/"cases"/digest(qid)[:32]/"completed.json" for qid in completed}
    if set((folder/"cases").glob("*/completed.json")) != expected:
        raise ValueError("foreign completion marker")
    return journal, by_case, completed


def summarize(folder, *, fixture_only=False):
    folder, protocol, state = load(folder, fixture_only=fixture_only)
    journal, terminal, completed = _audit_rows(folder, protocol, state)
    answer_rows = [r for case in terminal.values() for r in case.values() if r["status"] == "received"]
    failure_rows = [r for case in terminal.values() for r in case.values() if r["status"] != "received"]
    current = journal.current
    unknown = len(set(current.intents)-set(current.outcomes)) + sum(r["status"] == "unknown" for r in current.outcomes.values())
    outcomes = list(journal.outcomes.values())
    return {"version":VERSION, "dataset_key":protocol["dataset_key"], "planned_cases":protocol["case_count"],
        "completed_cases":len(completed), "planned_answers":protocol["planned_answers"],
        "received_answers":len(answer_rows), "technical_missing_answers":len(failure_rows),
        "planned_chunks":protocol["preparation_chunks"], "completed_chunks":len(state[4]),
        "failed_chunks":sum(c["status"] != "prepared" for c in state[4].values()),
        "arms":{a:{"planned":protocol["case_count"], "received":sum(r["arm"] == a for r in answer_rows),
                   "technical_missing":sum(r["arm"] == a for r in failure_rows)} for a in full.ARMS},
        "inherited_received_receipts":protocol["inherited_received_receipts"],
        "quarantined_unknown_attempts":len(protocol["quarantined_unknown_answers"]),
        "new_attempts":len(current.intents), "new_received_receipts":sum(r["status"] == "received" for r in current.outcomes.values()),
        "new_unknown_receipts":unknown, "new_call_cap":protocol["maximum_attempts"],
        "new_invalid_receipts":sum(r["status"] not in {"received","unknown"} for r in current.outcomes.values()),
        "provider_usage":{k:sum(r.get("usage",{}).get(k,0) for r in outcomes) for k in ("prompt_tokens","completion_tokens","total_tokens")},
        "receipt_binding_verified":True, "complete":len(completed) == protocol["case_count"] and unknown == 0,
        "all_provider_answers_received":len(answer_rows) == protocol["planned_answers"],
        "reference_accuracy_available":False, "clinical_or_human_outcome":False}


def grading_reader(folder, *, fixture_only=False):
    """The existing blind judge reads actual old/new receipts, never relabels them."""
    folder, protocol, state = load(folder, fixture_only=fixture_only)
    _, terminal, completed = _audit_rows(folder, protocol, state)
    cases, refs = state[1], {r["case_id"]:r for r in _json(folder/"references.json")}
    if len(completed) != len(cases):
        raise ValueError("reader is incomplete; every planned arm needs an answer or explicit terminal failure")
    answers, zeros = [], []
    for case in cases:
        reference = refs[case["case_id"]]
        for arm in full.ARMS:
            row = terminal[case["case_id"]][arm]
            common = {"case_id":case["case_id"], "arm":arm, "source_family_id":case["source_family_id"],
                      "category":reference["category"], "should_abstain":reference["should_abstain"]}
            if row["status"] == "received":
                answers.append((common, reference, row))
            else:
                zeros.append({**common, "score":0, "technical_failure":True, "failure_stage":"reader",
                    "failure_status":row["status"], "failure_reason":row.get("reason", row.get("error_type")),
                    "original_request_key":row.get("request_key"), "parent_protocol_sha256":row.get("parent_protocol_sha256")})
    snapshot = _snapshot(folder)
    return protocol, cases, answers, zeros, snapshot


def inspect_source_only(folder, *, fixture_only=False):
    """Recount a complete frozen formal source preparation without any send.

    In particular, a LongMemEval result here is a source/budget audit, never
    approval of its 39,232 possible DeepSeek sends or 1,650 grading sends.
    No credential, transport, embedding download, or current answer is used.
    """
    folder = Path(folder).resolve()
    _guard(folder, fixture_only)
    protocol = _protocol(folder/"protocol.json")
    cases, histories, chunks, references = (_json(folder/name) for name in
        ("cases.json","histories.json","chunks.json","references.json"))
    expected_count = {"locomo10":1540,"longmemeval_s_cleaned":500}.get(protocol.get("dataset_key"))
    if (protocol.get("version") != full.VERSION or expected_count is None
            or protocol.get("settings") != full.SETTINGS or protocol.get("provider") != full.IDENTITY
            or protocol.get("arms") != list(full.ARMS) or protocol.get("cases_sha256") != digest(cases)
            or len({c["case_id"] for c in cases}) != len(cases)
            or protocol.get("case_count") != len(cases) or protocol.get("planned_answers") != len(cases)*3
            or len(references) != len(cases) or {r["case_id"] for r in references} != {c["case_id"] for c in cases}):
        raise ValueError("complete original formal source roster differs")
    if fixture_only:
        if len(cases)>10 or len(chunks)>10:
            raise ValueError("tiny invented fixture limit exceeded")
    elif len(cases) != expected_count:
        raise ValueError("complete original formal cohort required")
    for name, expected in protocol["files_sha256"].items():
        if _hash(folder/name) != expected:
            raise ValueError("frozen source/reference checksum differs")
    pool = {}
    for record in histories.values():
        sources = full.source_fragments(record["original_history"])
        if sources != record["sources"]:
            raise ValueError("fragment mapping differs from complete original history")
        ids = []
        for source_chunk in full.source_chunks(sources):
            cid = digest(source_chunk)
            pool.setdefault(cid,{"case_id":"preparation:"+cid,"chunk_id":cid,"sources":source_chunk})
            ids.append(cid)
        if ids != record["chunk_ids"]:
            raise ValueError("source-only chronological chunk roster differs")
    chunk_map = {chunk["chunk_id"]:chunk for chunk in chunks}
    if len(chunk_map) != len(chunks) or pool != chunk_map:
        raise ValueError("source-only complete preparation chunks differ")
    for case in cases:
        record = histories.get(case["history_key"])
        if record is None or any(s["source_order"]>case["source_cutoff_order_inclusive"] for s in record["original_history"]):
            raise ValueError("missing history or source beyond the declared question cutoff")
    maximum = len(chunks)*2+len(cases)*3
    if (protocol.get("preparation_chunks") != len(chunks) or protocol.get("maximum_attempts") != maximum
            or protocol.get("mandatory_packet_preflight_passed") is not True):
        raise ValueError("source-derived cap or mandatory preflight differs")
    preflight = full.mandatory_packet_preflight(cases,histories)
    if not preflight["all_fit"] or preflight != _json(folder/"mandatory_packet_preflight.json"):
        raise ValueError("actual full mandatory packet source preflight differs")
    return {"dataset_key":protocol["dataset_key"], "source_protocol_sha256":protocol["protocol_sha256"],
        "source_only":True, "model_calls":0, "paid_execution_authorized":False,
        "planned_cases":len(cases), "planned_answers":len(cases)*3, "source_histories":len(histories),
        "preparation_chunks":len(chunks), "maximum_preparation_calls":len(chunks)*2,
        "maximum_answer_calls":len(cases)*3, "maximum_attempts_from_actual_sources":maximum,
        "maximum_independent_judge_calls":len(cases)*3+(len(cases)*3+9)//10,
        "existing_provider_intents":len(rows(folder/"intents.jsonl")),
        "mandatory_packet_preflight_all_fit":True, "complete_original_roster_verified":True}
