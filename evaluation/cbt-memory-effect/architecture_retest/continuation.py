"""New-directory continuation of the original professional7 / CACTUS600 grid.

Preparation is source-only. It freezes exact receipt provenance and bounded
remaining calls; running still requires genuine hosted Colab and explicit caps.
No import, preparation or offline audit accesses credentials or sends a request.
"""
from pathlib import Path

from . import CONTINUATION_VERSION
from . import runner
from .journal import atomic, digest, utc
from .receipt_resolver import (ReceiptResolver, collect, file_hash, load_manifest,
    read_json, validate_manifest)

ASSESSOR_VERSION = "cbt-v4-independent-memory-assertions-continuation-v3"


def _guard(folder, fixture_only):
    if type(fixture_only) is not bool:
        raise ValueError("explicit Boolean fixture mode required")
    if not fixture_only:
        runner.require_hosted()
        if not Path(folder).resolve().is_relative_to("/content") or Path(folder).resolve() == Path("/content"):
            raise ValueError("new named hosted /content continuation folder required")


def _received_count(manifest):
    legacy = manifest.get("legacy_inherited_target_receipts")
    return len(manifest["entries"]) + (len(legacy["entries"]) if legacy else 0)


def _validate_frozen_target(frozen,case,entries,identity):
    """Reconstruct frozen labels from genuine exact pre-answer receipts."""
    import json
    from . import memory_assessment as memory
    sources=runner.source_views(case)
    data={"history":sources,"input":case["input"]}
    def receipt(key,purpose,system,body):
        entry=entries.get(key)
        expected={"model":identity["model"],"messages":memory.p.request(system,body),
            "temperature":0,"max_tokens":4096,"thinking":{"type":"disabled"}}
        if (entry is None or entry["intent"]["case_id"]!=case["case_id"] or
                entry["intent"]["purpose"]!=purpose or entry["intent"]["payload"]!=expected or
                entry["intent"]["serialized_request"]!=json.dumps(expected,allow_nan=False)):
            raise ValueError("frozen target receipt is not the exact original case/source request")
        return entry["receipt"]
    proposed=receipt(frozen.get("target_request_key"),"fact_targets",memory.TARGETS,data)
    value=memory.p.strict_json(proposed["text"])
    bound,statuses=memory.bind_target_candidates(value,sources)
    if value["targets"] and not bound:
        raise ValueError("unbound target batch cannot be inherited as frozen labels")
    accepted=[]
    if bound:
        checked=receipt(frozen.get("verification_request_key"),"fact_target_check",memory.VERIFY,{**data,"targets":bound})
        accepted,_=memory.p.checked_items(checked["text"],[{"item_id":t["id"],**t} for t in bound])
        if not accepted:
            raise ValueError("unsupported target batch cannot be inherited as frozen labels")
    targets=[{k:v for k,v in t.items() if k!="item_id"} for t in accepted]
    expected_gold={"case_id":case["case_id"],"source_family":case["source_family_id"],"label_provenance":"model_proposed",
        "required_facts":[{"id":t["id"],"allowed_source_ids":list(dict.fromkeys(e["source_id"] for e in t["evidence"]))}
                          for t in targets if t["kind"]=="required"],
        "forbidden_or_stale_fact_ids":[t["id"] for t in targets if t["kind"]=="stale"]}
    if (frozen.get("case_id")!=case["case_id"] or frozen.get("targets")!=targets or
            frozen.get("gold")!=expected_gold or frozen.get("proposed_targets")!=value["targets"] or
            frozen.get("binding_statuses")!=statuses):
        raise ValueError("frozen labels differ from reconstruction of original pre-answer receipts")


def _seed_targets(source, destination, cases, imported):
    """Copy frozen labels with original bytes and explicit source-file hashes."""
    from .memory_assessment import evidence
    entries = {e["intent"]["key"]: e for e in imported["entries"]}
    legacy = imported.get("legacy_inherited_target_receipts")
    if legacy:
        entries.update({e["intent"]["key"]: e for e in legacy["entries"]})
    records = []
    for case in cases:
        old = Path(source) / "assessor_only" / case["case_id"]
        has_frozen=(old / "targets.json").exists()
        if has_frozen:
            frozen = read_json(old / "targets.json")
            evidence.validate_gold(frozen["gold"])
            _validate_frozen_target(frozen,case,entries,imported["source_protocol"]["provider"])
            if (frozen.get("case_id") != case["case_id"] or
                    frozen["gold"]["source_family"] != case["source_family_id"] or
                    frozen.get("target_request_key") not in entries or
                    entries[frozen["target_request_key"]]["intent"]["purpose"] != "fact_targets" or
                    (frozen.get("verification_request_key") is not None and
                     (frozen["verification_request_key"] not in entries or
                      entries[frozen["verification_request_key"]]["intent"]["purpose"] != "fact_target_check"))):
                raise ValueError("frozen target provenance is not an imported pre-answer receipt")
        for name in ("targets.json", "target_candidates.json", "target_unavailable.json"):
            path = old / name
            if not path.exists():
                continue
            value = read_json(path)
            if value.get("case_id", case["case_id"]) != case["case_id"]:
                raise ValueError("inherited target artifact belongs to another case")
            dest = destination / "inherited_target_artifacts" / case["case_id"] / name
            dest.parent.mkdir(exist_ok=True, parents=True)
            # Preserve original bytes; atomic() is used for new reports only.
            with dest.open("xb") as stream:
                stream.write(path.read_bytes())
            record={"relative_path": str(dest.relative_to(destination)),
                "source_path": str(path.resolve()), "sha256": file_hash(path),
                "source_protocol_sha256": imported["source_protocol"]["protocol_sha256"]}
            # Valid frozen labels are reused verbatim. Failed preparation audits
            # remain in the immutable import; their cached responses are parsed
            # again into new-run diagnostics without overwriting this snapshot.
            if has_frozen and name in {"targets.json","target_candidates.json"}:
                materialized=destination/"assessor_only"/case["case_id"]/name
                materialized.parent.mkdir(exist_ok=True,parents=True)
                with materialized.open("xb") as stream:
                    stream.write(path.read_bytes())
                record["materialized_relative_path"]=str(materialized.relative_to(destination))
            records.append(record)
    return {"version": "cbt-frozen-target-artifact-import-v1", "entries": records,
        "scope": "original pre-answer labels retained; old scores are reconstructed from exact received assessments"}


def validate_seed_artifacts(folder, protocol):
    folder = Path(folder)
    imported = read_json(folder / "imported_target_artifacts.json")
    if digest(imported) != protocol["imported_target_artifacts_sha256"]:
        raise ValueError("frozen target artifact provenance changed")
    seen = set()
    for row in imported["entries"]:
        path = folder / row["relative_path"]
        if (row["relative_path"] in seen or not path.resolve().is_relative_to((folder / "inherited_target_artifacts").resolve()) or
                file_hash(path) != row["sha256"] or file_hash(row["source_path"]) != row["sha256"]):
            raise ValueError("inherited frozen target artifact changed")
        if row.get("materialized_relative_path") is not None:
            active=folder/row["materialized_relative_path"]
            if not active.resolve().is_relative_to((folder/"assessor_only").resolve()) or file_hash(active)!=row["sha256"]:
                raise ValueError("reused active frozen target changed")
        seen.add(row["relative_path"])
    return imported


def prepare(source_reader, source_assessor, output_folder, *, fixture_only=False):
    """Preserve the exact original roster; never select by observed outcome."""
    from .journal import TERMINAL_RECEIPT_POLICY
    output = Path(output_folder).resolve()
    _guard(output, fixture_only)
    source_reader, source_assessor = Path(source_reader).resolve(), Path(source_assessor).resolve()
    if output == source_reader or output == source_assessor or output.is_relative_to(source_reader) or output.is_relative_to(source_assessor):
        raise ValueError("continuation must use an independent directory")
    old_protocol = read_json(source_reader / "protocol.json")
    cases = read_json(source_reader / "cases.json")
    if (old_protocol.get("arms") != list(runner.ARMS) or old_protocol.get("settings") != runner.SETTINGS or
            not old_protocol.get("code_sha256") or len({c["case_id"] for c in cases}) != len(cases) or
            any(type(c["case_id"]) is not str or not c["case_id"] or "/" in c["case_id"] or "\\" in c["case_id"] for c in cases)):
        raise ValueError("original frozen three-arm source-reader settings/roster required")
    if (not fixture_only and len(cases) not in {7,600}) or (fixture_only and not 0 < len(cases) <= 10):
        raise ValueError("complete original seven/600 roster or explicit tiny fixture required")
    if len(cases) == 600 and len({c["source_family_id"] for c in cases}) != 200:
        raise ValueError("original CACTUS 200-family / 600-case roster required")
    old_assessor = read_json(source_assessor / "protocol.json")
    identity = old_assessor["provider"]
    if identity["model_family"].casefold() == "deepseek":
        raise ValueError("independent assessor family required")
    gen_import = collect(source_reader,cases,runner.IDENTITY,fixture_only=fixture_only)
    assessment_import = collect(source_assessor,cases,identity,fixture_only=fixture_only,assessor=True)
    output.mkdir(parents=True,exist_ok=False)
    reader, assessor = output / "reader", output / "memory"
    reader.mkdir(); assessor.mkdir()
    preflight = runner.source_preflight(cases)
    generation_cap = len(cases)*5 - _received_count(gen_import) - len(gen_import.get("terminal_invalid", []))
    assessment_cap = (len(cases)*5 - _received_count(assessment_import) - len(assessment_import["unavailable"])
                      - len(assessment_import.get("terminal_invalid", [])))
    if min(generation_cap, assessment_cap) < 0:
        raise ValueError("imported receipts exceed the original full-grid attempt cap")
    protocol = {"version":CONTINUATION_VERSION,"fixture_only":fixture_only,"cases_sha256":digest(cases),
        "case_count":len(cases),"source_families":len({c["source_family_id"] for c in cases}),
        "provider":runner.IDENTITY,"settings":runner.SETTINGS,"arms":list(runner.ARMS),
        "maximum_attempts":generation_cap,"request_limits":{"answer":24000,"preparation":48000},
        "candidate_binding_version":runner.projection.BINDING_VERSION,
        "terminal_receipt_policy":TERMINAL_RECEIPT_POLICY,
        "source_preflight_sha256":digest(preflight),"receipt_import_sha256":digest(gen_import),
        "source_protocol_sha256":old_protocol["protocol_sha256"],"code_sha256":runner.code_hashes(),
        "family_roster_policy":"exact_immutable_original_cases_json",
        "authorization_reference":"user_requested_remaining_architecture_evaluations",
        "created_at_utc":utc()}
    protocol["protocol_sha256"] = digest(protocol)
    for name,value in (("cases.json",cases),("source_preflight.json",preflight),("receipt_import.json",gen_import),("protocol.json",protocol)):
        atomic(reader / name,value)
    # Empty new journals make the separation visible from preparation onward.
    for directory in (reader, assessor):
        for name in ("intents.jsonl","outcomes.jsonl"):
            (directory / name).touch(exist_ok=False)
    from . import memory_assessment as memory
    assessment_preflight=[]
    for case in cases:
        body={"model":identity["model"],"messages":memory.p.request(memory.TARGETS,
            {"history":runner.source_views(case),"input":case["input"]}),
            "temperature":0,"max_tokens":4096,"thinking":{"type":"disabled"}}
        import json
        chars=len(json.dumps(body))
        assessment_preflight.append({"case_id":case["case_id"],"target_request_chars":chars,
            "within_source_request_cap":chars<=96000,"scope":"full original source-only independent target request"})
    artifacts=_seed_targets(source_assessor,assessor,cases,assessment_import)
    ap={"version":ASSESSOR_VERSION,"fixture_only":fixture_only,"provider":identity,
        "terminal_receipt_policy":TERMINAL_RECEIPT_POLICY,
        "maximum_attempts":assessment_cap,"request_limits":{"answer":96000,"preparation":96000},
        "cases_sha256":digest(cases),"source_preflight_sha256":digest(assessment_preflight),
        "receipt_import_sha256":digest(assessment_import),"imported_target_artifacts_sha256":digest(artifacts),
        "natural_run":str(reader),"code_sha256":runner.code_hashes(),"created_at_utc":utc(),
        "label_provenance":"model_proposed","targets_before_answers":True,
        "inherited_received_attempts":_received_count(assessment_import),
        "maximum_total_attempts_including_inherited":len(cases)*5,
        "historical_unknown_policy":"exact_two_post_answer_requests_unavailable_never_resent",
        "authorization_reference":"user_requested_remaining_architecture_evaluations"}
    ap["protocol_sha256"]=digest(ap)
    for name,value in (("cases.json",cases),("source_preflight.json",assessment_preflight),
            ("receipt_import.json",assessment_import),("imported_target_artifacts.json",artifacts),("protocol.json",ap)):
        atomic(assessor/name,value)
    budget={"version":CONTINUATION_VERSION,"planned_cases":len(cases),"planned_answers":len(cases)*3,
        "reader_folder":str(reader),"assessment_folder":str(assessor),
        "maximum_new_generation_requests":generation_cap,"maximum_new_assertion_requests":assessment_cap,
        "quality_after_complete_grid":{"primary":len(cases)*3,"with_independent_10_percent_repeats":(len(cases)*33+9)//10},
        "inherited_generation_received":_received_count(gen_import),
        "inherited_generation_terminal_invalid":len(gen_import.get("terminal_invalid", [])),
        "inherited_assertion_received":_received_count(assessment_import),
        "inherited_assertion_terminal_invalid":len(assessment_import.get("terminal_invalid", [])),
        "inherited_frozen_unavailable_assessments":len(assessment_import["unavailable"]),
        "budget_is_a_cap_not_new_authorization":True,"mab_changed":False,
        "comparison_scope":"original fixed grid; exact reusable answers may originate in an earlier execution window",
        "denominators":"all planned cases/arms retained; report missing targets, answers and valid grades separately"}
    atomic(output/"budget_plan.json",budget)
    return budget


def run(output_folder, answer_transport, assessment_transport, *, max_new_calls,
        assessment_max_new_calls, workers=4, dense=None, fixture_only=False):
    output=Path(output_folder).resolve()
    _guard(output,fixture_only)
    budget=read_json(output/"budget_plan.json")
    if (type(max_new_calls) is not int or not 0<=max_new_calls<=budget["maximum_new_generation_requests"] or
            type(assessment_max_new_calls) is not int or not 0<=assessment_max_new_calls<=budget["maximum_new_assertion_requests"]):
        raise ValueError("explicit bounded remaining invocation budgets required")
    return runner.run(output/"reader",answer_transport,max_new_calls=max_new_calls,workers=workers,dense=dense,
        fixture_only=fixture_only,assessment_folder=output/"memory",assessment_transport=assessment_transport,
        assessment_max_new_calls=assessment_max_new_calls)
