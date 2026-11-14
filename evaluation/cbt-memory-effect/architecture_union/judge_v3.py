"""Blind R1--R5 grading of a union batch with the project's v3 indexed-evidence judge.

The older judge asks the model to retype exact quotations and was rejected for
almost every review in the pilot (descriptive dimension labels, extra fields).
The v3 judge, already in the repository for this reason, sends the visible
context and response as indexed segments and asks for segment IDs; its rubric,
prompt, segmenting and validator are reused here unchanged. Only the batch
(prior raw, frozen managed and union configs for the same cases) is new.
Importing this module sends nothing and reads no credential.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import sys

from architecture_retest.journal import (Journal, TerminalLengthReceipt, TERMINAL_RECEIPT_POLICY,
                                         atomic, canonical, digest, document_json, lock, rows, utc)
from auto_eval_v2.run_colab import require_hosted

from . import run as union_run

sys.path.insert(0, str(union_run.runner.ROOT / "evaluation/architecture-rerun"))
import quality_span_judge_v3 as v3  # noqa: E402  (reused unchanged)

VERSION = "cbt-union-judge-v3-indexed-1"
JUDGE_IDENTITY = v3.JUDGE_IDENTITY


def prepare(union_folder, output, *, include_no_rag=False, fixture_only=False):
    if not fixture_only:
        require_hosted()
    packets, mapping, union_protocol = union_run.quality_packets(union_folder, include_no_rag=include_no_rag,
                                                                  fixture_only=fixture_only)
    plan, manifests = [], {}
    for packet in packets:
        visible, manifest = v3.segment_packet(packet)
        manifests[packet["case_id"]] = manifest
        payload = {"model": JUDGE_IDENTITY["model"], "messages": [{"role": "system", "content": v3.GUIDE},
                   {"role": "user", "content": canonical(visible)}], "temperature": 0,
                   "max_tokens": v3.OUTPUT_TOKENS, "thinking": {"type": "disabled"}}
        wire = json.dumps(payload, allow_nan=False)
        if len(wire) > v3.MAX_REQUEST_CHARS:
            raise ValueError("a complete judge request exceeds the frozen v3 cap; nothing prepared")
        plan.append({"item_id": "q-" + digest([v3.SEED, packet["case_id"], digest(payload)])[:24],
                     "case_id": packet["case_id"], "source_family": packet["source_family"], "repeat_of": None,
                     "payload": payload, "serialized_request": wire,
                     "wire_sha256": sha256(wire.encode()).hexdigest()})
    plan.sort(key=lambda r: r["item_id"])
    repeats = sorted(plan, key=lambda r: digest([v3.SEED, "repeat", r["item_id"]]))[:(len(plan) + 9) // 10]
    plan += [{**r, "item_id": "repeat-" + r["item_id"], "repeat_of": r["item_id"]} for r in repeats]
    output = Path(output).resolve()
    if not fixture_only and (not output.is_relative_to("/content") or output == Path("/content")):
        raise ValueError("new hosted judge directory required")
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"version": VERSION, "judge_prompt_version": v3.VERSION, "provider": dict(JUDGE_IDENTITY),
        "fixture_only": fixture_only, "terminal_receipt_policy": TERMINAL_RECEIPT_POLICY,
        "union_protocol_sha256": union_protocol["protocol_sha256"],
        "maximum_attempts": len(plan), "request_limits": {"answer": v3.MAX_REQUEST_CHARS,
                                                          "preparation": v3.MAX_REQUEST_CHARS},
        "cases_sha256": digest(packets), "mapping_sha256": digest(mapping), "plan_sha256": digest(plan),
        "segment_manifest_sha256": digest(manifests), "code_sha256": union_run.code_hashes(),
        "rubric_sha256": sha256(v3.RUBRIC.encode()).hexdigest(), "repeat_fraction": .1, "created_at_utc": utc()}
    protocol["protocol_sha256"] = digest(protocol)
    for name, value in (("cases.json", packets), ("controller_arm_mapping.json", mapping), ("plan.json", plan),
                        ("segment_manifest.json", manifests), ("protocol.json", protocol)):
        atomic(output / name, value)
    return {"version": VERSION, "packets": len(packets), "planned_judge_requests": len(plan),
            "protocol_sha256": protocol["protocol_sha256"]}


def _load(folder, fixture_only):
    folder = Path(folder).resolve()
    protocol = json.loads((folder / "protocol.json").read_text())
    if not fixture_only:
        require_hosted()
    art = {n: json.loads((folder / n).read_text()) for n in ("cases.json", "controller_arm_mapping.json", "plan.json")}
    if (protocol["version"] != VERSION or protocol["fixture_only"] != fixture_only
            or protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            or protocol["code_sha256"] != union_run.code_hashes()
            or digest(art["cases.json"]) != protocol["cases_sha256"]
            or digest(art["controller_arm_mapping.json"]) != protocol["mapping_sha256"]
            or digest(art["plan.json"]) != protocol["plan_sha256"]):
        raise ValueError("frozen judge batch, mode or code differs")
    for item in art["plan.json"]:
        restored = document_json(item["serialized_request"])
        if restored != item["payload"] or sha256(item["serialized_request"].encode()).hexdigest() != item["wire_sha256"]:
            raise ValueError("judge request identity differs")
        item["payload"] = restored  # keep the original key order of the wire
    return folder, protocol, art


def run(output, transport, *, max_new_calls, workers=4, fixture_only=False):
    folder, protocol, art = _load(output, fixture_only)
    if (type(max_new_calls) is not int or max_new_calls < 0 or type(workers) is not int or not 1 <= workers <= 4
            or getattr(transport, "identity", None) != JUDGE_IDENTITY
            or fixture_only and getattr(transport, "fixture_only", False) is not True):
        raise ValueError("fixed judge, bounded calls/workers and correct fixture transport required")
    packets = {p["case_id"]: p for p in art["cases.json"]}
    with lock(folder):
        journal = Journal(folder, protocol, transport, max_new_calls)
        done = {r["item_id"] for r in rows(folder / "reviews.jsonl")}

        def one(item):
            purpose = "quality_v3" if item["repeat_of"] is None else "quality_v3_repeat:" + item["item_id"]
            try:
                receipt = journal.call({"case_id": item["case_id"]}, purpose, item["payload"])
                key, review = receipt["key"], v3.validate_review(receipt["text"], packets[item["case_id"]])
            except TerminalLengthReceipt as exc:
                key, review = exc.request_key, v3.terminal_review()
            with journal.mutex:
                journal.append("reviews.jsonl", {"item_id": item["item_id"], "case_id": item["case_id"],
                    "source_family": item["source_family"], "repeat_of": item["repeat_of"],
                    "request_key": key, "review": review})

        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(one, item): item for item in art["plan.json"] if item["item_id"] not in done}
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as exc:
                    journal.stop.set()
                    failures.append({"item_id": pending[future]["item_id"], "error_type": type(exc).__name__})
        result = summarize(folder)
        atomic(folder / "summary.json", result)
        atomic(folder / "last_invocation.json", {"new_calls": journal.new_calls, "errors": failures, "at_utc": utc()})
        return result


def summarize(folder):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    reviews = rows(folder / "reviews.jsonl")
    status = {}
    for r in reviews:
        status[r["review"]["status"]] = status.get(r["review"]["status"], 0) + 1
    return {"version": VERSION, "planned": len(plan), "reviewed": len(reviews), "review_status": status,
            "valid_fraction": (status.get("completed", 0) / len(reviews)) if reviews else None}
