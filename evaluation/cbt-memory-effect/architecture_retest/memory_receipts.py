"""Read-only inheritance of exact, completed pre-answer target receipts.

Inherited receipts remain separate from new-run paid intents/outcomes. Nothing
rewrites their model, prompt, evidence, usage or original protocol identity.
"""
from hashlib import sha256
import json
from pathlib import Path

from .journal import digest, rows


def validate_entry(entry, identity):
    intent, receipt = entry["intent"], entry["receipt"]
    key = digest({"purpose": intent["purpose"], "payload": intent["payload"]})
    if (intent["purpose"] != "fact_targets" or intent["key"] != key or receipt["key"] != key or
            intent["protocol_sha256"] != receipt["protocol_sha256"] or
            json.loads(intent["serialized_request"]) != intent["payload"] or
            sha256(intent["serialized_request"].encode()).hexdigest() != intent["wire_sha256"] or
            receipt.get("wire_sha256") != intent["wire_sha256"] or
            receipt.get("status") != "received" or receipt.get("finish_reason") != "stop" or
            receipt.get("provider_model") != identity["returned_model"] or
            intent["payload"]["model"] != identity["model"] or
            type(receipt.get("text")) is not str or not receipt["text"].strip()):
        raise ValueError("inherited target receipt identity or exact request differs")
    return key


def collect(source_folder, cases_sha256, identity, expected_requests, *, fixture_only):
    source = Path(source_folder).resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    if (protocol["protocol_sha256"] != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"}) or
            type(fixture_only) is not bool or protocol.get("fixture_only") is not fixture_only or
            protocol["provider"] != identity or protocol["cases_sha256"] != cases_sha256 or
            digest(json.loads((source / "cases.json").read_text())) != cases_sha256):
        raise ValueError("inherited target protocol, fixture mode, provider or original case roster differs")
    intents, outcomes = rows(source / "intents.jsonl"), rows(source / "outcomes.jsonl")
    by_key = {r["key"]:r for r in outcomes}
    if (len({r["key"] for r in intents}) != len(intents) or len(by_key) != len(outcomes) or
            {r["key"] for r in intents} != set(by_key) or
            any(r["status"] != "received" for r in outcomes)):
        raise ValueError("unreconciled inherited requests cannot be reused")
    accepted = []
    for intent in intents:
        if intent["purpose"] != "fact_targets":
            continue
        receipt = by_key[intent["key"]]
        entry = {"intent":intent,"receipt":receipt}
        validate_entry(entry, identity)
        if (intent["protocol_sha256"] != protocol["protocol_sha256"] or
                receipt["protocol_sha256"] != protocol["protocol_sha256"] or
                expected_requests.get(intent["case_id"]) != intent["payload"] or
                json.dumps(expected_requests[intent["case_id"]],allow_nan=False) != intent["serialized_request"]):
            raise ValueError("inherited target request is not the exact frozen request bytes")
        accepted.append(entry)
    return {"source_folder":str(source),"source_protocol":protocol,
            "source_file_sha256":{name:sha256((source/name).read_bytes()).hexdigest()
                                  for name in ("protocol.json","cases.json","intents.jsonl","outcomes.jsonl")},
            "scope":"pre-answer fact_targets only; raw received receipts unchanged",
            "entries":accepted}
