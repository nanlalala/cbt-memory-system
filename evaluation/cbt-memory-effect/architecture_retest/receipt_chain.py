"""Read-only flattening of already received CBT continuation receipts.

Each receipt retains its own original protocol and exact request/response. A
chain manifest is a new representation, never a replacement paid journal. All
source runs must be quiescent before preparing a new independent continuation.
"""
from contextvars import ContextVar
from hashlib import sha256
from pathlib import Path

from .journal import TerminalLengthReceipt, UnknownAttempt, digest, is_terminal_length, rows

VERSION = "cbt-exact-receipt-chain-import-v2"
_READING = ContextVar("cbt_receipt_chain_source_stack", default=())
KNOWN_INVALID_EXTRACTIONS = {
    "77c64dacc889884ec324d2a4f8d730a975cc2e42aea7b452019c987dfdc80a20": {
        "case_id":"cactus-fe2020202787546653a8625673a19bdf", "purpose":"extraction",
        "protocol_sha256":"48dd6aee167e84e7af4d3a14be9d0acd839e4b22c62123d2a4b603aaa754b3b7",
        "wire_sha256":"c2a100bff95fb3c4f82dbfef91680f9170c2d6767d9e6f4b714deadfca5aae92",
        "provider_model":"deepseek-v4-1-flash-260910", "finish_reason":"length", "error_type":"InvalidReceipt"},
}


class FrozenInvalidReceipt(TerminalLengthReceipt):
    """A preserved terminal response remains counted and unusable forever."""
    def __init__(self, entry):
        intent = entry["intent"]
        super().__init__(intent["key"], intent["protocol_sha256"], intent["wire_sha256"])


def validate_invalid(entry, protocol):
    from . import receipt_resolver as rr
    intent, outcome = entry["intent"], entry["receipt"]
    rr._intent(intent, protocol)
    if (outcome.get("key") != intent["key"]
            or any(outcome.get(key) != intent[key] for key in ("case_id", "purpose", "protocol_sha256"))):
        raise UnknownAttempt("terminal response has a foreign request/protocol binding")
    if is_terminal_length(outcome, protocol, intent["wire_sha256"]):
        return intent["key"]
    expected = KNOWN_INVALID_EXTRACTIONS.get(intent["key"])
    if (expected is None or any(intent.get(key) != expected[key]
            for key in ("case_id", "purpose", "protocol_sha256", "wire_sha256"))
            or any(outcome.get(key) != expected[key]
                for key in ("case_id", "purpose", "protocol_sha256", "wire_sha256", "provider_model", "finish_reason", "error_type"))
            or outcome.get("key") != intent["key"] or outcome.get("status") != "invalid"
            or type(outcome.get("text")) is not str or not outcome["text"].strip()):
        raise UnknownAttempt("nonreceived continuation send is not the exact preserved invalid extraction")
    return intent["key"]


def _protocol_for(entry, manifest):
    key = entry["intent"]["protocol_sha256"]
    if manifest.get("version") == VERSION:
        return manifest["source_protocols"][key]
    protocol = manifest["source_protocol"]
    if protocol["protocol_sha256"] == key:
        return protocol
    legacy = manifest.get("legacy_inherited_target_receipts")
    if legacy and legacy["source_protocol"]["protocol_sha256"] == key:
        return legacy["source_protocol"]
    raise ValueError("receipt has no authenticated original protocol")


def collect(source_folder, cases, identity, *, fixture_only, assessor=False):
    from . import receipt_resolver as rr
    source = Path(source_folder).resolve()
    stack = _READING.get()
    if str(source) in stack:
        raise ValueError("cyclic receipt-import source chain")
    token = _READING.set((*stack, str(source)))
    try:
        protocol = rr.read_json(source / "protocol.json")
        names = ("protocol.json", "cases.json", "intents.jsonl", "outcomes.jsonl", "receipt_import.json")
        hashes = {name:rr.file_hash(source / name) for name in names}
        if (protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"})
                or protocol.get("fixture_only") is not fixture_only or protocol.get("provider") != identity
                or protocol.get("cases_sha256") != digest(cases) or rr.read_json(source / "cases.json") != cases
                or protocol.get("case_count", len(cases)) != len(cases)):
            raise ValueError("continuation source protocol/provider/mode/roster differs")
        inherited = rr.load_manifest(source, protocol)
        if inherited is None:
            raise ValueError("a chain continuation requires its authenticated parent import")
        entries, unavailable, terminal_invalid, protocols, provenance = [], [], [], {}, {}
        case_ids = {case["case_id"] for case in cases}

        def add(entry, original, state, origin):
            key = entry["intent"]["key"]
            if key in provenance or entry["intent"]["case_id"] not in case_ids:
                raise ValueError("duplicate/overlapping or foreign-case receipt in chain")
            if state == "received":
                rr.validate_entry(entry, original)
                entries.append(entry)
            elif state == "invalid":
                validate_invalid(entry, original)
                terminal_invalid.append(entry)
            else:
                if not assessor:
                    raise UnknownAttempt("unreconciled generation receipt in chain")
                rr._frozen_unknown(entry, original)
                unavailable.append(entry)
            original_hash = original["protocol_sha256"]
            if original_hash in protocols and protocols[original_hash] != original:
                raise ValueError("conflicting original protocol in receipt chain")
            protocols[original_hash] = original
            provenance[key] = {"original_protocol_sha256":original_hash,
                "intent_sha256":digest(entry["intent"]), "receipt_sha256":digest(entry["receipt"]),
                "wire_sha256":entry["intent"]["wire_sha256"], "status":entry["receipt"]["status"],
                "response_text_sha256":sha256(entry["receipt"]["text"].encode()).hexdigest()
                    if type(entry["receipt"].get("text")) is str else None,
                "origin":origin}

        for state, inherited_rows in (("received", inherited["entries"]), ("unavailable", inherited["unavailable"]),
                                      ("invalid", inherited.get("terminal_invalid", []))):
            for entry in inherited_rows:
                original = _protocol_for(entry, inherited)
                add(entry, original, state, "authenticated_parent_import")
        legacy = inherited.get("legacy_inherited_target_receipts")
        if legacy:
            for entry in legacy["entries"]:
                add(entry, legacy["source_protocol"], "received", "authenticated_legacy_target_import")
        local_intents, local_outcomes = rows(source / "intents.jsonl"), rows(source / "outcomes.jsonl")
        by_key = {row["key"]:row for row in local_outcomes}
        if (len({row["key"] for row in local_intents}) != len(local_intents)
                or len(by_key) != len(local_outcomes) or {row["key"] for row in local_intents} != set(by_key)):
            raise UnknownAttempt("duplicate or outstanding continuation sends cannot be imported")
        for intent in local_intents:
            entry = {"intent":intent, "receipt":by_key[intent["key"]]}
            # No new local unknown may acquire the historical unavailable status.
            status = entry["receipt"].get("status")
            if status == "invalid":
                add(entry, protocol, "invalid", "source_local_journal")
                continue
            if status != "received":
                raise UnknownAttempt("new continuation send is unresolved; no resend permitted")
            add(entry, protocol, "received", "source_local_journal")
        if inherited.get("version") == VERSION:
            paid_origins = list(inherited["paid_attempt_origins"])
            legacy_outside = list(inherited["legacy_receipts_outside_current_local_envelope"])
        else:
            original_rows = inherited["entries"] + inherited["unavailable"]
            paid_origins = [{"source_folder":inherited["source_folder"],
                "protocol_sha256":inherited["source_protocol"]["protocol_sha256"],
                "intents_file_sha256":inherited["source_file_sha256"]["intents.jsonl"],
                "provider":identity, "paid_attempts_counted":len(original_rows),
                "received":len(inherited["entries"]), "unknown_preserved":len(inherited["unavailable"]),
                "invalid_preserved":0}]
            legacy_outside = ([{"source_protocol_sha256":legacy["source_protocol"]["protocol_sha256"],
                "received_receipts":len(legacy["entries"]), "new_paid_attempts_counted":0,
                "reason":"previously inherited target receipts outside the current local envelope"}]
                if legacy else [])
        # Count attempts by immutable origin, not by a request key that may
        # also identify a separately authorized, earlier unknown attempt.
        paid_origins.append({"source_folder":str(source), "protocol_sha256":protocol["protocol_sha256"],
            "intents_file_sha256":hashes["intents.jsonl"], "provider":identity,
            "paid_attempts_counted":len(local_intents),
            "received":sum(row.get("status") == "received" for row in local_outcomes),
            "invalid_preserved":sum(row.get("status") == "invalid" for row in local_outcomes), "unknown_preserved":0})
        if any(rr.file_hash(source / name) != expected for name, expected in hashes.items()):
            raise ValueError("continuation source changed while collecting receipt chain")
        return {"version":VERSION, "source_folder":str(source), "source_protocol":protocol,
            "source_file_sha256":hashes, "source_protocols":protocols, "entries":entries,
            "unavailable":unavailable, "terminal_invalid":terminal_invalid, "legacy_inherited_target_receipts":None,
            "source_receipt_import":inherited, "receipt_provenance":provenance,
            "local_received_attempts":sum(row.get("status") == "received" for row in local_outcomes), "assessor":assessor,
            "paid_attempt_origins":paid_origins,
            "legacy_receipts_outside_current_local_envelope":legacy_outside,
            "external_attempts_not_in_receipt_chain_require_separate_driver_debit":True,
            "scope":"all received ancestor/local receipts once; exact original protocols and bytes retained; no unknown retried"}
    finally:
        _READING.reset(token)


def validate(manifest, cases_sha256, identity, fixture_only):
    from . import receipt_resolver as rr
    source = Path(manifest["source_folder"])
    cases = rr.read_json(source / "cases.json")
    if digest(cases) != cases_sha256 or type(manifest.get("assessor")) is not bool:
        raise ValueError("receipt chain roster or assessor mode differs")
    expected = collect(source, cases, identity, fixture_only=fixture_only, assessor=manifest["assessor"])
    if manifest != expected:
        raise ValueError("receipt chain differs from its immutable authenticated source runs")
    return manifest


def source_dependencies(manifest):
    """Keep every available immutable ancestor in later grading snapshots."""
    result = {}
    current = manifest
    while current is not None:
        for name, expected in current["source_file_sha256"].items():
            path = str(Path(current["source_folder"]) / name)
            if path in result and result[path] != expected:
                raise ValueError("conflicting immutable source dependency")
            result[path] = expected
        current = current.get("source_receipt_import")
    return result
