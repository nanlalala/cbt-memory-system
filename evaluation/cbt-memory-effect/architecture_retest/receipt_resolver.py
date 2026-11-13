"""Exact received-receipt inheritance for an explicitly versioned continuation.

Original journals are read only. Their protocol identity, request bytes and
receipts live in a separate hash-bound manifest, never in the new paid journal.
Only the two enumerated historical post-answer unknowns can remain unavailable;
every other unknown still prevents continuation preparation/admission.
"""
from hashlib import sha256
import json
from pathlib import Path

from .journal import UnknownAttempt, digest, document_json, rows

MANIFEST_VERSION = "cbt-exact-receipt-import-v1"
FROZEN_ASSESSMENTS = {
    "8ac17fbf2183d707949f0ea5f89ca69934968f887944b110128fc810b478a0d3": {
        "case_id": "cactus-beaadb04f3c2f00a984bb009d1970567",
        "wire_sha256": "b6406640f48bc3d1d860b4290ee381fb72d40c83c6ff3572e9b7d19df8cebe5e"},
    "2796d496e8fc10f66c38756c2398ff5cbc635d24ede00875fbf72e1ad23638ec": {
        "case_id": "cactus-20c8d3e3f656216848fed9eb0bd4d2a3",
        "wire_sha256": "41d5ab159a302e7cf8a5ac601bbf7e103128f0589b01738d2137d743ecb39c85"},
}


class FrozenAssessmentUnavailable(ValueError):
    """A recorded historical post-answer send must never be sent again."""
    def __init__(self, key):
        self.request_key = key
        super().__init__("frozen historical post-answer assessment unavailable: " + key)


def file_hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular non-symlinked source artifact required")
    return sha256(path.read_bytes()).hexdigest()


def read_json(path):
    file_hash(path)
    return document_json(Path(path).read_text())


def _intent(intent, protocol):
    if (intent.get("protocol_sha256") != protocol["protocol_sha256"] or
            intent.get("key") != digest({"purpose": intent["purpose"], "payload": intent["payload"]}) or
            sha256(intent["serialized_request"].encode()).hexdigest() != intent.get("wire_sha256") or
            document_json(intent["serialized_request"]) != intent["payload"] or
            intent["payload"].get("model") != protocol["provider"]["model"]):
        raise ValueError("original intent identity or request bytes differ")


def validate_entry(entry, protocol):
    intent, receipt = entry["intent"], entry["receipt"]
    _intent(intent, protocol)
    if (receipt.get("key") != intent["key"] or
            receipt.get("protocol_sha256") != protocol["protocol_sha256"] or
            receipt.get("case_id") != intent["case_id"] or receipt.get("purpose") != intent["purpose"] or
            receipt.get("status") != "received" or receipt.get("finish_reason") != "stop" or
            receipt.get("provider_model") != protocol["provider"]["returned_model"] or
            receipt.get("wire_sha256") != intent["wire_sha256"] or
            type(receipt.get("text")) is not str or not receipt["text"].strip()):
        raise ValueError("original received receipt identity differs")
    return intent["key"]


def _frozen_unknown(entry, protocol):
    intent, outcome = entry["intent"], entry["receipt"]
    _intent(intent, protocol)
    expected = FROZEN_ASSESSMENTS.get(intent["key"])
    if (expected is None or intent["purpose"] != "memory_adjudication" or
            intent["case_id"] != expected["case_id"] or intent["wire_sha256"] != expected["wire_sha256"] or
            outcome.get("key") != intent["key"] or outcome.get("status") != "unknown" or
            outcome.get("case_id") != intent["case_id"] or outcome.get("purpose") != intent["purpose"] or
            outcome.get("protocol_sha256") != protocol["protocol_sha256"]):
        raise UnknownAttempt("unreconciled historical send is not an enumerated post-answer assessment")


def collect(source_folder, cases, identity, *, fixture_only, assessor=False):
    source = Path(source_folder).resolve()
    protocol = read_json(source / "protocol.json")
    names = ("protocol.json", "cases.json", "intents.jsonl", "outcomes.jsonl")
    hashes = {name: file_hash(source / name) for name in names}
    if (protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"}) or
            protocol.get("fixture_only") is not fixture_only or protocol.get("provider") != identity or
            protocol.get("cases_sha256") != digest(cases) or read_json(source / "cases.json") != cases or
            protocol.get("case_count", len(cases)) != len(cases)):
        raise ValueError("original protocol, provider, mode or exact full case roster differs")
    intents, outcomes = rows(source / "intents.jsonl"), rows(source / "outcomes.jsonl")
    by_key = {r["key"]: r for r in outcomes}
    if (len({r["key"] for r in intents}) != len(intents) or len(by_key) != len(outcomes) or
            {r["key"] for r in intents} != set(by_key)):
        raise UnknownAttempt("duplicate or outstanding original sends cannot be imported")
    ids = {c["case_id"] for c in cases}
    entries, unavailable = [], []
    for intent in intents:
        if intent["case_id"] not in ids:
            raise ValueError("original send belongs to a foreign case")
        entry = {"intent": intent, "receipt": by_key[intent["key"]]}
        if entry["receipt"].get("status") == "received":
            validate_entry(entry, protocol)
            entries.append(entry)
        else:
            if not assessor:
                raise UnknownAttempt("unreconciled original generation send")
            _frozen_unknown(entry, protocol)
            unavailable.append(entry)
    legacy = None
    if protocol.get("inherited_receipts_sha256") is not None:
        if not assessor:
            raise ValueError("legacy target inheritance is only valid for the independent assessor")
        legacy = read_json(source / "inherited_target_receipts.json")
        if digest(legacy) != protocol["inherited_receipts_sha256"]:
            raise ValueError("legacy frozen target receipt manifest differs")
        _validate_legacy(legacy, protocol)
        hashes["inherited_target_receipts.json"] = file_hash(source / "inherited_target_receipts.json")
    return {"version": MANIFEST_VERSION, "source_folder": str(source), "source_protocol": protocol,
        "source_file_sha256": hashes, "entries": entries, "unavailable": unavailable,
        "legacy_inherited_target_receipts": legacy,
        "scope": "unchanged original receipts; original unknown assessments never resent"}


def _validate_legacy(legacy, parent_protocol):
    """Authenticate the earlier target-only import through its frozen parent.

    The archived parent contains these raw entries and original upstream file
    hashes. It does not promise that the earlier upstream directory still exists.
    """
    original = legacy["source_protocol"]
    if (digest(legacy) != parent_protocol.get("inherited_receipts_sha256") or
            original.get("protocol_sha256") != digest({k:v for k,v in original.items() if k != "protocol_sha256"}) or
            original.get("provider") != parent_protocol["provider"] or
            original.get("cases_sha256") != parent_protocol["cases_sha256"] or
            original.get("fixture_only") is not parent_protocol["fixture_only"] or
            len(legacy["entries"]) != parent_protocol.get("inherited_received_attempts") or
            not legacy.get("source_file_sha256")):
        raise ValueError("legacy target import provenance differs from its frozen parent")
    keys = []
    for entry in legacy["entries"]:
        if entry["intent"]["purpose"] != "fact_targets":
            raise ValueError("legacy import supports pre-answer fact_targets only")
        keys.append(validate_entry(entry, original))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate inherited target receipt")


def validate_manifest(manifest, cases_sha256, identity, fixture_only):
    protocol = manifest["source_protocol"]
    if (manifest.get("version") != MANIFEST_VERSION or
            protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"}) or
            protocol.get("cases_sha256") != cases_sha256 or protocol.get("provider") != identity or
            protocol.get("fixture_only") is not fixture_only):
        raise ValueError("imported original protocol differs")
    source = Path(manifest["source_folder"])
    for name, expected in manifest["source_file_sha256"].items():
        if name not in {"protocol.json", "cases.json", "intents.jsonl", "outcomes.jsonl", "inherited_target_receipts.json"} or file_hash(source / name) != expected:
            raise ValueError("original immutable journal file changed")
    if read_json(source / "protocol.json") != protocol or digest(read_json(source / "cases.json")) != cases_sha256:
        raise ValueError("original immutable protocol or roster changed")
    # Bind embedded rows back to the untouched journal, not merely a user-edited
    # manifest bearing a plausible source hash.
    original_intents = rows(source / "intents.jsonl")
    original_outcomes = rows(source / "outcomes.jsonl")
    if (len({r["key"] for r in original_intents}) != len(original_intents) or
            len({r["key"] for r in original_outcomes}) != len(original_outcomes)):
        raise ValueError("duplicate immutable source rows")
    original = {r["key"]: r for r in original_intents}
    outcomes = {r["key"]: r for r in original_outcomes}
    imported = manifest["entries"] + manifest["unavailable"]
    if (len({e["intent"]["key"] for e in imported}) != len(imported) or
            {e["intent"]["key"] for e in imported} != set(original) or set(original) != set(outcomes)):
        raise ValueError("import manifest must retain the entire original journal")
    for entry in imported:
        key = entry["intent"]["key"]
        if entry["intent"] != original[key] or entry["receipt"] != outcomes[key]:
            raise ValueError("imported rows differ from immutable originals")
    for entry in manifest["entries"]:
        validate_entry(entry, protocol)
    for entry in manifest["unavailable"]:
        _frozen_unknown(entry, protocol)
    legacy = manifest.get("legacy_inherited_target_receipts")
    if protocol.get("inherited_receipts_sha256") is not None:
        if legacy != read_json(source / "inherited_target_receipts.json"):
            raise ValueError("legacy import differs from immutable parent artifact")
        _validate_legacy(legacy, protocol)
    elif legacy is not None:
        raise ValueError("unannounced legacy import")
    return manifest


def load_manifest(folder, protocol):
    expected = protocol.get("receipt_import_sha256")
    if expected is None:
        return None
    manifest = read_json(Path(folder) / "receipt_import.json")
    if digest(manifest) != expected:
        raise ValueError("frozen receipt import manifest changed")
    return validate_manifest(manifest, protocol["cases_sha256"], protocol["provider"], protocol["fixture_only"])


class ReceiptResolver:
    def __init__(self, folder, protocol, journal=None):
        self.journal = journal
        self.manifest = load_manifest(folder, protocol)
        self.received, self.unavailable = {}, {}
        if self.manifest:
            self.received = {e["intent"]["key"]: e for e in self.manifest["entries"]}
            self.unavailable = {e["intent"]["key"]: e for e in self.manifest["unavailable"]}
            legacy = self.manifest.get("legacy_inherited_target_receipts")
            if legacy:
                for entry in legacy["entries"]:
                    if entry["intent"]["key"] in self.received or entry["intent"]["key"] in self.unavailable:
                        raise ValueError("legacy and parent receipt identity overlaps")
                    self.received[entry["intent"]["key"]] = entry

    def call(self, case, purpose, request):
        key = digest({"purpose": purpose, "payload": request})
        entry = self.received.get(key) or self.unavailable.get(key)
        if entry:
            if (entry["intent"]["case_id"] != case["case_id"] or
                    json.dumps(request, allow_nan=False) != entry["intent"]["serialized_request"]):
                raise ValueError("inherited request is not the exact original case and wire")
            if key in self.unavailable:
                # Identity is checked at load AND immediately before refusing.
                _frozen_unknown(entry, self.manifest["source_protocol"])
                raise FrozenAssessmentUnavailable(key)
            return {**entry["receipt"], "receipt_origin": "inherited_received",
                "source_protocol_sha256": entry["intent"]["protocol_sha256"]}
        if self.journal is None:
            raise ValueError("no journal for a new request")
        return {**self.journal.call(case, purpose, request), "receipt_origin": "new_run_journal"}


class InheritedJournal:
    """Reuse exact receipts while keeping paid admission in the new Journal."""
    def __init__(self, journal):
        self.delegate = journal
        self.resolver = ReceiptResolver(journal.folder, journal.protocol, journal)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def call(self, case, purpose, request):
        return self.resolver.call(case, purpose, request)


def receipt_pairs(folder, protocol=None):
    """One resolver for answer auditing and blind-quality input authentication."""
    folder = Path(folder)
    protocol = read_json(folder / "protocol.json") if protocol is None else protocol
    resolver = ReceiptResolver(folder, protocol)
    intents, outcomes = {}, {}
    for entry in resolver.received.values():
        intents[entry["intent"]["key"]] = entry["intent"]
        outcomes[entry["receipt"]["key"]] = entry["receipt"]
    for name, target in (("intents.jsonl", intents), ("outcomes.jsonl", outcomes)):
        own = rows(folder / name)
        for row in own:
            if row["key"] in target:
                raise ValueError("duplicate inherited/new receipt identity")
            if protocol.get("protocol_sha256") is not None:
                if name=="intents.jsonl":
                    _intent(row,protocol)
                else:
                    intent=intents.get(row["key"])
                    if (intent is None or row.get("protocol_sha256")!=protocol["protocol_sha256"] or
                            row.get("case_id")!=intent["case_id"] or row.get("purpose")!=intent["purpose"]):
                        raise ValueError("new-run outcome has a foreign or missing intent")
                    if row.get("status")=="received":
                        validate_entry({"intent":intent,"receipt":row},protocol)
            target[row["key"]] = row
    return intents, outcomes
