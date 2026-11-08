"""Canonical, non-coercing memory construction and version-aware validation.

The bundled evaluator implements only the closed JSON Schema vocabulary used by
our checked-in schemas. Unknown schema keywords fail closed. It is not a general
JSON Schema implementation. JSON clients must run format validation followed by
``validate_memory_data`` for cross-field semantics, as the CLI does.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import datetime
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any

from memory_models import MemoryRecord, SourceEvidence

ROOT = Path(__file__).resolve().parent
CLASS_CATEGORIES = {
    "fixed_historical": frozenset({"F1", "F2", "F3", "F4"}),
    "mutable_state": frozenset({"V1", "V2", "V3", "V4", "V5"}),
}
PERMISSION_KEYS = ("save", "long_term_use", "response_external", "extraction_external")
UNKNOWN_PERMISSIONS = {key: None for key in PERMISSION_KEYS}
_VOCABULARY = frozenset({
    "$schema", "$id", "$defs", "$ref", "title", "description", "default", "$comment",
    "type", "enum", "const", "properties", "additionalProperties", "required",
    "items", "uniqueItems", "minItems", "maxItems", "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "format",
    "pattern", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
})
_DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


@lru_cache(maxsize=2)
def _load_schema(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def parse_rfc3339(value: str, *, path: str = "timestamp") -> datetime:
    """Parse the same strict timestamp grammar used by the memory schema.

    Ingress must call this before committing a source, queuing a job or admitting
    an external attempt. Broader ``fromisoformat`` forms (a space separator or
    omitted seconds) are rejected instead of being saved and failing later when
    the worker constructs its schema-validated record. Unknown source times are
    handled by their caller as ``None``; this parser never invents a date.
    """
    if type(value) is not str or not _DATE_TIME.fullmatch(value):
        raise ValueError(f"{path} must be an RFC3339 date-time with timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{path} must be a valid date-time") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{path} must include a timezone")
    return parsed


def _timestamp(value: str, path: str) -> datetime:
    return parse_rfc3339(value, path=path)


def _json_key(value: Any) -> str:
    def normalise(item: Any) -> Any:
        # JSON Schema regards 1 and 1.0 as the same JSON number, but true is
        # distinct. Normalise only numbers for enum/uniqueItems comparison.
        if type(item) is float and math.isfinite(item) and item.is_integer():
            return int(item)
        if type(item) is list:
            return [normalise(x) for x in item]
        if type(item) is dict:
            return {key: normalise(child) for key, child in item.items()}
        return item
    return json.dumps(normalise(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def _matches_type(value: Any, expected: str) -> bool:
    return {
        "object": lambda: type(value) is dict,
        "array": lambda: type(value) is list,
        "string": lambda: type(value) is str,
        "integer": lambda: type(value) is int or (type(value) is float and math.isfinite(value) and value.is_integer()),
        "number": lambda: type(value) is int or (type(value) is float and math.isfinite(value)),
        "boolean": lambda: type(value) is bool,
        "null": lambda: value is None,
    }[expected]()


def _schema_keywords(schema: Any, root: dict) -> None:
    if type(schema) is bool:
        return
    if type(schema) is not dict:
        raise ValueError("schema entries must be objects or booleans")
    unknown = set(schema) - _VOCABULARY
    if unknown:
        raise ValueError(f"unsupported JSON Schema keywords: {sorted(unknown)}")
    if "format" in schema and schema["format"] != "date-time":
        raise ValueError(f"unsupported schema format: {schema['format']}")
    if "type" in schema:
        types = schema["type"] if type(schema["type"]) is list else [schema["type"]]
        if not types or any(t not in {"object", "array", "string", "integer", "number", "boolean", "null"} for t in types):
            raise ValueError("unsupported JSON Schema type")
    for key in ("properties", "$defs"):
        for child in schema.get(key, {}).values():
            _schema_keywords(child, root)
    for key in ("items", "additionalProperties", "not", "if", "then", "else"):
        if key in schema:
            _schema_keywords(schema[key], root)
    for key in ("allOf", "anyOf", "oneOf"):
        for child in schema.get(key, []):
            _schema_keywords(child, root)
    if "$ref" in schema:
        if not schema["$ref"].startswith("#/$defs/"):
            raise ValueError("only local $defs references are supported")
        if schema["$ref"].split("/")[-1] not in root.get("$defs", {}):
            raise ValueError("unknown local schema reference")


def validate_schema_instance(value: Any, schema: dict, *, path: str = "record") -> None:
    """Execute our checked-in schema subset, including date-time formats.

    Callers using another schema get an explicit unsupported-keyword error
    rather than silently ignoring restrictions.
    """
    _schema_keywords(schema, schema)

    def valid(item: Any, spec: Any, here: str) -> bool:
        try:
            check(item, spec, here)
            return True
        except ValueError:
            return False

    def check(item: Any, spec: Any, here: str) -> None:
        if spec is True:
            return
        if spec is False:
            raise ValueError(f"{here} is not permitted")
        if "$ref" in spec:
            check(item, schema["$defs"][spec["$ref"].split("/")[-1]], here)
        if "type" in spec:
            expected = spec["type"] if type(spec["type"]) is list else [spec["type"]]
            if not any(_matches_type(item, typ) for typ in expected):
                raise ValueError(f"{here} must have JSON type {expected}")
        if "const" in spec and _json_key(item) != _json_key(spec["const"]):
            raise ValueError(f"{here} must equal {spec['const']!r}")
        if "enum" in spec and _json_key(item) not in {_json_key(v) for v in spec["enum"]}:
            raise ValueError(f"{here} must be one of {spec['enum']}")
        if type(item) is str:
            if len(item) < spec.get("minLength", 0) or len(item) > spec.get("maxLength", math.inf):
                raise ValueError(f"{here} has invalid string length")
            if "pattern" in spec and not re.search(spec["pattern"], item):
                raise ValueError(f"{here} does not match its schema pattern")
            if spec.get("format") == "date-time":
                _timestamp(item, here)
            elif "format" in spec:
                raise ValueError(f"unsupported schema format: {spec['format']}")
        if type(item) in {int, float}:
            if type(item) is float and not math.isfinite(item):
                raise ValueError(f"{here} must be finite")
            for key, fails in (
                ("minimum", lambda n: item < n), ("maximum", lambda n: item > n),
                ("exclusiveMinimum", lambda n: item <= n), ("exclusiveMaximum", lambda n: item >= n),
            ):
                if key in spec and fails(spec[key]):
                    raise ValueError(f"{here} is outside {key}")
        if type(item) is dict:
            missing = set(spec.get("required", [])) - item.keys()
            if missing:
                raise ValueError(f"{here} missing required fields: {sorted(missing)}")
            props = spec.get("properties", {})
            for name, child in item.items():
                if type(name) is not str:
                    raise ValueError(f"{here} contains a non-string object key")
                if name in props:
                    check(child, props[name], here + "." + name)
                elif "additionalProperties" in spec:
                    check(child, spec["additionalProperties"], here + "." + name)
        if type(item) is list:
            if len(item) < spec.get("minItems", 0) or len(item) > spec.get("maxItems", math.inf):
                raise ValueError(f"{here} has invalid array length")
            if spec.get("uniqueItems") and len({_json_key(v) for v in item}) != len(item):
                raise ValueError(f"{here} must contain unique values")
            for n, child in enumerate(item):
                check(child, spec.get("items", True), f"{here}[{n}]")
        for child in spec.get("allOf", []):
            check(item, child, here)
        for key in ("anyOf", "oneOf"):
            if key in spec:
                count = sum(valid(item, child, here) for child in spec[key])
                if count == 0 or (key == "oneOf" and count != 1):
                    raise ValueError(f"{here} does not satisfy {key}")
        if "not" in spec and valid(item, spec["not"], here):
            raise ValueError(f"{here} matches a prohibited schema")
        if "if" in spec:
            branch = "then" if valid(item, spec["if"], here) else "else"
            if branch in spec:
                check(item, spec[branch], here)

    check(value, schema, path)


def _nonempty_string(value: Any, path: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")


def validate_source_evidence_data(data: dict) -> None:
    """Reuse the checked-in source schema for standalone evidence ingress."""
    spec = _load_schema("memory_record.schema.json")["properties"]["source_evidence"]
    validate_schema_instance(data, spec, path="source_evidence")


def _semantic_rules(data: dict) -> None:
    """Shared cross-field/business rules after structural and format checks."""
    for name in ("memory_id", "user_id"):
        _nonempty_string(data[name], name)
    if data["lifecycle_status"] != "deleted":
        _nonempty_string(data["content"], "content")
    if _timestamp(data["updated_at"], "updated_at") < _timestamp(data["created_at"], "created_at"):
        raise ValueError("updated_at cannot be earlier than created_at")
    if data.get("valid_from") and data.get("valid_to"):
        if _timestamp(data["valid_to"], "valid_to") < _timestamp(data["valid_from"], "valid_from"):
            raise ValueError("valid_to cannot be earlier than valid_from")
    evidence = data["source_evidence"]
    for name in ("session_id", "supporting_text"):
        _nonempty_string(evidence[name], "source_evidence." + name)
    for turn_id in evidence["turn_ids"]:
        _nonempty_string(turn_id, "source_evidence.turn_ids")
    for key in ("tags", "related_memory_ids"):
        for item in data.get(key, []):
            _nonempty_string(item, key)
    if data.get("supersedes_memory_id") == data["memory_id"] or data["memory_id"] in data.get("related_memory_ids", []):
        raise ValueError("memory cannot supersede or relate to itself")
    payload = data.get("structured_payload", {})
    typ = data["memory_type"]
    required = {
        "emotion_record": {"emotion", "intensity", "situation"},
        "goal": {"description", "status"},
        "cbt_assignment": {"description", "status"},
        "assignment_outcome": {"assignment_memory_id", "completion_status"},
        "user_correction": {"corrected_content"},
        "cognitive_pattern": {"evidence_count"},
    }.get(typ, set())
    if data["schema_version"] == "2.0":
        # V2 task state lives in the multi-axis projector. A legacy status
        # string may remain source data but is not the current projection.
        if typ in {"goal", "cbt_assignment"}:
            required = {"description"}
        elif typ == "assignment_outcome":
            required = {"assignment_memory_id"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{typ} is missing structured_payload fields: {sorted(missing)}")
    for key in required - {"intensity", "evidence_count"}:
        _nonempty_string(payload[key], f"structured_payload.{key}")
    for key in {"description", "status", "completion_status", "assignment_memory_id", "corrected_content", "emotion", "situation"} & payload.keys():
        _nonempty_string(payload[key], f"structured_payload.{key}")
    if typ == "emotion_record" and (type(payload["intensity"]) is not int or not 0 <= payload["intensity"] <= 10):
        raise ValueError("emotion_record intensity must be an integer from 0 to 10")
    if typ == "cognitive_pattern":
        if data["origin"] != "agent_inference":
            raise ValueError("cognitive_pattern must be marked as agent_inference")
        if type(payload["evidence_count"]) is not int or payload["evidence_count"] < 1:
            raise ValueError("evidence_count must be an integer at least 1")
    if data["origin"] == "agent_inference" and data["confirmation_state"] not in {"unconfirmed", "user_confirmed"}:
        raise ValueError("agent inferences must remain unconfirmed until separately confirmed")
    if typ == "user_correction":
        if data["origin"] != "user_correction" or not data.get("supersedes_memory_id"):
            raise ValueError("user_correction requires origin=user_correction and supersedes_memory_id")
    if typ == "safety_signal" and data["sensitivity"] != "restricted":
        raise ValueError("safety_signal requires restricted sensitivity")
    if data["schema_version"] == "1.0":
        return
    status = data.get("classification_status", "pending")
    cls, category = data.get("memory_class"), data.get("category")
    if status == "pending":
        if cls is not None or category is not None:
            raise ValueError("pending classification requires null class and category")
    elif cls not in CLASS_CATEGORIES or category not in CLASS_CATEGORIES[cls]:
        raise ValueError("memory_class/category combination is invalid")
    source_time = data.get("source_time")
    if source_time and _timestamp(source_time, "source_time") > _timestamp(data["created_at"], "created_at"):
        raise ValueError("source_time cannot be later than created_at")
    if (cls == "fixed_historical" and source_time and data.get("event_time")
            and _timestamp(data["event_time"], "event_time") > _timestamp(source_time, "source_time")):
        raise ValueError("a fixed historical event cannot be later than its source report")
    if data.get("source_role") == "assistant" and data["origin"] in {"explicit_user_statement", "user_correction"}:
        raise ValueError("assistant source cannot be represented as a user statement")
    if data.get("state_version") is not None and type(data["state_version"]) is not int:
        raise ValueError("state_version must be a strict integer")
    support = data.get("semantic_support", {})
    if support.get("status") in {"rule_verified", "validator_verified"}:
        if not support.get("method") or not support.get("version") or not data.get("claim_bindings"):
            raise ValueError("verified semantic support requires method/version and claim bindings")
    claim_ids: set[str] = set()
    for binding in data.get("claim_bindings", []):
        if binding["claim_id"] in claim_ids:
            raise ValueError("claim_bindings claim_id must be unique")
        claim_ids.add(binding["claim_id"])
        if type(binding["start"]) is not int or type(binding["end"]) is not int:
            raise ValueError("claim binding offsets must be strict integers")
        if binding["turn_id"] not in evidence["turn_ids"]:
            raise ValueError("claim binding must reference a cited source turn")
        if binding["end"] <= binding["start"] or binding["end"] - binding["start"] != len(binding["quote"]):
            raise ValueError("claim binding offsets must describe its exact Unicode quote length")
    if category == "V5" and data["origin"] == "agent_inference":
        if data["lifecycle_status"] == "active":
            raise ValueError("V5 model interpretation cannot become active factual state")
        if data.get("semantic_support", {}).get("status") == "rule_verified":
            raise ValueError("V5 model interpretation is not a rule-verified user fact")
    mappings = data.get("migration_map", [])
    if mappings:
        targets = [item["target_memory_id"] for item in mappings]
        if len(targets) != len(set(targets)) or data["memory_id"] not in targets:
            raise ValueError("migration targets must be unique and include this output memory")


def validate_memory_data(data: dict, *, allow_legacy: bool = True) -> None:
    if type(data) is not dict:
        raise ValueError("memory record must be a JSON object")
    version = data.get("schema_version")
    if version == "1.0" and allow_legacy:
        schema_name = "memory_record.schema.json"
    elif version == "2.0":
        schema_name = "memory_record_v2.schema.json"
    else:
        raise ValueError("unsupported or disallowed schema_version")
    schema = _load_schema(schema_name)
    # Reject NaN/infinity and non-JSON Python values before either entry point.
    def json_values(item: Any) -> None:
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                json_values(child)
        elif type(item) is list:
            for child in item:
                json_values(child)
        elif type(item) not in {str, int, float, bool, type(None)}:
            raise ValueError("record must contain only JSON value types")
    try:
        json_values(data)
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("record must contain only finite JSON values") from exc
    validate_schema_instance(data, schema)
    _semantic_rules(data)


@dataclass(slots=True)
class MemoryRecordV2(MemoryRecord):
    memory_class: str | None = None
    category: str | None = None
    classification_status: str = "pending"
    representation_kind: str = "source_quote"
    source_role: str = "unknown"
    source_time: str | None = None
    permissions: dict = field(kw_only=True)
    semantic_support: dict = field(default_factory=lambda: {"status": "unknown", "method": None, "version": None})
    claim_bindings: list[dict] = field(default_factory=list)
    event_time: str | None = None
    subject_id: str | None = None
    goal_id: str | None = None
    task_instance_id: str | None = None
    period_id: str | None = None
    attempt_id: str | None = None
    state_version: int | None = None
    migration_map: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        # asdict() would silently turn an already-dict value into valid-looking
        # JSON, although manager consumers require SourceEvidence attributes.
        # Raw mappings belong at construct_memory(), which converts them once.
        if not isinstance(self.source_evidence, SourceEvidence):
            raise ValueError("source_evidence must be SourceEvidence; use construct_memory for JSON mappings")
        validate_memory_data(self.to_dict(), allow_legacy=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecordV2":
        record = construct_memory(data, allow_legacy=False)
        assert isinstance(record, cls)
        return record


def construct_memory(data: dict, *, allow_legacy: bool = True) -> MemoryRecord | MemoryRecordV2:
    """The shared Python/JSON/CLI/extraction entry point; never coerces inputs."""
    validate_memory_data(data, allow_legacy=allow_legacy)
    if data["schema_version"] == "1.0":
        return MemoryRecord.from_dict(deepcopy(data))
    payload = deepcopy(data)
    payload["source_evidence"] = SourceEvidence(**payload["source_evidence"])
    return MemoryRecordV2(**payload)


def decode_json_document(raw: str) -> Any:
    """Reject duplicate JSON object keys and non-JSON numeric constants."""
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def bad_constant(value: str) -> None:
        raise ValueError(f"non-JSON numeric constant: {value}")

    if type(raw) is not str:
        raise ValueError("JSON document must be text")
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=bad_constant)


def validate_json_memory(data: dict | str, *, allow_legacy: bool = True) -> MemoryRecord | MemoryRecordV2:
    """JSON ingress executes formats and the same cross-field semantics."""
    if type(data) is str:
        data = decode_json_document(data)
    return construct_memory(data, allow_legacy=allow_legacy)


def legacy_metadata(record: MemoryRecord) -> dict:
    """Read v1 without inventing provenance, permission or confirmation."""
    if record.schema_version != "1.0":
        raise ValueError("legacy_metadata requires a 1.0 record")
    return {"schema_version": "1.0", "source_role": "unknown", "source_time": None,
            "permissions": deepcopy(UNKNOWN_PERMISSIONS), "classification_status": "pending"}


def split_legacy_memory(record: MemoryRecord, fragments: list[dict]) -> list[MemoryRecordV2]:
    """Explicit migration only: retain source mapping and never copy confirmation.

    Fragments are caller-reviewed transformation instructions, not inferred facts.
    Their semantic support stays unknown. This function does not update a store.
    """
    record.validate()
    if record.schema_version != "1.0" or not fragments:
        raise ValueError("split requires a 1.0 source and at least one fragment")
    ids = [item.get("memory_id") for item in fragments]
    if any(type(i) is not str or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("split target memory ids must be nonempty and unique")
    mappings = [{"source_memory_id": record.memory_id, "target_memory_id": target, "operation": "split"} for target in ids]
    outputs = []
    allowed = {"memory_id", "content", "memory_class", "category", "memory_type", "structured_payload"}
    for fragment in fragments:
        if set(fragment) - allowed:
            raise ValueError("unsupported split transformation fields")
        data = record.to_dict()
        data.update(fragment)
        data.update(schema_version="2.0", classification_status="assigned" if fragment.get("memory_class") else "pending",
                    confirmation_state="unconfirmed", lifecycle_status="candidate", migration_map=mappings,
                    source_role="unknown", source_time=None, permissions=deepcopy(UNKNOWN_PERMISSIONS))
        outputs.append(construct_memory(data, allow_legacy=False))
    return outputs


def downgrade_memory(record: MemoryRecord | MemoryRecordV2) -> MemoryRecord:
    """Reject a downgrade whenever v2 meaning would be silently discarded."""
    record.validate()
    if record.schema_version == "1.0":
        return construct_memory(record.to_dict())
    data = record.to_dict()
    safe = {"memory_class": None, "category": None, "classification_status": "pending",
            "representation_kind": "source_quote", "source_role": "unknown", "source_time": None,
            "permissions": UNKNOWN_PERMISSIONS, "semantic_support": {"status": "unknown", "method": None, "version": None},
            "claim_bindings": [], "event_time": None, "subject_id": None, "goal_id": None,
            "task_instance_id": None, "period_id": None, "attempt_id": None, "state_version": None, "migration_map": []}
    if any(data[key] != value for key, value in safe.items()):
        raise ValueError("lossy schema 2.0 downgrade refused")
    old_fields = {item.name for item in fields(MemoryRecord)}
    payload = {key: value for key, value in data.items() if key in old_fields}
    payload["schema_version"] = "1.0"
    return construct_memory(payload)
