"""Output-only finite answer adapter; no question, source, model or gold access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

VERSION = "common-output-only-answer-adapter-v3"
STATUSES = (
    "active", "agreed", "approved", "cancelled", "canceled", "completed", "confirmed", "continued",
    "declined", "deferred", "expired", "failed", "helpful", "improved", "in_progress", "mixed",
    "no_change", "not_agreed", "not_reported", "not_started", "ongoing", "open", "partially_completed",
    "paused", "pending", "planned", "proposed", "rejected", "resumed", "stopped", "succeeded",
    "unchanged", "unhelpful", "unknown", "withdrawn", "worse", "worsened",
)
PREFIX = "answer:"
INTEGER = re.compile(r"(?:0|[1-9][0-9]{0,3})\Z", re.ASCII)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def policy():
    return {
        "version": VERSION,
        "inputs": "one raw response string only; no question, arm, gold, state or sources",
        "domain": {"integer_minimum": 0, "integer_maximum": 9999, "string_enum": list(STATUSES), "null_allowed": True},
        "accepted_forms": ["complete JSON object with answer as its sole key", "bare ASCII canonical integer",
            "bare fixed lowercase status", "JSON-quoted fixed lowercase status", "bare null",
            "exact lowercase answer: prefix followed by one accepted scalar literal"],
        "prefix": PREFIX,
        "prefix_may_precede_object_or_another_prefix": False,
        "whitespace": "outer str.strip(); scalar whitespace after answer: may also be stripped; standard JSON whitespace inside JSON",
        "json_escapes": "standard JSON string/key escapes are decoded before exact domain/key validation",
        "quoted_null": "rejected string; only JSON null denotes a null answer",
        "rejected": ["extra or duplicate object keys", "multiple values", "booleans", "floats or exponents",
            "negative integers including -0", "leading zeros", "plus signs", "prose", "markdown fences", "unknown status strings"],
        "canonical_form": "compact single-answer JSON object; accepted values are never semantically inferred or coerced",
        "raw_sha256": "SHA256 of original UTF-8 text including original whitespace",
        "rejection": "accepted=false; rule=adapter_rejected; canonical text and SHA are absent as None, never an answer null",
        "caller_contract": "text must be a UTF-8-encodable str; invalid caller types or Unicode raise before adaptation",
        "model_or_logits_modified": False,
        "factual_correctness_established": False,
    }


def identity():
    return {"version": VERSION, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy_sha256": _sha(_canonical(policy())), "policy": policy()}


class _Rejected(ValueError):
    pass


def _integer(token):
    if INTEGER.fullmatch(token) is None:
        raise _Rejected("invalid_integer_literal")
    return int(token)


def _noninteger_number(token):
    raise _Rejected("noninteger_or_nonfinite_number")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Rejected("duplicate_object_key")
        result[key] = value
    return result


def _json(text, reason):
    try:
        return json.loads(text, object_pairs_hook=_unique, parse_int=_integer,
            parse_float=_noninteger_number, parse_constant=_noninteger_number)
    except _Rejected:
        raise
    except (ValueError, TypeError, RecursionError):
        raise _Rejected(reason) from None


def _domain(value):
    if value is None or type(value) is int and 0 <= value <= 9999 or type(value) is str and value in STATUSES:
        return value
    if type(value) in (bool, float, list, dict):
        raise _Rejected("invalid_answer_type")
    raise _Rejected("answer_outside_fixed_domain")


def _scalar(text):
    if INTEGER.fullmatch(text) is not None:
        return int(text), "integer"
    if text in STATUSES:
        return text, "status"
    if text == "null":
        return None, "null"
    if text.startswith('"'):
        value = _json(text, "invalid_json_string_literal")
        if type(value) is not str or value not in STATUSES:
            raise _Rejected("quoted_value_outside_fixed_status_enum")
        return value, "json_quoted_status"
    if text in ("true", "false"):
        raise _Rejected("invalid_answer_type")
    if text and text[0] in "+-0123456789":
        raise _Rejected("invalid_integer_literal")
    raise _Rejected("unsupported_complete_scalar")


def adapt(text):
    """Normalize one complete accepted form, or return an explicit rejection.

    The canonical fields are None on rejection. An accepted null instead has
    canonical_response_text='{"answer":null}' and its corresponding text hash.
    """
    if type(text) is not str:
        raise TypeError("adapter_input_must_be_text")
    result = {"accepted": False, "raw_response_text": text, "raw_response_sha256": _sha(text),
        "canonical_response_text": None, "canonical_response_sha256": None,
        "rule": "adapter_rejected", "reason": None}
    stripped = text.strip()
    try:
        if not stripped:
            raise _Rejected("empty_output")
        if stripped.startswith("{"):
            value = _json(stripped, "invalid_complete_json_object")
            if type(value) is not dict or set(value) != {"answer"}:
                raise _Rejected("single_answer_key_required")
            answer, rule = _domain(value["answer"]), "single_answer_json"
        elif stripped.startswith(PREFIX):
            literal = stripped[len(PREFIX):].strip()
            if not literal:
                raise _Rejected("missing_answer_literal")
            answer, form = _scalar(literal)
            rule = "answer_prefix_" + form
        else:
            answer, form = _scalar(stripped)
            rule = "bare_" + form if form != "json_quoted_status" else form
        canonical_text = _canonical({"answer": answer})
        return {**result, "accepted": True, "canonical_response_text": canonical_text,
            "canonical_response_sha256": _sha(canonical_text), "rule": rule, "reason": "accepted"}
    except _Rejected as error:
        return {**result, "reason": str(error)}
