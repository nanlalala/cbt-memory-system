"""Validate full quotation/source bindings against an authorized visible catalog.

Pure response validation: no questions, retrieval, gold, database, model or I/O.
The caller must reconstruct ``catalog`` from the exact request and revalidate
current source controls. This module cannot grant permission or recover sources
that the response model did not see.
"""
from collections.abc import Mapping, Sequence
import json

VERSION = "visible-full-quotation-source-binding-v1"
FIELDS = frozenset({"quote", "source_id", "unknown"})


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError("non-finite JSON value")
    if type(text) is not str:
        raise ValueError("response must be JSON text")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def _norm(text):
    return " ".join(text.split())


def _wire(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _visible_users(catalog):
    if not isinstance(catalog, Sequence) or isinstance(catalog, (str, bytes)):
        raise ValueError("validated visible-user catalog sequence required")
    sources = {}
    for source in catalog:
        if (not isinstance(source, Mapping) or source.get("role") != "user" or
                type(source.get("source_id")) is not str or not source["source_id"].strip() or
                type(source.get("text")) is not str or not source["text"].strip()):
            raise ValueError("catalog must contain only bound nonempty visible user sources")
        sid, text = source["source_id"], source["text"]
        if sid in sources and sources[sid] != text:
            raise ValueError("conflicting visible source identity")
        sources[sid] = text
    return sources


def repair_source_quote_binding(response_text, catalog):
    """Keep coherent answers; repair only a unique complete-quote ID mismatch.

    Only whitespace is normalized for quotation matching. A correct declared
    binding is retained even if another visible source has identical wording.
    An ambiguous or absent quotation, malformed answer, or incoherent unknown
    becomes explicit unknown. An original unknown never becomes known.
    """
    sources = _visible_users(catalog)
    unknown_wire = _wire({"quote": "", "source_id": "", "unknown": True})

    def result(output, *, action, reason, input_status, output_status, matches=None):
        return {"version": VERSION, "output_text": output,
            "changed": output != response_text, "action": action, "reason": reason,
            "input_status": input_status, "output_status": output_status,
            "matched_visible_source_count": matches}

    def reject(reason, input_status="invalid", matches=None):
        return result(unknown_wire, action="rejected_to_unknown", reason=reason,
                      input_status=input_status, output_status="unknown", matches=matches)

    try:
        answer = _strict_json(response_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return reject("invalid_json")
    if (type(answer) is not dict or set(answer) != FIELDS or
            type(answer.get("quote")) is not str or type(answer.get("source_id")) is not str or
            type(answer.get("unknown")) is not bool):
        return reject("invalid_answer_schema")
    if answer["unknown"]:
        if answer["quote"] or answer["source_id"]:
            return reject("incoherent_unknown_answer", input_status="unknown")
        return result(response_text, action="passed_through", reason="original_unknown_retained",
                      input_status="unknown", output_status="unknown", matches=0)
    if not answer["quote"].strip() or not answer["source_id"].strip():
        return reject("known_answer_missing_quote_or_source", input_status="known")
    matches = [sid for sid, text in sources.items() if _norm(text) == _norm(answer["quote"])]
    if answer["source_id"] in matches:
        return result(response_text, action="passed_through", reason="coherent_visible_source_binding",
                      input_status="known", output_status="known", matches=len(matches))
    if len(matches) == 1:
        corrected = {**answer, "source_id": matches[0]}
        return result(_wire(corrected), action="repaired_source_id", reason="unique_visible_full_quote_source",
                      input_status="known", output_status="known", matches=1)
    return reject("ambiguous_visible_full_quote" if matches else "full_quote_not_in_visible_catalog",
                  input_status="known", matches=len(matches))
