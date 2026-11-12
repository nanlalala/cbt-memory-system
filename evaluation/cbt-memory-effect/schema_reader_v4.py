"""Prospective question-only answer schema, never an evaluator repair.

The same public question classifier selects an output *type* in every arm.
No evidence, task value, case identity, arm, or reference answer enters the
grammar. All values of the selected type remain possible. Unrecognized or
ambiguous questions retain the complete scalar domain rather than forcing null.
Actual tokenization and inference require genuine hosted Colab. Unit fixtures
can explicitly exercise preflight using tiny invented character tokens.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

VERSION = "question-only-complete-json-schema-reader-v4"
STATUSES = (
    "active", "agreed", "approved", "cancelled", "canceled", "completed",
    "confirmed", "continued", "declined", "deferred", "expired", "failed",
    "helpful", "improved", "in_progress", "mixed", "no_change", "not_agreed",
    "not_reported", "not_started", "ongoing", "open", "partially_completed",
    "paused", "pending", "planned", "proposed", "rejected", "resumed",
    "stopped", "succeeded", "unchanged", "unhelpful", "unknown", "withdrawn",
    "worse", "worsened",
)
HERE = Path(__file__).resolve().parent
FIELD_PATH = HERE.parents[1] / "modules/agent-integration/question_field_selection_v4.py"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _field_module():
    name = "cbt_question_field_selection_v4_schema"
    module = sys.modules.get(name)
    source_sha = hashlib.sha256(FIELD_PATH.read_bytes()).hexdigest()
    if module is None:
        spec = importlib.util.spec_from_file_location(name, FIELD_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError("owned_question_classifier_required")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        search_path = list(sys.path)
        try:
            sys.path.insert(0, str(FIELD_PATH.parent))
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.path[:] = search_path
        module._schema_source_sha256 = source_sha
    if Path(module.__file__).resolve() != FIELD_PATH.resolve():
        raise RuntimeError("foreign_question_classifier")
    if getattr(module, "_schema_source_sha256", None) != source_sha:
        raise RuntimeError("question_classifier_changed_after_import")
    return module


def policy():
    return {
        "version": VERSION,
        "selection_inputs": "current public question text only",
        "no_arm_gold_case_evidence_or_memory_value_input": True,
        "count_domain": {"integer_minimum": 0, "integer_maximum": 9999},
        "status_domain": list(STATUSES),
        "null_allowed_in_every_contract": True,
        "ambiguous_question_policy": "complete scalar union; never forced null",
        "wire_schema": {"type": "object", "required": ["answer"],
                        "additionalProperties": False},
        "generation": "complete compact JSON; no appended assistant prefill",
        "eos_policy": "only after one complete accepted JSON object",
        "is_semantic_validation": False,
    }


def schema_for_question(question: str):
    if type(question) is not str or not question.strip():
        raise ValueError("nonempty_current_question_required")
    field_contract = _field_module().question_field_contract(question)
    field = field_contract.get("field")
    kind = field_contract.get("value_kind")
    expected_kind = ("integer" if field in ("target_count", "reported_count")
                     else "status" if field in ("outcome", "acceptance", "engagement")
                     else "unrestricted_scalar" if field is None else None)
    if expected_kind is None or kind != expected_kind:
        raise ValueError("unsupported_or_inconsistent_question_field_contract")
    return {
        "version": VERSION,
        "question_sha256": digest(question),
        "question_classifier_sha256": hashlib.sha256(FIELD_PATH.read_bytes()).hexdigest(),
        "field": field,
        "value_kind": kind,
        "integer_allowed": kind in ("integer", "unrestricted_scalar"),
        "integer_minimum": 0,
        "integer_maximum": 9999,
        "status_allowed": kind in ("status", "unrestricted_scalar"),
        "string_enum": list(STATUSES),
        "null_allowed": True,
        "additional_properties": False,
        "policy_sha256": digest(policy()),
    }


def _checked_contract(contract):
    if type(contract) is not dict:
        raise ValueError("frozen_question_schema_required")
    field = contract.get("field")
    kind = ("integer" if field in ("target_count", "reported_count")
            else "status" if field in ("outcome", "acceptance", "engagement")
            else "unrestricted_scalar" if field is None else None)
    if (kind is None or contract.get("version") != VERSION
            or contract.get("value_kind") != kind
            or contract.get("integer_allowed") is not (kind in ("integer", "unrestricted_scalar"))
            or contract.get("status_allowed") is not (kind in ("status", "unrestricted_scalar"))
            or contract.get("integer_minimum") != 0
            or contract.get("integer_maximum") != 9999
            or contract.get("string_enum") != list(STATUSES)
            or contract.get("null_allowed") is not True
            or contract.get("additional_properties") is not False
            or contract.get("policy_sha256") != digest(policy())):
        raise ValueError("question_schema_domain_changed")
    return contract


def candidate_texts(contract):
    contract = _checked_contract(contract)
    values = [None]
    if contract["integer_allowed"]:
        values.extend(range(10000))
    if contract["status_allowed"]:
        values.extend(STATUSES)
    return [canonical({"answer": value}) for value in values]


def validate_response(text, contract):
    """Strict validation only; no markdown removal, extraction, coercion or fix."""
    contract = _checked_contract(contract)
    if type(text) is not str:
        return {"accepted": False, "reason": "not_text", "answer": None}

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = value
        return result

    def no_constant(value):
        raise ValueError("non_json_constant")

    try:
        obj = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"accepted": False, "reason": "invalid_whole_json", "answer": None}
    if type(obj) is not dict or set(obj) != {"answer"}:
        return {"accepted": False, "reason": "single_answer_object_required", "answer": None}
    answer = obj["answer"]
    valid = (answer is None or
             (contract["integer_allowed"] and type(answer) is int and 0 <= answer <= 9999) or
             (contract["status_allowed"] and type(answer) is str and answer in STATUSES))
    if not valid:
        return {"accepted": False, "reason": "value_outside_question_schema", "answer": None}
    # Integer -0 is valid JSON but is not the grammar's canonical representation.
    if type(answer) is int and text.strip() != canonical(obj):
        # Whitespace is harmless; reject alternate numeric spellings only.
        try:
            wire = json.loads(text, parse_int=lambda value: value)
        except ValueError:
            wire = {}
        if wire.get("answer") == "-0":
            return {"accepted": False, "reason": "noncanonical_negative_zero", "answer": None}
    return {"accepted": True, "reason": "valid_question_schema", "answer": answer}


def require_hosted_colab():
    try:
        hosted = (bool(os.environ.get("COLAB_RELEASE_TAG")) and Path("/content").is_dir()
                  and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        hosted = False
    if not hosted:
        raise RuntimeError("genuine_hosted_colab_required_before_tokenizer_or_model_io")


class AnswerTrie:
    def __init__(self, tokenizer, eos_token_ids, *, question: str, fixture_only=False):
        if not fixture_only:
            require_hosted_colab()
        self.schema = schema_for_question(question)
        raw_eos = [eos_token_ids] if type(eos_token_ids) is int else list(eos_token_ids)
        if not raw_eos or any(type(value) is not int or value < 0 for value in raw_eos):
            raise ValueError("explicit_nonempty_integer_eos_ids_required")
        self.eos = sorted(set(raw_eos))
        self.root = {}
        self.terminals = {}
        encoded = []
        for text in candidate_texts(self.schema):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if (not ids or any(type(value) is not int or value < 0 or value in self.eos for value in ids)
                    or tokenizer.decode(ids, skip_special_tokens=False) != text):
                raise ValueError("every_complete_candidate_must_roundtrip_without_eos")
            key = tuple(ids)
            if key in self.terminals:
                raise ValueError("ambiguous_tokenized_complete_candidates")
            self.terminals[key] = text
            node = self.root
            for token in ids:
                node = node.setdefault(token, {})
            encoded.append([text, ids])
        self.maximum_candidate_tokens = max(len(key) for key in self.terminals)
        self.identity = {
            "policy": policy(), "policy_sha256": digest(policy()),
            "question_schema": self.schema, "question_schema_sha256": digest(self.schema),
            "tokenizer_chat_template_sha256": digest(tokenizer.chat_template),
            "tokenizer_class": type(tokenizer).__name__,
            "encoded_language_sha256": digest(encoded), "candidate_count": len(encoded),
            "eos_token_ids": self.eos, "maximum_candidate_tokens": self.maximum_candidate_tokens,
        }

    def allowed(self, prefix):
        if any(type(value) is not int or value < 0 for value in prefix):
            raise ValueError("integer_token_prefix_required")
        key = tuple(prefix)
        node = self.root
        for token in key:
            if token not in node:
                raise ValueError("generated_prefix_outside_question_schema")
            node = node[token]
        result = sorted(node)
        if key in self.terminals:
            result += self.eos
        if not result:
            raise ValueError("nonterminal_dead_end")
        return result

    def callback(self, input_tokens):
        def allowed_tokens(batch_id, input_ids):
            if batch_id != 0:
                raise ValueError("single_example_generation_required")
            return self.allowed(input_ids.tolist()[input_tokens:])
        return allowed_tokens

    def finish(self, token_ids):
        if (not token_ids or any(type(value) is not int for value in token_ids)
                or token_ids[-1] not in self.eos):
            raise ValueError("complete_schema_output_must_end_with_eos")
        key = tuple(token_ids[:-1])
        if any(value in self.eos for value in key) or key not in self.terminals:
            raise ValueError("not_one_complete_question_schema_object")
        text = self.terminals[key]
        checked = validate_response(text, self.schema)
        if not checked["accepted"]:
            raise ValueError("grammar_validator_disagreement")
        return {"response_text": text, "chosen_answer": checked["answer"],
                "raw_generated_token_ids": list(token_ids),
                "decoding_identity_sha256": digest(self.identity)}


def _inputs(tokenizer, messages, trie):
    if (not isinstance(messages, list) or not messages
            or messages[-1].get("role") != "user"
            or digest(messages[-1].get("content")) != trie.schema["question_sha256"]):
        raise ValueError("last_user_question_must_match_frozen_question_schema")
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    if set(inputs) - {"input_ids", "attention_mask"}:
        raise ValueError("unsupported_chat_tensor_field")
    ids = inputs["input_ids"]
    if len(ids.shape) != 2 or ids.shape[0] != 1:
        raise ValueError("single_example_chat_input_required")
    count = int(ids.shape[-1])
    if count < 1 or count > 4096:
        raise ValueError("actual_input_token_limit_no_truncation")
    if "attention_mask" in inputs and tuple(inputs["attention_mask"].shape) != (1, count):
        raise ValueError("attention_mask_shape_changed")
    return inputs, count


def input_preflight(tokenizer, messages, trie, *, fixture_only=False):
    if not fixture_only:
        require_hosted_colab()
    inputs, count = _inputs(tokenizer, messages, trie)
    return {"input_tokens": count, "actual_input_token_count": count,
            "base_input_token_count": count, "prefilled_token_count": 0,
            "input_tensor_sha256": digest(inputs["input_ids"].tolist()),
            "question_schema_sha256": digest(trie.schema)}


def generate(model, tokenizer, messages, trie, *, max_new_tokens=64):
    require_hosted_colab()
    if type(max_new_tokens) is not int or max_new_tokens <= trie.maximum_candidate_tokens:
        raise ValueError("output_budget_must_fit_every_complete_candidate_plus_eos")
    import torch
    inputs, count = _inputs(tokenizer, messages, trie)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=trie.callback(count))
    ids = output[0][count:].tolist()
    result = trie.finish(ids)
    if tokenizer.decode(ids, skip_special_tokens=True) != result["response_text"]:
        raise ValueError("decoded_actual_wire_differs_from_accepted_schema")
    return {**result, "raw_generated_text": tokenizer.decode(ids, skip_special_tokens=False),
            "response_sha256": hashlib.sha256(result["response_text"].encode()).hexdigest(),
            "input_tensor_sha256": digest(inputs["input_ids"].tolist()),
            "finish_reason": "eos", "prefilled_token_count": 0,
            "usage": {"prompt_tokens": count, "completion_tokens": len(ids),
                      "total_tokens": count + len(ids)}}
