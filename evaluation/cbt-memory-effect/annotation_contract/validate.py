"""Validate source-span annotation structure; never certify clinical semantics.

Connects to cactus_materials' pending_annotations rows and natural generation
cases. Annotation files stay assessor-only. This module has no model/network path.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

VERSION = "natural-cbt-memory-source-annotation-v1"
STATUSES = {"pending_independent_source_annotation", "draft_source_annotation", "under_independent_review",
            "adjudication_required", "independently_reviewed"}
KINDS = {"situation", "thought", "emotion", "goal", "practice", "preference", "reported_count",
         "target_count", "participation", "user_consent", "assistant_suggestion", "reported_outcome"}
AXES = {"acceptance", "participation", "reported_count", "target_count", "preference", "goal_status", "reported_outcome"}


def text_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def case_hash(case):
    return hashlib.sha256(json.dumps({k: v for k, v in case.items() if k != "case_sha256"},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact_keys(value, required, optional=()):
    if type(value) is not dict or not set(required) <= set(value) or set(value) - set(required) - set(optional):
        raise ValueError("missing or unexpected annotation fields")


def draft_from_pending(pending_row, case):
    """Create a NEW annotation draft from the existing untouched pending row.

    Do not replace the original compiler output or its manifest hashes. This
    adapter only attaches identity and empty review/response-policy fields.
    """
    if (pending_row.get("case_id") != case["case_id"] or pending_row.get("case_sha256") != case["case_sha256"] or
            pending_row.get("annotation_status") != "pending_independent_source_annotation" or
            pending_row.get("memory_targets") is not None or pending_row.get("expected_current_state") is not None or
            pending_row.get("semantic_gold_available") is not False or pending_row.get("future_counselor_answer_is_gold") is not False):
        raise ValueError("only the unchanged existing pending row can seed a new annotation")
    annotation = {"annotation_schema_version": VERSION, "case_id": case["case_id"], "case_sha256": case["case_sha256"],
        "source_family_id": case["source_family_id"], "fixture_only": case.get("fixture_only") is True,
        "annotation_status": "pending_independent_source_annotation", "memory_targets": None, "expected_current_state": None,
        "expected_response_policy": {"abstain_on_fact_ids": [], "do_not_assert_fact_ids": [], "clarification": "unreviewed"},
        "review": {"label_origin": "unannotated", "reviewer_ids": [], "agreement": "unreviewed", "adjudicator_id": None, "human_record_refs": []},
        "semantic_gold_available": False, "future_counselor_answer_is_gold": False}
    validate(annotation, case)
    return annotation


def validate(annotation, case, *, verified_human_registry=None):
    exact_keys(annotation, {"annotation_schema_version", "case_id", "case_sha256", "source_family_id",
        "fixture_only", "annotation_status", "memory_targets", "expected_current_state", "expected_response_policy",
        "review", "semantic_gold_available", "future_counselor_answer_is_gold"})
    if annotation["annotation_schema_version"] != VERSION or annotation["annotation_status"] not in STATUSES:
        raise ValueError("unknown source annotation schema/status")
    if case.get("task_kind") != "natural_cbt_continuation" or case.get("case_sha256") != case_hash(case):
        raise ValueError("unchanged natural continuation case required")
    for field in ("case_id", "case_sha256", "source_family_id"):
        if annotation[field] != case[field]:
            raise ValueError("annotation case/family/cutoff identity changed")
    if type(annotation["fixture_only"]) is not bool or type(annotation["semantic_gold_available"]) is not bool:
        raise ValueError("explicit annotation identity flags required")
    fixture_identity = case.get("fixture_only") is True or case.get("source_kind") == "invented_fixture"
    if annotation["fixture_only"] != fixture_identity:
        raise ValueError("annotation fixture identity differs from source case")
    if annotation["future_counselor_answer_is_gold"] is not False:
        raise ValueError("future counselor continuation cannot supply memory truth")
    history = {turn["message_id"]: turn for turn in case["history"]}
    if len(history) != len(case["history"]):
        raise ValueError("duplicate original message IDs")
    cutoff = case["history_cutoff_order_exclusive"]
    if any(type(turn["source_order"]) is not int or not 0 <= turn["source_order"] < cutoff for turn in history.values()):
        raise ValueError("history source order exceeds cutoff")
    current = {"message_id": case["input_message_id"], "role": "user", "content": case["input"], "source_order": cutoff}
    sources = {**history, current["message_id"]: current}
    if case.get("calendar_time_available") is False and any(turn.get("date") is not None for turn in case["history"]):
        raise ValueError("ordinal sources cannot acquire invented calendar dates")
    if any(key in case for key in ("locator_cue", "disclosed_cue", "quote_prefix", "gold", "memory_targets")):
        raise ValueError("natural generation input cannot contain locator or assessor labels")
    targets = annotation["memory_targets"]
    states = annotation["expected_current_state"]
    if targets is None or states is None:
        if annotation["annotation_status"] != "pending_independent_source_annotation" or annotation["semantic_gold_available"]:
            raise ValueError("missing annotations cannot become completed truth")
        targets = []; states = []
    if type(targets) is not list or type(states) is not list:
        raise ValueError("target and state annotations must be arrays or pending null")
    by_id = {}
    evidence_orders = {}
    for target in targets:
        exact_keys(target, {"fact_id", "kind", "assertion", "value", "truth_status", "necessity", "source_scope",
                           "permitted_for_response", "evidence", "unknown_reason"})
        fid = target["fact_id"]
        if type(fid) is not str or not fid or fid in by_id or target["kind"] not in KINDS:
            raise ValueError("unique fact ID and known memory kind required")
        if target["truth_status"] not in {"known", "unknown", "not_annotated"} or target["necessity"] not in {"required_memory", "optional", "context_only"}:
            raise ValueError("truth and necessity must remain distinct")
        if target["source_scope"] not in {"history", "current_input"} or type(target["permitted_for_response"]) is not bool:
            raise ValueError("explicit history/current and permission scope required")
        if target["necessity"] == "required_memory" and (target["source_scope"] != "history" or target["permitted_for_response"] is not True):
            raise ValueError("memory coverage denominator contains only permitted historical facts")
        if type(target["assertion"]) is not str or not target["assertion"].strip() or type(target["evidence"]) is not list:
            raise ValueError("annotation assertion and source spans required")
        if target["truth_status"] != "known" and (target["value"] is not None or not target["unknown_reason"]):
            raise ValueError("unknown/unannotated value must stay null with a reason")
        if target["truth_status"] == "known" and (target["value"] is None or not target["evidence"]):
            raise ValueError("known fact requires a value and exact source evidence")
        roles = []; orders = []
        for span in target["evidence"]:
            exact_keys(span, {"message_id", "source_order", "source_text_sha256", "start", "end"})
            source = sources.get(span["message_id"])
            if source is None or span["source_text_sha256"] != text_hash(source["content"]) or span["source_order"] != source["source_order"]:
                raise ValueError("span original source identity changed")
            if type(span["start"]) is not int or type(span["end"]) is not int or not 0 <= span["start"] < span["end"] <= len(source["content"]):
                raise ValueError("source span must use exact Unicode offsets")
            if target["source_scope"] == "history" and span["message_id"] not in history:
                raise ValueError("current input cannot be relabelled as remembered history")
            roles.append(source["role"]); orders.append(source["source_order"])
        if target["truth_status"] == "known":
            if target["kind"] == "assistant_suggestion":
                if any(role != "assistant" for role in roles):
                    raise ValueError("assistant suggestion requires assistant source")
            elif any(role != "user" for role in roles):
                raise ValueError("assistant wording cannot establish a user fact or consent")
        by_id[fid] = target; evidence_orders[fid] = max(orders) if orders else -1
    state_axes = set()
    for state in states:
        exact_keys(state, {"state_id", "subject_id", "axis", "value", "truth_status", "basis_fact_ids", "superseded_fact_ids"})
        if state["axis"] not in AXES or not state["subject_id"] or not state["state_id"]:
            raise ValueError("explicit state subject/axis required")
        identity = (state["subject_id"], state["axis"])
        if identity in state_axes:
            raise ValueError("one latest state per subject and axis required")
        state_axes.add(identity)
        if state["axis"] in {"reported_count", "target_count"} and state["truth_status"] == "known" and (type(state["value"]) is not int or state["value"] < 0):
            raise ValueError("known practice counts must be explicit nonnegative integers")
        if state["truth_status"] not in {"known", "unknown", "not_annotated"}:
            raise ValueError("explicit current-state truth status required")
        if state["truth_status"] != "known" and state["value"] is not None:
            raise ValueError("unknown current state cannot contain a guessed value")
        refs = state["basis_fact_ids"] + state["superseded_fact_ids"]
        if not state["basis_fact_ids"] or any(fid not in by_id for fid in refs) or set(state["basis_fact_ids"]) & set(state["superseded_fact_ids"]):
            raise ValueError("current state needs distinct known source annotation references")
        if state["truth_status"] == "known" and any(by_id[fid]["truth_status"] != "known" or by_id[fid]["kind"] == "assistant_suggestion" for fid in state["basis_fact_ids"]):
            raise ValueError("unconfirmed proposal/unknown label cannot establish current user state")
        if state["superseded_fact_ids"] and max(evidence_orders[fid] for fid in state["superseded_fact_ids"]) >= min(evidence_orders[fid] for fid in state["basis_fact_ids"]):
            raise ValueError("a supposed latest update precedes its superseded evidence")
    policy = annotation["expected_response_policy"]
    exact_keys(policy, {"abstain_on_fact_ids", "do_not_assert_fact_ids", "clarification"})
    if policy["clarification"] not in {"none_needed", "relevant_unknown_only", "unreviewed"}:
        raise ValueError("clarification policy must be explicit")
    if any(fid not in by_id for fid in policy["abstain_on_fact_ids"] + policy["do_not_assert_fact_ids"]):
        raise ValueError("response policy references unknown fact")
    if any(by_id[fid]["truth_status"] == "known" for fid in policy["abstain_on_fact_ids"]):
        raise ValueError("abstention target must be explicitly unknown or unannotated")
    review = annotation["review"]
    exact_keys(review, {"label_origin", "reviewer_ids", "agreement", "adjudicator_id", "human_record_refs"})
    if review["label_origin"] not in {"human_draft", "model_proposal", "unannotated", "invented_fixture"}:
        raise ValueError("label origin must distinguish model proposals from independent truth")
    completed = annotation["annotation_status"] == "independently_reviewed"
    if annotation["semantic_gold_available"] != completed:
        raise ValueError("semantic gold requires completed independent source review")
    if completed:
        if annotation["fixture_only"] or review["label_origin"] != "human_draft" or len(set(review["reviewer_ids"])) < 2:
            raise ValueError("independent truth requires two distinct verified human source reviewers")
        registry = verified_human_registry or {}
        for rid in review["reviewer_ids"]:
            rater = registry.get(rid, {})
            if rater.get("human_not_model_attested") is not True or rater.get("independent_from_system_development") is not True or rater.get("arm_blinded") is not True:
                raise ValueError("external human reviewer registry verification required")
        if len(review["human_record_refs"]) < 2 or review["agreement"] not in {"agreed", "adjudicated"}:
            raise ValueError("human annotation records and agreement/adjudication required")
        if review["agreement"] == "adjudicated":
            if not review["adjudicator_id"] or review["adjudicator_id"] in review["reviewer_ids"]:
                raise ValueError("disagreement needs an independent third adjudicator")
            adjudicator = registry.get(review["adjudicator_id"], {})
            if any(adjudicator.get(flag) is not True for flag in
                   ("human_not_model_attested", "independent_from_system_development", "arm_blinded")):
                raise ValueError("third adjudicator also requires external registry verification")
        if any(target["truth_status"] == "not_annotated" for target in targets):
            raise ValueError("unfinished labels cannot be approved gold")
    return {"schema_version": VERSION, "structurally_valid": True, "semantic_review_certified_by_validator": False,
        "semantic_gold_available": completed, "memory_required_known_facts": sum(target["necessity"] == "required_memory" and target["truth_status"] == "known" for target in targets),
        "memory_required_unknown_facts": sum(target["necessity"] == "required_memory" and target["truth_status"] == "unknown" for target in targets),
        "user_or_professional_scores": None, "provider_calls": 0}
