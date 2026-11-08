"""Shared product policy for scoped refusal and low-burden clarification.

This module never confirms a memory or reads evaluation labels. Product and raw
comparison callers use the same pure functions and policy identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from zoneinfo import ZoneInfo

from dialogue_service import DialogueSession

POLICY_VERSION = "natural-memory-policy-v2.0"
PERMISSION_KEYS = ("save", "long_term_use", "response_external", "extraction_external")
POLICY_HASH = hashlib.sha256((POLICY_VERSION + ":subject,field,value;7d;one-question;scoped-refusal").encode()).hexdigest()


def moment(value: str) -> datetime:
    if type(value) is not str:
        raise ValueError("timestamp must be a string")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp requires a timezone")
    return dt


def permissions_for_input(text: str, defaults: dict) -> dict:
    """Separate retention, later use, response transmission and organization.

    Unknown is represented as None, and only literal True permits an operation.
    A no-save instruction does not imply permission to send to an extractor.
    """
    if type(text) is not str or type(defaults) is not dict or set(defaults) != set(PERMISSION_KEYS):
        raise ValueError("input text and four explicit permission axes are required")
    if any(v is not None and type(v) is not bool for v in defaults.values()):
        raise ValueError("permissions are bool or None (unknown)")
    result = dict(defaults)
    no_save = re.search(r"(?:这条|这句|这次|本条|本次).{0,5}(?:别|不要|不许).{0,3}(?:保存|记住|记录)|"
                        r"(?:不要|别)(?:保存|记住|记录)(?:这条|这句|这次|本条)?|"
                        r"\b(?:do not|don't)\s+(?:save|store|remember)\b", text, re.I)
    if no_save or DialogueSession._do_not_remember(text):
        result.update(save=False, long_term_use=False, extraction_external=False)
    if re.search(r"(?:不要|别).{0,4}(?:外发|发给模型|传给模型|上传)|\b(?:do not|don't)\s+(?:upload|send externally)\b", text, re.I):
        result.update(response_external=False, extraction_external=False)
    if re.search(r"(?:不要|别).{0,4}(?:整理|提取|抽取)|\b(?:do not|don't)\s+(?:extract|organize)\b", text, re.I):
        result["extraction_external"] = False
    if re.search(r"(?:以后|之后).{0,4}(?:不要|别).{0,4}(?:使用|引用)(?:这条|这个)?", text):
        result["long_term_use"] = False
    return result


def refusal_for_input(text: str, *, subject_id: str, as_of: str,
                      timezone_name: str = "UTC") -> dict | None:
    """A topic pause, a check refusal, and a task rejection are distinct."""
    if not subject_id:
        return None
    now = moment(as_of)
    if re.search(r"今天.{0,3}(?:不谈|别谈|不聊|别聊)|\bnot today\b", text, re.I):
        local = now.astimezone(ZoneInfo(timezone_name))
        end = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return {"subject_id": subject_id, "kind": "topic_pause", "scope": "today",
                "expires_at": end.isoformat(), "created_at": as_of}
    if re.search(r"(?:以后|今后|永远).{0,4}(?:别问|不要问|不再问)|\b(?:never ask|stop asking)\b", text, re.I):
        return {"subject_id": subject_id, "kind": "check_refusal", "scope": "until_changed",
                "expires_at": None, "created_at": as_of}
    if re.search(r"(?:这次|现在).{0,3}(?:别问|不要问)", text):
        return {"subject_id": subject_id, "kind": "check_refusal", "scope": "today",
                "expires_at": (now + timedelta(days=1)).isoformat(), "created_at": as_of}
    return None


@dataclass(frozen=True)
class CheckRequest:
    user_id: str
    subject_id: str
    field: str
    substantive_value: object
    uncertainty_kind: str
    question: str
    necessary: bool
    explicit_current: bool = False

    @property
    def identity(self) -> str:
        # Counts, timestamps and unrelated task changes are deliberately absent.
        value = {"user": self.user_id, "subject": self.subject_id, "field": self.field,
                 "value": self.substantive_value, "uncertainty": self.uncertainty_kind}
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         allow_nan=False).encode()).hexdigest()


def decide_check(request: CheckRequest, *, as_of: str, events=(), refusals=(),
                 already_shown_this_turn: bool = False) -> dict:
    if not isinstance(request, CheckRequest) or not request.subject_id or not request.field:
        raise ValueError("checks require a stable subject and field")
    if type(request.necessary) is not bool or type(request.explicit_current) is not bool:
        raise ValueError("check flags must be boolean")
    now = moment(as_of)
    reason = None
    if request.explicit_current:
        reason = "explicit_current_report"
    elif not request.necessary:
        reason = "not_material_to_current_answer"
    elif already_shown_this_turn:
        reason = "one_question_per_turn"
    for refusal in refusals:
        if refusal.get("subject_id") != request.subject_id:
            continue
        expiry = refusal.get("expires_at")
        if (expiry is None or now < moment(expiry)) and refusal.get("kind") in {"topic_pause", "check_refusal"}:
            reason = refusal["kind"]
    for event in events:
        if (event.get("identity") == request.identity and event.get("event") in
                {"check_shown", "check_answered", "check_declined"} and
                timedelta(0) <= now - moment(event["occurred_at"]) < timedelta(days=7)):
            reason = reason or "substantive_field_cooldown"
    return {"action": "suppress" if reason else "ask", "reason": reason or "material_ambiguity",
            "identity": request.identity, "policy_hash": POLICY_HASH,
            "subject_id": request.subject_id, "field": request.field,
            "question": request.question if reason is None else None,
            "not_memory_confirmation": True}


def bound_task_choice(text: str, proposals: list[dict], *, user_id: str, session_id: str,
                      reply_to_turn_id: str | None, as_of: str) -> dict | None:
    """Bind a yes only to a unique valid single-action product proposal.

    This returns a task selection witness, never user_confirmed memory status.
    A projector must independently validate proposal/source claim bindings.
    """
    if not re.fullmatch(r"\s*(?:yes|y|是的?|好(?:的)?|可以|愿意|同意)[。.!！]?\s*", text, re.I):
        return None
    now = moment(as_of)
    candidates = [p for p in proposals
                  if p.get("user_id") == user_id and p.get("session_id") == session_id
                  and p.get("status", "pending") in {"pending", "open"}
                  and p.get('consumed_at') is None
                  and moment(p["created_at"]) <= now < moment(p["expires_at"])]
    if len(candidates) != 1:
        return None
    proposal = candidates[0]
    if proposal.get('turn_id') != reply_to_turn_id or proposal.get('action_count') != 1 or proposal.get('question_count') != 1:
        return None
    return {"proposal_id": proposal["proposal_id"], "task_instance_id": proposal["task_instance_id"],
            "period_id": proposal["period_id"], "choice": "agreed",
            "not_memory_confirmation": True}
