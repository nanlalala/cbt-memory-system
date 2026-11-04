"""Evidence-aware conversational use of memory, independent of write confirmation.

This module never activates a record or changes user confirmation.  It is a
separate, opt-in reading policy for the conversational demonstration.  The caller
must authenticate source quotes independently, apply relevance gating first, and
rate-limit any resulting check.  A model's numerical ``confidence`` is not proof
of truth and is deliberately not used as an eligibility threshold.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import AbstractSet

from memory_models import MemoryRecord, is_valid_at


POLICY_VERSION = "conversational-use-v1"

# These cues only request clarification.  They do not infer a diagnosis, task
# agreement, preference, completion state or permanent user profile.
_DYNAMIC_CUES = re.compile(
    r"目标|作业|思维记录|行为实验|暴露练习|情绪|焦虑|压力|低落|难受|心情|"
    r"累|麻烦|负担|坚持|完成|练习|计划|这周|本周|最近|"
    r"\b(?:goals?|homework|assignments?|thought\s+records?|behaviou?ral\s+"
    r"(?:experiments?|activation)|exposure|mood|anxious|anxiety|stress|"
    r"overwhelmed|exhausted|burden|completed?|plans?|this\s+week|lately)\b",
    re.IGNORECASE,
)
_PREFERENCE_CUES = re.compile(
    r"喜欢|偏好|不想|不再|讨厌|愿意|兴趣|"
    r"\b(?:prefer|likes?|dislikes?|enjoy|hate|interested|no\s+longer|not\s+anymore)\b",
    re.IGNORECASE,
)
_CHANGE_CUES = re.compile(
    r"不再|改成|改为|换成|现在不|已经不|"
    r"\b(?:no\s+longer|anymore|any\s+longer|used\s+to|changed?|switched?)\b",
    re.IGNORECASE,
)
_DYNAMIC_TYPES = frozenset({
    "goal", "cbt_assignment", "assignment_outcome", "emotion_record",
    "cognitive_pattern", "safety_signal",
})


@dataclass(frozen=True, slots=True)
class MemoryUsePolicy:
    """Configurable engineering starting points, not clinical cutoffs.

    Dynamic CBT state is checked whenever the caller intends current use, even
    when recent.  These age thresholds describe freshness for other statements;
    they are not calibrated probabilities and need evaluation with real usage.
    """

    preference_recheck_days: float = 30.0
    general_recheck_days: float = 180.0
    recent_state_days: float = 7.0
    future_clock_tolerance_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "preference_recheck_days", "general_recheck_days", "recent_state_days",
            "future_clock_tolerance_seconds",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True, slots=True)
class MemoryUseDecision:
    memory_id: str
    action: str
    reason_codes: tuple[str, ...]
    evidence_label: str
    confidence_label: str
    freshness_label: str
    currentness_label: str
    reference_text: str = ""
    observed_at: str = ""
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _time(value: str | datetime) -> datetime:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(moment, datetime) or moment.tzinfo is None:
        raise ValueError("memory-use timestamps must include a timezone")
    return moment.astimezone(timezone.utc)


def assess_memory_use(
    record: MemoryRecord,
    *,
    user_id: str,
    as_of: str | datetime | None = None,
    trusted_quote_ids: AbstractSet[str] = frozenset(),
    verified_source_ids: AbstractSet[str] = frozenset(),
    conflicting_ids: AbstractSet[str] = frozenset(),
    current_use: bool = False,
    policy: MemoryUsePolicy | None = None,
) -> MemoryUseDecision:
    """Choose historical reference, just-in-time clarification, or exclusion.

    ``trusted_quote_ids`` means the caller verified the exact stored quote
    against its user-source audit. ``verified_source_ids`` authenticates only
    source evidence, not a model's summary. Neither collection may come from a
    model-controlled record field. ``current_use`` means the answer would apply
    a past report to the current situation; a dated historical query sets False.
    """

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id is required")
    config = policy or MemoryUsePolicy()
    now = _time(as_of) if as_of is not None else datetime.now(timezone.utc)

    def excluded(reason: str) -> MemoryUseDecision:
        # Never return the body/source of inaccessible or unsupported memories.
        return MemoryUseDecision(
            record.memory_id, "exclude", (reason,), "not_eligible", "不可使用",
            "not_assessed", "excluded",
        )

    if record.user_id != user_id:
        return excluded("different_user")
    if record.lifecycle_status not in {"active", "candidate"}:
        return excluded("inactive_lifecycle")
    if record.sensitivity == "restricted" or record.memory_type == "safety_signal":
        return excluded("restricted_memory")
    if not is_valid_at(record, now):
        return excluded("outside_validity_interval")

    age_seconds = (now - _time(record.observed_at)).total_seconds()
    if age_seconds < -config.future_clock_tolerance_seconds:
        return excluded("future_observation")
    age_days = max(0.0, age_seconds / 86400)

    confirmed = (
        record.lifecycle_status == "active"
        and record.confirmation_state in {"user_confirmed", "user_corrected"}
    )
    quote_verified = (
        record.memory_id in trusted_quote_ids
        and record.origin in {"explicit_user_statement", "user_correction"}
        and bool(record.source_evidence.supporting_text.strip())
        and record.content == record.source_evidence.supporting_text
    )
    source_verified = record.memory_id in verified_source_ids

    if confirmed:
        evidence = "user_confirmed"
        reference = record.content
    elif quote_verified:
        evidence = "source_verified_user_quote"
        reference = record.source_evidence.supporting_text
    elif source_verified and record.source_evidence.supporting_text.strip():
        # A real quote does not entail the model-generated summary. Only the
        # quote can be recalled as a past statement or used to ask a neutral
        # current-use question; the model's summary always stays hidden.
        evidence = "verified_source_only"
        reference = record.source_evidence.supporting_text
    else:
        return excluded("unverified_content_or_source")

    dynamic = record.memory_type in _DYNAMIC_TYPES or bool(_DYNAMIC_CUES.search(reference))
    preference = record.memory_type == "preference" or bool(_PREFERENCE_CUES.search(reference))
    limit_days = (
        config.recent_state_days if dynamic else
        config.preference_recheck_days if preference else config.general_recheck_days
    )
    stale = age_days >= limit_days
    freshness = "possibly_outdated" if stale else "recent_report"
    reasons: list[str] = [evidence]

    if evidence == "verified_source_only":
        reasons.append("summary_not_verified")
    if current_use:
        if record.memory_id in conflicting_ids:
            reasons.append("unresolved_conflict")
        if dynamic:
            reasons.append("dynamic_state_requires_current_check")
        if preference and _CHANGE_CUES.search(reference):
            reasons.append("changeable_preference")
        if stale:
            reasons.append("freshness_limit_reached")
        if record.sensitivity == "sensitive":
            reasons.append("sensitive_current_use")

    # An explicitly historical question does not need a present-day check.
    # Source-only evidence remains usable solely as the authenticated quote,
    # never as support for the model-generated summary attached to the record.
    needs_check = current_use and (evidence == "verified_source_only" or len(reasons) > 1)
    if needs_check:
        return MemoryUseDecision(
            record.memory_id, "check_current", tuple(reasons), evidence,
            "用户核对过，当前情况待核对" if confirmed else "原话有依据，当前情况待核对",
            freshness, "not_established_for_current_use",
            reference, record.observed_at,
        )
    return MemoryUseDecision(
        record.memory_id, "use_as_history", tuple(reasons), evidence,
        ("经用户核对的历史记录" if confirmed else
         "已核验原话，未使用摘要" if evidence == "verified_source_only" else "有来源的历史自述"),
        freshness, "past_statement_only",
        reference, record.observed_at,
    )
