"""Bounded source-time rules for dated task-attempt references.

No reference date means no physical attempt identity.  The reporting date is not
silently substituted for an unknown event date.  A resolved day is an interval:
``occurred_at`` is its UTC sorting boundary, never an asserted occurrence time.
Repeated unqualified references to one task/day share an identity.  A caller
with multiple known attempts on that date must ask which one was meant rather
than treat this unqualified reference as an extra attempt.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Mapping


_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
             "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)|(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日")
_WEEKDAY = re.compile(r"(?<![本这上下])(?:(上|本|这|下)周|(?:星期|周))([一二三四五六日天])|\b(?:(last|this|next)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_RELATIVE = re.compile(r"前天|昨天|今天|\b(?:today|yesterday|day before yesterday)\b", re.I)
_CUMULATIVE = re.compile(r"累计|累积|总共|一共|合计|\b(?:total|cumulative|so far|altogether)\b", re.I)
_PLAN = re.compile(r"打算|计划|准备|希望|想要|如果|假如|假设|比如|例如|下周|\b(?:plan|will|intend|hope|if|suppose|imagine|next week)\b", re.I)
_ATTEMPT = re.compile(r"那次|这次|做了|完成了|试了|做过|尝试了|做一次|做1次|\b(?:that attempt|that time|did|completed|tried|attempted)\b", re.I)
_ORDINAL = re.compile(r"第([一二两三四五六七八九十]|\d+)次|\b(first|second|third|fourth|fifth)\s+(?:attempt|time)\b", re.I)
_NUMBER_WORDS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                 "七": 7, "八": 8, "九": 9, "十": 10,
                 "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}


def _required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _source_moment(value):
    _required_string(value, "source_time")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("source_time must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None:
        raise ValueError("source_time must include a timezone")
    return result


def _utc(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def cumulative_coverage_interval(event):
    """Return a validated aggregate's source-bound coverage sorting interval.

    This helper does not validate linguistic support; the canonical task-event
    validator supplies that boundary. It never substitutes receipt time for a
    declared historical coverage date.
    """
    payload = event.get('payload') if isinstance(event, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get('report_mode') != 'cumulative':
        raise ValueError('coverage interval requires a cumulative report')
    cutoff = _source_moment(payload.get('covers_through', event['occurred_at']))
    if event.get('occurred_precision') == 'day':
        low, high = _source_moment(event['occurred_at']), _source_moment(event['occurred_before'])
        if not low <= cutoff < high:
            raise ValueError('coverage timestamp is outside its source day')
        return low, high
    return cutoff, cutoff


def resolve_event_date_reference(text, *, source_time, task_instance_id, period_id,
                                 timezone_name="UTC"):
    """Resolve an explicit day reference without claiming a physical attempt.

    Canonical weekly period IDs are ISO Monday dates.  A late date returns its
    historical Monday, not the reporting week's ID.  Opaque period IDs cannot
    safely identify a different historical week and therefore return ``None``.
    Unsupported/ambiguous language returns ``None``; malformed system metadata
    raises ``ValueError`` instead of guessing a clock or timezone.
    """
    _required_string(text, "text")
    _required_string(task_instance_id, "task_instance_id")
    _required_string(period_id, "period_id")
    _required_string(timezone_name, "timezone_name")
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("timezone_name must be an available IANA timezone") from exc
    local_source = _source_moment(source_time).astimezone(zone)
    today = local_source.date()
    monday = today - timedelta(days=today.weekday())
    candidates = []
    for match in _DATE.finditer(text):
        fields = match.groups()[:3] if match.group(1) else match.groups()[3:]
        try:
            candidates.append(date(*(int(x) for x in fields)))
        except ValueError:
            return None
    for match in _RELATIVE.finditer(text):
        value = match.group().casefold()
        shift = -2 if value in {"前天", "day before yesterday"} else -1 if value in {"昨天", "yesterday"} else 0
        candidates.append(today + timedelta(days=shift))
    for match in _WEEKDAY.finditer(text):
        qualifier = match.group(1) or (match.group(3) or "").casefold()
        weekday = _WEEKDAYS[(match.group(2) or match.group(4)).casefold()]
        shift = -7 if qualifier in {"上", "last"} else 7 if qualifier in {"下", "next"} else 0
        resolved = monday + timedelta(days=weekday + shift)
        if not qualifier and resolved > today:
            resolved -= timedelta(days=7)
        candidates.append(resolved)
    if not candidates or len(set(candidates)) != 1:
        return None
    event_day = candidates[0]
    if event_day > today:
        return None
    event_monday = event_day - timedelta(days=event_day.weekday())
    try:
        supplied_period = date.fromisoformat(period_id)
    except ValueError:
        if event_monday != monday:
            return None
        resolved_period = period_id
    else:
        if supplied_period.weekday() != 0:
            raise ValueError("ISO weekly period_id must be a Monday date")
        resolved_period = event_monday.isoformat()
    begin = datetime.combine(event_day, time.min, tzinfo=zone)
    end = datetime.combine(event_day + timedelta(days=1), time.min, tzinfo=zone)
    if begin.astimezone(timezone.utc) >= end.astimezone(timezone.utc):
        return None
    return {
        "event_date": event_day.isoformat(), "period_id": resolved_period,
        "occurred_at": _utc(begin), "occurred_before": _utc(end),
        "occurred_precision": "day", "event_timezone": timezone_name,
        "time_semantics": "calendar_day_interval_sorting_boundary_not_exact_occurrence",
    }


def resolve_attempt_reference(text, *, source_time, task_instance_id, period_id,
                              timezone_name="UTC", existing_events=None):
    """Resolve one explicitly dated attempt, conservatively and deterministically.

    Unqualified same-day reports share the day identity; a later report is not
    a new attempt.  An explicit ordinal is part of identity, not a completion
    count.  The caller must not parse ``second attempt`` as two completions.
    Aggregate reports, plans, an undated ``I did one``, vague extra attempts,
    conflicting dates and future dates all require fallback/clarification.
    """
    _required_string(text, "text")
    resolved = resolve_event_date_reference(text, source_time=source_time,
        task_instance_id=task_instance_id, period_id=period_id, timezone_name=timezone_name)
    if resolved is None:
        return None
    if _CUMULATIVE.search(text) or _PLAN.search(text) or not _ATTEMPT.search(text):
        return None
    ordinals = []
    for match in _ORDINAL.finditer(text):
        word = (match.group(1) or match.group(2)).casefold()
        value = int(word) if word.isdecimal() else _NUMBER_WORDS[word]
        if value < 1:
            return None
        ordinals.append(value)
    if len(set(ordinals)) > 1:
        return None
    if not ordinals and re.search(r"又|再做|另外|另一次|\b(?:another|again|additional|more)\b", text, re.I):
        return None
    # A numeral greater than one without an explicit ordinal is an aggregate,
    # never one new physical attempt.  Retain the original text for fallback.
    without_ordinals = _ORDINAL.sub("", text)
    for match in re.finditer(r"([一二两三四五六七八九十]|\d+)次|\b(\d+)\s+times?\b", without_ordinals, re.I):
        word = match.group(1) or match.group(2)
        value = int(word) if word.isdecimal() else _NUMBER_WORDS[word]
        if value > 1:
            return None
    # The default first-day identity is only safe when the day has no known
    # physical attempt. A later unqualified report must reuse a uniquely known
    # attempt even if its original explicit ordinal was 'second'.
    if existing_events is not None and not isinstance(existing_events, (list, tuple)):
        raise ValueError("existing_events must be a sequence of task events")
    if not ordinals and existing_events is not None:
        same_day = set()
        for event in existing_events:
            if not isinstance(event, Mapping):
                raise ValueError("existing task events must be mappings")
            payload = event.get("payload")
            if (event.get("event_type") == "progress_report"
                    and event.get("task_instance_id") == task_instance_id
                    and event.get("period_id") == resolved["period_id"]
                    and event.get("event_date") == resolved["event_date"]
                    and isinstance(payload, Mapping) and payload.get("report_mode") == "attempt"):
                same_day.add(_required_string(payload.get("attempt_id"), "existing attempt_id"))
        if len(same_day) > 1:
            return None
        if same_day:
            return {**resolved, "attempt_id": next(iter(same_day)), "attempt_ordinal": None,
                    "explicit_ordinal": False, "unqualified_day_reference": True,
                    "identity_rule": "reuse_unique_existing_task_period_day_attempt_v2",
                    "requires_existing_date_uniqueness_check": False}
    ordinal = ordinals[0] if ordinals else 1
    key = [task_instance_id, resolved["period_id"], resolved["event_date"], ordinal]
    identity = hashlib.sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:32]
    return {**resolved, "attempt_id": "attempt-day-" + identity,
            "attempt_ordinal": ordinal, "explicit_ordinal": bool(ordinals),
            "identity_rule": "task_period_local_day_and_explicit_ordinal_v1",
            "unqualified_day_reference": not ordinals,
            "requires_existing_date_uniqueness_check": not ordinals}


def parse_current_correction(text, task_payload):
    """Select an explicit numeric correction against a product task view.

    This is a candidate builder, not a new eligibility authority.  Its returned
    bindings still go through ``validate_task_event`` with real source records.
    Target/plan and cumulative progress are kept separate.  An ambiguous plain
    "change to two" with both a known target and known progress returns None.
    """
    _required_string(text, "text")
    if not isinstance(task_payload, Mapping):
        raise ValueError("task_payload must be a product task mapping")
    if re.search(r"如果|假如|假设|比如|例如|\b(?:if|suppose|example|maybe)\b", text, re.I):
        return None
    matches = list(re.finditer(r"(?:更正为?|改成|改为|纠正为?|而是|\bcorrect(?:ed)?\s+to\s+)([零〇一二两三四五六七八九十百]+|\d+)\s*(次|times?)", text, re.I))
    if len(matches) != 1:
        return None
    match = matches[0]
    word = match.group(1)
    if word.isdecimal() and word.isascii():
        value = int(word)
    elif word in {"零", "〇"}:
        value = 0
    elif word in _NUMBER_WORDS:
        value = _NUMBER_WORDS[word]
    else:
        return None  # Unsupported compound numeral, never guess.
    plan = bool(re.search(r"目标|计划|安排|每周|频率|\b(?:target|plan|weekly)\b", text, re.I))
    completed = bool(_CUMULATIVE.search(text) or re.search(r"已经做|做了|完成了|\b(?:completed|progress|did)\b", text, re.I))
    if plan and completed:
        return None
    field = "target_count" if plan else "count" if completed else None
    projection = task_payload.get("projection", {})
    progress = projection.get("progress", {}) if isinstance(projection, Mapping) else {}
    if field is None:
        known = []
        if isinstance(progress, Mapping) and type(progress.get("target_count")) is int:
            known.append("target_count")
        if isinstance(progress, Mapping) and type(progress.get("reported_count")) is int:
            known.append("count")
        if len(known) != 1:
            return None
        field = known[0]
    events = task_payload.get("events")
    if not isinstance(events, list):
        return None
    candidates = []
    for event in events:
        if not isinstance(event, Mapping) or not isinstance(event.get("payload"), Mapping):
            return None
        payload = event["payload"]
        if (event.get("task_instance_id") != task_payload.get("task_instance_id")
                or event.get("period_id") != task_payload.get("period_id")
                or event.get("goal_id") != task_payload.get("goal_id")):
            continue
        if field == "target_count" and event.get("event_type") == "agreed" and "target_count" in payload:
            candidates.append(event)
        elif field == "count" and event.get("event_type") == "progress_report" and payload.get("report_mode") == "cumulative":
            candidates.append(event)
    if not candidates:
        return None
    try:
        if len(candidates) == 1:
            # A single candidate needs no ordering assumption. This also
            # preserves the minimal legacy candidate-builder input contract;
            # the resulting event still requires full canonical validation.
            target = candidates[0]
            _source_moment(target['reported_at'])
        else:
            target = max(candidates, key=lambda event: (
                tuple(reversed(cumulative_coverage_interval(event))) if field == 'count' else (),
                _source_moment(event["reported_at"]), event["event_id"]))
    except (ValueError, KeyError, TypeError):
        return None
    start, end = match.start(1), match.end(2)
    return {"target_event_id": target["event_id"], "changes": {field: value},
            "claim_bindings": [{"start": start, "end": end,
                "claim": {"kind": "quantity", "value": value, "field": field}}],
            "selection_rule": "explicit_numeric_correction_to_current_product_axis_v1"}
