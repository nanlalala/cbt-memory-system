"""Source-bound task events and a multi-axis, append-only task projection.

This module is deliberately conservative.  A matching quotation is not a
semantic annotation: only the explicit rules below authorize a product state.
Unsupported language is rejected for quotation fallback or clarification.  No
evaluation reference, oracle label, model confidence or confirmation flag is an
input to eligibility.

Persistence, current permissions, source revocation and CAS writes belong to the
store/orchestrator.  In particular ``blocked=True`` is a persistent content-object
barrier, not a request to fall back to an earlier version.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any, Mapping, Sequence

from task_input_rules import resolve_attempt_reference, resolve_event_date_reference


_IDENTIFIERS = ("event_id", "user_id", "goal_id", "task_instance_id", "period_id", "source_id")
_KINDS = frozenset({
    "proposed", "agreed", "declined", "progress_report", "participation_changed",
    "outcome_report", "corrected", "due_review",
})
_EVENT_KEYS = frozenset({
    *_IDENTIFIERS, "event_type", "occurred_at", "reported_at", "origin", "payload",
    "claim_bindings", "session_id", "expected_version",
    "occurred_precision", "event_date", "occurred_before", "event_timezone",
})
_NEGATION = re.compile(r"没|没有|不|并非|未|别|无|\b(?:not|never|don['’]?t|doesn['’]?t|didn['’]?t|won['’]?t|wouldn['’]?t|shouldn['’]?t|couldn['’]?t|haven['’]?t|hasn['’]?t|hadn['’]?t|cannot|can['’]?t|mustn['’]?t|refus(?:e|ed|ing)|declin(?:e|ed|ing))\b", re.I)
_AMBIGUOUS = re.compile(r"也许|可能|如果|假如|假设|比如|例如|举例|要是|示例|假想|他说|她说|别人说|妈妈同意|母亲同意|父亲同意|朋友同意|未必|不是不|并非不|不是没|并非没|\b(?:maybe|might|perhaps|if|imagine|suppose|hypothetical|example|he said|she said|they said|he agreed|she agreed|they agreed|not unwilling)\b", re.I)
_QUOTED_REPORT = re.compile(r'["“”「」『』]|(?:^|[\s:：])[\'‘].+[\'’]|(?:说|表示|告诉|写道|声称|\bsaid|\bsays|\btold|\bwrote)\s*[:：]', re.I)
_HISTORICAL_QUALIFIER = re.compile(r"去年|前年|上个月|上月|上周|以前|之前|曾经|当时|那时候|\b(?:last year|last month|last week|previously|formerly|used to)\b", re.I)
_DATE_QUALIFIER = re.compile(r"(?<!\d)\d{4}(?:年|[-/]\d{1,2}(?:[-/]\d{1,2})?|\b)")
_BARE_YES = re.compile(r"\s*(?:yes|yeah|yep|ok|okay|好|好的|可以|行|嗯|是的)[.!。！\s]*", re.I)
_ORDINAL_CLAIM = re.compile(r"第([一二两三四五六七八九十]|\d+)次|\b(first|second|third|fourth|fifth)\s+(?:attempt|time)\b", re.I)
_QUANTITY = re.compile(r"(?P<n>\d+|[零〇一二两三四五六七八九十百]+)\s*(?P<u>次|遍|份|条|times?|sessions?|records?|attempts?)", re.I)
_CHOICES = {
    "proposed": re.compile(r"想试|想做|打算|计划|希望试|\b(?:plan|intend|would like|want to try|propose|try)\b", re.I),
    "agreed": re.compile(r"同意|答应|愿意|接受|可以试|确定要|\b(?:agree|agreed|accept|accepted|willing|i will|i'll)\b", re.I),
    "declined": re.compile(r"不愿|不想|拒绝|不接受|不同意|没答应|\b(?:declin(?:e|ed|ing)|refus(?:e|ed|ing)|do not agree|don['’]?t agree|don['’]?t want|do not want)\b", re.I),
    "paused": re.compile(r"暂停|先停|停一下|这周不做|暂时不做|\b(?:pause|paused|on hold|stop for now|take a break)\b", re.I),
    "active": re.compile(r"恢复|继续|重启|\b(?:resume|resumed|restart|continue)\b", re.I),
    "cancelled": re.compile(r"取消|不做这个了|\b(?:cancel|cancelled|canceled|drop this task)\b", re.I),
    "helpful": re.compile(r"有帮助|有点用|有用|\b(?:helpful|helped|useful)\b", re.I),
    "unhelpful": re.compile(r"没帮助|没有帮助|没用|没有用|\b(?:unhelpful|not helpful|not useful|didn't help|did not help)\b", re.I),
    "mixed": re.compile(r"有好有坏|有帮助也有|部分有用|\b(?:mixed|partly helpful|somewhat helpful)\b", re.I),
    "unknown": re.compile(r"不知道结果|还不清楚结果|不知道有没有用|\b(?:unknown|don't know the outcome|do not know the outcome)\b", re.I),
    "cumulative": re.compile(r"累计|累积|总共|一共|合计|\b(?:total|cumulative|so far|altogether)\b", re.I),
    "not_done": re.compile(r"还没做|尚未做|没有做过|一次也没做|\b(?:haven't done any|have not done any|none done)\b", re.I),
    "attempt": re.compile(r"做了|完成了|试了|做过|尝试了|\b(?:did|completed|tried|attempted)\b", re.I),
}
_PAYLOAD_KEYS = {
    "proposed": frozenset({"object_label", "target_count", "due_at", "proposal_id"}),
    "agreed": frozenset({"object_label", "target_count", "due_at", "proposal_id"}),
    "declined": frozenset({"object_label"}),
    "progress_report": frozenset({"object_label", "report_mode", "count", "attempt_id", "covers_through"}),
    "participation_changed": frozenset({"object_label", "participation"}),
    "outcome_report": frozenset({"object_label", "outcome"}),
    "corrected": frozenset({"object_label", "target_event_id", "changes"}),
    "due_review": frozenset({"object_label", "due_at"}),
}


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _moment(value: Any, name: str) -> datetime:
    _string(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    return result


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value: str) -> int:
    if value.isascii() and value.isdecimal():
        return int(value)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if value in digits:
        return digits[value]
    total, digit = 0, 0
    for char in value:
        if char in digits:
            digit = digits[char]
        elif char in {"十", "百"}:
            total += (digit or 1) * (10 if char == "十" else 100)
            digit = 0
        else:
            raise ValueError("unsupported number")
    return total + digit


def _scoped_clause_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    """Keep a predicate with its local object, including its own negation."""
    boundaries = ",，。;；!?！？\n"
    left, right = start, end
    while left and source[left - 1] not in boundaries:
        left -= 1
    while right < len(source) and source[right] not in boundaries:
        right += 1
    # Without punctuation, coordinated subjects still cannot donate a positive
    # predicate to another object's negative predicate.
    conjunctions = re.compile(r"但是|不过|而是|并且|同时|而|但|\b(?:but|instead|and|while|whereas)\b", re.I)
    for match in conjunctions.finditer(source, left, right):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return left, right


def _clause(source: str, start: int, end: int) -> str:
    left, right = _scoped_clause_bounds(source, start, end)
    return source[left:right]


def _negative_governs_predicate(prefix: str) -> bool:
    """Conservative predicate-prefix scope, not a whole-sentence 'not' test."""
    normalized = prefix.replace("’", "'").replace("‘", "'")
    return bool(re.search(
        r"不要|别|没有|并未|从未|不曾|没|不|未|并非|不是|拒绝|反对|"
        r"\b(?:not|never|no|don'?t|doesn'?t|didn'?t|haven'?t|hasn'?t|hadn'?t|won'?t|wouldn'?t|"
        r"shouldn'?t|couldn'?t|can'?t|cannot|mustn'?t|without|refus(?:e|ed|ing)|"
        r"declin(?:e|ed|ing)|deny|denied|denies|against)\b", normalized, re.I))


def _choice_predicate_supported(source: str, start: int, end: int, value: str) -> bool:
    left, right = _scoped_clause_bounds(source, start, end)
    clause, prefix = source[left:right], source[left:start]
    if _AMBIGUOUS.search(clause):
        return False
    if value in {"paused", "cancelled", "declined"}:
        if _negative_governs_predicate(prefix):
            return False
        # Declining a cancellation/pause is not declining the task itself.
        # Keep the opposite state unknown rather than infer active/agreed.
        if value == "declined" and re.search(
            r"暂停|取消|停止|恢复|继续|先停|停一下|\b(?:paus(?:e|ed|ing)|cancel(?:led|ed|ling)?|"
            r"stop|resume|continue|restart)\b", source[end:right], re.I):
            return False
        # If a single unresolved clause contains a second, negated occurrence
        # of this predicate, a positive occurrence for another subject cannot
        # establish this object's state.  Explicit conjunctions are scoped above.
        for other in _CHOICES[value].finditer(source, left, right):
            if other.start() != start and _negative_governs_predicate(source[left:other.start()]):
                return False
        return True
    if value in {"agreed", "proposed", "active", "helpful", "attempt", "cumulative"}:
        return not _NEGATION.search(clause.replace("’", "'").replace("‘", "'"))
    return True


def verify_claim_binding(source_text: str, start: int, end: int, claim: Mapping[str, Any]) -> bool:
    """Verify a small explicit product claim against an exact Unicode span.

    ``object`` values are exact object labels; quantities require a number and a
    unit; ``choice`` values use conservative bilingual lexical rules.  Bare yes
    never authorizes a historical or task state here; use the separately bound
    proposal contract in :func:`validate_bound_choice`.
    """
    if not isinstance(source_text, str) or type(start) is not int or type(end) is not int:
        return False
    if not 0 <= start < end <= len(source_text) or not isinstance(claim, Mapping):
        return False
    if set(claim) - {"kind", "value", "unit", "field", "proposal_id"}:
        return False
    text, context = source_text[start:end], _clause(source_text, start, end)
    kind, value = claim.get("kind"), claim.get("value")
    if "field" in claim and not isinstance(claim["field"], str):
        return False
    if "proposal_id" in claim and not isinstance(claim["proposal_id"], str):
        return False
    if kind == "time":
        try:
            _moment(value, "claim.time")
        except (ValueError, TypeError):
            return False
        return value in text
    if kind == "object":
        return isinstance(value, str) and bool(value.strip()) and value in text
    if kind == "negation":
        return type(value) is bool and bool(_NEGATION.search(context)) is value and not _AMBIGUOUS.search(context)
    if kind == "attempt_ordinal":
        if type(value) is not int or value < 1 or _NEGATION.search(context) or _AMBIGUOUS.search(context):
            return False
        english = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        return any((_number(m.group(1)) if m.group(1) else english[m.group(2).casefold()]) == value
                   for m in _ORDINAL_CLAIM.finditer(text))
    if kind == "quantity":
        if type(value) is not int or value < 0 or _AMBIGUOUS.search(context):
            return False
        if _NEGATION.search(context):
            # "Not two" is not zero.  Zero requires an explicit none-done form
            # with no competing numerical amount in this clause.
            return value == 0 and not _QUANTITY.search(context) and bool(re.search(
                r"还没做|尚未做|没有做过|一次也没做|\b(?:haven't done any|have not done any|none done)\b", context, re.I))
        unit = claim.get("unit")
        if unit is not None and (not isinstance(unit, str) or not unit):
            return False
        return any(_number(m.group("n")) == value and (unit is None or m.group("u").casefold() == unit.casefold())
                   for m in _QUANTITY.finditer(text))
    if kind != "choice" or not isinstance(value, str) or value not in _CHOICES or _AMBIGUOUS.search(context):
        return False
    if _BARE_YES.fullmatch(text):
        return False
    return any(_choice_predicate_supported(source_text, start + match.start(), start + match.end(), value)
               for match in _CHOICES[value].finditer(text))


def _source(sources: Mapping[str, Any], source_id: str, user_id: str) -> Mapping[str, Any]:
    if not isinstance(sources, Mapping):
        raise ValueError("sources must be a source mapping")
    value = sources.get(source_id)
    if not isinstance(value, Mapping):
        raise ValueError("task source does not exist")
    if value.get("user_id") != user_id:
        raise ValueError("task source belongs to another user")
    _string(value.get("text"), "source.text")
    _moment(value.get("source_time"), "source.source_time")
    if value.get("role") not in {"user", "assistant"}:
        raise ValueError("task source role is unknown")
    return value


def _verified_bindings(event: Mapping[str, Any], source: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = event.get("claim_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("every task claim needs source bindings")
    result, seen = [], set()
    for item in bindings:
        if not isinstance(item, dict) or set(item) != {"start", "end", "claim"}:
            raise ValueError("invalid claim binding")
        if not verify_claim_binding(source["text"], item["start"], item["end"], item["claim"]):
            raise ValueError("task claim is not supported by an explicit product rule")
        identity = (item["start"], item["end"], repr(sorted(item["claim"].items())))
        if identity in seen:
            raise ValueError("duplicate claim binding")
        seen.add(identity)
        result.append(deepcopy(item))
    return result


def _has(bindings: Sequence[Mapping[str, Any]], kind: str, value: Any, field: str | None = None) -> bool:
    return any(b["claim"].get("kind") == kind and b["claim"].get("value") == value
               and (field is None or b["claim"].get("field") in {None, field}) for b in bindings)



def _bound_context_object(event: Mapping[str, Any], source: Mapping[str, Any],
                          sources: Mapping[str, Any], context_objects: Mapping[str, Any] | None) -> bool:
    if not isinstance(context_objects, Mapping):
        return False
    context = context_objects.get(event["task_instance_id"])
    if not isinstance(context, Mapping):
        return False
    if any(context.get(key) != event.get(key) for key in ("goal_id", "period_id")):
        return False
    if context.get("object_label") != event["payload"]["object_label"]:
        return False
    try:
        original = _source(sources, context.get("source_id"), event["user_id"])
        if _moment(original["source_time"], "object.source_time") > _moment(source["source_time"], "source_time"):
            return False
        return verify_claim_binding(original["text"], context.get("start"), context.get("end"),
                                    {"kind": "object", "value": context["object_label"]})
    except (ValueError, TypeError, KeyError):
        return False


def _implicit_task_short_form(text: str) -> bool:
    """Allow a bound object's short update, not an unrecognized new activity."""
    # A terminal, explicit privacy-control clause is handled by orchestration.
    # Ignore it only for this short-form recognition; never change source text
    # or offsets, and never interpret its negation as task participation.
    semantic_text = re.sub(r"[,，;；]\s*(?:这条|这段|这句话|本条)?\s*(?:别保存|不要保存|不要记住|别记住|不要记录|别记录|别存|不要存)\s*[。.!！]?$", "", text)
    remaining = _QUANTITY.sub("", semantic_text)
    remaining = re.sub(r"这周|本周|上周|今天|昨天|前天|周[一二三四五六日天]|最近|现在|目前|刚刚|先|暂时|我|已经|还|只|又|才|一下|那次|这次|上次|第|更正为|更正|改成|纠正|说错|次数|想|吧|了|过|做|完成|尝试|暂停|停止|取消|恢复|继续|没|有|不|同意|愿意|试试|计划|一共|总共|累计|累积|总计|结果|不知道|还不清楚", "", remaining)
    remaining = re.sub(r"\b(?:i|we|today|yesterday|this|last|week|for|now|the|that|one|it|task|pause|paused|stop|stopped|completed|did|done|try|tried|total|cumulative|so|far|already|have|has|not|haven't|don't|cancel|cancelled|resume|continue|agree|agreed|accept|accepted|willing|to|times|time|unknown|outcome)\b", "", remaining, flags=re.I)
    return not re.sub(r"[\s,，。.!！?？;；:'’\-]", "", remaining)


def _claim_for_object(source: Mapping[str, Any], binding: Mapping[str, Any], label: str,
                      kind: str, value: Any, field: str | None, *, contextual: bool) -> bool:
    claim = binding["claim"]
    if claim.get("kind") != kind or claim.get("value") != value or (field is not None and claim.get("field") not in {None, field}):
        return False
    text = source["text"]
    selected = text[binding["start"]:binding["end"]]
    if kind == "quantity":
        matches = [m for m in _QUANTITY.finditer(selected) if _number(m.group("n")) == value]
        if value == 0 and not matches:
            matches = list(_CHOICES["not_done"].finditer(selected))
    else:
        matches = list(_CHOICES[value].finditer(selected))
    for match in matches:
        start, end = binding["start"] + match.start(), binding["start"] + match.end()
        if kind == "choice" and not _choice_predicate_supported(text, start, end, value):
            continue
        clause = _clause(text, start, end)
        explicit_subject = label in clause
        correction_reference = ((re.search(r"而是|更正|改成|纠正|instead|correct", clause, re.I)
                                 or re.search(r"(?:而是|instead)\s*$", text[:start], re.I))
                                and label in text[:start])
        implicit_subject = contextual and label not in text and _implicit_task_short_form(clause)
        if not (explicit_subject or correction_reference or implicit_subject):
            continue
        if kind == "quantity" and field == "count" and value > 0:
            if not correction_reference and not re.search(r"做了|做过|完成|试了|尝试了|而是|更正|改成|纠正|\b(?:did|done|completed|tried|attempted|instead|correct|actually)\b", clause, re.I):
                continue
        return True
    return False


def validate_bound_choice(event: Mapping[str, Any], *, sources: Mapping[str, Any],
                          proposals: Mapping[str, Mapping[str, Any]]) -> bool:
    """Accept yes only after one live, source-verified, single-action proposal.

    ``proposals`` is the persistent product proposal view, not evaluation data.
    Both the assistant proposal and the user reply must be visible source
    records with matching user/task/period/session and bounded times.
    """
    try:
        if event.get("event_type") != "agreed" or event.get("origin") != "explicit_user_statement":
            return False
        source = _source(sources, event["source_id"], event["user_id"])
        if source["role"] != "user" or not _BARE_YES.fullmatch(source["text"]):
            return False
        session = _string(event.get("session_id"), "session_id")
        reply_time = _moment(source["source_time"], "source_time")
        live = []
        for proposal in proposals.values():
            status = proposal.get("status", "open")
            historical_open = status in {"open", "consumed"}
            consumed = proposal.get("consumed_at")
            if consumed is not None:
                consumed_time = _moment(consumed, "proposal.consumed_at")
                historical_open = historical_open and (reply_time < consumed_time or
                    (reply_time == consumed_time and proposal.get("accepted_event_id") == event.get("event_id")))
            elif status != "open":
                historical_open = False
            if (proposal.get("user_id") == event["user_id"] and proposal.get("session_id") == session
                    and _moment(proposal.get("created_at"), "proposal.created_at") <= reply_time
                    < _moment(proposal.get("expires_at"), "proposal.expires_at")
                    and historical_open):
                live.append(proposal)
        if len(live) != 1:
            return False
        proposal = live[0]
        if proposal.get("proposal_id") != event["payload"].get("proposal_id") or type(proposal.get("action_count")) is not int or proposal["action_count"] != 1:
            return False
        if any(proposal.get(key) != event.get(key) for key in ("user_id", "goal_id", "task_instance_id", "period_id")):
            return False
        original = _source(sources, proposal["source_id"], event["user_id"])
        if original["role"] != "assistant" or _QUOTED_REPORT.search(original["text"]):
            return False
        if _moment(original["source_time"], "proposal.source_time") > reply_time:
            return False
        bindings = _verified_bindings(proposal, original)
        label = proposal.get("object_label")
        if not _has(bindings, "object", label) or not isinstance(label, str):
            return False
        # The displayed proposal itself must not contain multiple actions or
        # alternative/compound questions, even if action_count was misstated.
        remaining = original["text"].replace(label, "")
        question_count = proposal.get("question_count", 1)
        if type(question_count) is not int or question_count != 1 or len(re.findall(r"[?？]", original["text"])) != 1:
            return False
        if re.search(r"或者|还是|以及|并且|还要|另外|同时|\b(?:or|and|also)\b|和", remaining, re.I):
            return False
        count = proposal.get("target_count")
        if count is not None and (type(count) is not int or count < 0 or not _has(bindings, "quantity", count)):
            return False
        if "due_at" in proposal:
            _moment(proposal["due_at"], "proposal.due_at")
            if not _has(bindings, "time", proposal["due_at"]):
                return False
        if event["payload"].get("due_at") != proposal.get("due_at"):
            return False
        if event["payload"].get("target_count") != count or event["payload"].get("object_label") != label:
            return False
        choices = event.get("claim_bindings")
        if not isinstance(choices, list) or len(choices) != 1:
            return False
        binding = choices[0]
        if not isinstance(binding, dict) or set(binding) != {"start", "end", "claim"}:
            return False
        if binding["start"] != 0 or binding["end"] != len(source["text"]):
            return False
        return binding["claim"] == {"kind": "choice", "value": "agreed", "proposal_id": proposal["proposal_id"]}
    except (ValueError, TypeError, KeyError):
        return False


def validate_task_event(event: Mapping[str, Any], *, sources: Mapping[str, Any],
                        proposals: Mapping[str, Mapping[str, Any]] | None = None,
                        context_objects: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Construct a strict sourced task event, rejecting unsupported claims."""
    if not isinstance(event, Mapping) or set(event) - _EVENT_KEYS:
        raise ValueError("unknown task event fields")
    data = deepcopy(dict(event))
    for key in _IDENTIFIERS:
        _string(data.get(key), key)
    if data.get("event_type") not in _KINDS:
        raise ValueError("unsupported task event type")
    occurred = _moment(data.get("occurred_at"), "occurred_at")
    reported = _moment(data.get("reported_at"), "reported_at")
    if occurred > reported:
        raise ValueError("task occurrence cannot be after its report")
    if "expected_version" in data:
        _count(data["expected_version"], "expected_version")
    if "session_id" in data:
        _string(data["session_id"], "session_id")
    payload = data.get("payload")
    kind = data["event_type"]
    if not isinstance(payload, dict) or set(payload) - _PAYLOAD_KEYS[kind]:
        raise ValueError("unknown task payload fields")
    label = _string(payload.get("object_label"), "payload.object_label")
    source = _source(sources, data["source_id"], data["user_id"])
    if _moment(source["source_time"], "source_time") > reported:
        raise ValueError("task report precedes source visibility")
    precision = data.get("occurred_precision", "instant")
    if precision not in {"instant", "day"}:
        raise ValueError("unsupported occurred_precision")
    day_fields = {"event_date", "occurred_before", "event_timezone"}
    if precision == "instant" and any(key in data for key in day_fields):
        raise ValueError("day metadata requires occurred_precision=day")
    if precision == "day":
        if not day_fields <= data.keys():
            raise ValueError("day precision requires date, interval end and timezone")
        reference = resolve_event_date_reference(source["text"], source_time=source["source_time"],
            task_instance_id=data["task_instance_id"], period_id=data["period_id"],
            timezone_name=data["event_timezone"])
        if reference is None or any(reference[key] != data[key] for key in ("event_date", "occurred_at", "occurred_before", "event_timezone", "period_id")):
            raise ValueError("day interval is not supported by the source-time date rule")
        if _moment(data["occurred_before"], "occurred_before") <= occurred:
            raise ValueError("invalid occurred day interval")
    if _QUOTED_REPORT.search(source["text"]):
        raise ValueError("quoted or attributed speech is not a current self-report")
    # This already supported unknown-outcome assertion contains an embedded
    # interrogative word, but does not ask the assistant to establish a fact.
    question_text = source["text"].replace("不知道有没有用", "结果未知")
    if source["role"] == "user" and (re.search(r"[?？]|是否|是不是|能否|可否|要不要|有没有|请问", question_text)
            or re.search(r"吗(?:[。.!！\s]*$)", question_text)):
        raise ValueError("a user question is not a current task-state assertion")
    temporal_text = source["text"]
    # An explicitly bound deadline is a different time axis from occurrence.
    # Mask only its exact verified span in a deadline-labelled clause.
    for binding in data.get("claim_bindings", []):
        if not isinstance(binding, Mapping) or not isinstance(binding.get("claim"), Mapping):
            continue
        claim = binding["claim"]
        if (claim.get("kind") == "time" and claim.get("value") == payload.get("due_at")
                and verify_claim_binding(source["text"], binding.get("start"), binding.get("end"), claim)
                and re.search(r"截止|期限|到期|\b(?:due|deadline)\b", _clause(source["text"], binding["start"], binding["end"]), re.I)):
            temporal_text = temporal_text[:binding["start"]] + " " * (binding["end"] - binding["start"]) + temporal_text[binding["end"]:]
    if ((_HISTORICAL_QUALIFIER.search(temporal_text) or _DATE_QUALIFIER.search(temporal_text)) and precision != "day"
            and data["occurred_at"] not in source["text"]):
        raise ValueError("historical qualifier needs a source-verified occurrence time")
    if data.get("origin") == "agent_inference":
        raise ValueError("model inference cannot update factual task state")
    if kind == "proposed" and data.get("origin") == "assistant_proposal":
        if source["role"] != "assistant":
            raise ValueError("assistant proposal needs assistant source")
    elif data.get("origin") != "explicit_user_statement" or source["role"] != "user":
        raise ValueError("task state needs an explicit user source")
    contextual = _bound_context_object(data, source, sources, context_objects)
    bound_yes = kind == "agreed" and _BARE_YES.fullmatch(source["text"])
    if bound_yes:
        if not validate_bound_choice(data, sources=sources, proposals=proposals or {}):
            raise ValueError("yes is not bound to a unique valid proposal")
        bindings = data["claim_bindings"]
    else:
        bindings = _verified_bindings(data, source)
        if not _has(bindings, "object", label) and not (contextual and label not in source["text"] and _implicit_task_short_form(source["text"])):
            raise ValueError("task object lacks explicit or bound-context source support")
    def supported(kind_: str, value: Any, field: str | None = None) -> bool:
        return any(_claim_for_object(source, b, label, kind_, value, field, contextual=contextual) for b in bindings)
    if "target_count" in payload:
        _count(payload["target_count"], "target_count")
        if not bound_yes and not supported("quantity", payload["target_count"], "target_count"):
            raise ValueError("target quantity lacks explicit source support")
    if "due_at" in payload:
        _moment(payload["due_at"], "due_at")
        if not bound_yes and not _has(bindings, "time", payload["due_at"]):
            raise ValueError("deadline requires an explicit bound timestamp, not a guessed date")
    if kind in {"proposed", "agreed", "declined"}:
        # Typed assistant proposals are sourced product actions.  The assistant
        # may phrase a proposal as a question without saying "I propose".
        if not bound_yes and not (kind == "proposed" and data["origin"] == "assistant_proposal") and not supported("choice", kind):
            raise ValueError("task acceptance choice lacks explicit support")
    elif kind == "progress_report":
        count = _count(payload.get("count"), "count")
        mode = payload.get("report_mode")
        if mode not in {"attempt", "cumulative"}:
            raise ValueError("progress report must distinguish attempt and cumulative")
        ordinal_one = False
        if mode == "attempt" and count == 1 and precision == "day":
            reference = resolve_attempt_reference(source["text"], source_time=source["source_time"],
                task_instance_id=data["task_instance_id"], period_id=data["period_id"],
                timezone_name=data["event_timezone"])
            ordinal_one = bool(reference and reference["explicit_ordinal"]
                and reference["attempt_id"] == payload.get("attempt_id")
                and _has(bindings, "attempt_ordinal", reference["attempt_ordinal"]))
        if not (supported("quantity", count, "count") or ordinal_one) or not (supported("choice", mode) or (mode == "attempt" and count == 0 and supported("choice", "not_done"))):
            raise ValueError("progress quantity or report mode lacks explicit support")
        if mode == "attempt":
            _string(payload.get("attempt_id"), "attempt_id")
            if count not in {0, 1} or "covers_through" in payload:
                raise ValueError("one attempt is zero or one, not an aggregate")
        else:
            if "attempt_id" in payload:
                raise ValueError("cumulative reports are not new attempt identities")
            cutoff = _moment(payload.get("covers_through", data["occurred_at"]), "covers_through")
            if cutoff > reported:
                raise ValueError("cumulative coverage cannot exceed report time")
            if precision == "day" and not occurred <= cutoff < _moment(data["occurred_before"], "occurred_before"):
                raise ValueError("cumulative coverage lies outside its source-supported day")
    elif kind == "participation_changed":
        value = payload.get("participation")
        if value not in {"active", "paused", "cancelled"} or not supported("choice", value):
            raise ValueError("participation choice lacks explicit support")
    elif kind == "outcome_report":
        value = payload.get("outcome")
        if value not in {"unknown", "helpful", "unhelpful", "mixed"} or not supported("choice", value):
            raise ValueError("outcome choice lacks explicit support")
    elif kind == "corrected":
        _string(payload.get("target_event_id"), "target_event_id")
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes or set(changes) - {"target_count", "count", "participation", "outcome", "acceptance"}:
            raise ValueError("unsupported correction fields")
        for field, value in changes.items():
            if field in {"target_count", "count"}:
                _count(value, field)
                if not supported("quantity", value, field):
                    raise ValueError("corrected quantity lacks source support")
            else:
                options = {"participation": {"active", "paused", "cancelled"},
                           "outcome": {"unknown", "helpful", "unhelpful", "mixed"},
                           "acceptance": {"proposed", "agreed", "declined"}}
                if value not in options[field] or not supported("choice", value):
                    raise ValueError("corrected choice lacks source support")
        if not re.search(r"更正|改成|不是|纠正|说错|\b(?:correct|correction|actually|not)\b", source["text"], re.I):
            raise ValueError("correction must be explicit")
    return data


def _full_clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Keep the entire original clause, including negation and contrast."""
    separators = ",，。;；!?！？\n"
    left, right = start, end
    while left and text[left - 1] not in separators:
        left -= 1
    while right < len(text) and text[right] not in separators:
        right += 1
    return left, right


def _source_claim_witness(event: Mapping[str, Any], source: Mapping[str, Any],
                          kind: str, value: Any, field: str | None, *, contextual: bool):
    """Find a witness using the *same* object/claim rule as eligibility."""
    text, label = source["text"], event["payload"]["object_label"]
    for binding in event["claim_bindings"]:
        claim = binding["claim"]
        if (claim.get("kind") != kind or claim.get("value") != value
                or (field is not None and claim.get("field") not in {None, field})):
            continue
        selected = text[binding["start"]:binding["end"]]
        if kind == "quantity":
            matches = [m for m in _QUANTITY.finditer(selected) if _number(m.group("n")) == value]
            if value == 0 and not matches:
                matches = list(_CHOICES["not_done"].finditer(selected))
        elif kind == "choice":
            matches = list(_CHOICES[value].finditer(selected))
        elif kind == "time":
            offset = selected.find(value)
            if offset >= 0:
                return _full_clause_bounds(text, binding["start"] + offset, binding["start"] + offset + len(value))
            continue
        else:
            continue
        for match in matches:
            narrow = {"start": binding["start"] + match.start(),
                      "end": binding["start"] + match.end(), "claim": claim}
            if not verify_claim_binding(text, narrow["start"], narrow["end"], claim):
                continue
            if _claim_for_object(source, narrow, label, kind, value, field, contextual=contextual):
                return _full_clause_bounds(text, narrow["start"], narrow["end"])
    return None


def supporting_spans_for_event(event: Mapping[str, Any], sources: Mapping[str, Any],
                               proposals: Mapping[str, Mapping[str, Any]] | None = None,
                               context_objects: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Return sufficient original spans for an independently validated event.

    Witness selection shares product eligibility's object/clause rules; it does
    not search for the first generic state word.  Separate necessary clauses in
    one source remain separate spans.  A complex correction or day reference
    conservatively retains its complete source, rather than clipping negation,
    dates or conditions.  Invalid/unsupported events raise; full-source fallback
    is only available *after* validation.
    """
    data = validate_task_event(event, sources=sources, proposals=proposals, context_objects=context_objects)
    source_id = data["source_id"]
    source = _source(sources, source_id, data["user_id"])
    text = source["text"]
    spans = []
    if data["event_type"] == "agreed" and _BARE_YES.fullmatch(text):
        proposal = (proposals or {})[data["payload"]["proposal_id"]]
        original = _source(sources, proposal["source_id"], data["user_id"])
        spans.extend([{"source_id": source_id, "start": 0, "end": len(text)},
                      {"source_id": proposal["source_id"], "start": 0, "end": len(original["text"])}])
    elif (data["event_type"] == "corrected" or data.get("occurred_precision") == "day"
          or data["origin"] == "assistant_proposal"):
        spans.append({"source_id": source_id, "start": 0, "end": len(text)})
    else:
        payload, kind = data["payload"], data["event_type"]
        contextual = _bound_context_object(data, source, sources, context_objects)
        requirements = []
        if kind in {"proposed", "agreed", "declined"}:
            requirements.append(("choice", kind, None))
        elif kind == "progress_report":
            choice = payload["report_mode"]
            if choice == "attempt" and payload["count"] == 0 and _has(data["claim_bindings"], "choice", "not_done"):
                choice = "not_done"
            requirements.extend([("choice", choice, None), ("quantity", payload["count"], "count")])
        elif kind == "participation_changed":
            requirements.append(("choice", payload["participation"], None))
        elif kind == "outcome_report":
            requirements.append(("choice", payload["outcome"], None))
        if "target_count" in payload:
            requirements.append(("quantity", payload["target_count"], "target_count"))
        if "due_at" in payload:
            requirements.append(("time", payload["due_at"], None))
        for claim_kind, value, field in requirements:
            witness = _source_claim_witness(data, source, claim_kind, value, field, contextual=contextual)
            if witness is None:
                # An already validated binding may contain an indivisible
                # reference.  Retain the complete source; do not invent a span.
                spans = [{"source_id": source_id, "start": 0, "end": len(text)}]
                break
            spans.append({"source_id": source_id, "start": witness[0], "end": witness[1]})
        if not spans:
            spans.append({"source_id": source_id, "start": 0, "end": len(text)})
        label = payload["object_label"]
        if label in text and not any(label in text[s["start"]:s["end"]] for s in spans):
            for binding in data["claim_bindings"]:
                if binding["claim"].get("kind") == "object" and binding["claim"].get("value") == label:
                    selected = text[binding["start"]:binding["end"]]
                    offset = selected.find(label)
                    if offset >= 0:
                        start = binding["start"] + offset
                        spans.append({"source_id": source_id, "start": start, "end": start + len(label)})
                        break
    if data["payload"]["object_label"] not in text and _bound_context_object(data, source, sources, context_objects):
        context = context_objects[data["task_instance_id"]]
        spans.append({"source_id": context["source_id"], "start": context["start"], "end": context["end"]})
    merged = []
    for span in sorted(spans, key=lambda item: (item["source_id"], item["start"], item["end"])):
        if (merged and merged[-1]["source_id"] == span["source_id"] and span["start"] <= merged[-1]["end"]):
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
        else:
            merged.append(dict(span))
    return merged


def project_task(events: Sequence[Mapping[str, Any]], *, user_id: str, task_instance_id: str,
                 period_id: str, sources: Mapping[str, Any], as_of: str | None = None,
                 source_cutoff: str | None = None, blocked: bool = False,
                 proposals: Mapping[str, Mapping[str, Any]] | None = None,
                 context_objects: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build current task axes without erasing progress when participation stops.

    A late report only affects its explicit period.  Cumulative totals replace
    prior totals rather than adding to them.  Repeated attempt identity adds
    evidence, not an extra attempt.  Corrections patch the effective projection
    while preserving the original input events and quotations unchanged.
    """
    for value, name in [(user_id, "user_id"), (task_instance_id, "task_instance_id"), (period_id, "period_id")]:
        _string(value, name)
    if type(blocked) is not bool:
        raise ValueError("blocked must be boolean")
    now = _moment(as_of, "as_of") if as_of else None
    visible = _moment(source_cutoff, "source_cutoff") if source_cutoff else None
    result: dict[str, Any] = {
        "user_id": user_id, "goal_id": None, "task_instance_id": task_instance_id,
        "period_id": period_id, "state_version": 0, "acceptance": "unknown",
        "participation": "unknown", "outcome": "unknown", "due_review": False,
        "blocked": blocked, "progress": {"reported_count": None, "target_count": None,
        "attempt_ids": [], "known_attempt_count": 0, "reported_total": None,
        "count_conflict": False, "completion": "unknown"},
        "source_ids": [], "object_support_source_ids": [], "pending_proposals": [], "field_support_event_ids": {}, "event_ids": [], "eligibility": "blocked" if blocked else "unknown",
    }
    if blocked:
        return result
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise ValueError("events must be an event sequence")
    unique: dict[str, dict[str, Any]] = {}
    for raw in events:
        if not isinstance(raw, Mapping):
            raise ValueError("each task event must be an object")
        if raw.get("user_id") != user_id or raw.get("task_instance_id") != task_instance_id or raw.get("period_id") != period_id:
            continue
        if visible:
            source = _source(sources, raw.get("source_id"), user_id)
            if _moment(source["source_time"], "source_time") > visible:
                continue
        event = validate_task_event(raw, sources=sources, proposals=proposals, context_objects=context_objects)
        existing = unique.get(event["event_id"])
        if existing is not None and existing != event:
            raise ValueError("event_id cannot identify different events")
        unique[event["event_id"]] = event
    ordered = sorted(unique.values(), key=lambda e: (_moment(e["reported_at"], "reported_at"), e["event_id"]))
    effective: dict[str, dict[str, Any]] = {}
    goal_ids = {e["goal_id"] for e in ordered}
    if len(goal_ids) > 1:
        raise ValueError("one task instance cannot change goal identity")
    result["goal_id"] = next(iter(goal_ids), None)
    if len({e["payload"]["object_label"] for e in ordered}) > 1:
        raise ValueError("task object label changes require explicit object identity handling")
    for event in ordered:
        if event["event_type"] != "corrected":
            effective[event["event_id"]] = deepcopy(event)
            continue
        payload = event["payload"]
        target = effective.get(payload["target_event_id"])
        if target is None or target["event_type"] == "corrected":
            raise ValueError("correction target must be an existing event in this task period")
        allowed = {"proposed": {"target_count"}, "agreed": {"target_count", "acceptance"},
                   "declined": {"acceptance"}, "progress_report": {"count"},
                   "participation_changed": {"participation"}, "outcome_report": {"outcome"}}
        if set(payload["changes"]) - allowed.get(target["event_type"], set()):
            raise ValueError("correction fields do not match target event")
        if "count" in payload["changes"] and target["payload"].get("report_mode") == "attempt" and payload["changes"]["count"] not in {0, 1}:
            raise ValueError("corrected single attempt must remain zero or one")
        # A correction to an old aggregate/plan is not permission to roll
        # back a later explicit current report.  The orchestrator may retain
        # the quotation for historical adjudication, but cannot guess that it
        # replaced the newer current value.
        for field in payload["changes"]:
            def relevant(other: Mapping[str, Any]) -> bool:
                other_kind, other_payload = other["event_type"], other["payload"]
                if field == "count":
                    return (target["payload"].get("report_mode") == "cumulative"
                            and other_kind == "progress_report" and other_payload.get("report_mode") == "cumulative")
                if field == "target_count":
                    return other_kind == "agreed" and "target_count" in other_payload
                if field == "acceptance":
                    return other_kind in {"proposed", "agreed", "declined"}
                return other_kind == {"participation": "participation_changed", "outcome": "outcome_report"}.get(field)
            target_time = _moment(target.get("projection_field_times", {}).get(field, target["reported_at"]), "target field time")
            for other in effective.values():
                if other is not target and relevant(other):
                    other_time = _moment(other.get("projection_field_times", {}).get(field, other["reported_at"]), "other field time")
                    if other_time > target_time:
                        raise ValueError("correction targets a superseded current field; clarification required")
        target["payload"].update(deepcopy(payload["changes"]))
        target["projection_support_source_id"] = event["source_id"]
        target["projection_corrected_at"] = event["reported_at"]
        for field in payload["changes"]:
            target.setdefault("projection_field_times", {})[field] = event["reported_at"]
            target.setdefault("projection_field_sources", {})[field] = event["source_id"]
            target.setdefault("projection_field_event_ids", {})[field] = event["event_id"]
    result["state_version"] = len(ordered)
    supporting_objects = set()
    for event in ordered:
        source = _source(sources, event["source_id"], user_id)
        if (event["payload"]["object_label"] not in source["text"]
                and _bound_context_object(event, source, sources, context_objects)):
            supporting_objects.add(context_objects[task_instance_id]["source_id"])
    result["object_support_source_ids"] = sorted(supporting_objects)
    result["source_ids"] = sorted({e["source_id"] for e in ordered} | supporting_objects)
    result["event_ids"] = [e["event_id"] for e in ordered]
    attempts: dict[str, dict[str, Any]] = {}
    cumulative: list[dict[str, Any]] = []
    conflict = False
    due_at = None
    axis_times: dict[str, tuple[datetime, str]] = {}
    result["field_support_source_ids"] = {}
    support_ids = result["field_support_event_ids"]
    def effective_support(event, field):
        return event.get("projection_field_event_ids", {}).get(field, event["event_id"])
    def apply_axis(field: str, value: Any, event: Mapping[str, Any], payload_field: str) -> None:
        time = event.get("projection_field_times", {}).get(payload_field, event["reported_at"])
        key = (_moment(time, "field update"), event["event_id"])
        if key >= axis_times.get(field, (datetime.min.replace(tzinfo=key[0].tzinfo), "")):
            axis_times[field] = key
            if field == "target_count":
                result["progress"][field] = value
            else:
                result[field] = value
            result["field_support_source_ids"][field] = event.get("projection_field_sources", {}).get(payload_field, event["source_id"])
            support_ids[field] = [effective_support(event, payload_field)]
    user_decision_times = [_moment(e.get("projection_field_times", {}).get("acceptance", e["reported_at"]), "decision time")
                           for e in effective.values() if e["event_type"] in {"agreed", "declined"}]
    last_decision = max(user_decision_times, default=None)
    attempt_conflict_support = set()
    for event in effective.values():
        kind, payload = event["event_type"], event["payload"]
        if kind == "proposed":
            if last_decision is None:
                apply_axis("acceptance", "proposed", event, "acceptance")
            if last_decision is None or _moment(event["reported_at"], "proposal time") > last_decision:
                result["pending_proposals"].append({"event_id": event["event_id"], "source_id": event["source_id"],
                    "origin": event["origin"], "target_count": payload.get("target_count"),
                    "due_at": payload.get("due_at"), "object_label": payload["object_label"]})
        elif kind in {"agreed", "declined"}:
            apply_axis("acceptance", payload.get("acceptance", kind), event, "acceptance")
            if "target_count" in payload:
                apply_axis("target_count", payload["target_count"], event, "target_count")
            if "due_at" in payload:
                due_at = _moment(payload["due_at"], "due_at")
        elif kind == "participation_changed":
            apply_axis("participation", payload["participation"], event, "participation")
        elif kind == "outcome_report":
            apply_axis("outcome", payload["outcome"], event, "outcome")
        elif kind == "due_review":
            due_at = _moment(payload["due_at"], "due_at")
        elif kind == "progress_report":
            if payload["report_mode"] == "cumulative":
                cumulative.append(event)
            else:
                attempt_id = payload["attempt_id"]
                previous = attempts.get(attempt_id)
                if previous and (previous["payload"]["count"] != payload["count"] or previous["occurred_at"] != event["occurred_at"]):
                    # A correction to the same attempt is explicit and wins;
                    # two uncorrected contradictory reports remain unresolved.
                    if "projection_corrected_at" in previous or "projection_corrected_at" in event:
                        prior_time = previous.get("projection_corrected_at", previous["reported_at"])
                        this_time = event.get("projection_corrected_at", event["reported_at"])
                        if _moment(prior_time, "correction time") > _moment(this_time, "correction time"):
                            continue
                    else:
                        conflict = True
                        attempt_conflict_support.update([effective_support(previous, "count"), effective_support(event, "count")])
                attempts[attempt_id] = event
    progress = result["progress"]
    progress["attempt_ids"] = sorted(attempts)
    progress["known_attempt_count"] = sum(e["payload"]["count"] for e in attempts.values())
    count: int | None = progress["known_attempt_count"] if attempts else None
    count_support = {effective_support(e, "count") for e in attempts.values() if e["payload"]["count"]}
    if attempts and not count_support:
        count_support.add(effective_support(max(attempts.values(), key=lambda e: e["reported_at"]), "count"))
    count_support |= attempt_conflict_support
    support_ids["known_attempt_count"] = sorted(count_support)
    if cumulative:
        cumulative.sort(key=lambda e: (_moment(e.get("projection_field_times", {}).get("count", e["reported_at"]), "count report"), e["event_id"]))
        latest = cumulative[-1]
        base = latest["payload"]["count"]
        cutoff = _moment(latest["payload"].get("covers_through", latest["occurred_at"]), "coverage time")
        cutoff_low = _moment(latest["occurred_at"], "coverage start") if latest.get("occurred_precision") == "day" else cutoff
        cutoff_high = _moment(latest["occurred_before"], "coverage end") if latest.get("occurred_precision") == "day" else cutoff
        covered = later = 0
        ambiguous_attempts = []
        count_support = {effective_support(latest, "count")} | attempt_conflict_support
        covered_support = set()
        for attempt_id, attempt in attempts.items():
            value = attempt["payload"]["count"]
            if not value:
                continue
            begin = _moment(attempt["occurred_at"], "attempt start")
            end = _moment(attempt["occurred_before"], "attempt end") if attempt.get("occurred_precision") == "day" else begin
            if attempt.get("occurred_precision") != "day" and latest.get("occurred_precision") != "day":
                if begin <= cutoff:
                    covered += value
                    covered_support.add(effective_support(attempt, "count"))
                else:
                    later += value
                    count_support.add(effective_support(attempt, "count"))
            elif end <= cutoff_low and (attempt.get("occurred_precision") == "day" or end < cutoff_low):
                covered += value
                covered_support.add(effective_support(attempt, "count"))
            elif begin >= cutoff_high and (latest.get("occurred_precision") == "day" or begin > cutoff_high):
                later += value
                count_support.add(effective_support(attempt, "count"))
            else:
                ambiguous_attempts.append(attempt_id)
                count_support.add(effective_support(attempt, "count"))
        if ambiguous_attempts:
            conflict = True
        progress["coverage_conflict_attempt_ids"] = sorted(ambiguous_attempts)
        if covered > base:
            conflict = True
            count_support |= covered_support
        # Unexplained decreasing totals conflict; explicit append-only
        # correction has already replaced the earlier effective total.
        if len(cumulative) > 1 and base < cumulative[-2]["payload"]["count"] and "projection_corrected_at" not in latest:
            conflict = True
            count_support.add(effective_support(cumulative[-2], "count"))
        count = base + later
        progress["reported_total"] = base
        support_ids["reported_total"] = [effective_support(latest, "count")]
    support_ids["reported_count"] = sorted(count_support)
    progress["count_conflict"] = conflict
    progress["reported_count"] = None if conflict else count
    target = progress["target_count"]
    if not conflict and count is not None:
        support_ids["completion"] = sorted(set(support_ids["reported_count"]) | set(support_ids.get("target_count", [])))
        progress["completion"] = ("not_started" if count == 0 else "unknown" if target is None
                                  else "partial" if count < target else "target_met" if count == target else "over_target")
    if now and due_at:
        result["due_review"] = now >= due_at
    result["eligibility"] = ("proposal_only" if ordered and all(e["origin"] == "assistant_proposal" for e in ordered)
                             else "sourced_current_report" if ordered else "unknown")
    return result
