"""Same v4 field selection with an isolated v8 evidence-type import.

Only the import namespace differs. Original selection functions, wire policy
and question contracts remain unchanged. No bare product module is registered.

This pure helper does not infer a fact, read targets, answer a question, or grant
permission. It preserves all original quotation claims and direct recent
messages are outside its input. A caller must still apply the original source,
subject, cutoff, version and durable send/publication gates to the returned
bundles. It is deliberately not wired into any frozen v1/v2/v3 experiment.

For a clearly single-axis factual question, unrelated known task fields can be
withheld only if every span they would remove is already present verbatim in a
retained original quotation. Unknown values and controls remain visible. The
same question classifier and selection policy are suitable for either arm;
quotation-only RAG therefore retains every original quotation unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import re

import hashlib
import importlib.util
from pathlib import Path
import sys


def _owned_private_evidence():
    """Load the frozen evidence types without registering a product module name."""
    path = Path(__file__).with_name("evidence_bundles.py").resolve()
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if source_sha != "53eea74e99ddbe38068084bee6a2c5532c980fe7af3231426965a815d2390d17":
        raise RuntimeError("private_evidence_source_bytes_changed")
    name = "cbt_v8_question_selection_private_evidence"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("owned_private_evidence_module_required")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        module._selection_source_sha256 = source_sha
    if (Path(module.__file__).resolve() != path
            or getattr(module, "_selection_source_sha256", None) != source_sha):
        raise RuntimeError("foreign_private_evidence_module")
    return module


_PRIVATE_EVIDENCE = _owned_private_evidence()
EvidenceBundle = _PRIVATE_EVIDENCE.EvidenceBundle
EvidenceClaim = _PRIVATE_EVIDENCE.EvidenceClaim
EvidenceSpan = _PRIVATE_EVIDENCE.EvidenceSpan

VERSION = "question-only-task-field-selection-v4-development-v1"
TASK_FIELDS = frozenset({"target_count", "reported_count", "outcome", "acceptance", "engagement"})
# Participation and all unknown/tombstone/sensitive/version controls must survive
# even a question about another axis. Unrecognised fields also remain untouched.
CONTROL_FIELDS = frozenset({
    "engagement", "participation", "permission", "negation", "current_status",
    "deleted", "tombstone", "lifecycle_status", "version", "valid_from", "valid_to",
    "sensitivity", "expires_at", "blocked", "availability", "activity_status",
    "paused", "due_review", "due_at",
})
_QUOTE_FIELDS = frozenset({"history_quote", "source_quote"})
_QUERY = re.compile(
    r"[?？]|(?:^|[.;!。；！]\s*)(?:what|which|how|did|do|does|is|are|am|has|have|was|were|"
    r"can|could|whether)\b|是什么|怎么样|如何|几次|多少|是否|有没有|吗(?:[。！\s]*$)", re.I)
_COUNT = re.compile(r"\b(?:how many|count|number of|total)\b|次数|几次|多少次|累计|总共|一共", re.I)
_TARGET = re.compile(r"\b(?:target|goal|planned|plan to|committed to)\b|目标|计划|打算", re.I)
_AGREED_QUANTITY = re.compile(
    r"\b(?:agree|agreed|committed|promise|promised)\b.{0,48}\b(?:to do|to complete|to try|target|count|times?)\b|"
    r"(?:同意|答应|承诺).{0,24}(?:做|完成|试|目标|次数)", re.I)
_COMPLETION = re.compile(
    r"\b(?:complet(?:e|ed|ing)|done|did|attempt(?:ed|s)?|reported count|cumulative|so far|altogether)\b|"
    r"完成|做了|做过|尝试了|累计|总共|一共", re.I)
_OUTCOME = re.compile(
    r"\b(?:outcome|benefit|helpful|unhelpful|useful|effectiveness|helped|did it help|did that help)\b|"
    r"效果|有没有帮助|是否有帮助|有用|有没有用|是否有效|结果", re.I)
_AGREEMENT = re.compile(r"\b(?:agree|agreed|accept|accepted|consent|consented|willing)\b|同意|答应|接受|愿意", re.I)
_PARTICIPATION = re.compile(
    r"\b(?:paused?|on hold|cancel(?:led|ed)?|resum(?:e|ed)|participation|continue|active)\b|"
    r"暂停|先停|继续|恢复|取消|参与", re.I)
# A numerical symptom rating or symptom change is a distinct clinical axis, not
# a practice completion count or the user's helpful/unhelpful practice report.
_SYMPTOM_RATING = re.compile(r"\b(?:severity|score|rating|symptom level|anxiety level|mood level)\b|程度|评分|分数|强度", re.I)
_SYMPTOM_CHANGE = re.compile(
    r"\b(?:anxiety|depression|mood|symptoms?)\b.{0,32}\b(?:improv(?:e|ed|ing)|worsen(?:ed|ing)?|changed?|better|worse)\b|"
    r"\b(?:improv(?:e|ed|ing)|worsen(?:ed|ing)?|changed?|better|worse)\b.{0,32}\b(?:anxiety|depression|mood|symptoms?)\b|"
    r"(?:焦虑|抑郁|情绪|症状).{0,16}(?:改善|好转|变好|恶化|加重|变化)", re.I)


def select_question_field(question: str) -> str | None:
    """Return one explicitly requested task axis, or None conservatively.

    Only question text is examined. Unsupported and multi-axis queries keep the
    original complete field context. No values, source IDs, case identities,
    arm labels or released answer type are accepted by this function.
    """
    if type(question) is not str:
        raise TypeError("question must be text")
    if not question.strip() or not _QUERY.search(question):
        return None
    if _SYMPTOM_RATING.search(question) or _SYMPTOM_CHANGE.search(question):
        return None
    count = bool(_COUNT.search(question))
    target = count and bool(_TARGET.search(question) or _AGREED_QUANTITY.search(question))
    completed = count and bool(_COMPLETION.search(question))
    outcome = bool(_OUTCOME.search(question))
    agreement = bool(_AGREEMENT.search(question))
    participation = bool(_PARTICIPATION.search(question))
    fields = set()
    if target:
        fields.add("target_count")
    if completed:
        # 'How many did I agree to complete?' requests a target, not progress.
        if not target or re.search(r"\b(?:actually|already|so far|and|versus|vs)\b|已经|实际|以及|并且|分别", question, re.I):
            fields.add("reported_count")
    if count and not target and not completed:
        return None
    if outcome:
        fields.add("outcome")
        # Number of helpful occasions is not a single outcome status.
        if count:
            return None
    if agreement and not target:
        fields.add("acceptance")
    if participation:
        fields.add("engagement")
    return next(iter(fields)) if len(fields) == 1 else None


def question_field_contract(question: str) -> dict[str, str | None]:
    field = select_question_field(question)
    return {"version": VERSION, "field": field,
            "value_kind": ("integer" if field in {"target_count", "reported_count"}
                           else "status" if field else "unrestricted_scalar")}


def _span_covered(span: EvidenceSpan, quote: EvidenceClaim) -> bool:
    if quote.field not in _QUOTE_FIELDS or quote.support_status != "source_quote":
        return False
    if quote.origin == "agent_inference" or type(quote.value) is not str:
        return False
    for option in quote.evidence_options:
        for original in option:
            if (quote.value != original.text or original.source_id != span.source_id
                    or original.role != span.role or original.source_time != span.source_time
                    or original.start > span.start or original.end < span.end):
                continue
            start = span.start - original.start
            if original.text[start:start + len(span.text)] == span.text:
                return True
    return False


def _all_spans(claim: EvidenceClaim) -> tuple[EvidenceSpan, ...]:
    # Retain every alternative's provenance, not only the first packed option.
    return tuple(span for option in claim.evidence_options for span in option)


def _unknown(value) -> bool:
    return value is None or (type(value) is str and value.casefold() in {"unknown", "blocked", "deleted", "expired", "revoked"})


@dataclass(frozen=True, slots=True)
class QuestionFieldSelection:
    bundles: tuple[EvidenceBundle, ...]
    field: str | None
    omitted_claim_ids: tuple[str, ...]
    retained_for_provenance: tuple[str, ...]
    version: str = VERSION


def select_question_fields(question: str, bundles: Sequence[EvidenceBundle]) -> QuestionFieldSelection:
    """Withhold only redundant optional fields; never add evidence or values.

    All quotation claims and all control/unknown/unrecognised fields are the
    very same EvidenceClaim objects as the input. An omitted field's exact
    evidence is already covered by one of those retained quotation objects.
    A caller may safely retain the original dependency gate as a superset.
    """
    if not isinstance(bundles, Sequence) or isinstance(bundles, (str, bytes)):
        raise TypeError("bundles must be an ordered sequence")
    if any(not isinstance(bundle, EvidenceBundle) for bundle in bundles):
        raise TypeError("every bundle must be an EvidenceBundle")
    field = select_question_field(question)
    if field is None:
        return QuestionFieldSelection(tuple(bundles), None, (), ())
    quotes = tuple(claim for bundle in bundles for claim in bundle.claims if claim.field in _QUOTE_FIELDS)
    kept, omitted, retained = [], [], []
    for bundle in bundles:
        claims = []
        for claim in bundle.claims:
            optional = (claim.field in TASK_FIELDS and claim.field not in CONTROL_FIELDS
                        and claim.field != field and not _unknown(claim.value)
                        and claim.support_status == "rule_verified")
            if not optional:
                claims.append(claim)
                continue
            spans = _all_spans(claim)
            if spans and all(any(_span_covered(span, quote) for quote in quotes) for span in spans):
                omitted.append(claim.claim_id)
            else:
                retained.append(claim.claim_id)
                claims.append(claim)
        if claims:
            kept.append(bundle if tuple(claims) == bundle.claims else replace(bundle, claims=tuple(claims)))
    return QuestionFieldSelection(tuple(kept), field, tuple(omitted), tuple(retained))
