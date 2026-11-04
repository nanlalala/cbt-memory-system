"""Frozen secondary response diagnostics for the v3 adapted comparison.

The released official scorer remains unchanged. These conservative short-answer
rules diagnose format compliance and literal commitment, not semantic or clinical
correctness. References are accepted only by the scoring-side function.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
import unicodedata

from score_objective_comparison import (
    deterministic_score,
    mab_normalize_answer,
)

VERSION = "v3-committed-answer-diagnostics-v1"
MAX_ANSWER_CHARACTERS = 160
MAX_ANSWER_WORDS = 24

# Deliberately language/format cues, not checks for particular benchmark answers.
# A cue can have false positives (for example a work title); this diagnostic is
# not an independent semantic judgement and must be reported beside the official
# metric, never silently substituted for it.
_ABSTENTION_PATTERNS = (
    r"^UNKNOWN$",
    r"\bi\s+(?:do\s+not|don't|don’t)\s+know\b",
    r"\b(?:i\s+)?(?:cannot|can't|can’t|could\s+not|couldn't|couldn’t)\s+(?:(?:confidently|reliably)\s+)?(?:determine|identify|tell|answer|establish|say)\b",
    r"\bi(?:\s+am|'m|’m)\s+not\s+(?:sure|certain|confident)\b",
    r"\b(?:not\s+enough|insufficient)\s+(?:information|evidence|context)\b",
    r"\b(?:history|context|information|evidence)\s+(?:does\s+not|doesn't|doesn’t)\s+(?:state|contain|provide|establish|identify|mention)\b",
    r"\b(?:it\s+is|it's|it’s)\s+(?:unclear|unknown|uncertain)\b",
    r"\b(?:unable\s+to\s+(?:determine|identify|answer)|cannot\s+be\s+determined)\b",
    r"无法(?:确定|判断|回答)|不知道|信息不足|证据不足|无法确认",
)
_ABSTENTION = tuple(re.compile(p, re.IGNORECASE) for p in _ABSTENTION_PATTERNS)
_OPTION = re.compile(r"(?:([a-d])|\(([a-d])\))", re.IGNORECASE)


def policy_descriptor() -> dict:
    """JSON-serializable protocol policy, frozen before any v3 generation."""
    return {
        "version": VERSION,
        "official_metric": "unchanged upstream deterministic scoring; separately reported",
        "secondary_metric": "exact normalized full short answer plus format/abstention checks",
        "mab_normalization": "same mab_normalize_answer as frozen official scorer; equality, not substring",
        "optional_wrapper": "one leading case-insensitive Answer: only",
        "maximum_answer_characters": MAX_ANSWER_CHARACTERS,
        "maximum_answer_whitespace_words": MAX_ANSWER_WORDS,
        "required_nonempty_lines": 1,
        "abstention_patterns": list(_ABSTENTION_PATTERNS),
        "personamem_format": "entire answer is one a-d label, bare or parenthesized",
        "interpretation": "conservative format/literal-alias diagnostic, not semantic truth, independently reviewed conflict resolution or clinical usefulness",
        "no_posthoc_threshold_selection": True,
    }


def _answer_text(prediction: str) -> str:
    if not isinstance(prediction, str):
        raise ValueError("prediction must be text")
    return re.sub(r"^Answer:\s*", "", prediction.strip(), count=1, flags=re.IGNORECASE).strip()


def response_format_diagnostics(prediction: str) -> dict:
    """Gold-blind observations; do not infer factual correctness from these."""
    answer = _answer_text(prediction)
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    matched = [i for i, pattern in enumerate(_ABSTENTION) if pattern.search(answer)]
    within_limits = len(answer) <= MAX_ANSWER_CHARACTERS and len(answer.split()) <= MAX_ANSWER_WORDS
    option = _OPTION.fullmatch(answer)
    return {
        "diagnostic_version": VERSION,
        "answer_text": answer,
        "nonempty": bool(answer),
        "nonempty_line_count": len(lines),
        "characters": len(answer),
        "whitespace_words": len(answer.split()),
        "within_short_answer_limits": within_limits,
        "abstention_cue_detected": bool(matched),
        "abstention_pattern_indices": matched,
        "disjunction_cue_detected": bool(re.search(r"\b(?:either|or)\b", answer, re.IGNORECASE)),
        "enumeration_cue_detected": bool(re.search(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", answer)),
        "bare_option_label": (option.group(1) or option.group(2)).lower() if option else None,
        "short_single_line_without_abstention": bool(answer) and len(lines) == 1 and within_limits and not matched,
        "boundary": "lexical cues and length only; a short non-abstaining response may still be wrong or ambiguous",
    }


def score_supplemental(prediction: str, reference: dict) -> dict:
    """Scoring-side only. This function never prepares a model request."""
    official, _ = deterministic_score(prediction, reference)
    diagnostic = response_format_diagnostics(prediction)
    answer = diagnostic["answer_text"]
    if reference["metric"] == "mab_substring_exact_match":
        normalized = mab_normalize_answer(answer)
        exact = bool(normalized) and any(normalized == mab_normalize_answer(alias)
                                         for alias in reference["accepted_answers"])
    elif reference["metric"] == "personamem_option_exact_match":
        exact = diagnostic["bare_option_label"] == reference["correct_option"].strip("() ").lower()
    else:  # deterministic_score also refuses this; retain a local explicit guard.
        raise ValueError("unsupported supplemental metric")
    committed = bool(exact and diagnostic["short_single_line_without_abstention"])
    return {
        "official_score": official,
        "exact_alias_or_option_score": float(exact),
        "strict_committed_score": float(committed),
        "official_positive_with_abstention_cue": bool(official and diagnostic["abstention_cue_detected"]),
        "official_positive_not_strict": bool(official and not committed),
        "format": diagnostic,
        "secondary_interpretation": "strict literal answer and declared response-format compliance; not independently reviewed semantic correctness",
    }


def _question_fingerprint(case: dict) -> str | None:
    query = case.get("query")
    if query is None:
        return None
    if not isinstance(query, str):
        raise ValueError("case query must be text")
    normalized = unicodedata.normalize("NFC", query).casefold().translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"', "？": "?", "！": "!", "。": ".",
    }))
    normalized = " ".join(normalized.split()).rstrip(" ?!.")
    return hashlib.sha256((str(case.get("dataset_key", "")) + "\0" + normalized).encode()).hexdigest()


def development_partition(cases: list[dict], exposed_case_ids: list[str]) -> dict:
    """Gold-free accounting of development exposure, including variant overlap.

    'Question-held-out' does not mean independent held-out users or source
    histories. Same source families/prefixes remain dependent and development
    exposed. Call this before generation and freeze the returned IDs in protocol.
    """
    ids = [case.get("case_id") for case in cases]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("cases require unique nonempty IDs")
    if len(set(exposed_case_ids)) != len(exposed_case_ids):
        raise ValueError("duplicate exposed case IDs")
    missing = sorted(set(exposed_case_ids) - set(ids))
    if missing:
        raise ValueError("exposed cases absent from frozen cohort; cannot infer their query identity")
    exposed = [case for case in cases if case["case_id"] in set(exposed_case_ids)]
    fingerprints = {_question_fingerprint(case) for case in exposed} - {None}
    development, heldout, reasons = [], [], {}
    for case in cases:
        cid = case["case_id"]
        why = []
        if cid in exposed_case_ids:
            why.append("exact_case_previously_exposed")
        fingerprint = _question_fingerprint(case)
        if fingerprint is not None and fingerprint in fingerprints:
            why.append("same_normalized_question_text")
        if why:
            development.append(cid)
            reasons[cid] = why
        else:
            heldout.append(cid)
    return {
        "version": "question-development-exposure-v1",
        "explicit_exposed_case_ids": list(exposed_case_ids),
        "development_case_ids": development,
        "question_heldout_case_ids": heldout,
        "development_count": len(development),
        "question_heldout_count": len(heldout),
        "development_reasons": reasons,
        "reason_counts": dict(Counter(reason for values in reasons.values() for reason in values)),
        "question_identity_rule": "same dataset key and NFC/casefold/whitespace normalized query text; normalize curly quotes/full-width terminal punctuation and strip terminal ?!. only; source ordinal numbers are not stable cross-length question identities",
        "group_independence_claim": False,
        "boundary": "question-held-out accounting only; shared source histories, context variants and benchmark adaptation remain development-exposed; no independent-participant or confirmatory claim",
    }


def partition_from_exposed_cases(cases: list[dict], exposed_cases: list[dict]) -> dict:
    """Use previously frozen case metadata, not scoring references, for exposure.

    The caller reads the old run's checksummed ``cases.jsonl`` and freezes this
    return value in the new protocol. Case IDs present in both files must still
    have matching normalized queries/dataset; a changed dataset cannot silently
    inherit the old exposure accounting.
    """
    cohort_by_id = {case["case_id"]: case for case in cases}
    for old in exposed_cases:
        current = cohort_by_id.get(old.get("case_id"))
        if current is None:
            raise ValueError("previously exposed case absent from cohort")
        if _question_fingerprint(old) is None or _question_fingerprint(old) != _question_fingerprint(current):
            raise ValueError("previously exposed question identity changed")
    return development_partition(cases, [case["case_id"] for case in exposed_cases])
