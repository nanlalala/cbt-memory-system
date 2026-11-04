"""Conservative source-only version views for synthetic fact benchmarks.

This is a grammar adapter, not a general semantic extractor or a statement that
the quoted facts are true. It only recognises explicit positive assertions in a
small declared English grammar. Unsupported, negated and qualified text remains
available verbatim. No question, answer label, model knowledge or confirmation
event is consulted. Callers must apply the common access/time boundary first.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import re

PARSER_VERSION = "source-slot-version-view-v1"


@dataclass(frozen=True, slots=True)
class Assertion:
    subject: str
    relation: str
    object: str


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


# These are syntax declarations, not a list learned from released gold labels.
# Longest relations go first, so a compound relation's internal "of" is not
# accidentally parsed as the subject delimiter.
_OF_RELATIONS = {
    "country of citizenship": "citizenship",
    "country of origin": "country of origin",
    "place of birth": "place of birth",
    "date of birth": "date of birth",
    "place of death": "place of death",
    "date of death": "date of death",
    "field of work": "field of work",
    "official language": "official language",
    "chief executive officer": "chief executive officer",
    "capital city": "capital",
    "work location": "work location",
    "notable work": "notable work",
    "political party": "political party",
    "headquarters": "headquarters",
    "director": "director",
    "author": "author",
    "performer": "performer",
    "developer": "developer",
    "creator": "creator",
    "founder": "founder",
    "producer": "producer",
    "publisher": "publisher",
    "designer": "designer",
    "composer": "composer",
    "partner": "partner",
    "spouse": "spouse",
    "religion": "religion",
    "continent": "continent",
    "capital": "capital",
    "country": "country",
    "position": "position",
    "sport": "sport",
    "citizenship": "citizenship",
    "nationality": "nationality",
    "occupation": "occupation",
    "employer": "employer",
    "location": "location",
    "language": "language",
    "manufacturer": "manufacturer",
    "owner": "owner",
    "president": "president",
    "leader": "leader",
    "coach": "coach",
    "manager": "manager",
    "population": "population",
    "currency": "currency",
}
_RELATION_PATTERN = "|".join(re.escape(s) for s in sorted(_OF_RELATIONS, key=len, reverse=True))
_OF = re.compile(rf"^(?:The\s+)?(?P<relation>{_RELATION_PATTERN}) of (?P<subject>.+?) (?:is|was) (?P<object>.+)$", re.I)
_POSSESSIVE = re.compile(rf"^(?P<subject>.+?)(?:'s|’s) (?P<relation>{_RELATION_PATTERN}) (?:is|was) (?P<object>.+)$", re.I)
_QUALIFIED = re.compile(
    r"\b(?:not|never|no longer|may|might|could|maybe|perhaps|possibly|probably|"
    r"allegedly|reportedly|uncertain|unknown|unclear|seems?|appears?|believes?|"
    r"thinks?|suspects?|used to|previously|formerly|historically)\b", re.I)
_NEGATIVE_PREFIX = re.compile(r"^(?:No|Neither|None)\b", re.I)

_FORMS = [
    (re.compile(r"^(?P<subject>.+?) (?:is|was) a citizen of (?P<object>.+)$", re.I), "citizenship"),
    (re.compile(r"^(?P<subject>.+?) plays (?:at )?the position of (?P<object>.+)$", re.I), "position"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) famous for (?P<object>.+)$", re.I), "notable work"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) headquartered in (?P<object>.+)$", re.I), "headquarters"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) located in (?P<object>.+)$", re.I), "location"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) a member of (?P<object>.+)$", re.I), "membership"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) affiliated with (?P<object>.+)$", re.I), "affiliation"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) educated at (?P<object>.+)$", re.I), "educated at"),
    (re.compile(r"^(?P<subject>.+?) (?:is|was) employed by (?P<object>.+)$", re.I), "employer"),
    (re.compile(r"^(?P<subject>.+?) works in (?P<object>.+)$", re.I), "work location"),
    (re.compile(r"^(?P<subject>.+?) (?:was|is) born in (?P<object>.+)$", re.I), "place of birth"),
    (re.compile(r"^(?P<subject>.+?) died in (?P<object>.+)$", re.I), "place of death"),
    (re.compile(r"^(?P<subject>.+?) has the occupation (?:of )?(?P<object>.+)$", re.I), "occupation"),
]
for _participle, _relation in {
    "written": "author", "directed": "director", "performed": "performer",
    "developed": "developer", "created": "creator", "founded": "founder",
    "designed": "designer", "produced": "producer", "published": "publisher",
    "composed": "composer", "manufactured": "manufacturer", "owned": "owner",
}.items():
    _FORMS.append((re.compile(rf"^(?P<subject>.+?) (?:was|is) {_participle} by (?P<object>.+)$", re.I), _relation))


def parse_assertion(text: str) -> Assertion | None:
    """Parse only declared positive single-line templates, without rewriting source.

    Leading benchmark sequence numbers are presentation metadata. Their value
    is never used as chronology; the caller's validated ``order`` field is used.
    Unsupported grammar is deliberately retained, rather than guessed.
    """
    if not isinstance(text, str):
        raise TypeError("source text must be a string")
    line = text.strip()
    if not line or "\n" in line or "\r" in line:
        return None
    line = re.sub(r"^\d+[.)]\s+", "", line)
    if _QUALIFIED.search(line) or _NEGATIVE_PREFIX.search(line) or "?" in line or ";" in line:
        return None
    # A terminal full stop is punctuation, not part of the relation value.
    if line.endswith("."):
        line = line[:-1].rstrip()
    # Reject obvious second clauses. A single source line may carry several
    # facts; this adapter must not supersede that entire line using one slot.
    if re.search(r"\b(?:and|but|whereas|although|however)\b", line, re.I):
        return None
    match = _OF.fullmatch(line) or _POSSESSIVE.fullmatch(line)
    if match:
        subject, relation, value = match["subject"].strip(), _OF_RELATIONS[_normalise(match["relation"])], match["object"].strip()
    else:
        match, relation = None, None
        for pattern, candidate_relation in _FORMS:
            candidate = pattern.fullmatch(line)
            if candidate:
                match, relation = candidate, candidate_relation
                break
        if match is None:
            return None
        subject, value = match["subject"].strip(), match["object"].strip()
    if not subject or not value or len(subject) > 512 or len(value) > 1024:
        return None
    # Do not treat an embedded assertion or several sentences as one object.
    # Initials in entity names (e.g. W. G. Sebald) remain accepted.
    if re.search(r"[.!]\s+[A-Za-z]{2,}\b", value) or re.search(r"\b(?:is|was|are|were)\b", value, re.I):
        return None
    return Assertion(subject, relation, value)


def consolidate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Return a non-destructive latest-version view over already eligible history.

    Original rows are neither mutated nor deleted. Source session is part of
    the slot, so distinct benchmark contexts cannot affect one another. Equal
    maximum-order observations are all retained; no arbitrary ID tie-break
    silently chooses a conflicting version.
    """
    parsed, groups, identities = [], defaultdict(list), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError("source rows must be objects")
        for name in ("session_id", "turn_id"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError("source identity is required")
        if type(row.get("order")) is not int or row["order"] < 0:
            raise ValueError("source order must be a nonnegative integer")
        if row.get("lifecycle_status", "candidate") not in {"candidate", "active"} or row.get("sensitivity") == "restricted":
            raise ValueError("version view requires common eligible sources; inactive/restricted source received")
        identity = row["session_id"], row["turn_id"]
        if identity in identities:
            raise ValueError("duplicate source identity")
        identities.add(identity)
        assertion = parse_assertion(row.get("text"))
        slot = (row["session_id"], _normalise(assertion.subject), _normalise(assertion.relation)) if assertion else None
        parsed.append((assertion, slot))
        if slot is not None:
            groups[slot].append(index)
    kept = {i for i, (assertion, _) in enumerate(parsed) if assertion is None}
    winners = {}
    for slot, indices in groups.items():
        latest_order = max(rows[i]["order"] for i in indices)
        current = [i for i in indices if rows[i]["order"] == latest_order]
        winners[slot] = current
        kept.update(current)
    entries = []
    for index, row in enumerate(rows):
        assertion, slot = parsed[index]
        current = winners.get(slot, [])
        latest_sources = [{"session_id": rows[i]["session_id"], "turn_id": rows[i]["turn_id"], "order": rows[i]["order"]} for i in current]
        status = ("unparsed" if assertion is None else "superseded" if index not in kept else
                  "unresolved_latest_tie" if len(current) > 1 else "latest")
        entries.append({"session_id": row["session_id"], "turn_id": row["turn_id"], "order": row["order"],
                        "parsed": assertion is not None, "assertion": asdict(assertion) if assertion else None,
                        "slot": list(slot) if slot else None, "status": status,
                        "latest_source": latest_sources[0] if len(latest_sources) == 1 else None,
                        "latest_sources": latest_sources})
    selected = [row for i, row in enumerate(rows) if i in kept]
    selected.sort(key=lambda row: (row["order"], row["session_id"], row["turn_id"]))
    return selected, {"parser_version": PARSER_VERSION, "source_rows": len(rows),
                      "parsed_rows": sum(a is not None for a, _ in parsed),
                      "unparsed_rows": sum(a is None for a, _ in parsed), "version_groups": len(groups),
                      "superseded_rows": len(rows) - len(selected),
                      "unresolved_latest_tie_groups": sum(len(value) > 1 for value in winners.values()),
                      "entries": entries,
                      "interpretation": "synthetic benchmark source-order view; not confirmed user facts, semantic entailment or clinical state"}
