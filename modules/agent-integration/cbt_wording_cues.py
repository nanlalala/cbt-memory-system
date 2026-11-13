"""Deterministic, no-network detector for tentative wording cues.

A cue is only a hypothesis about how a user may be feeling, inferred from their
own wording. It never states a fact, rating, severity or diagnosis, and it
keeps the exact quoted words as evidence. Output items match the frozen
`cbt_projection_contract.bind_candidates` envelope: section `working_formulation`,
origin `agent_inference`, exact unique literal evidence.

Wording is a weak signal (see the project literature notes); treat every cue as
something to check with the user, never as a conclusion. Self-harm language is
deliberately out of scope here: it belongs to the deterministic safety router.
"""
import re

# (cue label, compiled pattern). Patterns match user wording only.
CUES = [
    ("possibly discouraged",
     re.compile(r"(?:just )?(?:isn't|is not|isnt|not) for me|(?:want|going) to give up|no point(?: in)?|what's the point", re.I)),
    ("possibly overwhelmed",
     re.compile(r"too much for me|(?:it'?s|is) too much|overwhelm(?:ed|ing)|can't cope|cannot cope", re.I)),
    ("possibly anxious about it",
     re.compile(r"dreading|dread (?:it|this|the)|terrified|panic(?:king|ky)?|freez(?:e|ing)", re.I)),
]
ABSOLUTIST = re.compile(r"\b(always|never|completely|totally|nothing|everything|entire(?:ly)?)\b", re.I)
MAX_ITEMS = 8


def wording_cues(sources, prefix="w"):
    """Return candidate items from user-role sources only."""
    items = []
    for src in sources:
        if src.get("role") != "user":
            continue
        text = src["content"]
        for label, pattern in CUES:
            m = pattern.search(text)
            if not m or text.count(m.group(0)) != 1:
                continue  # exact unique quote required for binding
            value = {"cue": label, "basis": "user wording", "status": "tentative; check with the user"}
            a = ABSOLUTIST.search(text)
            if a and text.count(a.group(0)) == 1 and a.group(0) != m.group(0):
                value["also_notes"] = "absolutist wording (not diagnostic)"
            evidence = [{"source_id": src["source_id"], "quote": m.group(0)}]
            if "also_notes" in value:
                evidence.append({"source_id": src["source_id"], "quote": a.group(0)})
            items.append({"item_id": f"{prefix}{len(items)}", "section": "working_formulation",
                          "value": value, "origin": "agent_inference", "evidence": evidence})
            if len(items) >= MAX_ITEMS:
                return items
    return items
