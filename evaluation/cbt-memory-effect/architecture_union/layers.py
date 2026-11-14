"""Pure selection helpers for the additive managed-memory layers (draft).

Layer numbering follows the private design note:
  1  raw-RAG whole messages (invariant; handled in union_requests)
  2  fact-key channel: extra whole messages found through the items that cite them
  3  pinned CBT profile: corrections and current case/task state tried first
  4  question-ranked processed items (frozen relevance order, optionally re-ranked)

Nothing here calls a model or changes stored memory. Every value is computed at
read time from the case's own sources and extracted items, so a source that is
not in the eligible pool (deleted, blocked, future, recent window) can never be
brought back by these layers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

# Sections whose content describes a state that can change over the session.
# Historical background is not decayed (an event that happened stays true).
MUTABLE_SECTIONS = frozenset({"case_state", "intervention_learning", "conversation_progress"})

# Layer 3: what is pinned regardless of the question. Stable background
# (relevant_background) is deliberately not pinned: it enters only by relevance.
PINNED_SECTIONS = ("case_state", "intervention_learning")

# Importance is set by item kind, not by a model-written score.
IMPORTANCE = {"user_correction": 1.0, "intervention_learning": 0.8, "case_state": 0.7,
              "relevant_background": 0.5, "conversation_progress": 0.4,
              "working_formulation": 0.3, "turn_guidance": 0.2}

# Evidence grade (memory_use_policy idea): a model confidence number is never used.
EVIDENCE_GRADE = {"user_correction": 1.0, "explicit_user_statement": 0.75,
                  "assistant_proposal": 0.5, "benchmark_source": 0.5, "agent_inference": 0.25}


@dataclass(frozen=True)
class UnionConfig:
    """Which additive layers are on. The default reproduces the tested 1+4 union."""
    renderer: str = "v4"            # "v5" uses the draft renderer that pins user corrections
    fact_key_sources: int = 0       # layer 2: extra whole messages (design value 3)
    pin_profile: bool = False       # layer 3
    profile_limit: int | None = None
    rerank: bool = False            # layer 4 re-ranking by recency, importance, reliability
    recency_weight: float = 0.08    # product retriever defaults; tune on the dev split only
    importance_weight: float = 0.04
    reliability_weight: float = 0.03

    def __post_init__(self):
        if self.renderer not in {"v4", "v5"}:
            raise ValueError("renderer must be v4 or v5")
        if type(self.fact_key_sources) is not int or self.fact_key_sources < 0:
            raise ValueError("fact_key_sources must be a nonnegative integer")
        if self.profile_limit is not None and (type(self.profile_limit) is not int or self.profile_limit < 0):
            raise ValueError("profile_limit must be a nonnegative integer or None")
        for name in ("recency_weight", "importance_weight", "reliability_weight"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name + " must be within [0, 1]")

    def as_dict(self):
        return asdict(self)


def _item_sources(item):
    return list(dict.fromkeys(e["source_id"] for e in item["evidence"]))


def _canonical(value):
    import json
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)


def fact_key_documents(older, items):
    """Layer 2 index: key = original message + values of the items that cite it."""
    cited = {}
    for item in items:
        for sid in _item_sources(item):
            cited.setdefault(sid, []).append(_canonical(item["value"]))
    return [{"source_id": s["source_id"],
             "content": "\n".join([s["content"], *cited.get(s["source_id"], [])])} for s in older]


def fact_key_candidates(query, older, items, exclude_ids, rank, limit):
    """Ranked extra source ids from the eligible pool, never one already quoted."""
    if limit <= 0 or not older:
        return []
    documents = fact_key_documents(older, items)
    order = rank(query, documents)
    exclude = set(exclude_ids)
    return [documents[i]["source_id"] for i in order if documents[i]["source_id"] not in exclude][:limit]


def profile_items(items, source_orders, limit=None):
    """Layer 3: corrections first, then current task/outcome and case state.

    Hypotheses are never pinned. Within a group the item whose evidence is latest
    comes first, so a later report is tried before an earlier one.
    """
    def latest(item):
        orders = [source_orders[s] for s in _item_sources(item) if s in source_orders]
        return max(orders) if orders else -1

    corrections = [i for i in items if i["origin"] == "user_correction"]
    rest = [i for i in items if i["origin"] not in {"user_correction", "agent_inference"}
            and i["section"] in PINNED_SECTIONS]
    order = {name: n for n, name in enumerate(PINNED_SECTIONS)}
    pinned = (sorted(corrections, key=lambda i: -latest(i)) +
              sorted(rest, key=lambda i: (order[i["section"]], -latest(i))))
    return pinned if limit is None else pinned[:limit]


def reliability(item, source_roles):
    """Read-time reliability tier with reasons; repetition never lifts a hypothesis."""
    grade = EVIDENCE_GRADE.get(item["origin"], 0.25)
    reasons = [item["origin"]]
    sources = _item_sources(item)
    if item["origin"] in {"explicit_user_statement", "user_correction"}:
        if not any(source_roles.get(s) == "user" for s in sources):
            grade = min(grade, 0.25)
            reasons.append("no_user_role_evidence")
        independent = sum(1 for s in sources if source_roles.get(s) == "user")
        if independent >= 2:
            grade = min(1.0, grade + 0.1 * (independent - 1))
            reasons.append("independent_mentions=" + str(independent))
    tier = "high" if grade >= 0.75 else "medium" if grade >= 0.5 else "low"
    return grade, tier, reasons


def rerank_items(ranked_items, source_orders, source_roles, history_length, config):
    """Relevance order stays primary; small weights only reorder near neighbours.

    Relevance is the frozen ranking turned into 1 - rank/n. Recency is ordinal
    (CACTUS has no calendar dates) and applies to mutable sections only.
    """
    n = len(ranked_items)
    rows = []
    for rank, item in enumerate(ranked_items):
        relevance = 1 - rank / n if n else 0.0
        orders = [source_orders[s] for s in _item_sources(item) if s in source_orders]
        if item["section"] in MUTABLE_SECTIONS and orders and history_length:
            recency = max(orders) / max(history_length - 1, 1)
        else:
            recency = 1.0
        kind = "user_correction" if item["origin"] == "user_correction" else item["section"]
        importance = IMPORTANCE.get(kind, 0.3)
        grade, tier, reasons = reliability(item, source_roles)
        score = (relevance + config.recency_weight * recency + config.importance_weight * importance
                 + config.reliability_weight * grade)
        rows.append((score, rank, item, {"item_id": item["item_id"], "relevance": round(relevance, 6),
                     "recency": round(recency, 6), "importance": importance, "reliability": round(grade, 6),
                     "reliability_tier": tier, "reliability_reasons": reasons, "score": round(score, 6)}))
    rows.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in rows], [row[3] for row in rows]
