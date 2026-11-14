"""Managed arm = the raw-RAG selection plus additive memory layers (draft).

In ``architecture_retest`` the managed arm replaces the raw retrieval channel
with item-derived evidence, so a message the raw arm retrieved can be absent from
the managed request whenever extraction did not turn it into an item. Here the
managed request always carries the same whole-message selection as the raw arm
(layer 1). Further layers only add, each switched by ``UnionConfig``:

  layer 3  pinned CBT profile items, tried first
  layer 2  fact-key channel: up to N extra whole messages found through items
  layer 4  question-ranked items (frozen order, optionally re-ranked)

Addition order is 1, 3, 2, 4, so when the complete-request ceiling binds the
layer added last loses first. A candidate that does not fit is skipped; the raw
selection is never dropped (if it alone cannot fit, the build fails loudly).
The default config reproduces the earlier tested union (layers 1 and 4).
Not used by any frozen evaluation; a run needs its own protocol and result
directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from architecture_retest import runner  # noqa: E402  (read-only reuse)
from . import VERSION  # noqa: E402
from .layers import (UnionConfig, fact_key_candidates, profile_items, rerank_items,  # noqa: E402
                     reliability)

UNION_CONDITION = "managed_union_raw_plus_items"


def _renderer(name):
    api, Span = runner._api()
    if name == "v5":
        import cbt_context_v5_draft as api
    return api, Span


def quote_claim_ids(quote_ids, compact):
    """Claim IDs for whole-message quotations: ``quote:<source_id>`` or a short unique form."""
    if not compact:
        return {sid: "quote:" + sid for sid in quote_ids}
    short = {sid: "q:" + sid.rsplit(":", 1)[-1] for sid in quote_ids}
    if len(set(short.values())) != len(short):
        return {sid: "quote:" + sid for sid in quote_ids}
    return short


def assemble_union(case, sources, quote_ids, items, *, extra_source_ids=(), renderer="v4",
                   compact=False, representation="processed_with_evidence"):
    """Like ``runner.assemble`` for processed_with_evidence, with whole-message
    quotations for ``quote_ids`` and item evidence sources admitted alongside."""
    api, Span = _renderer(renderer)
    from evidence_bundles import EvidenceBundle, EvidenceClaim
    uid = case["source_family_id"]
    by_id = {s["source_id"]: s for s in sources}
    recent = sources[-runner.SETTINGS["recent_messages"]:]
    registry = {s["source_id"]: dict(user_id=uid, source_id=s["source_id"], text=s["content"], role=s["role"],
                source_time=s.get("date"), source_order=s["source_order"],
                permissions={"save": True, "long_term_use": True, "response_external": True}) for s in sources}
    registry["CURRENT"] = dict(user_id=uid, source_id="CURRENT", text=case["input"], role="user",
        source_time=case.get("input_date") if case.get("calendar_time_available") else None,
        source_order=case["history_cutoff_order_exclusive"],
        permissions={"save": True, "long_term_use": True, "response_external": True})
    contextual = []
    for item in items:
        evidence = tuple(Span(e["source_id"], item["item_id"] + ":" + str(i), e["text"], e["start"], e["end"],
                              by_id[e["source_id"]].get("date"), by_id[e["source_id"]]["role"])
                         for i, e in enumerate(item["evidence"]))
        contextual.append(api.ContextualItem(item_id=item["item_id"], section=item["section"], value=item["value"],
            evidence=evidence, origin=item["origin"],
            support_status="source_supported" if item["origin"] != "agent_inference" else "hypothesis",
            verifier_revision=item.get("verifier_revision", runner.projection.VERIFIER_VERSION), version=1,
            source_scope={"kind": "dated_historical_reports" if case.get("calendar_time_available")
                          else "historical_report_at_ordinal_cutoff",
                          "cutoff_order": case["history_cutoff_order_exclusive"] - 1, "calendar_date": None}))
    bundles = []
    ids = quote_claim_ids([s["source_id"] for s in sources], compact)  # one mapping per case
    for rank, sid in enumerate(quote_ids):
        source = by_id[sid]
        span = Span(sid, ids[sid], source["content"], 0, len(source["content"]), None, source["role"])
        claim = EvidenceClaim(ids[sid], "history_quote", source["content"], (span,),
            support_status="source_quote", priority=-rank,
            origin="assistant_proposal" if source["role"] == "assistant" else "explicit_user_statement")
        bundles.append(EvidenceBundle(ids[sid], (claim,)))
    selected_ids = list(dict.fromkeys([*quote_ids, *extra_source_ids]))
    options = {"model": runner.MODEL, "temperature": 0, "thinking": {"type": "disabled"}}
    compact_kw = {"compact_quotes": True} if compact else {}
    return api.build_cbt_context([{"role": "system", "content": api.CBT_POLICY},
                                  {"role": "user", "content": case["input"]}],
        bundles=bundles, source_registry=registry, subject_id=uid,
        max_outbound_chars=runner.SETTINGS["max_answer_chars"],
        max_response_tokens=runner.SETTINGS["answer_output_tokens"], request_metadata=options,
        representation=representation, contextual_items=contextual,
        recent_messages=[{"role": s["role"], "content": s["content"]} for s in recent],
        recent_source_ids=[s["source_id"] for s in recent], selected_source_ids=selected_ids,
        current_source_id="CURRENT", **compact_kw)


def prepare_case_requests_union(case, sources, items, dense, dense_mutex, config=None):
    """Return the three frozen arms, with the managed arm made a raw superset.

    no_rag and raw_history_rag are exactly what ``runner.prepare_case_requests``
    produces; only the managed arm is rebuilt.
    """
    config = config or UnionConfig()
    plans = runner.prepare_case_requests(case, sources, items, dense, dense_mutex)
    audit = plans["raw_history_rag"]["selection_audit"]
    raw_ids = list(audit["raw_selected_source_ids"])
    by_item = {item["item_id"]: item for item in items}
    ranked = [by_item[i] for i in audit["managed_ranked_item_ids"]]
    recent_ids = {s["source_id"] for s in sources[-runner.SETTINGS["recent_messages"]:]}
    older = sources[:-runner.SETTINGS["recent_messages"]]
    source_orders = {s["source_id"]: s["source_order"] for s in sources}
    source_roles = {s["source_id"]: s["role"] for s in sources}

    representation = "processed" if config.evidence_once else "processed_with_evidence"

    def all_quotes(quotes, chosen):
        """With evidence_once, every older evidence message is quoted whole, once."""
        if not config.evidence_once:
            return list(quotes)
        return list(dict.fromkeys([*quotes, *(e["source_id"] for it in chosen for e in it["evidence"]
                                              if e["source_id"] not in recent_ids)]))

    def build(quotes, chosen):
        extra = list(dict.fromkeys(e["source_id"] for it in chosen for e in it["evidence"]
                                   if e["source_id"] not in recent_ids))
        return assemble_union(case, sources, all_quotes(quotes, chosen), chosen, extra_source_ids=extra,
                              renderer=config.renderer, compact=config.compact_quotes,
                              representation=representation)

    def quoted(package, quote_list):
        ids = quote_claim_ids([s["source_id"] for s in sources], config.compact_quotes)
        return all(ids[s] in package.included_claim_ids for s in quote_list)

    # Layer 1: the raw selection alone must fit; never drop any of it.
    package = build(raw_ids, [])
    quotes, managed, rejections, tried = list(raw_ids), [], [], set()

    def try_item(item, layer):
        nonlocal package, managed
        tried.add(item["item_id"])
        try:
            trial = build(quotes, [*managed, item])
        except ValueError as exc:
            rejections.append({"item_id": item["item_id"], "layer": layer, "reason": type(exc).__name__})
            return
        # The builder drops an unverifiable item silently (reported in dropped_claims);
        # only an item actually rendered counts as added.
        if item["item_id"] in trial.included_claim_ids:
            managed, package = [*managed, item], trial
        else:
            rejections.append({"item_id": item["item_id"], "layer": layer,
                               "reason": trial.dropped_claims.get(item["item_id"], "not_rendered")})

    # Layer 3: pinned profile.
    pinned = profile_items(items, source_orders, config.profile_limit) if config.pin_profile else []
    for item in pinned:
        try_item(item, "profile")

    # Layer 2: fact-key channel, drawn only from the eligible older pool.
    fact_ids, fact_rejections = [], []
    if config.fact_key_sources:
        with dense_mutex:
            candidates = fact_key_candidates(case["input"], older, items, quotes,
                                             lambda q, docs: runner.rank(q, docs, dense),
                                             config.fact_key_sources)
        for sid in candidates:
            try:
                trial = build([*quotes, sid], managed)
            except ValueError as exc:
                fact_rejections.append({"source_id": sid, "reason": type(exc).__name__})
                continue
            if quoted(trial, all_quotes([*quotes, sid], managed)):
                quotes, package = [*quotes, sid], trial
                fact_ids.append(sid)
            else:
                fact_rejections.append({"source_id": sid, "reason": "not_rendered"})

    # Layer 4: question-ranked items.
    rerank_rows = None
    if config.rerank:
        ranked, rerank_rows = rerank_items(ranked, source_orders, source_roles, len(sources), config)
    for item in ranked:
        if item["item_id"] not in tried:
            try_item(item, "ranked")

    final_quotes = all_quotes(quotes, managed)
    assert quoted(package, final_quotes)
    assert all(i["item_id"] in package.included_claim_ids for i in managed)
    row = runner._package(package)
    row.update(arm="managed_memory_rag", case_id=case["case_id"], source_family=case["source_family_id"])
    row["selection_audit"] = {**audit, "condition": UNION_CONDITION, "union_version": VERSION,
        "union_config": config.as_dict(),
        "union_raw_source_ids": raw_ids, "union_item_ids": [i["item_id"] for i in managed],
        "union_item_rejections": rejections,
        "union_profile_item_ids": [i["item_id"] for i in pinned if i in managed],
        "union_fact_key_source_ids": fact_ids, "union_fact_key_rejections": fact_rejections,
        "union_rerank": rerank_rows,
        "union_item_reliability": {i["item_id"]: reliability(i, source_roles)[1] for i in managed},
        "union_evidence_quote_source_ids": [s for s in final_quotes if s not in quotes],
        "raw_selection_is_subset_of_managed": quoted(package, raw_ids) if raw_ids else True}
    plans["managed_memory_rag"] = row
    return plans
