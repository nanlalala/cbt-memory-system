"""Frozen CBT knowledge packets for a union run (the textbook RAG channel).

The knowledge channel is ``modules/agent-integration/knowledge_adapter_v2.py``:
it ranks chunks of reviewed CBT sources (WHO mhGAP, VA CBT manuals, Beck/Tolin
sample chapters, ...) and returns evidence cards only for sources whose
retrieval *and* external-model use are approved in ``source_registry.json``.

``build_packets`` asks the adapter once per case, with the case's current user
message as the query, and freezes the answer. A run prepared with these packets
gives the same cards to no-RAG, raw-history RAG and every managed config, so the
compared arms still differ only in memory. Nothing here approves a source: with
the current registry (all sources disabled) every packet is empty and the run
is refused, rather than silently reporting textbook grounding.
"""
from __future__ import annotations

import json
from pathlib import Path

from architecture_retest.journal import atomic, digest

PACKET_VERSION = "cbt-union-knowledge-packets-v1"
CARD_FIELDS = ("source_id", "chunk_id", "citation", "page_start", "page_end", "text",
               "chunk_sha256", "source_sha256", "evidence_status")
MAX_CARD_CHARS = 1200


def card_view(item):
    """The part of an adapter evidence card that is sent to the model."""
    card = {k: item[k] for k in CARD_FIELDS}
    if type(card["text"]) is not str or not card["text"].strip():
        raise ValueError("knowledge card text required")
    card["text"] = card["text"][:MAX_CARD_CHARS]
    return card


def build_packets(cases, adapter, output, *, task_kind="general"):
    """Query the knowledge adapter once per case and freeze the result."""
    if getattr(adapter, "mode", None) != "K1":
        raise ValueError("knowledge packets need an adapter in mode K1")
    output = Path(output)
    if output.exists():
        raise ValueError("knowledge packet file already exists; use a new path")
    packets = {}
    for case in cases:
        selection = adapter.retrieve(case["input"], task_kind=task_kind, external=True)
        packets[case["case_id"]] = {"status": selection.status, "reason": selection.reason,
                                    "items": [card_view(i) for i in selection.items]}
    statuses = sorted({p["status"] for p in packets.values()})
    body = {"version": PACKET_VERSION, "policy_hash": adapter.policy_hash,
            "corpus_fingerprints": sorted({s for s in [getattr(adapter.retriever, "corpus_fingerprint", None)] if s}),
            "fixture_only": bool(getattr(adapter, "fixture_only", False)),
            "task_kind": task_kind, "status_counts": {s: sum(p["status"] == s for p in packets.values())
                                                      for s in statuses},
            "packets": packets}
    body["packets_sha256"] = digest(packets)
    atomic(output, body)
    return {k: v for k, v in body.items() if k != "packets"}


def load_packets(path, case_ids, *, fixture_only):
    """Validated packets restricted to these cases; refuses an all-empty channel."""
    body = json.loads(Path(path).read_text())
    if body.get("version") != PACKET_VERSION or body.get("packets_sha256") != digest(body.get("packets")):
        raise ValueError("knowledge packets are not an intact frozen packet file")
    if body["fixture_only"] != fixture_only:
        raise ValueError("fixture knowledge packets cannot be used for a real run, or the reverse")
    if not set(case_ids) <= set(body["packets"]):
        raise ValueError("knowledge packets must cover every case of the run")
    body["packets"] = {c: body["packets"][c] for c in sorted(case_ids)}
    body["packets_sha256"] = digest(body["packets"])
    statuses = sorted({p["status"] for p in body["packets"].values()})
    body["status_counts"] = {s: sum(p["status"] == s for p in body["packets"].values()) for s in statuses}
    for packet in body["packets"].values():
        for card in packet["items"]:
            if card != card_view(card):
                raise ValueError("knowledge card fields differ from the frozen card view")
    if not any(p["items"] for p in body["packets"].values()):
        raise ValueError("no case received any knowledge card; approve and pin a source first")
    return body
