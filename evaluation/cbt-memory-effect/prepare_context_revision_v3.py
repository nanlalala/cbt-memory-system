"""Hosted-Colab-only zero-provider diagnostic over an existing v2 checkpoint.

Reads generation inputs, prepared-context provenance and immutable SQLite only.
Never imports a scoring runner, reads assessor_only, accesses Secrets or sends
requests. Outputs a separately frozen development protocol and not-send-ready
context/rank/packing traces. The original v2 directory is not modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import context_assembly_v3 as assembly

VERSION = "checkpoint-context-v3-development-dryrun-v2"
EPOCH = datetime(2000, 1, 3, tzinfo=timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular nonsymlink checkpoint file required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    file_hash(path)
    return json.loads(Path(path).read_text())


def read_rows(path):
    file_hash(path)
    body = Path(path).read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError("incomplete JSONL checkpoint")
    return [json.loads(line) for line in body.splitlines() if line]


def require_colab():
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and
            importlib.util.find_spec("google.colab")):
        raise RuntimeError("real checkpoint diagnostic requires genuine hosted Colab")


def product_modules(product_repo):
    expected = Path(product_repo).resolve() / "modules/agent-integration/controlled_dialogue.py"
    sys.path.insert(0, str(expected.parent))
    module = importlib.import_module("controlled_dialogue")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError("foreign cached product module: use a fresh Python process")
    return module


class TrackedEncoder:
    """Measure local embedding work without recording encoded text or vectors."""
    def __init__(self, encoder):
        self.encoder = encoder
        self.model_identity = encoder.model_identity
        self.calls_attempted = 0
        self.calls_completed = 0
        self.texts_attempted = 0
        self.texts_completed = 0
        self.query_texts_attempted = 0
        self.document_texts_attempted = 0
        self.elapsed_seconds = 0.0

    def encode(self, texts, *, is_query):
        self.calls_attempted += 1
        self.texts_attempted += len(texts)
        if is_query:
            self.query_texts_attempted += len(texts)
        else:
            self.document_texts_attempted += len(texts)
        started = time.perf_counter()
        try:
            encoded = self.encoder.encode(texts, is_query=is_query)
            # Product encoders commonly return NumPy float32 arrays. Normalize
            # at this boundary so scores and all downstream JSON traces contain
            # native finite Python floats, without changing retrieval policy.
            result = [[float(component) for component in vector] for vector in encoded]
            if len(result) != len(texts) or any(not vector for vector in result):
                raise ValueError("encoder returned missing or empty embedding vectors")
            if any(not math.isfinite(component) for vector in result for component in vector):
                raise ValueError("encoder returned nonfinite embedding values")
            self.calls_completed += 1
            self.texts_completed += len(texts)
            return result
        finally:
            self.elapsed_seconds += time.perf_counter() - started

    def statistics(self):
        return {"calls_attempted": self.calls_attempted, "calls_completed": self.calls_completed,
            "texts_attempted": self.texts_attempted, "texts_completed": self.texts_completed,
            "query_texts_attempted": self.query_texts_attempted,
            "document_texts_attempted": self.document_texts_attempted,
            "local_embedding_seconds_sum": self.elapsed_seconds, "embedding_api_requests": 0}


@contextmanager
def readonly_durable(database, product):
    """Reuse v2 read/control methods without running its writing constructor.

    A read-only connection plus an explicit read transaction prevents SQLite
    mutation. The frozen binding and independent control anchor are still checked
    by v2's original _verify_control_anchor/get_source/get_object_state methods.
    """
    if Path(database).is_symlink():
        raise ValueError("symlinked checkpoint database")
    database = Path(database).resolve(strict=True)
    if Path(str(database) + "-wal").exists() or Path(str(database) + "-shm").exists():
        raise ValueError("live WAL checkpoint requires explicit immutable export first")
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("BEGIN")
    try:
        identity = connection.execute("SELECT storage_id,database_path FROM dm_storage_identity WHERE singleton=1").fetchone()
        if identity is None or identity["database_path"] != str(database):
            raise ValueError("restore the checkpoint at its bound original path; no rebinding")
        durable = object.__new__(product.DurableMemoryStore)
        durable.connection = connection
        durable.store = SimpleNamespace(_control_fault=False)
        durable._file_identity = dict(identity)
        durable._binding_path = Path(str(database) + ".control.json")
        durable._anchor_path = database.parent / ".cbt-memory-control" / (identity["storage_id"] + ".json")
        durable._verify_control_anchor()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        yield durable
        durable._verify_control_anchor()
    finally:
        connection.rollback()
        connection.close()


def checkpoint_sources(durable, prepared, case):
    trace = prepared["trace"]
    user = trace["user_id"]
    receipts = {r["message_id"]: r for r in trace["source_receipts"]}
    cutoff = case["history_cutoff_order_exclusive"]
    cutoff_stamp = (EPOCH + timedelta(seconds=cutoff)).isoformat()
    result = []; exclusions = []
    for turn in case["history"]:
        receipt = receipts.get(turn["message_id"])
        if receipt is None:
            raise ValueError("history has no original checkpoint source binding")
        sid = receipt["source_id"]
        if (receipt["role"] != turn["role"] or receipt["text_sha256"] !=
                hashlib.sha256(turn["content"].encode()).hexdigest()):
            raise ValueError("history differs from prepared source provenance")
        actual = durable.get_source(user, sid, as_of=cutoff_stamp)
        partial_withdrawal = durable.connection.execute("SELECT 1 FROM dm_revocations WHERE user_id=? AND target_kind='source' AND target_id=? AND scope IN ('span','span_stop_use') LIMIT 1", (user, sid)).fetchone()
        if actual is None or partial_withdrawal or actual["permissions"]["response_external"] is not True:
            exclusions.append({"source_id": sid, "reason": "current_source_control_or_permission"})
            continue
        if actual["text"] != turn["content"] or actual["role"] != turn["role"]:
            raise ValueError("stored source changed since frozen generation input")
        order = turn["source_order"]
        if type(order) is not int or not 0 <= order < cutoff:
            raise ValueError("history exceeds source-order cutoff")
        expected_stamp = (EPOCH + timedelta(seconds=order)).isoformat()
        if actual["source_time"] != expected_stamp:
            raise ValueError("surrogate source-order encoding differs")
        result.append({**actual, "source_order": order})
    return result, exclusions


def candidate_ranks(durable, product, sources, query, user_id, *, encoder=None):
    pool = {source["source_id"]: source for source in sources}
    records = []; binding = {}; excluded = []
    for row in durable.connection.execute("SELECT object_id FROM dm_objects WHERE user_id=? ORDER BY object_id", (user_id,)):
        state = durable.get_object_state(user_id, row["object_id"])
        payload = state.get("payload") or {}
        if state["status"] != "available" or payload.get("kind") != "candidate":
            continue
        record = product.construct_memory(payload["record"], allow_legacy=False)
        permissions = record.permissions
        if record.user_id != user_id or any(permissions.get(key) is not True for key in ("save", "long_term_use", "response_external")):
            excluded.append({"memory_id": record.memory_id, "reason": "candidate_permission_or_subject"})
            continue
        ids = list(record.source_evidence.turn_ids)
        if not ids or any(sid not in pool or product.restore_source_quote(pool[sid],
                pointer_text=record.source_evidence.supporting_text, span_id="v3-index:" + record.memory_id + ":" + sid) is None for sid in ids):
            excluded.append({"memory_id": record.memory_id, "reason": "missing_or_withdrawn_complete_source_binding"})
            continue
        records.append(replace(record, valid_from=None, valid_to=None))
        binding[record.memory_id] = ids
    hits = product.MemoryRetriever(records, encoder=encoder, policy=product.RetrievalPolicy(
        allowed_lifecycle=frozenset({"candidate"}), recency_weight=0,
        importance_weight=0, confidence_weight=0)).search(query, user_id=user_id) if records else []
    ranked = [{"memory_id": hit.record.memory_id, "source_ids": binding[hit.record.memory_id],
               "score": hit.score, "components": hit.components} for hit in hits]
    return assembly.project_candidate_sources(ranked), {"method": ("candidate_summary_TFIDF_plus_pinned_E5" if encoder is not None else "candidate_summary_TFIDF_no_dense_encoder"),
        "eligible_candidate_count": len(records), "ranked_index_hits": ranked, "excluded": excluded}


def prepare(old_run, output, *, product_repo, max_outbound_chars=6000, recent_messages=6,
            fixture_only=False, encoder_name=None, encoder_factory=None):
    if not fixture_only:
        require_colab()
    if encoder_name not in {None, "multilingual-e5-small"}:
        raise ValueError("unknown encoder; no silent retrieval fallback")
    if encoder_factory is not None and not fixture_only:
        raise ValueError("injected encoders are restricted to tiny fixture tests")
    old_run = Path(old_run).resolve(strict=True); output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(old_run):
        raise ValueError("new independent development output directory required")
    product = product_modules(product_repo)
    protocol_path = old_run / "protocol.json"
    old_protocol = read_json(protocol_path)
    if old_protocol.get("protocol_sha256") != digest({k: v for k, v in old_protocol.items() if k != "protocol_sha256"}):
        raise ValueError("v2 protocol checksum mismatch")
    if old_protocol.get("version") != "cactus-source-quotation-recall-v2" or old_protocol.get("clock_mode") != "ordinal-only-v1":
        raise ValueError("explicit frozen ordinal v2 checkpoint required")
    product_hashes = old_protocol.get("code_sha256", {})
    if not product_hashes:
        raise ValueError("frozen product code hashes required")
    for relative, expected in product_hashes.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or file_hash(Path(product_repo) / rel) != expected:
            raise ValueError("use the unchanged frozen v2 product checkout")
    generation = old_run / "generation/cases.jsonl"
    if file_hash(generation) != old_protocol["files_sha256"]["generation/cases.jsonl"]:
        raise ValueError("frozen generation inputs changed")
    cases = read_rows(generation)
    for case in cases:
        if case.get("case_sha256") != digest({k: v for k, v in case.items() if k != "case_sha256"}):
            raise ValueError("generation case checksum mismatch")
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate frozen generation case IDs")
    anchors = sorted((old_run / "cases").glob("*/prepared_contexts.json"))
    anchors = [path for path in anchors if path.parent.name in by_id and (path.parent / "memory.sqlite").is_file()]
    if not anchors:
        raise ValueError("checkpoint contains no complete SQLite and prepared source provenance")
    anchor = anchors[0]; anchor_case = by_id[anchor.parent.name]
    prepared_envelope = read_json(anchor)
    if (prepared_envelope.get("protocol_sha256") != old_protocol["protocol_sha256"] or
            prepared_envelope.get("study_case_sha256") != anchor_case["case_sha256"]):
        raise ValueError("prepared provenance differs from old frozen case")
    prepared = prepared_envelope["prepared"]
    selected = [case for case in cases if case["source_family_id"] == anchor_case["source_family_id"]]
    if len(selected) != 3 or any(case["history"] != anchor_case["history"] or
            case["history_cutoff_order_exclusive"] != anchor_case["history_cutoff_order_exclusive"] for case in selected):
        raise ValueError("expected three probes sharing exactly one frozen history")
    database = anchor.parent / "memory.sqlite"
    input_paths = [protocol_path, generation, anchor, database, Path(str(database) + ".control.json")]
    input_paths += list((database.parent / ".cbt-memory-control").glob("*.json"))
    before = {str(path): file_hash(path) for path in input_paths}
    encoder = None
    encoder_load_seconds = None
    if encoder_name is not None:
        started = time.perf_counter()
        # The product adapter verifies the pinned snapshot bytes and loads with
        # trust_remote_code=False. A failure propagates; never fall back silently.
        model = (encoder_factory() if encoder_factory is not None else
                 importlib.import_module(product.MemoryRetriever.__module__).MultilingualE5Encoder(
                     model_name="intfloat/multilingual-e5-small"))
        if not getattr(model, "model_identity", None):
            raise ValueError("pinned embedding model identity required")
        encoder = TrackedEncoder(model)
        encoder_load_seconds = time.perf_counter() - started
    rows = []
    with readonly_durable(database, product) as durable:
        for case in selected:
            sources, exclusions = checkpoint_sources(durable, prepared, case)
            user = prepared["trace"]["user_id"]
            cutoff_stamp = (EPOCH + timedelta(seconds=case["history_cutoff_order_exclusive"])).isoformat()
            query = case.get("retrieval_query", case["input"])
            embedding_before = encoder.statistics() if encoder is not None else None
            if len(sources) > 64:
                raise ValueError("checkpoint source pool exceeds bounded 64-turn protocol")
            # Both channels may rank the entire same eligible original pool.
            # The exact complete-request budget, not a smaller raw top-k quota,
            # decides which quotations fit in each arm. Candidate summaries add
            # a ranking signal; they do not grant managed extra source capacity.
            raw_depth = max(1, len(sources))
            raw = product.StableSourceRetriever(sources, encoder=encoder).search(query, user_id=user,
                source_cutoff=cutoff_stamp, as_of=None, for_external_response=True,
                top_k=raw_depth, lexical_k=raw_depth, dense_k=raw_depth)
            raw_ids = list(dict.fromkeys(hit.source_id for hit in raw.hits))
            candidate_ids, candidate_trace = candidate_ranks(durable, product, sources, query, user, encoder=encoder)
            recent_ids = [source["source_id"] for source in sources][-recent_messages:] if recent_messages else []
            result = assembly.assemble_contexts(sources=sources, user_id=user,
                cutoff_order_exclusive=case["history_cutoff_order_exclusive"], current_input=case["input"],
                current_source_id="v3-query:" + case["case_id"],
                system_prompt=("Answer from the supplied original conversation. Preserve speaker roles and "
                    "source identity. Source order is not a calendar date. Do not invent missing information."),
                recent_source_ids=recent_ids, raw_source_ids=raw_ids, candidate_source_ids=candidate_ids,
                request_metadata={"model": old_protocol["settings"]["model"], "temperature": 0,
                    "thinking": {"type": "disabled"}, "max_tokens": old_protocol["settings"]["output_tokens"]["response"]},
                max_outbound_chars=max_outbound_chars, max_recent_messages=recent_messages)
            rows.append({"case_id": case["case_id"], "source_family_id": case["source_family_id"],
                "case_sha256": case["case_sha256"], "not_send_ready": True,
                "retrieval_query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "source_exclusions": exclusions, "eligible_source_count": len(sources),
                "raw_candidate_depth": len(sources), "raw_retrieval_trace": raw.trace,
                "candidate_index_trace": candidate_trace,
                "local_embedding_work": ({key: value - embedding_before[key] for key, value in encoder.statistics().items()}
                    if encoder is not None else None), "assembly": result})
    if before != {str(path): file_hash(path) for path in input_paths}:
        raise ValueError("original checkpoint changed during diagnostic")
    development = {"version": VERSION, "fixture_only": fixture_only, "not_send_ready": True,
        "old_protocol_sha256": old_protocol["protocol_sha256"], "source_family_id": anchor_case["source_family_id"],
        "selected_cases": [case["case_id"] for case in selected], "source_checkpoint_hashes": before,
        "max_outbound_chars": max_outbound_chars, "recent_messages": recent_messages,
        "retrieval": {"raw": ("StableSourceRetriever_TFIDF_plus_pinned_E5_eligible_pool" if encoder is not None else "StableSourceRetriever_TFIDF_eligible_pool"),
            "raw_candidate_depth": "eligible_source_pool_size_up_to_64",
            "raw_lexical_k_and_dense_k": "eligible_source_pool_size_up_to_64",
            "quotation_capacity": "all arms fill the same complete-request character budget; no arm-specific quote quota",
            "candidate": ("MemoryRetriever_TFIDF_plus_pinned_E5_top3_minimum0.12" if encoder is not None else "MemoryRetriever_TFIDF_top3_minimum0.12"),
            "encoder": encoder_name, "encoder_model_identity": encoder.model_identity if encoder is not None else None,
            "same_encoder_instance_for_both_channels": encoder is not None, "fusion": "equal_weight_RRF_k60"},
        "embedding_load_seconds": encoder_load_seconds,
        "local_embedding_work": encoder.statistics() if encoder is not None else None,
        "actual_model_calls": 0, "actual_provider_calls": 0,
        "local_embedding_model_used": encoder is not None, "secret_access": False, "gold_read": False,
        "product_code_sha256": product_hashes,
        "code_sha256": {str(Path(__file__).resolve()): file_hash(__file__),
            str(Path(assembly.__file__).resolve()): file_hash(assembly.__file__),
            str(Path(product.__file__).resolve()): file_hash(product.__file__)},
        "remaining_requirements": ["fresh evaluation families and approved new frozen protocol",
            "durable send/publication integration", "real model evaluation before any effect claim"]}
    development["protocol_sha256"] = digest(development)
    output.mkdir(parents=True, mode=0o700)
    (output / "protocol.json").write_text(json.dumps(development, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    (output / "context_rank_pack_traces.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    summary = {"version": VERSION, "fixture_only": fixture_only, "not_send_ready": True,
        "prepared_cases": len(rows), "actual_model_calls": 0, "actual_provider_calls": 0,
        "local_embedding_model_used": encoder is not None, "secret_access": False, "gold_read": False,
        "encoder": encoder_name, "encoder_model_identity": encoder.model_identity if encoder is not None else None,
        "embedding_load_seconds": encoder_load_seconds,
        "local_embedding_work": encoder.statistics() if encoder is not None else None,
        "original_checkpoint_unchanged": True, "protocol_sha256": development["protocol_sha256"],
        "by_case": [{"case_id": row["case_id"], "raw_managed_same_request":
            row["assembly"]["contexts"]["raw_history_rag"]["request_sha256"] == row["assembly"]["contexts"]["managed_memory_rag"]["request_sha256"],
            "eligible_source_count": row["eligible_source_count"],
            "raw_candidate_depth": row["raw_candidate_depth"],
            "arm_context_sizes": {arm: context["outbound_chars"] for arm, context in row["assembly"]["contexts"].items()},
            "arm_source_selection": {arm: {"recent_count": len(context["recent_source_ids"]),
                "retrieved_count": len(context["retrieved_source_ids"]),
                "dropped_count": len(context["dropped_sources"])}
                for arm, context in row["assembly"]["contexts"].items()}}
                    for row in rows], "effect_scores": None}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--product-repo", required=True, type=Path)
    parser.add_argument("--encoder", choices=["multilingual-e5-small"], default=None,
                        help="optional pinned local E5 encoder, shared by both retrieval channels; no fallback")
    args = parser.parse_args()
    print(json.dumps(prepare(args.old_run, args.output, product_repo=args.product_repo,
                             encoder_name=args.encoder), sort_keys=True))


if __name__ == "__main__":
    main()
