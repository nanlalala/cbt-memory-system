"""Zero-API development ablation: source keys versus source-plus-candidate keys.

Real checkpoint work requires genuine hosted Colab. This script reads one
already exposed source family's three inputs and its immutable SQLite. It never
reads assessor_only, accesses credentials, generates answers or changes old runs.
Frozen context-assembly and checkpoint-reader modules remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_context_revision_v3 as base
import context_assembly_v3 as assembly
import memoized_embedding

VERSION = "source-key-augmentation-development-v2"
KEY_SEPARATOR = "\n[Unconfirmed indexing description]\n"
POLICY = {"candidate_binding": "one_distinct_original_user_source_only",
    "candidate_order": "memory_id_ascending", "candidate_identity_deduplication": "once_per_memory_id",
    "candidate_lifecycle": "candidate_only", "candidate_sensitivity": "restricted_excluded",
    "candidate_calendar_validity": "ignored_in_ordinal_index_view_as_in_v3_no_expiry_claim", "content_deduplication": "normalized_whitespace_per_source",
    "max_candidates_per_source": 2, "max_total_candidate_characters_per_source": 800,
    "overflow": "skip_whole_candidate_never_truncate", "key_order": "full_original_first_then_index_descriptions",
    "ranker": "same_source_level_TFIDF_cosine_plus_dense_RRF_for_raw_and_augmented_keys",
    "rrf_k": 60, "minimum_relevance": 0.05, "rank_depth": "entire_eligible_source_pool_up_to_64",
    "recent_messages": 6, "max_outbound_chars": 6000,
    "wire": "original_source_text_only_with_common_recent_source_map"}


def load_checkpoint(old_run, product_repo):
    old_run = Path(old_run).resolve(strict=True)
    protocol = base.read_json(old_run / "protocol.json")
    if protocol.get("protocol_sha256") != base.digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("old protocol checksum mismatch")
    if protocol.get("version") != "cactus-source-quotation-recall-v2" or protocol.get("clock_mode") != "ordinal-only-v1":
        raise ValueError("frozen ordinal v2 checkpoint required")
    if not protocol.get("code_sha256"):
        raise ValueError("frozen product identity required")
    for relative, expected in protocol["code_sha256"].items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts or base.file_hash(Path(product_repo) / rel) != expected:
            raise ValueError("frozen product code differs")
    generation = old_run / "generation/cases.jsonl"
    if base.file_hash(generation) != protocol["files_sha256"]["generation/cases.jsonl"]:
        raise ValueError("frozen generation inputs changed")
    cases = base.read_rows(generation)
    for case in cases:
        if case.get("case_sha256") != base.digest({k: v for k, v in case.items() if k != "case_sha256"}):
            raise ValueError("frozen case checksum mismatch")
    by_id = {case["case_id"]: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate generation IDs")
    anchors = [path for path in sorted((old_run / "cases").glob("*/prepared_contexts.json"))
               if path.parent.name in by_id and (path.parent / "memory.sqlite").is_file()]
    if not anchors:
        raise ValueError("complete checkpoint source provenance and SQLite required")
    anchor = anchors[0]; original = by_id[anchor.parent.name]
    envelope = base.read_json(anchor)
    if envelope.get("protocol_sha256") != protocol["protocol_sha256"] or envelope.get("study_case_sha256") != original["case_sha256"]:
        raise ValueError("checkpoint provenance differs from frozen case")
    selected = [case for case in cases if case["source_family_id"] == original["source_family_id"]]
    if len(selected) != 3 or any(case["history"] != original["history"] or
            case["history_cutoff_order_exclusive"] != original["history_cutoff_order_exclusive"] for case in selected):
        raise ValueError("exactly three probes sharing one exposed history required")
    database = anchor.parent / "memory.sqlite"
    paths = [old_run / "protocol.json", generation, anchor, database, Path(str(database) + ".control.json")]
    paths += list((database.parent / ".cbt-memory-control").glob("*.json"))
    return protocol, selected, envelope["prepared"], database, {str(path): base.file_hash(path) for path in paths}


def build_keys(durable, product, sources, user_id):
    """Use bounded unconfirmed descriptions as indexing data, never evidence."""
    pool = {source["source_id"]: source for source in sources}
    original = {sid: source["text"] for sid, source in pool.items()}
    additions = {sid: [] for sid in pool}; seen = {sid: set() for sid in pool}
    excluded = []; accepted = []
    records = []
    for row in durable.connection.execute("SELECT object_id FROM dm_objects WHERE user_id=? ORDER BY object_id", (user_id,)):
        state = durable.get_object_state(user_id, row["object_id"])
        payload = state.get("payload") or {}
        if state["status"] == "available" and payload.get("kind") == "candidate":
            records.append(product.construct_memory(payload["record"], allow_legacy=False))
    seen_ids = set()
    for record in sorted(records, key=lambda record: record.memory_id):
        if record.memory_id in seen_ids:
            excluded.append({"memory_id": record.memory_id, "reason": "duplicate_candidate_identity"}); continue
        seen_ids.add(record.memory_id)
        reason = None
        ids = list(dict.fromkeys(record.source_evidence.turn_ids))
        if record.lifecycle_status != "candidate":
            reason = "non_candidate_lifecycle"
        elif record.sensitivity == "restricted":
            reason = "restricted_candidate"
        elif record.user_id != user_id or any(record.permissions.get(key) is not True
                for key in ("save", "long_term_use", "response_external")):
            reason = "candidate_subject_or_permission"
        elif len(ids) != 1:
            reason = "multiple_or_missing_source_binding"
        elif ids[0] not in pool:
            reason = "missing_or_withdrawn_source"
        else:
            source = pool[ids[0]]
            if source["role"] != "user" or record.source_role != "user":
                reason = "non_user_index_source"
            elif product.restore_source_quote(source, pointer_text=record.source_evidence.supporting_text,
                    span_id="key-index:" + record.memory_id) is None:
                reason = "invalid_exact_source_locator"
            elif not record.content.strip():
                reason = "empty_index_description"
        if reason:
            excluded.append({"memory_id": record.memory_id, "reason": reason}); continue
        sid = ids[0]; normalized = " ".join(record.content.split())
        if normalized in seen[sid]:
            excluded.append({"memory_id": record.memory_id, "reason": "duplicate_content_for_source"}); continue
        seen[sid].add(normalized)
        if len(additions[sid]) >= POLICY["max_candidates_per_source"] or sum(len(item["text"]) for item in additions[sid]) + len(record.content) > POLICY["max_total_candidate_characters_per_source"]:
            excluded.append({"memory_id": record.memory_id, "reason": "fixed_source_augmentation_limit"}); continue
        additions[sid].append({"memory_id": record.memory_id, "text": record.content})
        accepted.append({"memory_id": record.memory_id, "source_id": sid,
            "candidate_text_sha256": hashlib.sha256(record.content.encode()).hexdigest(),
            "candidate_characters": len(record.content), "confirmation": "unconfirmed_index_only"})
    augmented = {sid: text + "".join(KEY_SEPARATOR + item["text"] for item in additions[sid])
                 for sid, text in original.items()}
    manifest = {sid: {"original_characters": len(original[sid]), "augmented_key_characters": len(augmented[sid]),
        "added_candidate_characters": sum(len(item["text"]) for item in additions[sid]),
        "candidate_ids": [item["memory_id"] for item in additions[sid]],
        "key_sha256": hashlib.sha256(augmented[sid].encode()).hexdigest()} for sid in pool}
    return original, augmented, {"accepted": accepted, "excluded": excluded, "sources": manifest}


def rank_keys(keys, query, *, product, encoder):
    """One identical ranker for both raw and augmented source-level keys."""
    retriever = importlib.import_module(product.MemoryRetriever.__module__)
    ids = sorted(keys); documents = [keys[sid] for sid in ids]
    if not documents or not query.strip():
        return [], {"ranking": [], "key_count": len(ids)}
    lexical = retriever._lexical_scores(query, documents)
    q = encoder.encode([query], is_query=True)[0]
    vectors = encoder.encode(documents, is_query=False)
    dense = [max(0.0, retriever._cosine(q, vector)) for vector in vectors]
    def ranks(scores):
        included = [i for i, score in enumerate(scores) if score > 0 and score >= POLICY["minimum_relevance"]]
        ordered = sorted(included, key=lambda i: (-scores[i], ids[i]))
        return {index: rank for rank, index in enumerate(ordered, 1)}
    lexical_rank, dense_rank = ranks(lexical), ranks(dense)
    ranking = []
    for index in sorted(set(lexical_rank) | set(dense_rank)):
        score = sum(1 / (POLICY["rrf_k"] + ranks_[index]) for ranks_ in (lexical_rank, dense_rank) if index in ranks_)
        ranking.append({"source_id": ids[index], "score": score,
            "lexical": lexical[index], "dense": dense[index],
            "lexical_rank": lexical_rank.get(index), "dense_rank": dense_rank.get(index)})
    ranking.sort(key=lambda row: (-row["score"], row["source_id"]))
    return [row["source_id"] for row in ranking], {"ranking": ranking, "key_count": len(ids),
        "algorithm": POLICY["ranker"], "minimum_relevance": POLICY["minimum_relevance"]}


def truncation_report(encoder, keys):
    """Token-count estimates are separate from verified model execution."""
    model = getattr(encoder.encoder, "model", None)
    tokenizer = getattr(model, "tokenizer", None)
    limit = getattr(model, "max_seq_length", None)
    result = {}
    for sid, text in keys.items():
        row = {"key_characters": len(text), "encoder_max_seq_length": limit if type(limit) is int else None,
               "estimated_passage_tokens": None, "may_exceed_encoder_limit": None,
               "actual_embedding_truncation_verified": False, "status": "unknown"}
        if callable(tokenizer) and type(limit) is int and limit > 0:
            try:
                encoded = tokenizer("passage: " + text, add_special_tokens=True, truncation=False, padding=False)
                count = len(encoded["input_ids"])
                row.update(estimated_passage_tokens=count, may_exceed_encoder_limit=count > limit,
                           status="tokenizer_length_estimate_only")
            except Exception as error:
                row["tokenizer_error_type"] = type(error).__name__
        result[sid] = row
    return result


def physical_work(encoder, memoizer):
    if memoizer is not None:
        return memoizer.statistics()
    work = encoder.statistics()
    return {"physical_encode_calls_attempted": work["calls_attempted"],
        "physical_encode_calls_completed": work["calls_completed"],
        "physical_encode_texts_attempted": work["texts_attempted"],
        "physical_encode_texts_completed": work["texts_completed"],
        "physical_encode_seconds_sum": work["local_embedding_seconds_sum"],
        "cache_hit_texts": 0, "cached_entries": 0}


def compare_reference(reference_output, rows):
    """Compare all nine complete wire hashes and source orders, without gold."""
    if reference_output is None:
        return {"status": "not_checked", "all_nine_wire_hashes_and_source_orders_equal": None}
    root = Path(reference_output).resolve(strict=True)
    protocol = base.read_json(root / "protocol.json")
    if protocol.get("protocol_sha256") != base.digest({k:v for k,v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("reference protocol checksum mismatch")
    if protocol.get("version") != "source-key-augmentation-development-v1":
        raise ValueError("unchanged v1 reference output required")
    for field in ("code_sha256", "source_checkpoint_hashes"):
        hashes = protocol.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("reference code and source checkpoint hashes required")
        for path, expected in hashes.items():
            if not isinstance(path, str) or base.file_hash(path) != expected:
                raise ValueError("reference code or checkpoint dependency changed")
    prior = base.read_rows(root / "traces.jsonl")
    before = {(row["case_id"], arm): context for row in prior for arm, context in row["contexts"].items()}
    current = {(row["case_id"], arm): context for row in rows for arm, context in row["contexts"].items()}
    if len(before) != 9 or set(before) != set(current):
        raise ValueError("the same nine reference case/condition payloads are required")
    for context in [*before.values(), *current.values()]:
        payload = context["request_payload"]
        wire = assembly.canonical(payload)
        actual_hash = hashlib.sha256(wire.encode()).hexdigest()
        if (context["request_sha256"] != actual_hash or context["serialized_request"] != wire or
                context["outbound_chars"] != len(wire) or payload["messages"] != context["messages"]):
            raise ValueError("reference/current context differs from its complete canonical wire")
    comparisons = [{"case_id": key[0], "condition": key[1],
        "wire_hash_equal": current[key]["request_sha256"] == before[key]["request_sha256"],
        "recent_source_order_equal": current[key]["recent_source_ids"] == before[key]["recent_source_ids"],
        "retrieved_source_order_equal": current[key]["retrieved_source_ids"] == before[key]["retrieved_source_ids"]}
        for key in sorted(current)]
    return {"status": "checked", "reference_protocol_sha256": protocol["protocol_sha256"],
        "reference_files_sha256": {str(root/name):base.file_hash(root/name) for name in ("protocol.json", "traces.jsonl")},
        "all_nine_wire_hashes_and_source_orders_equal": all(all(row[field] for field in
            ("wire_hash_equal", "recent_source_order_equal", "retrieved_source_order_equal")) for row in comparisons),
        "comparisons": comparisons}


def prepare(old_run, output, *, product_repo, fixture_only=False, encoder_factory=None,
            memoize_embeddings=False, reference_output=None):
    if not fixture_only:
        base.require_colab()
    if encoder_factory is not None and not fixture_only:
        raise ValueError("test encoder injection is restricted to fixtures")
    if type(memoize_embeddings) is not bool:
        raise ValueError("explicit boolean memoization policy required")
    old_run = Path(old_run).resolve(strict=True); output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(old_run):
        raise ValueError("new independent development directory required")
    product = base.product_modules(product_repo)
    old_protocol, cases, prepared, database, checkpoint_hashes = load_checkpoint(old_run, product_repo)
    started = time.perf_counter()
    model = (encoder_factory() if encoder_factory else importlib.import_module(product.MemoryRetriever.__module__).MultilingualE5Encoder(
        model_name="intfloat/multilingual-e5-small"))
    memoizer = memoized_embedding.MemoizedEmbedding(model) if memoize_embeddings else None
    if memoizer is not None:
        model = memoizer
    encoder = base.TrackedEncoder(model)
    model_load_seconds = time.perf_counter() - started
    rows = []
    with base.readonly_durable(database, product) as durable:
        sources, exclusions = base.checkpoint_sources(durable, prepared, cases[0])
        if not 0 < len(sources) <= 64:
            raise ValueError("bounded eligible source pool required")
        user = prepared["trace"]["user_id"]
        original_keys, augmented_keys, key_manifest = build_keys(durable, product, sources, user)
        truncation = {"original": truncation_report(encoder, original_keys), "augmented": truncation_report(encoder, augmented_keys)}
        for case in cases:
            current_sources, current_exclusions = base.checkpoint_sources(durable, prepared, case)
            if current_sources != sources or current_exclusions != exclusions:
                raise ValueError("source control snapshot changed between paired probes")
            query = case.get("retrieval_query", case["input"])
            work_before = encoder.statistics()
            physical_before = physical_work(encoder, memoizer)
            raw_rank, raw_trace = rank_keys(original_keys, query, product=product, encoder=encoder)
            augmented_rank, augmented_trace = rank_keys(augmented_keys, query, product=product, encoder=encoder)
            candidate_rank, candidate_trace = base.candidate_ranks(durable, product, sources, query, user, encoder=encoder)
            kwargs = dict(sources=sources, user_id=user, cutoff_order_exclusive=case["history_cutoff_order_exclusive"],
                current_input=case["input"], current_source_id="key-dev-query:" + case["case_id"],
                system_prompt=("Answer from the supplied original conversation. Preserve speaker roles and "
                    "source identity. Source order is not a calendar date. Do not invent missing information."),
                recent_source_ids=[source["source_id"] for source in sources][-POLICY["recent_messages"]:],
                request_metadata={"model": old_protocol["settings"]["model"], "temperature": 0,
                    "thinking": {"type": "disabled"}, "max_tokens": old_protocol["settings"]["output_tokens"]["response"]},
                max_outbound_chars=POLICY["max_outbound_chars"], max_recent_messages=POLICY["recent_messages"])
            first = assembly.assemble_contexts(**kwargs, raw_source_ids=raw_rank, candidate_source_ids=candidate_rank)
            augmented = assembly.assemble_contexts(**kwargs, raw_source_ids=augmented_rank, candidate_source_ids=[])
            contexts = {"raw_original": first["contexts"]["raw_history_rag"],
                "managed_rrf": first["contexts"]["managed_memory_rag"],
                "managed_key_augmentation": augmented["contexts"]["raw_history_rag"]}
            rows.append({"case_id": case["case_id"], "case_sha256": case["case_sha256"],
                "source_family_id": case["source_family_id"], "development_only": True, "not_send_ready": True,
                "retrieval_query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "source_exclusions": exclusions, "eligible_source_count": len(sources),
                "raw_key_ranking": raw_trace, "augmented_key_ranking": augmented_trace,
                "candidate_index_trace": candidate_trace, "rrf_ranking": first["fusion"],
                "contexts": contexts, "local_embedding_work": {key: value - work_before[key] for key, value in encoder.statistics().items()}})
            rows[-1]["physical_embedding_work"] = {key:value-physical_before[key]
                for key,value in physical_work(encoder, memoizer).items()}
    if checkpoint_hashes != {path: base.file_hash(path) for path in checkpoint_hashes}:
        raise ValueError("original checkpoint changed")
    reference_comparison = compare_reference(reference_output, rows)
    cache_config = {"enabled": memoize_embeddings, "version": memoized_embedding.VERSION,
        "scope": "new_in_memory_cache_per_prepare_invocation_no_disk_persistence",
        "key": "pinned_model_identity_plus_is_query_plus_exact_text",
        "vectors": "original_dtype_deep_copied_on_storage_and_return",
        "cache_hits_are_physical_encode_calls": False}
    protocol = {"version": VERSION, "fixture_only": fixture_only, "development_only": True,
        "not_send_ready": True, "old_protocol_sha256": old_protocol["protocol_sha256"],
        "policy": POLICY, "source_checkpoint_hashes": checkpoint_hashes,
        "source_family_id": cases[0]["source_family_id"], "selected_cases": [case["case_id"] for case in cases],
        "encoder_model_identity": encoder.model_identity, "same_encoder_for_every_condition": True,
        "model_load_seconds": model_load_seconds, "local_embedding_work": encoder.statistics(),
        "embedding_cache": cache_config, "physical_embedding_work": physical_work(encoder, memoizer),
        "reference_comparison": reference_comparison,
        "key_manifest": key_manifest, "index_truncation": truncation,
        "actual_provider_calls": 0, "secret_access": False, "gold_read": False,
        "original_checkpoint_unchanged": True,
        "code_sha256": {str(Path(__file__).resolve()): base.file_hash(__file__),
            str(Path(assembly.__file__).resolve()): base.file_hash(assembly.__file__),
            str(Path(base.__file__).resolve()): base.file_hash(base.__file__),
            str(Path(memoized_embedding.__file__).resolve()):base.file_hash(memoized_embedding.__file__)},
        "product_code_sha256": old_protocol["code_sha256"],
        "holdout_policy": "current family and all previously inspected or tuned source families are development; all three probes stay in one family partition; freeze before evaluating new families"}
    protocol["protocol_sha256"] = base.digest(protocol)
    summary = {"version": VERSION, "fixture_only": fixture_only, "development_only": True,
        "not_send_ready": True, "prepared_cases": len(rows), "source_families": 1,
        "actual_provider_calls": 0, "secret_access": False, "gold_read": False, "effect_scores": None,
        "original_checkpoint_unchanged": True, "protocol_sha256": protocol["protocol_sha256"],
        "eligible_source_count": len(sources), "accepted_index_candidates": len(key_manifest["accepted"]),
        "excluded_index_candidates": len(key_manifest["excluded"]), "model_load_seconds": model_load_seconds,
        "local_embedding_work": encoder.statistics(),
        "embedding_cache": cache_config, "physical_embedding_work": physical_work(encoder, memoizer),
        "reference_comparison": reference_comparison,
        "by_case": [{"case_id": row["case_id"], "raw_augmented_same_rank":
            [hit["source_id"] for hit in row["raw_key_ranking"]["ranking"]] == [hit["source_id"] for hit in row["augmented_key_ranking"]["ranking"]],
            "conditions": {arm: {"outbound_chars": context["outbound_chars"],
                "retrieved_count": len(context["retrieved_source_ids"]), "dropped_count": len(context["dropped_sources"]),
                "request_sha256": context["request_sha256"]} for arm, context in row["contexts"].items()}} for row in rows]}
    output.mkdir(parents=True, mode=0o700)
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (output / "traces.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--product-repo", required=True, type=Path)
    parser.add_argument("--memoize-embeddings", action="store_true", default=False)
    parser.add_argument("--reference-output", type=Path, help="unchanged v1 output for all nine wire/source-order checks")
    args = parser.parse_args()
    print(json.dumps(prepare(args.old_run, args.output, product_repo=args.product_repo,
        memoize_embeddings=args.memoize_embeddings, reference_output=args.reference_output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
