"""Hosted-only read-only assessor of key-ablation retrieval and packing.

Verifies protocol, original source controls and reconstructed request bytes before
opening frozen targets. Reports input availability only; no generation, embedding,
Secret access or answer scoring. No private quotation is printed or exported.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_key_augmentation as keys
import audit_context_revision as assessor

base = keys.base
assembly = keys.assembly
sender = assessor.sender
VERSION = "key-augmentation-readonly-input-assessor-v1"
CONDITIONS = ("raw_original", "managed_rrf", "managed_key_augmentation")


def verify_rank(trace, pool_ids):
    """Validate declared source-level score/rank arithmetic without embeddings."""
    rows = trace["ranking"]; ids = [row["source_id"] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= pool_ids:
        raise ValueError("rank contains duplicate or unbound source")
    for row in rows:
        for field in ("score", "lexical", "dense"):
            if type(row[field]) not in {int, float} or not math.isfinite(row[field]):
                raise ValueError("nonfinite or nonnumeric ranking score")
        expected = 0.0
        for field in ("lexical_rank", "dense_rank"):
            rank = row[field]
            if rank is not None:
                if type(rank) is not int or rank < 1:
                    raise ValueError("invalid component rank")
                expected += 1 / (keys.POLICY["rrf_k"] + rank)
        if not math.isclose(row["score"], expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError("ranking RRF arithmetic differs")
    if rows != sorted(rows, key=lambda row: (-row["score"], row["source_id"])):
        raise ValueError("declared ranking order differs")
    return ids


def verify_candidate_hits(durable, product, sources, user, trace):
    pool = {source["source_id"]: source for source in sources}
    hits = trace["ranked_index_hits"]
    if len(hits) > 3 or len({hit["memory_id"] for hit in hits}) != len(hits):
        raise ValueError("candidate rank outside frozen depth")
    for hit in hits:
        state = durable.get_object_state(user, "memory:" + hit["memory_id"])
        payload = state.get("payload") or {}
        if state["status"] != "available" or payload.get("kind") != "candidate":
            raise ValueError("candidate no longer available")
        record = product.construct_memory(payload["record"], allow_legacy=False)
        ids = list(record.source_evidence.turn_ids)
        if record.user_id != user or record.lifecycle_status != "candidate" or record.sensitivity == "restricted":
            raise ValueError("candidate subject, lifecycle or sensitivity differs")
        if ids != hit["source_ids"] or any(record.permissions.get(k) is not True for k in ("save", "long_term_use", "response_external")):
            raise ValueError("candidate permission or source binding differs")
        if any(sid not in pool or product.restore_source_quote(pool[sid],
                pointer_text=record.source_evidence.supporting_text, span_id="key-audit:" + hit["memory_id"]) is None for sid in ids):
            raise ValueError("candidate locator no longer resolves original source")
    return assembly.project_candidate_sources(hits)


def assess(key_dir, proposal_dir, output, *, expected_key_protocol=None, fixture_only=False):
    if not fixture_only:
        base.require_colab()
    key_dir = Path(key_dir).resolve(strict=True); proposal_dir = Path(proposal_dir).resolve(strict=True)
    output = Path(output).absolute()
    protocol = base.read_json(key_dir / "protocol.json")
    if protocol.get("protocol_sha256") != base.digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("key protocol checksum mismatch")
    if expected_key_protocol is not None and protocol["protocol_sha256"] != expected_key_protocol:
        raise ValueError("unexpected frozen key protocol")
    if protocol.get("version") != keys.VERSION or protocol.get("fixture_only") != fixture_only or protocol.get("policy") != keys.POLICY:
        raise ValueError("key protocol identity or fixed policy differs")
    if protocol.get("actual_provider_calls") != 0 or protocol.get("gold_read") is not False:
        raise ValueError("key preparation provenance is not zero-provider/gold-free")
    sender._verify_hashes(protocol["code_sha256"])
    sender._verify_hashes(protocol["source_checkpoint_hashes"])
    proposal = base.read_json(proposal_dir / "proposal.json")
    if proposal.get("proposal_sha256") != sender.sha({k: v for k, v in proposal.items() if k != "proposal_sha256"}):
        raise ValueError("original proposal checksum mismatch")
    if proposal.get("version") != sender.VERSION or proposal.get("fixture_only") != fixture_only:
        raise ValueError("proposal version or fixture/live identity differs")
    sender._verify_hashes(proposal["code_sha256"])
    old_run = Path(proposal["old_run"]).resolve(strict=True)
    product_repo = Path(proposal["product_repo"]).resolve(strict=True)
    if output.exists() or any(output.is_relative_to(root) for root in (key_dir, proposal_dir, old_run)):
        raise ValueError("new independent audit output directory required")
    product = base.product_modules(product_repo)
    old_protocol, cases, prepared, database, checkpoint_hashes = keys.load_checkpoint(old_run, product_repo)
    if protocol["old_protocol_sha256"] != old_protocol["protocol_sha256"] or checkpoint_hashes != protocol["source_checkpoint_hashes"]:
        raise ValueError("key protocol belongs to a different original checkpoint")
    if protocol["selected_cases"] != [case["case_id"] for case in cases] or proposal["selected_cases"] != protocol["selected_cases"]:
        raise ValueError("paired case selection differs")
    traces = base.read_rows(key_dir / "traces.jsonl")
    if [row["case_id"] for row in traces] != protocol["selected_cases"] or len(traces) != 3:
        raise ValueError("exactly three matching key traces required")
    paths = {**checkpoint_hashes, **protocol["code_sha256"], **proposal["code_sha256"]}
    paths.update({str(path): base.file_hash(path) for path in (key_dir / "protocol.json", key_dir / "traces.jsonl", proposal_dir / "proposal.json")})
    validated = []
    with base.readonly_durable(database, product) as durable:
        for case, trace in zip(cases, traces):
            if trace.get("case_sha256") != case["case_sha256"] or trace.get("source_family_id") != case["source_family_id"]:
                raise ValueError("key trace case binding differs")
            query = case.get("retrieval_query", case["input"])
            if trace.get("retrieval_query_sha256") != hashlib.sha256(query.encode()).hexdigest():
                raise ValueError("trace retrieval query differs from frozen input")
            sources, _ = base.checkpoint_sources(durable, prepared, case)
            user = prepared["trace"]["user_id"]
            _, _, manifest = keys.build_keys(durable, product, sources, user)
            if manifest != protocol["key_manifest"]:
                raise ValueError("index source/candidate admission or key contents changed")
            pool_ids = {source["source_id"] for source in sources}
            raw_ids = verify_rank(trace["raw_key_ranking"], pool_ids)
            aug_ids = verify_rank(trace["augmented_key_ranking"], pool_ids)
            candidate_ids = verify_candidate_hits(durable, product, sources, user, trace["candidate_index_trace"])
            common = dict(sources=sources, user_id=user,
                cutoff_order_exclusive=case["history_cutoff_order_exclusive"], current_input=case["input"],
                current_source_id="key-dev-query:" + case["case_id"], system_prompt=sender.SYSTEM,
                recent_source_ids=[source["source_id"] for source in sources][-keys.POLICY["recent_messages"]:],
                request_metadata={"model": old_protocol["settings"]["model"], "temperature": 0,
                    "thinking": {"type": "disabled"}, "max_tokens": old_protocol["settings"]["output_tokens"]["response"]},
                max_outbound_chars=keys.POLICY["max_outbound_chars"], max_recent_messages=keys.POLICY["recent_messages"])
            first = assembly.assemble_contexts(**common, raw_source_ids=raw_ids, candidate_source_ids=candidate_ids)
            augmented = assembly.assemble_contexts(**common, raw_source_ids=aug_ids, candidate_source_ids=[])
            expected = {"raw_original": first["contexts"]["raw_history_rag"],
                "managed_rrf": first["contexts"]["managed_memory_rag"],
                "managed_key_augmentation": augmented["contexts"]["raw_history_rag"]}
            if trace["contexts"] != expected or trace["rrf_ranking"] != first["fusion"]:
                raise ValueError("wire/source bytes or rank packing differs from reconstruction")
            for context in expected.values():
                if context["request_sha256"] != sender.sha(context["request_payload"]):
                    raise ValueError("context payload checksum mismatch")
            validated.append((case, trace, {"raw_original": raw_ids, "managed_rrf": [row["source_id"] for row in first["fusion"]["source_ranks"]],
                                            "managed_key_augmentation": aug_ids}))
    # Only the assessor opens targets, after verifying every wire/source byte.
    targets_path = old_run / "assessor_only/targets.jsonl"
    target_hash = old_protocol["files_sha256"]["assessor_only/targets.jsonl"]
    if base.file_hash(targets_path) != target_hash:
        raise ValueError("old target checksum mismatch")
    paths[str(targets_path)] = target_hash
    targets = {row["case_id"]: row for row in base.read_rows(targets_path)}
    _, _, scorer_hash = assessor.frozen_scorer(product_repo, old_protocol)
    receipts = {receipt["message_id"]: receipt for receipt in prepared["trace"]["source_receipts"]}
    result_rows = []
    for case, trace, ranks in validated:
        target = targets.get(case["case_id"])
        if target is None or target.get("case_sha256") != case["case_sha256"] or target.get("source_family_id") != case["source_family_id"]:
            raise ValueError("target case or family binding differs")
        turn = next((turn for turn in case["history"] if turn["message_id"] == target["target_message_id"]), None)
        receipt = receipts.get(target["target_message_id"])
        if turn is None or receipt is None or turn["role"] != "user" or receipt["role"] != "user" or turn["content"] != target["target_quote"]:
            raise ValueError("target does not identify complete Client source")
        if not 0 <= turn["source_order"] < case["history_cutoff_order_exclusive"]:
            raise ValueError("target exceeds original cutoff")
        sid = receipt["source_id"]
        for condition in CONDITIONS:
            context = trace["contexts"][condition]
            packed, recent = assessor._packet_presence(context, target, sid)
            rank = assessor._rank(ranks[condition], sid)
            stage = ("present_in_recent" if recent else "present_in_retrieved_packet" if packed else
                     "packing_omission" if rank is not None else "target_not_retrieved")
            result_rows.append({"case_id": case["case_id"], "position": target.get("position"), "condition": condition,
                "target_source_id": sid, "target_quote_sha256": hashlib.sha256(target["target_quote"].encode()).hexdigest(),
                "target_rank": rank, "complete_quote_and_source_present": packed or recent,
                "target_in_recent": recent, "target_in_retrieved_packet": packed, "packing_stage": stage,
                "packing_drop_recorded": sid in context["dropped_sources"], "request_sha256": context["request_sha256"],
                "outbound_chars": context["outbound_chars"], "answer_accuracy": None, "accuracy": None})
    sender._verify_hashes(paths)
    by_condition = {condition: {"planned_rows": 3,
        "targets_present_in_input": sum(row["complete_quote_and_source_present"] for row in result_rows if row["condition"] == condition),
        "answer_accuracy": None} for condition in CONDITIONS}
    compact = {"version": VERSION, "development_only": True, "key_protocol_sha256": protocol["protocol_sha256"],
        "original_proposal_sha256": proposal["proposal_sha256"], "planned_rows": 9, "by_condition": by_condition,
        "cases": [{k: row[k] for k in ("position", "condition", "target_rank", "complete_quote_and_source_present", "packing_stage", "accuracy")}
                  for row in result_rows], "actual_provider_calls": 0, "new_embedding_calls": 0,
        "secret_access": False, "answer_accuracy": None, "input_presence_is_answer_accuracy": False,
        "frozen_inputs_unchanged": True}
    report = {**compact, "fixture_only": fixture_only, "role": "assessor_only",
        "original_target_file_sha256": target_hash, "original_scorer_sha256": scorer_hash,
        "verified_input_hashes": paths, "rows": result_rows,
        "limitation": "Source ranks are audited from frozen recorded embedding scores; no embedding is recomputed. Input presence is not generated-answer accuracy or a held-out effect."}
    output.mkdir(parents=True, mode=0o700)
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (output / "compact_report.json").write_text(json.dumps(compact, sort_keys=True, allow_nan=False) + "\n")
    with (output / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-dir", required=True, type=Path)
    parser.add_argument("--proposal-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-key-protocol")
    args = parser.parse_args()
    print(json.dumps(assess(args.key_dir, args.proposal_dir, args.output,
        expected_key_protocol=args.expected_key_protocol), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
