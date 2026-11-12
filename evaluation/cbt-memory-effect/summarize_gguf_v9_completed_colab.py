"""Hosted-only post-completion review; no inference, relabeling or response repair."""
import argparse, ast, base64, csv, hashlib, importlib.util, json, os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
if not __debug__: raise RuntimeError("do_not_disable_completion_and_integrity_guards")

STUDY = "be618cc1492351111f76d6c97dd3dd6c26a138cb3b8212a994272bd778eeac94"
FREEZE = "c3638f7f9492e57d2e5179bfaa4ebd57db57152242a07315abf179b3ee54edb5"
SCORER = "b4dc236135c421e2451afd2b264adcd24c52fd16a2a305e458e3ade9107d12ed"
LAUNCH = "c38c1fd1a52fdedaee22a6680dd7799a96973f5b203bf900b48eb1346c9bad78"
ARMS, SIDES = ("raw_history_rag", "managed_memory_rag"), ("before", "after")
canonical = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest = lambda x: hashlib.sha256(canonical(x).encode()).hexdigest()
INPUTS = {}

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate_json_key")
        result[key] = value
    return result

def decode(text):
    return json.loads(text, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))

def read(path, lines=False):
    path = Path(path); INPUTS[str(path)] = sha(path)
    return [decode(x) for x in path.read_text().splitlines() if x.strip()] if lines else decode(path.read_text())

reader = SimpleNamespace(_unique=unique)

def strict_answer(text):
    """Same single-key/type parser as the frozen original primary metric."""
    value = json.loads(text, object_pairs_hook=reader._unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_answer")))
    if type(value) is not dict or set(value) != {"answer"}:
        raise ValueError("single_answer_key_required")
    answer = value["answer"]
    if answer is not None and type(answer) not in (int, str):
        raise ValueError("invalid_answer_type")
    return answer

def strict_result(text, expected):
    try:
        value = strict_answer(text)
    except (ValueError, TypeError):
        return {"correct": False, "reason": "malformed_output", "observed_answer": None}
    correct = type(value) is type(expected) and value == expected
    reason = ("correct" if correct else "unsupported_assertion" if expected is None
        else "incorrect_abstention" if value is None else "output_type_error" if type(value) is not type(expected) else "wrong_value")
    return {"correct": correct, "reason": reason, "observed_answer": value}

def checked(row):
    assert row["record_sha256"] == digest({k: v for k, v in row.items() if k != "record_sha256"}), "journal_checksum"
    return row

def tally(cells):
    known = [c for c in cells if c["gold"] is not None]; unknown = [c for c in cells if c["gold"] is None]
    asserted = [c for c in cells if c["strict"]["reason"] != "malformed_output" and c["strict"]["observed_answer"] is not None]
    return {"n": len(cells), "strict_correct": sum(c["strict"]["correct"] for c in cells),
        "known": len(known), "known_correct": sum(c["strict"]["correct"] for c in known),
        "known_correct_coverage": sum(c["strict"]["correct"] for c in known) / len(known) if known else None,
        "assertion_precision": sum(c["strict"]["correct"] for c in asserted) / len(asserted) if asserted else None,
        "known_wrong_nonnull": sum(not c["strict"]["correct"] and c["strict"]["observed_answer"] is not None for c in known),
        "omissions": sum(c["strict"]["reason"] == "incorrect_abstention" for c in cells),
        "unknown": len(unknown), "false_assertions": sum(c["strict"]["reason"] == "unsupported_assertion" for c in cells),
        "error_reasons": dict(Counter(c["strict"]["reason"] for c in cells if not c["strict"]["correct"]))}

def visibility(row, preflight, target, case, prepared_case):
    messages = row["request_payload"]["messages"]; body = decode(messages[1]["content"].split("\n", 1)[1])
    claims = body["claims"]; field = preflight["question_schema"]["field"]
    typed = [c for c in claims if c["field"] not in ("history_quote", "source_quote")]
    relevant = [c for c in typed if c["field"] == field]
    required = []; receipts = {r["message_id"]: r for r in prepared_case["parent_trace"]["source_receipts"]}
    recent = decode(messages[0]["content"].split("\nRecent original source map: ", 1)[1])
    assert len(recent) == len(messages[2:-1])
    for message_id in target["required_message_ids"]:
        original = next(h for h in case["history"] if h["message_id"] == message_id)
        receipt = receipts[message_id]; assert receipt["role"] == original["role"] and receipt["text_sha256"] == hashlib.sha256(original["content"].encode()).hexdigest()
        evidence = [e for c in claims for e in c["evidence"] if e["source_id"] == receipt["source_id"] and e["source_time"] == original["date"] and e["role"] == original["role"] and e["text"] == original["content"] and e["start"] == 0 and e["end"] == len(original["content"])]
        direct = any(meta["source_id"] == receipt["source_id"] and meta["source_time"] == original["date"] and meta["role"] == m["role"] == original["role"] and m["content"] == original["content"] for meta, m in zip(recent, messages[2:-1]))
        required.append({"message_id": message_id, "source_id": receipt["source_id"], "source_time": original["date"], "role": original["role"], "text": original["content"], "direct_recent_visible": direct, "full_source_spans_visible": evidence})
    return {"question_field": field, "answer_kind": preflight["question_schema"]["value_kind"],
        "input_type": "original_quotes_plus_structured_fields" if typed else "original_quotes_and_recent_messages",
        "direct_recent_count": len(messages[2:-1]), "structured_requested_fields": relevant,
        "gold_matching_rule_verified_field_visible": target["answer"] is not None and any(c["support_status"] == "rule_verified" and canonical(c["value"]) == canonical(target["answer"]) for c in relevant),
        "required_source_text_visibility": required, "causal_attribution": "未定"}

def summarize(root, execution, output):
    assert os.environ.get("COLAB_RELEASE_TAG") and importlib.util.find_spec("google.colab") is not None, "hosted_colab_only"
    root, execution, output = [Path(p).resolve() for p in (root, execution, output)]
    assert all(p.is_relative_to(Path("/content")) for p in (root, execution, output)) and not output.exists()
    assert sha(root / "study_protocol.json") == STUDY and sha(root / "main_freeze.json") == FREEZE
    done = read(execution / "all-phases-completed.json"); launch = read(execution / "execution_intent.json")
    assert done["before_after_logical_rows"] == 96 and done["actual_frozen_unique"] == 51 and launch["source_sha256"] == LAUNCH and launch["freeze_sha256"] == FREEZE
    phase_wall = {}
    for phase in ("before", "after", "score"):
        intent = read(execution / (phase + "-intent.json")); completion = read(execution / (phase + "-completion.json"))
        assert completion["returncode"] == 0 and completion["phase"] == phase and completion["argv"] == intent["argv"]
        phase_wall[phase] = completion["whole_process_wall_seconds"]
    assert read(root / "score_phase_timing.json")["complete"] is True
    protocol = read(root / "study_protocol.json"); plan = read(root / "main_freeze.json")
    scorer = next(Path(p) for p in protocol["code_sha256"] if Path(p).name == "gguf_state_comparison_v9_colab.py")
    assert sha(scorer) == SCORER; INPUTS[str(scorer)] = SCORER
    trees = [ast.parse(p.read_text()) for p in (Path(__file__), scorer)]
    for name in ("strict_answer", "strict_result"):
        assert len({ast.dump(next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name)) for t in trees}) == 1, "fixed_parser_changed"
    summary = read(root / "reports/summary.json"); resources = read(root / "reports/resources.json")
    assert summary["main_logical_rows"] == 96 and summary["main_unique_generations"] == summary["new_study_unique_generations"] == plan["main_unique_payloads"] == 51
    journal = [checked(r) for r in read(root / "generation_ledger.jsonl", True)]
    intents = [r for r in journal if r["event"] == "generation_intent"]; outcomes = [r for r in journal if r["event"] == "generation_outcome"]
    assert len(intents) == len(outcomes) == len({r["canonical_key"] for r in outcomes}) == 51 and all(r["status"] == "published" for r in outcomes)
    cost = resources["new_study_unique_generation_cost"]
    assert cost["unique_admitted_attempts"] == 51 and cost["status_counts"] == {"published": 51}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(r.get("usage") or {}).get(field) for r in outcomes]; observed = [v for v in values if type(v) is int and v >= 0]
        assert cost["observed_tokens"][field] == (sum(observed) if observed else None) and cost["usage_missing_attempts"][field] == 51 - len(observed)
    pairs, cells, review = {}, [], []
    for side in SIDES:
        section = plan["sections"][side]; prepared = Path(section["fresh_prepared"])
        prepared_protocol = read(prepared / "protocol.json"); assert sha(prepared / "protocol.json") == section["fresh_protocol_sha256"]
        for name, key in (("requests.jsonl", "requests_sha256"), ("assessor_only/targets.json", "targets_file_sha256")):
            assert sha(prepared / name) == section[key]
        requests = read(prepared / "requests.jsonl", True); targets = {t["case_id"]: t for t in read(prepared / "assessor_only/targets.json")}
        cases = {c["case_id"]: c for c in read(prepared / "generation_cases.json")}
        assert sha(prepared / "generation_cases.json") == section["case_file_sha256"] and len(cases) == 24
        report = read(root / "reports" / (side + "_metrics.json")); folder = root / ("answers_" + side)
        archive = read(folder / "completion_summary.json"); run = read(folder / "run_protocol.json")
        assert archive["complete"] is True and archive["logical_completed"] == 48 and archive["answers_sha256"] == report["answers_sha256"] == sha(folder / "answers_unscored.jsonl")
        assert report["main_freeze_sha256"] == FREEZE and report["scorer_code_sha256"] == SCORER and digest(run) == archive["run_protocol_sha256"]
        answers = [checked(a) for a in read(folder / "answers_unscored.jsonl", True)]; scored = report["per_question"]
        assert len(answers) == len(requests) == len(scored) == len(run["actual_native_input_preflight"]) == 48
        assert len({(r["case_id"], r["arm"]) for r in requests}) == 48
        for row, answer, score, preflight in zip(requests, answers, scored, run["actual_native_input_preflight"]):
            cid, arm = row["case_id"], row["arm"]; target = targets[cid]; question = row["request_payload"]["messages"][-1]["content"]
            assert (cid, arm) == (answer["case_id"], answer["arm"]) == (score["case_id"], score["arm"]) and answer["source_checks_passed"] is True
            assert row["request_sha256"] == answer["request_sha256"] == score["request_sha256"] == digest(row["request_payload"])
            assert hashlib.sha256(answer["response_text"].encode()).hexdigest() == answer["response_sha256"] == score["response_sha256"]
            assert base64.b64decode(answer["response_bytes_base64"], validate=True).decode() == answer["response_text"]
            complete = answer.get("finish_reason") == "eos" and answer.get("schema_validation", {}).get("accepted") is True
            strict = strict_result(answer["response_text"], target["answer"]) if complete else {"correct": False, "reason": "unfinished_or_invalid_native_output", "observed_answer": None}
            assert canonical(strict) == canonical(score["raw_strict"]) and canonical(target["answer"]) == canonical(score["expected_answer"])
            pair = pairs.setdefault(cid, {"case_id": cid, "question": question, "gold": target["answer"], "family": target["source_family_id"], "language": target["language"], "cells": {}})
            assert pair["question"] == question and canonical(pair["gold"]) == canonical(target["answer"])
            prepared_case_path = Path(row["prepared_case_path"]); assert prepared_case_path.resolve().is_relative_to(prepared)
            assert sha(prepared_case_path) == prepared_protocol["prepared_files_sha256"][str(prepared_case_path.relative_to(prepared))]
            cell = {"side": side, "arm": arm, "gold": target["answer"], "family": pair["family"], "language": pair["language"], "strict": strict,
                "gold_json_type": type(target["answer"]).__name__, "observed_json_type": type(strict["observed_answer"]).__name__,
                "response_text": answer["response_text"], "raw_native_text": answer["raw_generated_text"], "raw_generated_token_ids": answer["raw_generated_token_ids"],
                "request_sha256": row["request_sha256"], "response_sha256": answer["response_sha256"], "input_evidence": visibility(row, preflight, target, cases[cid], read(prepared_case_path))}
            pair["cells"][side + "/" + arm] = cell; cells.append(cell)
            if not strict["correct"] and type(target["answer"]) is str:
                review.append({"case_id": cid, "cell": side + "/" + arm, "question": question, "gold": target["answer"], "observed": strict["observed_answer"], "review_status": "待人工核验状态语义或标签歧义；原严格分数保留，尚不能归因为记错"})
    assert len(pairs) == 24 and len(cells) == 96 and all(len(p["cells"]) == 4 for p in pairs.values())
    assert len({p["family"] for p in pairs.values()}) == 4 and {p["language"] for p in pairs.values()} == {"zh", "en"}
    paired = {}
    for name, left, right in [(s + "_raw_to_managed", s + "/" + ARMS[0], s + "/" + ARMS[1]) for s in SIDES] + [(a + "_before_to_after", "before/" + a, "after/" + a) for a in ARMS]:
        states = [(p["cells"][left]["strict"]["correct"], p["cells"][right]["strict"]["correct"]) for p in pairs.values()]
        paired[name] = {"left": left, "right": right, "right_wins": sum(b and not a for a, b in states), "left_wins": sum(a and not b for a, b in states), "both_correct": sum(a and b for a, b in states), "both_wrong": sum(not a and not b for a, b in states), "pairs": 24}
    groups = {axis: {str(key): {side + "/" + arm: tally([c for c in cells if c[axis] == key and c["side"] == side and c["arm"] == arm]) for side in SIDES for arm in ARMS} for key in sorted({c[axis] for c in cells})} for axis in ("family", "language")}
    result = {"all_96_frozen_native_authentication_completed_by_original_scorer": True, "review_scope": "exposed_controlled_development_not_natural_CBT_or_clinical", "all_24_pairs": list(pairs.values()),
        "totals": {side + "/" + arm: tally([c for c in cells if c["side"] == side and c["arm"] == arm]) for side in SIDES for arm in ARMS},
        "groups": groups, "paired_comparisons": paired, "posthoc_status_review_not_relabeling": review, "costs_original_nonadditive": resources,
        "outer_phase_wall_seconds_nonadditive_with_original_substages": phase_wall, "input_sha256": INPUTS,
        "metric_note": "Frozen whole-JSON exact-type parser unchanged. Known coverage=correct known/known; omissions=incorrect abstention; false assertions=nonnull when gold null. Wrong known values reported separately. Input visibility is posthoc evidence, not causal proof. Precision excludes null and malformed predictions.", "summarizer_source_sha256": sha(__file__)}
    output.mkdir(mode=0o700); (output / "all_pairs_and_costs.json").write_text(canonical(result) + "\n")
    with (output / "all_24_pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["case_id", "family", "language", "question", "gold"] + [s + "/" + a for s in SIDES for a in ARMS])
        for p in pairs.values(): writer.writerow([p[k] for k in ("case_id", "family", "language", "question", "gold")] + [canonical({"native": p["cells"][s + "/" + a]["raw_native_text"], "strict": p["cells"][s + "/" + a]["strict"]}) for s in SIDES for a in ARMS])
    print(canonical({"complete": True, "pairs": 24, "logical_cells": 96, "unique_generations": 51, "totals": result["totals"], "outputs": {p.name: sha(p) for p in output.iterdir()}}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", default="/content/cbt-qwen7b-gpu-state-study-20261112-v9")
    parser.add_argument("--execution-root", default="/content/cbt-gpu-v9-main-execution-20261112-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); summarize(args.study_root, args.execution_root, args.output)
