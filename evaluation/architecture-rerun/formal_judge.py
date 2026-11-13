"""Independent reference grading for the full three-arm readout replications.

The pinned earlier LongMemEval templates are reused, including their declared
transfer to LoCoMo. This is adapted model-assisted grading, not an official
benchmark model score, human assessment or clinical outcome. No import accesses
a credential, downloads data or calls a model.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
import reference_judge as legacy
from architecture_retest.journal import (Journal, CallLimit, UnknownAttempt,
    InvalidReceipt, atomic, digest, lock, rows, utc)
from architecture_retest.transport import require_hosted

VERSION = "full-readout-independent-reference-judge-v1"
ARMS = ("no_rag", "raw_history_rag", "managed_memory_rag")
COUNTS = {"longmemeval_s_cleaned": 500, "locomo10": 1540}
SEED, RESAMPLES = 20261021, 10000
BOUNDARY = "adapted reference-model grading; not official benchmark, human or clinical evaluation"


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    paths = [Path(__file__), Path(__file__).with_name("formal_continuation.py"),
        ROOT / "evaluation/end-to-end-dialogue/reference_judge.py",
        legacy.SNAPSHOT, ROOT / "evaluation/cbt-memory-effect/architecture_retest/journal.py",
        ROOT / "evaluation/cbt-memory-effect/architecture_retest/transport.py",
        ROOT / "evaluation/cbt-memory-effect/auto_eval_v2/run_colab.py",
        ROOT / "modules/agent-integration/cbt_counselor_policy.py",
        ROOT / "modules/agent-integration/cbt_runtime_guard.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def _json(path):
    from auto_eval_v2.run_colab import strict_json
    if Path(path).is_symlink():
        raise ValueError("symlinked frozen artifact")
    return strict_json(Path(path).read_text())


def _guard(folder, fixture_only):
    if not fixture_only:
        require_hosted()
        if not Path(folder).resolve().is_relative_to("/content"):
            raise ValueError("full grading is restricted to hosted /content")


def _identity(identity):
    if (type(identity) is not dict or set(identity) != {"provider", "model", "returned_model", "model_family"}
            or identity["provider"] != "https://tokendance.space/gateway/v1"
            or any(type(v) is not str or not v.strip() or v != v.strip() for v in identity.values())
            or identity["model_family"].casefold() == "deepseek"
            or "deepseek" in identity["model"].casefold()
            or "deepseek" in identity["returned_model"].casefold()):
        raise ValueError("explicit independent judge identity required")


def _reader(run_dir, fixture_only):
    run_dir = Path(run_dir).resolve()
    _guard(run_dir, fixture_only)
    reader = _json(run_dir / "protocol.json")
    # A continuation has a separate send journal and immutable original
    # receipts. Its adapter verifies both provenances rather than rewriting an
    # old receipt's protocol hash to make it look like a new observation.
    import formal_continuation
    if reader.get("version") == formal_continuation.VERSION:
        return formal_continuation.grading_reader(run_dir, fixture_only=fixture_only)
    cases, references = _json(run_dir / "cases.json"), _json(run_dir / "references.json")
    if reader.get("protocol_sha256") != digest({k:v for k,v in reader.items() if k != "protocol_sha256"}):
        raise ValueError("reader protocol checksum differs")
    if (reader.get("dataset_key") not in COUNTS or reader.get("arms") != list(ARMS)
            or reader.get("case_count") != len(cases)
            or len({c["case_id"] for c in cases}) != len(cases)
            or {r["case_id"] for r in references} != {c["case_id"] for c in cases}
            or len(references) != len(cases)):
        raise ValueError("full reader roster or three-arm grid differs")
    if fixture_only:
        if len(cases) > 10 or sum(p.stat().st_size for p in run_dir.rglob("*.json")) > 1_000_000:
            raise ValueError("tiny software fixture limit exceeded")
    elif len(cases) != COUNTS[reader["dataset_key"]]:
        raise ValueError("the complete original source cohort is required")
    if "cases_sha256" in reader and reader["cases_sha256"] != digest(cases):
        raise ValueError("reader case roster checksum differs")
    expected_model = reader.get("provider", {}).get("returned_model")
    receipt_rows, intent_rows = rows(run_dir / "outcomes.jsonl"), rows(run_dir / "intents.jsonl")
    receipts, intents = {r["key"]:r for r in receipt_rows}, {r["key"]:r for r in intent_rows}
    if len(receipts) != len(receipt_rows) or len(intents) != len(intent_rows):
        raise ValueError("duplicate reader receipt or intent")
    snapshot = {name:file_hash(run_dir / name) for name in ("protocol.json", "cases.json", "references.json")}
    for name in ("intents.jsonl", "outcomes.jsonl"):
        if (run_dir/name).exists():
            snapshot[name] = file_hash(run_dir/name)
    answers, zeros = [], []
    refs = {r["case_id"]:r for r in references}
    for case in cases:
        qid = case["case_id"]
        dest = run_dir / "cases" / digest(qid)[:32]
        ref = refs[qid]
        for arm in ARMS:
            path, failed = dest / (arm + ".answer.json"), dest / (arm + ".failure.json")
            if path.exists() and failed.exists():
                raise ValueError("reader answer and terminal failure both present")
            common = {"case_id":qid, "arm":arm, "source_family_id":case["source_family_id"],
                      "category":ref["category"], "should_abstain":ref["should_abstain"]}
            if path.exists():
                answer = _json(path)
                if (answer.get("case_id") != qid or answer.get("arm") != arm
                        or answer.get("status") != "received" or type(answer.get("response")) is not str
                        or not answer["response"].strip() or type(answer.get("request_key")) is not str
                        or (expected_model is not None and answer.get("provider_model") != expected_model)):
                    raise ValueError("reader answer identity or completion differs")
                receipt = receipts.get(answer["request_key"])
                intent = intents.get(answer["request_key"])
                if not fixture_only and (receipt is None or receipt.get("status") != "received"
                        or receipt.get("text") != answer["response"]
                        or receipt.get("provider_model") != answer["provider_model"]
                        or receipt.get("finish_reason") != "stop" or intent is None
                        or intent.get("purpose") != "answer"
                        or receipt.get("protocol_sha256") != reader["protocol_sha256"]
                        or intent.get("protocol_sha256") != reader["protocol_sha256"]
                        or intent.get("wire_sha256") != receipt.get("wire_sha256")
                        or digest({"purpose":intent["purpose"], "payload":intent["payload"]}) != answer["request_key"]):
                    raise ValueError("reader answer has no matching complete provider receipt")
                answers.append((common, ref, answer))
                snapshot[str(path.relative_to(run_dir))] = file_hash(path)
            elif failed.exists():
                failure = _json(failed)
                if (failure.get("case_id") != qid or failure.get("arm") != arm
                        or failure.get("status") not in {"preparation_failed", "terminal_failure"}):
                    raise ValueError("reader terminal failure schema differs")
                zeros.append({**common, "score":0, "technical_failure":True,
                              "failure_stage":"reader", "failure_status":failure["status"]})
                snapshot[str(failed.relative_to(run_dir))] = file_hash(failed)
            else:
                raise ValueError("reader is incomplete; every planned arm needs an answer or explicit terminal failure")
    return reader, cases, answers, zeros, snapshot


def prepare(run_dir, output, identity, *, fixture_only=False):
    """Freeze every anonymous grading packet before any credential or send."""
    _identity(identity)
    _guard(output, fixture_only)
    reader, cases, answers, zeros, snapshot = _reader(run_dir, fixture_only)
    requests, mapping = [], []
    for common, reference, answer in answers:
        opaque = "grade-" + digest([VERSION, common["case_id"], common["arm"]])[:32]
        messages = [{"role":"system", "content":legacy.JUDGE_GUARD},
                    {"role":"user", "content":legacy.reference_prompt(reference, answer["response"], reader["dataset_key"])}]
        payload = {"model":identity["model"], "messages":messages, "temperature":0,
                   "max_tokens":64, "thinking":{"type":"disabled"}}
        if len(json.dumps(payload, allow_nan=False)) > 24000:
            raise ValueError("full grading request exceeds the unchanged complete request cap")
        requests.append({"case_id":opaque, "purpose":"reference_grade", "payload":payload, "repeat":False})
        mapping.append({**common, "item_id":opaque, "reader_response_sha256":digest(answer)})
    # Fixed, outcome-blind 10% repeat sample: purpose differs so the exact-wire
    # journal does not silently deduplicate the repeated model observation.
    rng = random.Random(SEED)
    selected = rng.sample(requests, k=(len(requests)+9)//10) if requests else []
    requests += [{**r, "case_id":r["case_id"]+"-repeat", "base_item_id":r["case_id"],
                  "purpose":"reference_grade_repeat", "repeat":True} for r in selected]
    rng.shuffle(requests)
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    atomic(folder / "requests.json", requests)
    atomic(folder / "controller_mapping.json", mapping)
    atomic(folder / "reader_zeros.json", zeros)
    protocol = {"version":VERSION, "fixture_only":fixture_only, "reader_run":str(Path(run_dir).resolve()),
        "dataset_key":reader["dataset_key"], "source_case_count":len(cases), "arms":list(ARMS),
        "planned_answers":len(cases)*len(ARMS), "primary_grading_rows":len(answers),
        "repeat_rows":len(selected), "provider":dict(identity), "maximum_attempts":len(requests),
        "request_limits":{"answer":24000, "preparation":24000}, "source_manifest_name":"requests.json",
        "cases_sha256":digest(requests), "reader_snapshot_sha256":snapshot,
        "files_sha256":{name:file_hash(folder/name) for name in ("controller_mapping.json", "reader_zeros.json")},
        "code_sha256":code_hashes(), "created_at_utc":utc(), "boundary":BOUNDARY,
        "upstream_prompt_snapshot_sha256":legacy.SNAPSHOT_SHA256,
        "blindness":"judge receives only the pinned rubric, question, reference answer and anonymous reader answer",
        "scoring":"strict pinned yes/no parser; unusable labels retained as technical-failure zeros",
        "retry_policy":"no automatic retry; unknown sends block continuation",
        "primary_contrast":["managed_memory_rag", "raw_history_rag"],
        "auxiliary_contrast":["managed_memory_rag", "no_rag"],
        "statistics":{"aggregation":"equal-weight source-family means", "seed":SEED,
                      "bootstrap_resamples":RESAMPLES, "confidence_level":0.95},
        "authorization_reference":"user_requested_full_architecture_rerun"}
    protocol["protocol_sha256"] = digest(protocol)
    atomic(folder / "protocol.json", protocol)
    return protocol


def load(folder, fixture_only=False):
    folder = Path(folder).resolve()
    _guard(folder, fixture_only)
    protocol = _json(folder / "protocol.json")
    requests = _json(folder / "requests.json")
    if (protocol.get("version") != VERSION or protocol.get("fixture_only") != fixture_only
            or protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"})
            or protocol.get("cases_sha256") != digest(requests)
            or protocol.get("code_sha256") != code_hashes()):
        raise ValueError("frozen judge code, mode, protocol or requests differ")
    _identity(protocol["provider"])
    for name, expected in protocol["files_sha256"].items():
        if file_hash(folder/name) != expected:
            raise ValueError("frozen judge mapping differs")
    for name, expected in protocol["reader_snapshot_sha256"].items():
        if file_hash(Path(protocol["reader_run"])/name) != expected:
            raise ValueError("reader changed after grading freeze")
    if fixture_only and protocol["source_case_count"] > 10:
        raise ValueError("tiny software fixture limit exceeded")
    return folder, protocol, requests


def _grade(row, folder, journal):
    path = folder / "scores" / (row["case_id"]+".json")
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        value = _json(path)
        if value.get("request_key") != digest({"purpose":row["purpose"], "payload":row["payload"]}):
            raise ValueError("retained grade belongs to a different request")
        return
    receipt = journal.call(row, row["purpose"], row["payload"])
    try:
        label = legacy.parse_label(receipt["text"])
        failure, score = False, int(label)
    except ValueError:
        failure, score = True, 0
    atomic(path, {"item_id":row["case_id"], "request_key":receipt["key"], "score":score,
        "technical_failure":failure, "failure_stage":"judge_label" if failure else None,
        "judge_text":receipt["text"], "provider_model":receipt["provider_model"],
        "repeat":row["repeat"], "base_item_id":row.get("base_item_id"),
        "reason_available":False})


def run(folder, transport, *, max_new_calls, workers=4, fixture_only=False):
    folder, protocol, requests = load(folder, fixture_only)
    if (type(max_new_calls) is not int or max_new_calls < 0 or type(workers) is not int or not 1 <= workers <= 4):
        raise ValueError("finite invocation call cap and at most four workers required")
    if fixture_only and getattr(transport, "fixture_only", False) is not True:
        raise ValueError("software fixtures cannot invoke a real provider")
    with lock(folder):
        journal = Journal(folder, protocol, transport, max_new_calls)
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_grade, r, folder, journal):r["case_id"] for r in requests}
            for future in as_completed(futures):
                try:
                    future.result()
                except (CallLimit, UnknownAttempt, InvalidReceipt) as exc:
                    journal.stop.set()
                    errors.append({"item_id":futures[future], "error_type":type(exc).__name__})
                except Exception as exc:
                    journal.stop.set()
                    errors.append({"item_id":futures[future], "error_type":type(exc).__name__})
        atomic(folder / "last_invocation.json", {"new_calls":journal.new_calls, "errors":errors, "at_utc":utc()})
        result = summarize(folder, fixture_only=fixture_only)
        atomic(folder / "summary.json", result)
        return result


def summarize(folder, *, fixture_only=False):
    folder, protocol, requests = load(folder, fixture_only)
    mapping = _json(folder / "controller_mapping.json")
    zero_rows = _json(folder / "reader_zeros.json")
    grades = {r["item_id"]:r for p in (folder / "scores").glob("*.json") for r in [_json(p)]}
    if set(grades) - {r["case_id"] for r in requests}:
        raise ValueError("foreign grade output")
    outcomes = rows(folder / "outcomes.jsonl")
    by_receipt = {r["key"]:r for r in outcomes}
    requests_by_id = {r["case_id"]:r for r in requests}
    for item_id, grade in grades.items():
        request = requests_by_id[item_id]
        key = digest({"purpose":request["purpose"], "payload":request["payload"]})
        receipt = by_receipt.get(key)
        if (grade.get("request_key") != key or receipt is None or receipt.get("status") != "received"
                or grade.get("judge_text") != receipt.get("text")
                or grade.get("provider_model") != protocol["provider"]["returned_model"]
                or receipt.get("provider_model") != grade.get("provider_model")
                or receipt.get("finish_reason") != "stop"
                or grade.get("repeat") != request["repeat"]
                or grade.get("base_item_id") != request.get("base_item_id")):
            raise ValueError("grade does not bind its frozen request and complete provider receipt")
        try:
            expected_score, expected_failure = int(legacy.parse_label(receipt["text"])), False
        except ValueError:
            expected_score, expected_failure = 0, True
        if grade.get("score") != expected_score or grade.get("technical_failure") is not expected_failure:
            raise ValueError("grade was changed after strict parsing")
    scored = list(zero_rows)
    for item in mapping:
        grade = grades.get(item["item_id"])
        scored.append({**item, "score":grade["score"] if grade else 0,
            "technical_failure":grade["technical_failure"] if grade else True,
            "received_grade":grade is not None,
            "failure_stage":grade["failure_stage"] if grade else "judge_pending"})
    expected = protocol["planned_answers"]
    if len(scored) != expected or len({(r["case_id"],r["arm"]) for r in scored}) != expected:
        raise ValueError("planned three-arm denominator differs")
    families = sorted({r["source_family_id"] for r in scored})
    by_arm, means = {}, {}
    for arm in ARMS:
        selected = [r for r in scored if r["arm"] == arm]
        means[arm] = {f:statistics.mean(r["score"] for r in selected if r["source_family_id"] == f) for f in families}
        by_arm[arm] = {"planned":len(selected), "received_grades":sum(r.get("received_grade",False) for r in selected),
            "technical_failures_or_pending":sum(r["technical_failure"] for r in selected),
            "question_micro_accuracy":statistics.mean(r["score"] for r in selected),
            "source_family_equal_accuracy":statistics.mean(means[arm].values()),
            "by_category":{c:{"planned":sum(r["category"]==c for r in selected),
                "accuracy":statistics.mean(r["score"] for r in selected if r["category"]==c)}
                for c in sorted({r["category"] for r in selected})},
            "by_abstention":{str(a).lower():{"planned":sum(r["should_abstain"]==a for r in selected),
                "accuracy":statistics.mean(r["score"] for r in selected if r["should_abstain"]==a)
                    if any(r["should_abstain"]==a for r in selected) else None} for a in (False,True)}}
    contrasts = {}
    for name, pair in (("primary_managed_minus_raw",protocol["primary_contrast"]),
                       ("auxiliary_managed_minus_no_rag",protocol["auxiliary_contrast"])):
        treatment, baseline = pair
        deltas = [means[treatment][f]-means[baseline][f] for f in families]
        interval = legacy.cluster_interval(deltas, seed=SEED, resamples=RESAMPLES, confidence=0.95)
        contrasts[name] = {"treatment":treatment, "baseline":baseline,
            "mean_source_family_accuracy_delta":statistics.mean(deltas), "cluster_bootstrap_95_interval":interval,
            "source_family_count":len(families), "interval_status":"estimated" if interval else "not_estimable_one_source_family"}
    repeated = [r for r in requests if r["repeat"]]
    pairs = [(grades[r["base_item_id"]],grades[r["case_id"]]) for r in repeated
             if r["base_item_id"] in grades and r["case_id"] in grades
             and not grades[r["base_item_id"]]["technical_failure"] and not grades[r["case_id"]]["technical_failure"]]
    return {"version":VERSION, "dataset_key":protocol["dataset_key"], "boundary":BOUNDARY,
        "official_evaluator_status":"adapted pinned templates with a different judge model",
        "complete":len(grades)==len(requests), "planned_answers":expected, "scored_denominator":len(scored),
        "planned_grading_rows":len(requests), "received_grading_rows":len(grades),
        "source_families":len(families), "conditions":by_arm, "paired_contrasts":contrasts,
        "repeatability":{"planned":len(repeated), "valid_pairs":len(pairs),
            "label_agreement":statistics.mean(a["score"]==b["score"] for a,b in pairs) if pairs else None},
        "provider_usage":{k:sum(r.get("usage",{}).get(k,0) for r in outcomes) for k in
                          ("prompt_tokens","completion_tokens","total_tokens")},
        "attempts":len(rows(folder / "intents.jsonl")),
        "unknown_receipts":sum(r["status"]=="unknown" for r in outcomes),
        "interpretation":"Incomplete runs are progress only; all missing rows remain in the planned denominator. Prior development exposure is retained; no confirmatory or clinical claim is inferred."}
