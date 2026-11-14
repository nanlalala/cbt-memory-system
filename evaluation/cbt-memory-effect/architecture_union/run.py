"""Hosted, resumable run of the additive managed-memory layers.

Reuses a completed ``architecture_retest`` run (the "prior" folder):
  * its frozen case roster and the memory items it already extracted and checked;
  * its received no-RAG, raw-RAG and frozen managed answers.
Only the new managed-arm answers, one per layer config, are generated here, so the
raw arm is literally the same answered request. A case is skipped (and recorded)
unless the prior raw answer is a verified received receipt and the rebuilt raw
request serializes to exactly the bytes that were sent for it (same
``serialized_request`` and ``wire_sha256``): otherwise the superset claim would
not hold for the answer that is scored.

Source families are split once, deterministically, into a development set and a
test set. Run and tune on ``dev`` only; freeze the configs; then run ``test`` once.
Importing this module sends nothing and reads no credential.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path

from architecture_retest import runner, judge
from architecture_retest.journal import (Journal, UnknownAttempt, CallLimit, InvalidReceipt,
                                         TerminalLengthReceipt, atomic, digest, lock, rows, utc)
from architecture_retest.retrieval import DENSE_MODEL, DENSE_REVISION, DenseRetriever
from auto_eval_v2 import quality
from auto_eval_v2.evidence import _cluster_interval
from auto_eval_v2.run_colab import require_hosted

from . import VERSION
from .layers import UnionConfig
from .union_requests import prepare_case_requests_union

RUN_VERSION = "cbt-managed-union-run-v1"
SPLIT_SALT = "cbt-managed-union-dev-test-split-v1"
HERE = Path(__file__).resolve().parent

# The layer ladder from the design note: one change per step.
_COMPACT = dict(renderer="v5", compact_quotes=True, evidence_once=True)
DEFAULT_LADDER = {
    # Reference: the first union, where each quoted message is rendered twice with long
    # IDs and item evidence repeats it again; the diagnosis showed most items crowded out.
    "L0_naive_union": UnionConfig(),
    "L1_raw_plus_items_compact": UnionConfig(**_COMPACT),
    "L2_plus_rerank": UnionConfig(rerank=True, **_COMPACT),
    "L3_plus_fact_key": UnionConfig(rerank=True, fact_key_sources=3, **_COMPACT),
    "L4_plus_pinned_profile": UnionConfig(rerank=True, fact_key_sources=3, pin_profile=True, **_COMPACT),
}


def code_hashes():
    """Union files plus every frozen file the reused runner already pins."""
    own = {str(p.relative_to(runner.ROOT)): sha256(p.read_bytes()).hexdigest()
           for p in sorted(HERE.glob("*.py"))}
    v5 = runner.ROOT / "modules/agent-integration/cbt_context_v5_draft.py"
    own[str(v5.relative_to(runner.ROOT))] = sha256(v5.read_bytes()).hexdigest()
    return {**runner.code_hashes(), **own}


def split_families(family_ids, dev_families):
    """Deterministic, outcome-blind family split; a family never spans both sets."""
    ordered = sorted(set(family_ids), key=lambda f: sha256((SPLIT_SALT + f).encode()).hexdigest())
    if type(dev_families) is not int or not 0 < dev_families <= len(ordered):
        raise ValueError("dev family count must be between 1 and the number of families")
    return {"dev": sorted(ordered[:dev_families]), "test": sorted(ordered[dev_families:])}


def prepare_prior(source, folder, *, family_ids=None, fixture_only=False):
    """Freeze a new natural CACTUS roster with the frozen runner (no model call).

    Without ``family_ids`` the public deterministic compiler chooses 200 source
    families; pass the historical list only if its frozen copy still exists.
    """
    return runner.prepare(source, folder, family_limit=200 if not fixture_only else 1,
                          fixture_only=fixture_only, family_ids=family_ids)


def run_prior(folder, transport, *, max_new_calls, split="dev", dev_families=40, workers=4,
              dense=None, fixture_only=False):
    """Run the frozen three-arm pipeline only for one split of a fresh roster.

    Each case goes through the unchanged ``runner._case``: extraction, automatic
    support check, then no-RAG, raw-RAG and frozen managed answers, with the same
    journal and receipts. The independent pre-answer assessor (direct memory
    endpoints) is not run here; R1--R5 judging still is, later, in one batch.
    """
    folder, protocol, cases = runner.load(folder, fixture_only)
    if protocol["version"] != runner.VERSION:
        raise ValueError("a fresh roster prepared by prepare_prior is required")
    if type(max_new_calls) is not int or max_new_calls < 0 or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("explicit bounded invocation budget and workers required")
    if fixture_only and getattr(transport, "fixture_only", False) is not True:
        raise ValueError("fixture execution cannot send to a real provider")
    families = set(split_families([c["source_family_id"] for c in cases], dev_families)[split])
    selected = [c for c in cases if c["source_family_id"] in families]
    scope = {"split": split, "dev_families": dev_families, "split_salt": SPLIT_SALT,
             "case_ids": sorted(c["case_id"] for c in selected), "independent_assessor": "not_run",
             "code_sha256": code_hashes()}
    scope_path = folder / ("union_prior_scope_" + split + ".json")
    other = folder / ("union_prior_scope_" + ("test" if split == "dev" else "dev") + ".json")
    if scope_path.exists():
        # The prior run depends only on the frozen pipeline, so only its hashes must match;
        # union-only files may change between batches without blocking the prior run.
        stored, frozen = json.loads(scope_path.read_text()), runner.code_hashes()
        same_cases = {k: v for k, v in stored.items() if k != "code_sha256"} == \
                     {k: v for k, v in scope.items() if k != "code_sha256"}
        same_frozen = {k: stored["code_sha256"].get(k) for k in frozen} == frozen
        if not (same_cases and same_frozen):
            raise ValueError("this split was already run with a different family split or frozen code")
    if other.exists() and json.loads(other.read_text())["dev_families"] != dev_families:
        raise ValueError("dev and test splits must use the same family split")
    with lock(folder):
        atomic(scope_path, scope)
        journal = Journal(folder, protocol, transport, max_new_calls)
        dense = dense if dense is not None else DenseRetriever()
        if not fixture_only and (getattr(dense, "model_id", None), getattr(dense, "revision", None)) != (DENSE_MODEL, DENSE_REVISION):
            raise ValueError("dense ranker identity differs from frozen E5 model/revision")
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(runner._case, c, folder, journal, dense, None): c["case_id"] for c in selected}
            for future in as_completed(futures):
                try:
                    future.result()
                except (UnknownAttempt, CallLimit, InvalidReceipt) as exc:
                    journal.stop.set()
                    failures.append({"case_id": futures[future], "status": type(exc).__name__})
                except Exception as exc:
                    journal.stop.set()
                    failures.append({"case_id": futures[future], "status": "runner_error",
                                     "error_type": type(exc).__name__})
        answers, _ = runner.verified_answers(folder)
        wanted = set(scope["case_ids"])
        result = {"split": split, "planned_cases": len(selected),
                  "completed_cases": sum((folder / "cases" / c / "completed.json").exists() for c in wanted),
                  "arms": {arm: sum(a["arm"] == arm and a["case_id"] in wanted for a in answers) for arm in runner.ARMS},
                  "new_calls": journal.new_calls, "errors": failures}
        atomic(folder / ("union_prior_summary_" + split + ".json"), result)
        return result


def _load_prior(prior, fixture_only):
    prior = Path(prior).resolve()
    protocol = json.loads((prior / "protocol.json").read_text())
    cases = json.loads((prior / "cases.json").read_text())
    if protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("prior protocol checksum differs")
    if protocol["cases_sha256"] != digest(cases) or protocol["fixture_only"] != fixture_only:
        raise ValueError("prior roster or fixture mode differs")
    if protocol["settings"] != runner.SETTINGS or protocol["provider"] != runner.IDENTITY:
        raise ValueError("prior run used different settings or answer model")
    return prior, protocol, cases


def prior_completed_case_ids(prior, *, split="dev", dev_families=40):
    """Cases of the split whose prior three-arm run finished with all three answers received."""
    prior = Path(prior)
    cases = json.loads((prior / "cases.json").read_text())
    families = set(split_families([c["source_family_id"] for c in cases], dev_families)[split])
    answers, _ = runner.verified_answers(prior)
    received = {}
    for a in answers:
        received.setdefault(a["case_id"], set()).add(a["arm"])
    return sorted(c["case_id"] for c in cases if c["source_family_id"] in families
                  and received.get(c["case_id"]) == set(runner.ARMS))


def prepare(prior, folder, *, split="dev", dev_families=40, ladder=None, case_ids=None, fixture_only=False):
    """Freeze roster, split, layer configs and code for one new run directory.

    ``case_ids`` makes an exploratory pilot on a subset of the split (for example
    the cases whose prior run already finished). A pilot never replaces the full
    split run and is labelled as such in its protocol.
    """
    if not fixture_only:
        require_hosted()
    prior, prior_protocol, prior_cases = _load_prior(prior, fixture_only)
    ladder = DEFAULT_LADDER if ladder is None else ladder
    if not ladder or any(type(c) is not UnionConfig for c in ladder.values()):
        raise ValueError("named UnionConfig ladder required")
    if any(not isinstance(n, str) or not n or "/" in n or n in runner.ARMS for n in ladder):
        raise ValueError("path-safe config names that differ from the frozen arm names required")
    families = split_families([c["source_family_id"] for c in prior_cases], dev_families)
    if split not in families:
        raise ValueError("split must be dev or test")
    cases = [c for c in prior_cases if c["source_family_id"] in set(families[split])]
    if case_ids is not None:
        wanted = set(case_ids)
        if not wanted or not wanted <= {c["case_id"] for c in cases}:
            raise ValueError("pilot case IDs must be a nonempty subset of the split")
        cases = [c for c in cases if c["case_id"] in wanted]
    folder = Path(folder).resolve()
    if not fixture_only and (not folder.is_relative_to("/content") or folder == Path("/content")):
        raise ValueError("new hosted run directory required")
    folder.mkdir(parents=True, exist_ok=False)
    protocol = {"version": RUN_VERSION, "union_version": VERSION, "fixture_only": fixture_only,
        "prior_folder": str(prior), "prior_protocol_sha256": prior_protocol["protocol_sha256"],
        "split": split, "scope": "pilot_subset_exploratory" if case_ids is not None else "full_split",
        "split_salt": SPLIT_SALT, "dev_families": dev_families, "family_split": families,
        "configs": {name: config.as_dict() for name, config in ladder.items()},
        "case_count": len(cases), "cases_sha256": digest(cases), "provider": runner.IDENTITY,
        "settings": runner.SETTINGS, "maximum_attempts": len(cases) * len(ladder),
        "request_limits": {"answer": runner.SETTINGS["max_answer_chars"],
                           "preparation": runner.SETTINGS["max_preparation_chars"]},
        "reused_from_prior": ["extracted_and_checked_items", "no_rag_answers", "raw_history_rag_answers",
                              "frozen_managed_answers"],
        "tuning_rule": "tune and choose on dev only; run test once with frozen configs",
        "code_sha256": code_hashes(), "created_at_utc": utc()}
    protocol["protocol_sha256"] = digest(protocol)
    atomic(folder / "cases.json", cases)
    atomic(folder / "protocol.json", protocol)
    return protocol


def load(folder, fixture_only=False):
    folder = Path(folder).resolve()
    protocol = json.loads((folder / "protocol.json").read_text())
    cases = json.loads((folder / "cases.json").read_text())
    if not fixture_only:
        require_hosted()
    if (protocol["version"] != RUN_VERSION or protocol["fixture_only"] != fixture_only
            or protocol["cases_sha256"] != digest(cases)
            or protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})):
        raise ValueError("frozen union run protocol differs")
    if protocol["code_sha256"] != code_hashes() or protocol["settings"] != runner.SETTINGS:
        raise ValueError("code or settings changed since prepare; prepare a new run")
    prior, prior_protocol, _ = _load_prior(protocol["prior_folder"], fixture_only)
    if prior_protocol["protocol_sha256"] != protocol["prior_protocol_sha256"]:
        raise ValueError("prior run differs from the one this run was prepared on")
    ladder = {name: UnionConfig(**values) for name, values in protocol["configs"].items()}
    return folder, protocol, cases, prior, ladder


def _prior_items(prior, case):
    """Accepted items the prior run actually checked, or None if preparation failed."""
    dest = prior / "cases" / case["case_id"]
    if (dest / "memory_preparation_failure.json").exists() or not (dest / "projections.json").exists():
        return None
    return json.loads((dest / "projections.json").read_text()).get("accepted")


def _prior_raw_wires(prior):
    """Exact sent bytes of every verified prior raw answer, by case."""
    from architecture_retest.receipt_resolver import receipt_pairs
    answers, _ = runner.verified_answers(prior)
    intents, _ = receipt_pairs(prior)
    wires = {}
    for answer in answers:
        if answer["arm"] == "raw_history_rag":
            intent = intents[answer["request_key"]]
            wires[answer["case_id"]] = (intent["serialized_request"], intent["wire_sha256"])
    return wires


def raw_bytes_match(payload, wire):
    serialized = json.dumps(payload, allow_nan=False)
    return serialized == wire[0] and sha256(serialized.encode()).hexdigest() == wire[1]


def _case(case, folder, prior, ladder, journal, dense, raw_wires):
    dest = folder / "cases" / case["case_id"]
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "completed.json").exists():
        return
    items = _prior_items(prior, case)
    if items is None:
        atomic(dest / "completed.json", {"case_id": case["case_id"], "status": "skipped",
                                         "reason": "prior_memory_preparation_unavailable"})
        return
    wire = raw_wires.get(case["case_id"])
    if wire is None:
        atomic(dest / "completed.json", {"case_id": case["case_id"], "status": "skipped",
                                         "reason": "prior_raw_answer_unavailable"})
        return
    sources = runner.source_views(case)
    rows_out = []
    for name, config in ladder.items():
        try:
            plans = prepare_case_requests_union(case, sources, items, dense, journal.dense_mutex, config)
        except ValueError as exc:
            rows_out.append({"config": name, "status": "preparation_failed", "reason": type(exc).__name__})
            continue
        if not raw_bytes_match(plans["raw_history_rag"]["payload"], wire):
            atomic(dest / "completed.json", {"case_id": case["case_id"], "status": "skipped",
                                             "reason": "raw_request_bytes_differ_from_prior_receipt"})
            return
        row = plans["managed_memory_rag"]
        atomic(dest / (name + ".request.json"), row)
        try:
            receipt = journal.call(case, "answer", row["payload"])
        except TerminalLengthReceipt as exc:
            rows_out.append({"config": name, "status": "terminal_length_response", "request_key": exc.request_key})
            continue
        answer = {"config": name, "case_id": case["case_id"], "source_family": case["source_family_id"],
                  "status": "received", "request_key": receipt["key"], "response": receipt["text"],
                  "provider_model": receipt["provider_model"], "usage": receipt["usage"]}
        atomic(dest / (name + ".answer.json"), answer)
        rows_out.append({"config": name, "status": "received", "request_key": receipt["key"]})
    atomic(dest / "completed.json", {"case_id": case["case_id"], "status": "completed", "rows": rows_out})


def run(folder, transport, *, max_new_calls, workers=4, dense=None, fixture_only=False):
    folder, protocol, cases, prior, ladder = load(folder, fixture_only)
    if type(max_new_calls) is not int or max_new_calls < 0 or type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("explicit bounded invocation budget and workers required")
    if fixture_only and getattr(transport, "fixture_only", False) is not True:
        raise ValueError("fixture execution cannot send to a real provider")
    unfinished = [c["case_id"] for c in cases if not (prior / "cases" / c["case_id"] / "completed.json").exists()]
    if unfinished:
        # Otherwise a case whose prior answers are still pending would be skipped for good.
        raise ValueError("prior run not finished for %d cases of this split; finish it first" % len(unfinished))
    with lock(folder):
        journal = Journal(folder, protocol, transport, max_new_calls)
        dense = dense if dense is not None else DenseRetriever()
        if not fixture_only and (getattr(dense, "model_id", None), getattr(dense, "revision", None)) != (DENSE_MODEL, DENSE_REVISION):
            raise ValueError("dense ranker identity differs from frozen E5 model/revision")
        raw_wires = _prior_raw_wires(prior)
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_case, c, folder, prior, ladder, journal, dense, raw_wires): c["case_id"]
                       for c in cases}
            for future in as_completed(futures):
                try:
                    future.result()
                except (UnknownAttempt, CallLimit, InvalidReceipt) as exc:
                    journal.stop.set()
                    failures.append({"case_id": futures[future], "status": type(exc).__name__})
                except Exception as exc:
                    journal.stop.set()
                    failures.append({"case_id": futures[future], "status": "runner_error",
                                     "error_type": type(exc).__name__})
        result = summarize(folder)
        atomic(folder / "summary.json", result)
        atomic(folder / "last_invocation.json", {"new_calls": journal.new_calls, "errors": failures, "at_utc": utc()})
        return result


def diagnose(prior, case_ids, *, ladder=None, dense=None, fixture_only=False):
    """Build every config's managed request for finished prior cases; no model call.

    Reports, per config, whether the raw selection is inside, how many items fit
    compared with the frozen managed arm, the extra fact-key messages and the
    request size against the ceiling.
    """
    if not fixture_only:
        require_hosted()
    prior, _, cases = _load_prior(prior, fixture_only)
    ladder = DEFAULT_LADDER if ladder is None else ladder
    dense = dense if dense is not None else DenseRetriever()
    by_id = {c["case_id"]: c for c in cases}
    mutex, rows_out = __import__("threading").Lock(), []
    for cid in case_ids:
        case = by_id[cid]
        items = _prior_items(prior, case)
        frozen = json.loads((prior / "cases" / cid / "managed_memory_rag.request.json").read_text())
        row = {"case_id": cid, "items_available": None if items is None else len(items),
               "frozen_managed_items": len(frozen.get("selection_audit", {}).get("managed_selected_item_ids", [])),
               "frozen_managed_chars": frozen["trace"]["request_chars"], "configs": {}}
        if items is not None:
            for name, config in ladder.items():
                plans = prepare_case_requests_union(case, runner.source_views(case), items, dense, mutex, config)
                audit, trace = plans["managed_memory_rag"]["selection_audit"], plans["managed_memory_rag"]["trace"]
                row["configs"][name] = {"raw_subset": audit["raw_selection_is_subset_of_managed"],
                    "items_included": len(audit["union_item_ids"]),
                    "items_rejected_for_budget": sum(r["reason"] == "OutboundBudgetError" for r in audit["union_item_rejections"]),
                    "items_rejected_other": sum(r["reason"] != "OutboundBudgetError" for r in audit["union_item_rejections"]),
                    "fact_key_messages": len(audit["union_fact_key_source_ids"]),
                    "pinned_profile_items": len(audit["union_profile_item_ids"]),
                    "request_chars": trace["request_chars"]}
        rows_out.append(row)
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 2) if values else None
    usable = [r for r in rows_out if r["configs"]]
    summary = {"cases": len(rows_out), "cases_with_items": len(usable),
               "ceiling_chars": runner.SETTINGS["max_answer_chars"],
               "frozen_managed_mean_items": mean([r["frozen_managed_items"] for r in usable]),
               "configs": {name: {key: mean([r["configs"][name][key] for r in usable])
                                  for key in ("items_included", "items_rejected_for_budget", "items_rejected_other",
                                              "fact_key_messages", "pinned_profile_items", "request_chars")}
                           | {"raw_subset_all": all(r["configs"][name]["raw_subset"] for r in usable)}
                           for name in ladder}}
    return {"summary": summary, "cases": rows_out}


def _answers(folder):
    """Only answer files bound to a received receipt with the same text and request."""
    outcomes = {r["key"]: r for r in rows(folder / "outcomes.jsonl")}
    intents = {r["key"]: r for r in rows(folder / "intents.jsonl")}
    accepted = []
    for path in (folder / "cases").glob("*/*.answer.json"):
        answer = json.loads(path.read_text())
        receipt, intent = outcomes.get(answer.get("request_key"), {}), intents.get(answer.get("request_key"), {})
        request = json.loads(path.with_name(answer["config"] + ".request.json").read_text())
        if (receipt.get("status") == "received" and receipt.get("text") == answer.get("response")
                and request.get("payload") == intent.get("payload")):
            accepted.append(answer)
    return accepted


def summarize(folder):
    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text())
    completed = [json.loads(p.read_text()) for p in (folder / "cases").glob("*/completed.json")]
    answers = _answers(folder)
    sizes = {}
    for path in (folder / "cases").glob("*/*.request.json"):
        sizes.setdefault(path.name.removesuffix(".request.json"), []).append(
            json.loads(path.read_text())["trace"]["request_chars"])
    return {"version": RUN_VERSION, "split": protocol["split"], "planned_cases": protocol["case_count"],
        "completed_cases": sum(c["status"] == "completed" for c in completed),
        "skipped_cases": {r: sum(c.get("reason") == r for c in completed)
                          for r in {c.get("reason") for c in completed if c["status"] == "skipped"}},
        "configs": {name: {"received": sum(a["config"] == name for a in answers),
                           "mean_request_chars": (sum(sizes.get(name, [])) / len(sizes[name])) if sizes.get(name) else None}
                    for name in protocol["configs"]},
        "new_receipts": sum(r["status"] == "received" for r in rows(folder / "outcomes.jsonl")),
        "response_quality_scores_available": False}


def export_quality(folder, judge_identity, output, *, maximum_attempts, include_no_rag=False, fixture_only=False):
    """One blinded judge batch: prior raw (+ frozen managed, + optional no-RAG) and every config.

    Re-judging the prior answers in the same batch keeps one judge, one prompt and
    one session for every compared condition.
    """
    folder, protocol, cases, prior, ladder = load(folder, fixture_only)
    unfinished = [c["case_id"] for c in cases if not (folder / "cases" / c["case_id"] / "completed.json").exists()]
    if unfinished:
        raise ValueError("union run not finished for %d cases; export only after every case is completed or skipped"
                         % len(unfinished))
    prior_answers, _ = runner.verified_answers(prior)
    old = {(a["case_id"], a["arm"]): a for a in prior_answers}
    new = {(a["case_id"], a["config"]): a for a in _answers(folder)}
    arms = [*(["no_rag"] if include_no_rag else []), "raw_history_rag", "managed_memory_rag", *ladder]
    packets, mapping = [], []
    for case in cases:
        marker = folder / "cases" / case["case_id"] / "completed.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "completed":
            continue
        history = [{"id": s["source_id"], "role": s["role"], "content": s["content"]} for s in runner.source_views(case)]
        for arm in arms:
            answer = old.get((case["case_id"], arm)) if arm in runner.ARMS else new.get((case["case_id"], arm))
            if answer is None:
                mapping.append({"case_id": case["case_id"], "source_family": case["source_family_id"],
                                "arm": arm, "packet_id": None, "status": "not_received"})
                continue
            opaque = "response-" + digest([protocol["protocol_sha256"], case["case_id"], arm])[:32]
            packets.append({"case_id": opaque, "source_family": case["source_family_id"], "answer_model_family": "deepseek",
                            "history": history, "input": case["input"], "response": answer["response"]})
            mapping.append({"case_id": case["case_id"], "source_family": case["source_family_id"], "arm": arm,
                            "packet_id": opaque, "status": "received"})
    result = judge.prepare(packets, judge_identity, output, maximum_attempts=maximum_attempts, fixture_only=fixture_only)
    atomic(Path(output) / "controller_arm_mapping.json", mapping)
    return result


def compare(quality_folder, *, baseline="raw_history_rag", draws=10000, seed=20261114):
    """Paired difference of every condition against the raw arm, per R1--R5.

    Clustered by source family. 10,000 draws only stabilise the percentile
    computation; they add no independent observations. Exploratory on dev.
    """
    folder = Path(quality_folder)
    mapping = json.loads((folder / "controller_arm_mapping.json").read_text())
    reviews = {r["case_id"]: r for r in rows(folder / "reviews.jsonl") if r["repeat_of"] is None}
    grouped = {}
    for row in mapping:
        grouped.setdefault(row["case_id"], {})[row["arm"]] = row
    arms = list(dict.fromkeys(r["arm"] for r in mapping))
    if baseline not in arms:
        raise ValueError("baseline condition is not in this judge batch")

    def score(row, dimension):
        review = reviews.get(row["packet_id"], {}).get("review", {}) if row and row["packet_id"] else {}
        if review.get("status") != "completed":
            return None
        value = review["assessment"][dimension]
        return int(value == "pass") if dimension == "R5" else value

    report = {"baseline": baseline, "interpretation": "automatic independent assessment; exploratory unless test split",
              "comparisons": {}}
    for arm in arms:
        if arm == baseline:
            continue
        report["comparisons"][arm] = {}
        for dimension in quality.DIMENSIONS:
            differences = []
            for pair in grouped.values():
                a, b = score(pair.get(baseline), dimension), score(pair.get(arm), dimension)
                if a is not None and b is not None:
                    differences.append((pair[baseline]["source_family"], b - a))
            report["comparisons"][arm][dimension] = {"planned_pairs": len(grouped), "scored_pairs": len(differences),
                "wins": sum(d > 0 for _, d in differences), "ties": sum(d == 0 for _, d in differences),
                "losses": sum(d < 0 for _, d in differences),
                "family_cluster_difference": _cluster_interval(differences, draws=draws, seed=seed)}
    atomic(folder / "union_comparison.json", report)
    return report
