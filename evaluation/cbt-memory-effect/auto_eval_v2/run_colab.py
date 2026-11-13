"""Freeze and journal a supplied independent judge in genuine hosted Colab.

There is no credential lookup, automatic retry, dataset download or default
model transport. The notebook supplies a transport with an explicit identity.
Invented-only fixture tests can run without Colab; no actual model is included.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

from . import VERSION
from . import quality


def _task(name):
    if name == "quality":
        return quality
    if name == "calibration":
        from . import calibration_prompt
        return calibration_prompt
    raise ValueError("unknown automatic evaluation task")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_json(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def require_hosted():
    try:
        valid = (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
                 and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ModuleNotFoundError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("genuine hosted Colab required before experiment I/O")


def _guard(fixture_only, identity, packets=()):
    if fixture_only:
        if identity.get("provider") != "fixture" or identity.get("model_family") != "invented-judge":
            raise ValueError("fixture mode cannot select a real provider or judge")
        if any(not p["source_family"].startswith("invented:") for p in packets):
            raise ValueError("fixture mode requires explicitly invented source families")
    else:
        require_hosted()


def _code_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def _calibration_bank_identity(bank):
    from .calibration import verify_source_manifest
    bank = Path(bank).resolve()
    if not bank.is_relative_to(Path("/content")):
        raise ValueError("calibration bank must remain in hosted Colab")
    manifest_path = bank / "source_bank_manifest.json"
    data_path = bank / "assessor_only/therapyjudgebench.jsonl"
    if manifest_path.is_symlink() or data_path.is_symlink():
        raise ValueError("symlinked calibration bank rejected")
    manifest_bytes, data_bytes = manifest_path.read_bytes(), data_path.read_bytes()
    source = verify_source_manifest(strict_json(manifest_bytes), hashlib.sha256(data_bytes).hexdigest())
    return {**source, "bank_path": str(bank),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}


def _validate_comparisons(packets, comparisons, task):
    if type(comparisons) not in (list, tuple):
        raise ValueError("comparison grid must be a list")
    if task != "quality" and comparisons:
        raise ValueError("calibration is not a raw/managed quality comparison")
    by_id = {p["case_id"]: p for p in packets}
    seen, used, result = set(), set(), []
    for row in comparisons:
        if type(row) is not dict or set(row) != {"case_id", "source_family", "raw_case_id", "managed_case_id"}:
            raise ValueError("exact paired comparison schema required")
        if any(type(row[k]) is not str or not row[k] for k in ("case_id", "source_family")) or row["case_id"] in seen:
            raise ValueError("unique nonempty comparison identities required")
        ids = [row["raw_case_id"], row["managed_case_id"]]
        present_ids = [i for i in ids if i is not None]
        if any(type(i) is not str or not i for i in present_ids):
            raise ValueError("missing generated response must be explicit null")
        if len(set(present_ids)) != len(present_ids) or any(i not in by_id or i in used for i in present_ids):
            raise ValueError("comparison requires two distinct unused response packets")
        existing = [by_id[i] for i in present_ids]
        if len(existing) == 2 and any(existing[0][k] != existing[1][k] for k in (
                "source_family", "answer_model_family", "history", "input")):
            raise ValueError("judge histories/current inputs/answer families must match across arms")
        if any(p["source_family"] != row["source_family"] for p in existing):
            raise ValueError("comparison source-family binding changed")
        seen.add(row["case_id"])
        used.update(present_ids)
        result.append(dict(row))
    return result


def _paired_quality(comparisons, rows):
    # Repeats audit judge consistency and never become extra effect cases.
    from .evidence import _cluster_interval
    by_case = {r["case_id"]: r for r in rows if r["repeat_of"] is None}
    report = {"planned_pairs": len(comparisons), "dimensions": {},
              "same_canonical_receipt_pairs": sum(
                  by_case[c["raw_case_id"]]["canonical_key"] == by_case[c["managed_case_id"]]["canonical_key"]
                  for c in comparisons if c["raw_case_id"] is not None and c["managed_case_id"] is not None),
              "repeat_scores_are_effect_observations": False}
    for dimension in ("R1", "R2", "R3", "R4", "R5"):
        raw_values, managed_values, differences = [], [], []
        for case in comparisons:
            entries = [by_case.get(case[key], {}) for key in ("raw_case_id", "managed_case_id")]
            if not all(entry.get("status") == "received" for entry in entries):
                continue
            reviews = [entry.get("review") for entry in entries]
            if not all(type(r) is dict and r.get("status") == "completed" for r in reviews):
                continue
            values = [r["assessment"][dimension] for r in reviews]
            if dimension == "R5":
                values = [int(v == "pass") for v in values]
            raw_values.append(values[0])
            managed_values.append(values[1])
            differences.append((case["source_family"], values[1] - values[0]))
        n = len(differences)
        report["dimensions"][dimension] = {
            "planned_pairs": len(comparisons), "scored_pairs": n,
            "missing_or_unresolved_pairs": len(comparisons) - n,
            "raw_mean_or_safety_pass_rate": sum(raw_values) / n if n else None,
            "managed_mean_or_safety_pass_rate": sum(managed_values) / n if n else None,
            "managed_wins": sum(d > 0 for _, d in differences),
            "ties": sum(d == 0 for _, d in differences),
            "raw_wins": sum(d < 0 for _, d in differences),
            "family_cluster_difference": _cluster_interval(differences, draws=1000, seed=20261112)}
    return report


def _repeat_consistency(rows):
    by_item = {r["item_id"]: r for r in rows}
    differences = {d: [] for d in ("R1", "R2", "R3", "R4", "R5")}
    planned, completed = 0, 0
    for row in rows:
        if row["repeat_of"] is None:
            continue
        planned += 1
        entries = [row, by_item[row["repeat_of"]]]
        if not all(entry.get("status") == "received" for entry in entries):
            continue
        reviews = [entry.get("review") for entry in entries]
        if not all(type(r) is dict and r.get("status") == "completed" for r in reviews):
            continue
        completed += 1
        for dimension in differences:
            values = [r["assessment"][dimension] for r in reviews]
            differences[dimension].append(abs(values[0] - values[1]) if dimension != "R5"
                                          else int(values[0] != values[1]))
    return {"planned_repeats": planned, "completed_repeat_pairs": completed,
            "missing_or_unresolved_repeat_pairs": planned - completed,
            "mean_absolute_score_difference_or_safety_disagreement_rate": {
                d: sum(v) / len(v) if v else None for d, v in differences.items()}}


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _signed(value):
    return {**value, "record_sha256": digest(value)}


def _append(path, value):
    if Path(path).is_symlink():
        raise ValueError("symlink journal rejected")
    with Path(path).open("a", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical(_signed(value)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rows(path):
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("symlink journal rejected")
    raw = path.read_text()
    if raw and not raw.endswith("\n"):
        raise ValueError("incomplete journal tail retained; inspect before resuming")
    result = []
    for line in raw.splitlines():
        row = strict_json(line)
        if type(row) is not dict or row.get("record_sha256") != digest(
                {k: v for k, v in row.items() if k != "record_sha256"}):
            raise ValueError("journal checksum mismatch")
        result.append(row)
    return result


@contextmanager
def _lock(folder):
    path = folder / "execution.lock"
    if path.is_symlink():
        raise ValueError("symlink lock rejected")
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def prepare(packets, judge_identity, output, *, maximum_attempts=0,
            authorization_reference="not-authorized-for-new-calls",
            max_tokens=2048, max_request_chars=30000,
            repeat_fraction=0.1, seed=20261112, fixture_only=False, task="quality",
            comparisons=(), calibration_bank=None):
    """Freeze caller-approved limits; zero is a preparation-only default.

    authorization_reference records the session decision, it does not establish
    authorization by itself. No historic attempt allowance is inherited.
    """
    _guard(fixture_only, judge_identity, packets)
    if task == "calibration" and not fixture_only and calibration_bank is None:
        raise ValueError("existing checked calibration bank required; no automatic acquisition")
    source_binding = (_calibration_bank_identity(calibration_bank)
                      if task == "calibration" and not fixture_only else None)
    if type(maximum_attempts) is not int or maximum_attempts < 0:
        raise ValueError("explicit nonnegative new-run attempt cap required")
    if not isinstance(authorization_reference, str) or not authorization_reference.strip():
        raise ValueError("authorization reference required")
    if any(type(v) is not int or v <= 0 for v in (max_tokens, max_request_chars)):
        raise ValueError("positive generation bounds required")
    items = _task(task).plan_quality(packets, judge_identity,
                                    repeat_fraction=repeat_fraction, seed=seed)
    comparisons = _validate_comparisons(packets, comparisons, task)
    for row in items:
        payload = {"model": judge_identity["model"], "messages": row["messages"],
                   "temperature": 0, "max_tokens": max_tokens}
        if len(json.dumps(payload, ensure_ascii=True, allow_nan=False)) > max_request_chars:
            raise ValueError("complete escaped judge request exceeds frozen cap")
        row["payload"] = payload
        # Identical primary requests share one receipt; repeat slots really run
        # independently, with their own intent and cost.
        slot = "primary" if row["repeat_of"] is None else row["item_id"]
        row["canonical_key"] = digest({"judge": judge_identity, "payload": payload, "slot": slot})
    output = Path(output).resolve()
    if not fixture_only and (not output.is_relative_to(Path("/content")) or output == Path("/content")):
        raise ValueError("new hosted Colab /content output directory required")
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    protocol = {"version": VERSION, "fixture_only": fixture_only, "task": task,
                "judge_identity": judge_identity, "maximum_attempts": maximum_attempts,
                "authorization_reference": authorization_reference,
                "max_tokens": max_tokens, "max_request_chars": max_request_chars,
                "repeat_fraction": repeat_fraction, "seed": seed,
                "packets_sha256": digest(packets), "plan_sha256": digest(items),
                "comparisons_sha256": digest(comparisons),
                "calibration_source_binding": source_binding,
                "code_sha256": _code_hashes(), "human_review_required": False,
                "existing_results_modified": False, "new_model_calls_at_preparation": 0}
    protocol["protocol_sha256"] = digest(protocol)
    _write_new(output / "packets.json", packets)
    _write_new(output / "plan.json", items)
    _write_new(output / "comparisons.json", comparisons)
    _write_new(output / "protocol.json", protocol)
    return {"status": "prepared_not_executed", "output": str(output),
            "items": len(items), "unique_requests_including_repeats": len({i["canonical_key"] for i in items}),
            "maximum_attempts": maximum_attempts, "protocol_sha256": protocol["protocol_sha256"]}


def _load(folder, fixture_only):
    # A real run is guarded before any source-containing file is read.
    if not fixture_only:
        require_hosted()
    for name in ("protocol.json", "packets.json", "plan.json", "comparisons.json"):
        if (folder / name).is_symlink():
            raise ValueError("symlink frozen input rejected")
    protocol = strict_json((folder / "protocol.json").read_text())
    packets = strict_json((folder / "packets.json").read_text())
    plan = strict_json((folder / "plan.json").read_text())
    comparisons = strict_json((folder / "comparisons.json").read_text())
    _guard(fixture_only, protocol["judge_identity"], packets)
    if (protocol["version"] != VERSION or protocol["fixture_only"] is not fixture_only
            or protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            or protocol["code_sha256"] != _code_hashes()
            or protocol["packets_sha256"] != digest(packets)
            or protocol["plan_sha256"] != digest(plan)
            or protocol["comparisons_sha256"] != digest(comparisons)):
        raise ValueError("frozen protocol/code/input identity changed")
    _validate_comparisons(packets, comparisons, protocol["task"])
    if protocol["task"] == "calibration" and not fixture_only:
        binding = protocol["calibration_source_binding"]
        if not binding or _calibration_bank_identity(binding["bank_path"]) != binding:
            raise ValueError("frozen published calibration source changed")
    return protocol, packets, plan


class UnknownAttempt(RuntimeError):
    """An intent may already be billed; inspection is required, not a retry."""


def _usage(value):
    value = value if type(value) is dict else {}
    return {k: value.get(k) if type(value.get(k)) is int and value[k] >= 0 else None
            for k in ("input_tokens", "output_tokens", "total_tokens")}


def _state(folder, protocol, plan, packets, *, allow_unknown=False):
    intents, outcomes = {}, {}
    allowed = {r["canonical_key"]: r["payload"] for r in plan}
    item_for_key = {r["canonical_key"]: r for r in plan}
    packet_by_id = {p["case_id"]: p for p in packets}
    for name, collection in (("intents.jsonl", intents), ("outcomes.jsonl", outcomes)):
        for row in _rows(folder / name):
            key = row["canonical_key"]
            if (key in collection or key not in allowed or row["protocol_sha256"] != protocol["protocol_sha256"]
                    or row["payload_sha256"] != digest(allowed[key])):
                raise ValueError("foreign, duplicate or unbound journal record")
            collection[key] = row
    if set(outcomes) - set(intents) or len(intents) > protocol["maximum_attempts"]:
        raise ValueError("journal outside frozen attempt grid")
    for row in outcomes.values():
        if row["status"] not in {"received", "invalid", "unknown"}:
            raise ValueError("invalid outcome state")
        if row.get("usage") != _usage(row.get("usage")):
            raise ValueError("invalid preserved usage")
        seconds = row.get("observed_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid preserved latency")
        review = row.get("review")
        if row["status"] != "received" and type(review) is dict and (
                review.get("accepted") or review.get("status") in {"completed", "unresolved"}):
            raise ValueError("invalid or unknown delivery cannot carry an accepted score")
        if row["status"] == "received" and (
                row.get("provider_model") != protocol["judge_identity"]["model"]
                or type(row.get("native_text")) is not str
                or type(row.get("review")) is not dict or not row["review"].get("accepted")):
            raise ValueError("received outcome lacks bound native review")
        if row["status"] == "received":
            expected = _task(protocol["task"]).validate_review(
                row["native_text"], packet_by_id[item_for_key[row["canonical_key"]]["case_id"]])
            if expected != row["review"]:
                raise ValueError("preserved scores do not match native model response")
    if not allow_unknown and (set(intents) - set(outcomes)
                              or any(r["status"] == "unknown" for r in outcomes.values())):
        raise UnknownAttempt("unfinished or unknown send blocks further calls; do not retry")
    return intents, outcomes


def summarize(folder, *, fixture_only=False):
    folder = Path(folder).resolve()
    protocol, packets, plan = _load(folder, fixture_only)
    # Unknown sends are reported, but cannot be made resumable by analysis.
    intent_map, by_key = _state(folder, protocol, plan, packets, allow_unknown=True)
    intents, outcomes = list(intent_map.values()), list(by_key.values())
    token_totals, missing = {}, {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [r.get("usage", {}).get(field) for r in outcomes]
        observed = [v for v in values if type(v) is int and v >= 0]
        token_totals[field] = sum(observed) if observed else None
        missing[field] = len(intents) - len(observed)
    rows = [{"item_id": r["item_id"], "case_id": r["case_id"],
             "source_family": r["source_family"], "repeat_of": r["repeat_of"],
             "canonical_key": r["canonical_key"],
             "status": by_key.get(r["canonical_key"], {}).get("status", "pending"),
             "review": by_key.get(r["canonical_key"], {}).get("review")}
            for r in plan]
    review_states = Counter(r.get("review", {}).get("status", "invalid_or_unknown")
                           if r["status"] == "received" and type(r.get("review")) is dict
                           else "invalid_or_unknown" for r in outcomes)
    comparisons = strict_json((folder / "comparisons.json").read_text())
    return {"version": VERSION, "task": protocol["task"], "protocol_sha256": protocol["protocol_sha256"],
            "planned_items": len(plan), "actual_attempts": len(intents),
            "delivery_states": dict(Counter(r["status"] for r in outcomes)),
            "review_states": dict(review_states),
            "scored_unique_reviews": review_states["completed"],
            "unresolved_unique_reviews": review_states["unresolved"],
            "unknown_attempts": len({r["canonical_key"] for r in intents} - set(by_key))
                + sum(r["status"] == "unknown" for r in outcomes),
            "tokens_reported": token_totals, "attempts_missing_usage": missing,
            "observed_invocation_seconds": sum(r.get("observed_seconds", 0) for r in outcomes),
            "monetary_cost": None, "human_reviews_required": False,
            "source_of_scores": "automatic_model_review", "items": rows,
            "paired_quality": _paired_quality(comparisons, rows) if protocol["task"] == "quality" else None,
            "repeat_consistency": _repeat_consistency(rows) if protocol["task"] == "quality" else None}


def score_calibration(folder, gold_rows=None, *, fixture_only=False):
    """Read labels only after every planned primary model attempt is frozen.

    Caller projects existing published labels to the exact frozen case grid;
    model-generated labels must not be passed off as published expert labels.
    This does not certify R1/R3 or human treatment outcomes.
    """
    from .calibration import calibrate
    summary = summarize(folder, fixture_only=fixture_only)
    if summary["task"] != "calibration" or summary["unknown_attempts"]:
        raise ValueError("completed calibration run required before gold scoring")
    primary = [r for r in summary["items"] if r["repeat_of"] is None]
    if any(r["status"] == "pending" for r in primary):
        raise ValueError("all primary predictions must be frozen before label comparison")
    if not fixture_only:
        if gold_rows is not None:
            raise ValueError("real calibration labels must come from the bound published bank")
        protocol, packets, _ = _load(Path(folder).resolve(), False)
        bank = Path(protocol["calibration_source_binding"]["bank_path"])
        records = {}
        for line in (bank / "assessor_only/therapyjudgebench.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = strict_json(line)
            if record["id"] in records:
                raise ValueError("duplicate published calibration case")
            records[record["id"]] = record
        gold_rows = []
        from .calibration import project_dialogue
        for packet in packets:
            record = records.get(packet["case_id"])
            if not record or project_dialogue(record)["messages"] != packet["messages"]:
                raise ValueError("prediction input does not match published rated dialogue")
            labels = record["expert_rating"]
            gold_rows.append({"case_id": packet["case_id"], "ctrs": labels["ctrs_ratings"],
                              "safety": labels["safety_ethics"]})
    if type(gold_rows) is not list or {r.get("case_id") for r in gold_rows} != {r["case_id"] for r in primary}:
        raise ValueError("published labels must match frozen calibration case grid")
    predictions = []
    for row in primary:
        review = row.get("review")
        if type(review) is dict and review.get("status") == "completed":
            assessment = review["assessment"]
            predictions.append({"case_id": row["case_id"], "status": "completed",
                                "ctrs": assessment["ctrs"], "safety": assessment["safety"]})
        else:
            predictions.append({"case_id": row["case_id"],
                                "status": "unresolved" if type(review) is dict and review.get("status") == "unresolved"
                                else "invalid"})
    return {"protocol_sha256": summary["protocol_sha256"],
            "gold_use": "published expert labels; no newly recruited reviewers",
            "calibration": calibrate(gold_rows, predictions)}


def run(folder, transport, *, max_new_calls=0, fixture_only=False):
    """transport(payload) -> {text, provider_model, usage}; no automatic retries.

    transport.identity must equal the frozen judge identity. The notebook owns
    provider access and must journal any additional provider-side retries: such
    retries are forbidden by this transport contract.
    """
    folder = Path(folder).resolve()
    protocol, packets, plan = _load(folder, fixture_only)
    if type(max_new_calls) is not int or max_new_calls < 0:
        raise ValueError("nonnegative explicit invocation limit required")
    if getattr(transport, "identity", None) != protocol["judge_identity"]:
        raise ValueError("transport identity differs from frozen independent judge")
    packet_by_id = {p["case_id"]: p for p in packets}
    with _lock(folder):
        intents, outcomes = _state(folder, protocol, plan, packets)
        new_calls = 0
        for item in plan:
            key = item["canonical_key"]
            if key in outcomes:
                continue
            if new_calls >= max_new_calls or len(intents) >= protocol["maximum_attempts"]:
                break
            if not fixture_only:
                preflight = getattr(transport, "preflight", None)
                if not callable(preflight) or preflight(
                        strict_json(canonical(item["payload"])),
                        strict_json(canonical(packet_by_id[item["case_id"]]))) is not True:
                    raise PermissionError("current source/permission preflight required before any send intent")
            shared = {"canonical_key": key, "protocol_sha256": protocol["protocol_sha256"],
                      "payload_sha256": digest(item["payload"])}
            intent = {**shared, "at_utc": datetime.now(timezone.utc).isoformat()}
            _append(folder / "intents.jsonl", intent)
            intents[key] = intent
            new_calls += 1
            started = time.monotonic()
            outcome = {**shared, "status": "unknown", "usage": _usage(None),
                       "native_text": None, "provider_model": None, "review": None}
            try:
                transport_payload = strict_json(canonical(item["payload"]))
                received = transport(transport_payload)
                if type(received) is not dict:
                    raise ValueError("transport receipt is not an object")
                outcome["usage"] = _usage(received.get("usage"))
                outcome["provider_model"] = received.get("provider_model")
                text = received.get("text")
                if type(text) is not str or len(text) > 200000:
                    raise ValueError("missing or oversized native model response")
                outcome["native_text"] = text
                outcome["status"] = "invalid"
                if (transport_payload == item["payload"]
                        and received.get("provider_model") == protocol["judge_identity"]["model"]):
                    checked = _task(protocol["task"]).validate_review(text, packet_by_id[item["case_id"]])
                    outcome["review"] = checked
                    if checked["accepted"]:
                        outcome["status"] = "received"
            except Exception as error:
                # No exception text: provider exceptions may contain credentials.
                outcome["error_type"] = type(error).__name__
            outcome["observed_seconds"] = time.monotonic() - started
            _append(folder / "outcomes.jsonl", outcome)
            outcomes[key] = outcome
            if outcome["status"] == "unknown":
                raise UnknownAttempt("new send result unknown; retained, no retry or later call") from None
        return summarize(folder, fixture_only=fixture_only)
