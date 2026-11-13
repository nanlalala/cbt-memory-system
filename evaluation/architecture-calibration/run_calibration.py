"""Frozen published-label calibration for a different-family automatic judge.

Real data acquisition, preparation and model calls require genuine hosted Colab.
The pure packet projection is covered only with tiny invented fixtures locally.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
from auto_eval_v2 import calibration, calibration_prompt
from auto_eval_v2.run_colab import require_hosted
from architecture_retest import journal

VERSION = "published-ctrs-safety-calibration-v4"
SOURCE_BYTES = 1446326
SOURCE_COUNT = 116
SOURCE_BASE = ("https://huggingface.co/datasets/" + calibration.REPOSITORY +
               "/resolve/" + calibration.REVISION + "/")
PROVIDER = {"provider": "https://tokendance.space/gateway/v1", "model": "qwen3.5-plus",
            "returned_model": "qwen3.5-plus-2026-04-20", "model_family": "qwen"}
FAMILIES = {"PHI_3_5": "phi", "CLAUDE_3_7": "claude", "GEMINI_2_0": "gemini",
            "DEEPSEEK": "deepseek", "O3_MINI": "openai", "QWEN3_4B_LOCAL": "qwen",
            "LLAMA_4_SCOUT": "llama", "O1": "openai"}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _download(name, maximum):
    with urlopen(SOURCE_BASE + name, timeout=60) as response:
        body = response.read(maximum + 1)
    if len(body) > maximum:
        raise ValueError("published calibration download exceeds declared size bound")
    return body


def verify_bank(bank):
    bank = Path(bank).resolve()
    path = bank / "assessor_only/therapyjudgebench.jsonl"
    if path.is_symlink() or bank.is_symlink():
        raise ValueError("symlinked calibration source")
    manifest = json.loads((bank / "source_bank_manifest.json").read_text())
    binding = calibration.verify_source_manifest(manifest, file_hash(path))
    if path.stat().st_size != SOURCE_BYTES:
        raise ValueError("published source byte count changed")
    for name in ("README.md", "LICENSE", "DATASHEET.md"):
        if file_hash(bank / name) != manifest["metadata_sha256"][name]:
            raise ValueError("frozen source metadata changed")
    return binding


def acquire(bank):
    require_hosted()
    bank = Path(bank).resolve()
    if not bank.is_relative_to(Path("/content")) or bank == Path("/content"):
        raise ValueError("hosted source-bank subdirectory required")
    if (bank / "source_bank_manifest.json").exists():
        return verify_bank(bank)
    bank.mkdir(parents=True, exist_ok=True)
    (bank / "assessor_only").mkdir(exist_ok=True)
    metadata = {name: _download(name, 65536) for name in ("README.md", "LICENSE", "DATASHEET.md")}
    if b"license: cc-by-4.0" not in metadata["README.md"] or b"2.0.0" not in metadata["README.md"]:
        raise ValueError("pinned calibration permission/version metadata differs")
    body = _download("therapyjudgebench.jsonl", SOURCE_BYTES)
    if len(body) != SOURCE_BYTES or sha256(body).hexdigest() != calibration.SOURCE_SHA256:
        raise ValueError("pinned published calibration bytes differ")
    for name, data in metadata.items():
        (bank / name).write_bytes(data)
    (bank / "assessor_only/therapyjudgebench.jsonl").write_bytes(body)
    manifest = {"repository": calibration.REPOSITORY, "revision": calibration.REVISION,
                "dataset_version": "2.0.0", "license": "CC-BY-4.0",
                "files": [{"filename": "therapyjudgebench.jsonl", "sha256": calibration.SOURCE_SHA256}],
                "metadata_sha256": {name: sha256(data).hexdigest() for name, data in metadata.items()}}
    journal.atomic(bank / "source_bank_manifest.json", manifest)
    return verify_bank(bank)


def records_from_bank(bank):
    require_hosted()
    verify_bank(bank)
    records = [json.loads(line) for line in (Path(bank) / "assessor_only/therapyjudgebench.jsonl").read_text().splitlines()
               if line.strip()]
    if len(records) != SOURCE_COUNT or len({r["id"] for r in records}) != SOURCE_COUNT:
        raise ValueError("published source roster changed")
    return records


def project_packets(records):
    """Read visible messages and explicit IDs, never rating values or hidden profiles."""
    packets, excluded, seen = [], [], set()
    for record in records:
        case_id, model = record["id"], record["therapist_model"]
        if type(case_id) is not str or not case_id or case_id in seen or model not in FAMILIES:
            raise ValueError("unique published identities and known therapist families required")
        seen.add(case_id)
        if FAMILIES[model] == PROVIDER["model_family"]:
            excluded.append({"case_id": case_id, "reason": "same_therapist_and_judge_model_family"})
            continue
        profile = record.get("patient_profile")
        if type(profile) is not str or not profile:
            raise ValueError("explicit published profile ID required")
        packets.append({"case_id": case_id, "source_family": "published-profile:" + profile,
                        "answer_model_family": FAMILIES[model],
                        "messages": calibration.project_dialogue(record)["messages"]})
    return sorted(packets, key=lambda r: r["case_id"]), excluded


def code_hashes():
    files = [Path(__file__), *sorted((ROOT / "evaluation/cbt-memory-effect/auto_eval_v2").glob("*.py")),
             *sorted((ROOT / "evaluation/cbt-memory-effect/architecture_retest").glob("*.py"))]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def prepare(bank, folder):
    require_hosted()
    bank, folder = Path(bank).resolve(), Path(folder).resolve()
    if not folder.is_relative_to(Path("/content")) or folder == Path("/content"):
        raise ValueError("new hosted calibration directory required")
    records = records_from_bank(bank)
    packets, excluded = project_packets(records)
    logical_judge = {k: PROVIDER[k] for k in ("provider", "model", "model_family")}
    plan = calibration_prompt.plan_quality(packets, logical_judge, repeat_fraction=0.1, seed=20261113)
    for item in plan:
        item["payload"] = {"model": PROVIDER["model"], "messages": item["messages"],
                           "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": 2048}
        if len(json.dumps(item["payload"])) > 48000:
            raise ValueError("full published-dialogue judge request exceeds fixed cap")
    # A repeat is a separate provider attempt while retaining the identical rated input.
    # Its separate purpose creates a separate exact-wire journal identity.
    for item in plan:
        item["purpose"] = "calibration" if item["repeat_of"] is None else "calibration_repeat:" + item["item_id"]
    maximum = len({journal.digest({"purpose": p["purpose"], "payload": p["payload"]}) for p in plan})
    protocol = {"version": VERSION, "provider": PROVIDER, "bank_path": str(bank),
                "cases_sha256": journal.digest(packets), "plan_sha256": journal.digest(plan),
                "published_source_sha256": calibration.SOURCE_SHA256,
                "published_source_count": SOURCE_COUNT, "eligible_cases": len(packets), "excluded": excluded,
                "grouping": "exact published patient_profile IDs only; parent-family lineage unavailable",
                "source_profile_group_count": len({p["source_family"] for p in packets}),
                "split": "fixed-prompt evaluation of all different-family published dialogues; no prompt tuning",
                "repeat_fraction": 0.1, "seed": 20261113, "maximum_attempts": maximum,
                "request_limits": {"answer": 48000, "preparation": 48000},
                "code_sha256": code_hashes(), "human_reviews_added": 0,
                "authorization_reference": "user approved architecture update and full previous experiment rerun"}
    protocol["protocol_sha256"] = journal.digest(protocol)
    folder.mkdir(parents=True, exist_ok=False)
    journal.atomic(folder / "cases.json", packets)
    journal.atomic(folder / "plan.json", plan)
    journal.atomic(folder / "protocol.json", protocol)
    return {"status": "prepared", "eligible_cases": len(packets), "excluded_cases": len(excluded),
            "maximum_attempts": maximum, "protocol_sha256": protocol["protocol_sha256"]}


def load(folder):
    require_hosted()
    folder = Path(folder).resolve()
    protocol = json.loads((folder / "protocol.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    cases = json.loads((folder / "cases.json").read_text())
    if (protocol["version"] != VERSION or protocol["provider"] != PROVIDER or
        protocol["code_sha256"] != code_hashes() or protocol["cases_sha256"] != journal.digest(cases) or
        protocol["plan_sha256"] != journal.digest(plan) or protocol["protocol_sha256"] != journal.digest(
            {k:v for k,v in protocol.items() if k != "protocol_sha256"})):
        raise ValueError("frozen source/code/model/plan changed")
    verify_bank(protocol["bank_path"])
    return folder, protocol, cases, plan


def summarize(folder):
    folder, protocol, cases, plan = load(folder)
    intents = journal.rows(folder / "intents.jsonl")
    outcomes = {r["key"]: r for r in journal.rows(folder / "outcomes.jsonl")}
    by_case = {p["case_id"]: p for p in cases}
    ratings, states = [], []
    for item in plan:
        key = journal.digest({"purpose": item["purpose"], "payload": item["payload"]})
        received = outcomes.get(key)
        review = calibration_prompt.validate_review(received["text"], by_case[item["case_id"]]) if received and received["status"] == "received" else None
        row = {"item_id": item["item_id"], "case_id": item["case_id"], "repeat_of": item["repeat_of"],
               "delivery": received["status"] if received else "pending", "review": review}
        states.append(row)
        if item["repeat_of"] is None:
            if review and review["accepted"] and review["status"] == "completed":
                assessment = review["assessment"]
                ratings.append({"case_id": item["case_id"], "status": "completed", "ctrs": assessment["ctrs"], "safety": assessment["safety"]})
            else:
                ratings.append({"case_id": item["case_id"], "status": "missing" if not received else "unresolved" if review and review["status"] == "unresolved" else "invalid"})
    complete = len(outcomes) == protocol["maximum_attempts"] and all(r["status"] == "received" for r in outcomes.values())
    summary = {"protocol_sha256": protocol["protocol_sha256"], "planned_cases": len(cases),
               "planned_attempts": protocol["maximum_attempts"], "attempts": len(intents),
               "outcomes": len(outcomes), "complete": complete, "items": states,
               "tokens": {field: sum(r.get("usage", {}).get(field, 0) for r in outcomes.values())
                          for field in ("prompt_tokens", "completion_tokens", "total_tokens")},
               "scope": "published nine-aspect CTRS and four safety labels; custom R1/R3 uncalibrated"}
    summary["repeat_consistency"] = repeat_consistency(states)
    if complete:
        records = {r["id"]: r for r in records_from_bank(protocol["bank_path"])}
        gold = [{"case_id": p["case_id"], "ctrs": records[p["case_id"]]["expert_rating"]["ctrs_ratings"],
                 "safety": records[p["case_id"]]["expert_rating"]["safety_ethics"]} for p in cases]
        summary["calibration"] = calibration.calibrate(gold, ratings)
    journal.atomic(folder / "summary.json", summary)
    return summary


def repeat_consistency(states):
    by_id = {row["item_id"]: row for row in states}
    repeats = [row for row in states if row["repeat_of"] is not None]
    result = {"planned_repeats": len(repeats), "completed_review_pairs": 0,
              "ctrs_mean_absolute_difference": {}, "safety_disagreement_rate": {}}
    differences = {dimension: [] for dimension in (*calibration.CTRS, *calibration.SAFETY)}
    for repeat in repeats:
        original = by_id[repeat["repeat_of"]]
        pair = [row.get("review") for row in (original, repeat)]
        if not all(review and review["accepted"] and review["status"] == "completed" for review in pair):
            continue
        result["completed_review_pairs"] += 1
        for section, dimensions in (("ctrs", calibration.CTRS), ("safety", calibration.SAFETY)):
            for dimension in dimensions:
                a, b = [review["assessment"][section][dimension] for review in pair]
                if a is not None and b is not None:
                    differences[dimension].append(abs(a-b) if section == "ctrs" else int(a != b))
    for section, dimensions in (("ctrs_mean_absolute_difference", calibration.CTRS),
                                ("safety_disagreement_rate", calibration.SAFETY)):
        result[section] = {dimension: {"n_compared": len(differences[dimension]),
                          "value": sum(differences[dimension])/len(differences[dimension])
                          if differences[dimension] else None} for dimension in dimensions}
    return result


def run(folder, transport, *, max_new_calls=None, workers=4):
    folder, protocol, cases, plan = load(folder)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("one to four fixed concurrent workers required")
    cap = protocol["maximum_attempts"] if max_new_calls is None else max_new_calls
    if type(cap) is not int or cap < 0:
        raise ValueError("nonnegative invocation cap required")
    with journal.lock(folder):
        calls = journal.Journal(folder, protocol, transport, cap)
        def process(item):
            return calls.call({"case_id": item["case_id"]}, item["purpose"], item["payload"])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(process, plan))
    return summarize(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire", "prepare", "run", "summarize"))
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-calls", type=int)
    args = parser.parse_args()
    require_hosted()
    if args.command == "acquire":
        result = acquire(args.bank)
    elif args.command == "prepare":
        result = prepare(args.bank, args.output)
    elif args.command == "summarize":
        result = summarize(args.output)
    else:
        from architecture_retest.transport import HostedGatewayTransport
        from google.colab import userdata
        transport = HostedGatewayTransport(PROVIDER, userdata.get("Capstone"))
        result = run(args.output, transport, max_new_calls=args.max_new_calls)
    print(json.dumps({k:v for k,v in result.items() if k not in {"items", "calibration"}}, sort_keys=True))


if __name__ == "__main__":
    main()
