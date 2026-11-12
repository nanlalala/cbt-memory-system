"""Three fixed fictional cases for a model-matched readout replication in hosted Colab."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import cactus_materials
from auto_eval_v2.context_assembly import ARMS, assemble
from auto_eval_v2.retrieval import canonical
from run_cactus_effect import ALIAS, API_BASE, MODEL, append, atomic, locked, read_json, read_rows, stamp, strict_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
from followup_transport import ResponseFailure, generate_response

INPUT = ROOT / "docs/context-assembly-v3/mechanism_cases.json"
TARGETS = {"correction": 2, "pause": True, "proposal": False}
LIMIT = 64
# Hashes of the actual renderer-level requests observed in the hosted Qwen pilot.
# Replacing provider options must preserve each previously measured messages array.
PREVIOUS = (
    "21eb23dd9a697270af05b202a6af69a851954a168bc0763cf00371aa26e723a0",
    "75f0da6db839908ba154671bc794095ae05b175ba5ce707394c791aaec51458a",
    "015f5dac8cc75fd6032fa961d43b6665009d1e7ce2bb230c649eb1fed91ff691",
    "d1081a1fc0776c013118e7e87f9852cf7651f783f48473117b2c65d91c5af046",
    "87ee5675cd7771cb7b3bb56c00c5c8cb79cc8f3bbaa414d0089bbc25b57e92a0",
    "20c820fa51dc02fa2d1913a2523847203a5d9537807c6fce4af0a3f20943b0d5",
    "bb064906577470fde74a31638d099ce4bb971a63afb756c2f5ad3f9d63ead0e2",
    "05ea26eb97c49e0a887d779244caaa13d26a6c53364142240564e5c2e29e55b9",
    "d0bb5e57dfe62429cbb1290ae4b2cceccf5766df27384e9b0e7ced9194f96dd3",
)


def digest(text):
    return sha256(text.encode("utf-8")).hexdigest()


def plan():
    rows = []
    for case in read_json(INPUT):
        cid = case["case_id"]
        subject = "invented:" + cid
        sources = [dict(turn, source_order=i, source_id=f"{cid}:s{i}", subject_id=subject,
                        retained=True, use_allowed=True, external_allowed=True)
                   for i, turn in enumerate(case["sources"])]
        source_ids = [s["source_id"] for s in sources]
        spans = [dict(source_id=s["source_id"], start=0, end=len(s["content"]), text=s["content"]) for s in sources]
        verifier = "invented-fixture-reviewed-v1"
        memory = dict(memory_id=cid, subject_id=subject, version=1, active=True, deleted=False,
                      valid_at_cutoff=True, superseded_by=None, support="verified_projection",
                      verifier_revision=verifier, epistemic_status="supported_state",
                      summary=case["supported_summary"], source_ids=source_ids, evidence=spans)
        for arm in ARMS:
            package = assemble(sources=sources, selected_ids=source_ids, memories=[memory], recent=[],
                               current_input=case["current_input"], subject_id=subject,
                               cutoff_order=len(sources)-1, max_request_chars=6000, arm=arm,
                               verifier_revision=verifier, request_options={"model": MODEL, "temperature": 0,
                                   "max_tokens": LIMIT, "thinking": {"type": "disabled"}})
            messages = package["request"]["messages"]
            previous = dict(model="Qwen/Qwen2.5-1.5B-Instruct", temperature=0, max_tokens=64, messages=messages)
            if digest(canonical(previous)) != PREVIOUS[len(rows)]:
                raise ValueError("messages differ from hosted pilot")
            # Preserve HTTP key ordering and spacing used by followup_transport.generate_response.
            body = dict(model=MODEL, messages=messages, temperature=0, max_tokens=LIMIT, thinking={"type": "disabled"})
            wire = json.dumps(body)
            if body != package["request"] or len(wire) > 6000:
                raise ValueError("unexpected wire payload or size")
            rows.append(dict(case_id=cid, arm=arm, request=body, trace=package["trace"],
                             wire_sha256=digest(wire), wire_chars=len(wire)))
    if len(rows) != len(PREVIOUS):
        raise ValueError("incomplete case grid")
    return {"study":"deepseek-context-mechanism-v1", "cases":"three fictional development cases with authored summaries",
            "model":MODEL, "required_returned_model":ALIAS, "endpoint":API_BASE, "max_attempts":9,
            "max_output_tokens_per_request":LIMIT, "targets_sha256":digest(canonical(TARGETS)),
            "rows":rows}


def correct(answer, target):
    try:
        parsed = strict_json(answer)
    except ValueError:
        return False
    return (type(parsed) is dict and set(parsed) == {"answer"} and
            type(parsed["answer"]) is type(target) and parsed["answer"] == target)


def run(directory, approved_attempts):
    cactus_materials.require_colab()
    expected = plan()
    with locked(directory):
        if read_json(directory / "plan.json") != expected or approved_attempts != expected["max_attempts"]:
            raise ValueError("supply approved nine-attempt ceiling for this exact prepared plan")
        journal = directory / "journal.jsonl"
        if read_rows(journal):
            raise RuntimeError("existing paid attempt: inspect receipts before any new run")
        from google.colab import userdata
        key = userdata.get("Capstone")
        if not key:
            raise RuntimeError("no credential available")
        results = []
        for index, row in enumerate(expected["rows"]):
            messages = row["request"]["messages"]
            identity = {k: row[k] for k in ("case_id", "arm", "wire_sha256")}
            append(journal, dict(identity, event="intent", index=index, utc=stamp()))
            try:
                answer = generate_response(messages, model=MODEL, allowed_model=ALIAS, api_base=API_BASE,
                                           key=key, max_response_tokens=LIMIT)
            except ResponseFailure as exc:
                append(journal, dict(identity, event="rejected", reason=exc.failure_kind,
                                     usage=exc.reported_usage, utc=stamp()))
                raise
            except Exception:
                append(journal, dict(identity, event="outcome_unknown", utc=stamp()))
                raise RuntimeError("unknown send; inspect journal") from None
            result = dict(identity, event="received", **answer, utc=stamp(),
                          strict_correct=correct(answer["hypothesis"], TARGETS[row["case_id"]]))
            append(journal, result)
            results.append(result)
        atomic(directory / "results.json", {"by_arm":{arm:sum(r["strict_correct"] for r in results
               if r["arm"] == arm) for arm in ARMS}, "planned_per_arm":3,"rows":results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "send"])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--approved-attempts", type=int)
    args = parser.parse_args()
    cactus_materials.require_colab()
    if args.command == "send":
        run(args.directory, args.approved_attempts)
        return
    expected = plan()
    args.directory.mkdir(exist_ok=False, parents=True)
    atomic(args.directory / "plan.json", expected)
    print(json.dumps({"prepared":len(expected["rows"]), "outbound_calls":0, "model":MODEL,
                      "plan_sha256":digest(canonical(expected)),
                      "wire_chars":[r["wire_chars"] for r in expected["rows"]]}))


if __name__ == "__main__":
    main()
