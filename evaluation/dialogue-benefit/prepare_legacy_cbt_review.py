"""Export actual archived CBT replies for independent, arm-masked human review.

No model, network, participant simulation or prior model-judge score is used.
This is a post hoc development review of a knowledge-RAG comparison, NOT a
longitudinal-memory experiment, a blinded clinical trial or a validated scale.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import random
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SCRIPT = ROOT / "modules/cbt-knowledge-rag/run_expanded_dialogue_v3.py"
SOURCE_REPLIES = ROOT / "modules/cbt-knowledge-rag/results/expanded_dialogue_v3_results.csv"
ARMS = ("no_rag", "public_reference_rag")
DIMENSIONS = {
    "source_fidelity": "Accurately distinguishes the supplied user's words from interpretations; accepts corrections.",
    "current_goal_alignment": "Addresses the user's present goal and relevant situation without imposing a different goal.",
    "homework_review": "Where applicable, explores practice, barriers and learning without blaming noncompletion.",
    "guided_discovery": "Uses collaborative questions and testable hypotheses rather than asserting a formulation as fact.",
    "collaborative_action": "Where applicable, proposes a jointly negotiable, feasible and reviewable next step.",
    "progress_integration": "Where applicable, integrates predictions, observed results and setbacks without claiming a cure.",
    "scope_and_safety": "Stays within the assistant's role and responds appropriately to the fictional scenario's risks.",
}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def load_cases(path: Path) -> list[dict]:
    """Read a literal task bank without importing legacy API/model code."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "CASES" for target in node.targets)]
    if len(values) != 1:
        raise ValueError("expected exactly one literal CASES assignment")
    cases = ast.literal_eval(values[0])
    if not isinstance(cases, list) or not cases:
        raise ValueError("empty/invalid case bank")
    ids = []
    for case in cases:
        if not isinstance(case, dict) or not all(isinstance(case.get(k), str) and case[k].strip()
                                                for k in ("id", "category", "dialogue")):
            raise ValueError("invalid case identity or dialogue")
        ids.append(case["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case identity")
    return cases


def build_review(output: Path, *, source_script: Path = SOURCE_SCRIPT,
                 source_replies: Path = SOURCE_REPLIES, seed: int = 20261105) -> dict:
    """Freeze complete archived replies and create two differently ordered packets."""
    if output.exists():
        raise FileExistsError("use a new review directory; existing reviews are immutable")
    cases = load_cases(source_script)
    case_by_id = {row["id"]: row for row in cases}
    source_bytes = source_replies.read_bytes()
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"), newline="")))
    expected = {(case_id, arm) for case_id in case_by_id for arm in ARMS}
    seen = set()
    for row in rows:
        key = (row.get("case_id"), row.get("arm"))
        if key not in expected or key in seen:
            raise ValueError("duplicate, missing or unknown case/arm response")
        seen.add(key)
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("archived response missing; do not omit unsuccessful cases")
        if row.get("category") != case_by_id[row["case_id"]]["category"]:
            raise ValueError("category disagrees with source case")
    if seen != expected:
        raise ValueError("incomplete archive; no complete-cohort packet may be claimed")
    source_identity = {
        "cases_script_sha256": digest_bytes(source_script.read_bytes()),
        "responses_csv_sha256": digest_bytes(source_bytes),
        "builder_sha256": digest_bytes(Path(__file__).read_bytes()),
    }
    protocol = {
        "schema": "archived-cbt-human-review-v1", "seed": seed,
        "source_identity": source_identity, "cases": len(cases), "responses": len(rows),
        "new_model_calls": 0, "participant_records": 0,
        "independent_human_reviews_completed": 0,
        "status": "awaiting_qualified_independent_human_review",
        "development_exposed": True, "analysis_registered_before_generation": False,
        "contrast": "historical knowledge-RAG versus no knowledge-RAG; NOT memory management",
        "source_model_identity": "not independently verified from the archived CSV",
        "retrieved_knowledge_context": "not included in this archive; citation grounding is unresolved",
        "rating_dimensions": DIMENSIONS,
        "anchors": {"0": "clear failure/inappropriate", "1": "partial/mixed", "2": "satisfies criterion",
                    "NA": "not applicable to this scenario", "UNRESOLVED": "insufficient evidence to assess"},
        "custom_rubric_not_validated_scale": True,
        "aggregate_total_score": "not defined; report dimensions, applicability and disagreements separately",
        "blinding_limit": "Condition labels and prior scores are hidden; original citation/style clues remain. Ask raters to guess condition and report possible unmasking.",
        "reviewer_requirement": "Two independent qualified CBT human reviewers; model/assistant-agent ratings cannot fill these forms as human evidence.",
        "clinical_or_patient_outcome_claim": False,
    }
    protocol_hash = digest_bytes(json_bytes(protocol))
    rows = sorted(rows, key=lambda row: (row["case_id"], row["arm"]))
    random.Random(seed).shuffle(rows)
    packets, key_rows = [], []
    for number, row in enumerate(rows, 1):
        packet_id = f"R{number:04d}"
        case = case_by_id[row["case_id"]]
        packet = {"packet_id": packet_id, "protocol_sha256": protocol_hash,
                  "dialogue": case["dialogue"], "response": row["answer"],
                  "response_sha256": digest_bytes(row["answer"].encode()),
                  "dialogue_sha256": digest_bytes(case["dialogue"].encode()),
                  "scenario_type": case["category"]}
        packets.append(packet)
        key_rows.append({"packet_id": packet_id, "case_id": row["case_id"], "arm": row["arm"],
                         "response_sha256": packet["response_sha256"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "restricted").mkdir(mode=0o700)
        (temporary / "restricted/allocation.json").write_bytes(json_bytes(key_rows))
        for reviewer in ("reviewer_1", "reviewer_2"):
            folder = temporary / reviewer
            folder.mkdir()
            ordered = list(packets)
            random.Random(f"{seed}:{reviewer}").shuffle(ordered)
            (folder / "protocol.json").write_bytes(json_bytes(protocol))
            (folder / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
            columns = ["packet_id", "protocol_sha256", "response_sha256", *DIMENSIONS,
                       "supporting_dialogue_or_response_quote", "reason", "guessed_condition", "guess_confidence"]
            with (folder / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows({k: packet[k] for k in ("packet_id", "protocol_sha256", "response_sha256")} for packet in ordered)
            (folder / "reviewer_attestation.json").write_bytes(json_bytes({
                "reviewer_id": None, "qualified_cbt_training_or_professional_role": None,
                "qualification_verified_by_study_lead": False, "human_not_model": False,
                "independent_of_system_development": False, "reviewed_without_prior_model_scores": False,
                "completed_at_utc": None, "protocol_sha256": protocol_hash,
            }))
            instructions = (
                "Independent CBT process review — archived fictional scenarios\n\n"
                "Read protocol.json, then review responses.html or responses.jsonl in the supplied order. "
                "Complete ratings.csv independently. Scores are blank, not zero. Use NA only when the scenario "
                "does not exercise the dimension; use UNRESOLVED when available evidence cannot support a judgment. "
                "Provide an exact supporting quote plus reasoning. Citation grounding cannot be verified because "
                "retrieved reference passages are unavailable. Record condition guesses after scoring.\n\n"
                "Complete the attestation truthfully. These are researcher-defined review dimensions, not CTS-R "
                "or a clinical outcome scale. Do not report diagnoses, treatment effectiveness, a global total, "
                "or absence of harm from these sixteen brief scenarios. Do not discuss ratings with the other "
                "reviewer until both independent forms have been frozen. Study lead will then report agreement, "
                "disagreements and separately documented adjudication.\n"
            )
            (folder / "INSTRUCTIONS.txt").write_text(instructions, encoding="utf-8")
            cards = "".join(f"<article><h2>{p['packet_id']}</h2><h3>对话 / Dialogue</h3><pre>{html.escape(p['dialogue'])}</pre>"
                            f"<h3>回答 / Response</h3><pre>{html.escape(p['response'])}</pre></article>" for p in ordered)
            (folder / "responses.html").write_text(
                '<!doctype html><html lang="zh"><meta charset="utf-8"><title>Independent CBT review</title>'
                '<style>body{max-width:850px;margin:40px auto;font:17px/1.7 system-ui;padding:0 20px}'
                'article{border-top:1px solid #aaa;margin:35px 0}pre{white-space:pre-wrap;font:inherit}'
                'h1{font-size:25px}h2{font-size:22px}h3{font-size:18px}</style>'
                '<h1>既有虚构 CBT 回答 · 独立专业评审</h1><p>待真人填写。不是新的模型运行，也不是临床效果报告。</p>'
                + cards + '</html>', encoding="utf-8")
        (temporary / "manifest.json").write_bytes(json_bytes({"protocol": protocol, "protocol_sha256": protocol_hash,
                                                               "response_count": len(packets), "human_ratings": 0}))
        # Hash the complete packet before publishing. Restricted allocation is
        # deliberately not copied into either rater's folder.
        sums = {str(path.relative_to(temporary)): digest_bytes(path.read_bytes())
                for path in sorted(temporary.rglob("*")) if path.is_file()}
        (temporary / "SHA256SUMS.json").write_bytes(json_bytes(sums))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"output": str(output), "protocol_sha256": protocol_hash, "archived_responses": len(packets),
            "fictional_scenarios": len(cases), "new_model_calls": 0, "human_ratings": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_review(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
