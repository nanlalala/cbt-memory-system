"""Minimally revise the verified v8 thesis after complete Colab scoring.

This authoring script reads authenticated, already scored result JSON; it never
loads a reader, runs an experiment, changes reference answers or manufactures
scores.  Independent expected report SHA256 values are required on the command
line.  A matching digest establishes byte identity with the supplied provenance,
not cryptographic proof that a clinical experiment occurred.  The script refuses
fixture/incomplete reports and writes a new working DOCX only after validation.

Before executing an edit, run the documents artifact marker in its own tool call.
Afterwards render with the packaged render_docx.py and inspect every page before
publishing.  This script never writes to ChatGPT Library or GitHub itself.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re


TITLE = "A Hierarchical Memory System for Psychotherapy Reflection and Care Tracking"
SOURCE_V8_SHA256 = "6aaee7eb8630ce2d76d75c7c7d337550e1e9b8ec37560a5d235f2936686fff1b"
SOURCE_LIBRARY_ID = "libfile_fde3_f-7a3ae0a52ee081918951eb2fd98525ca_7e9d829b317c8191a098e8015cc46514"
SOURCE_VERSION = 8
ARMS = ("raw_history_rag", "managed_memory_rag")
FAMILIES = {
    "target_correction": "Corrected target",
    "cumulative_repeat": "Repeated cumulative count",
    "proposal_not_agreement": "Proposal and agreement",
    "outcome_not_progress": "Progress and reported benefit",
}
HEX = re.compile(r"[0-9a-f]{64}\Z")
MATERIAL = "invented_controlled_mechanism_not_natural_CBT"
EVIDENCE_VERSION = "complete-controlled-paper-evidence-v1"
MAX_EVIDENCE_BYTES = 32_000_000
# These values are previous verified Colab observations, not provisional scores.
# The new task-state results must still be supplied and validated at execution.
SOURCE_PILOT = {
    "source_kind": "previously_verified_hosted_colab_development_run",
    "cases": 3, "source_families": 1, "answer_calls": 9,
    "answer_tokens": 11239, "api_seconds_sum": 18.523712973998045,
    "original_correct": {"short_term": 1, "raw_history_rag": 2, "managed_memory_rag": 2},
    "common_validator_replay_correct": {"short_term": 1, "raw_history_rag": 3, "managed_memory_rag": 3},
    "repair_additional_model_calls": 0,
    "transformed_rows_sha256": "b123c1e283d41e76ff24b7fe117c367c30e4c2442c55f3904477c4fc319d7309",
    "protocol_sha256": "a600c93183d23247824cb30fe077ef7291996e4edae32865297f440e22c965c6",
    "result_report": "https://chatgpt.com/library?file_id=libfile_fde3_f-9f30125726c88191b0e10d63dee2a29d_7e9d829b317c8191a098e8015cc46514&account_id=personal",
}


def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def read_json(path, expected_sha256=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000:
        raise ValueError("bounded_regular_json_file_required")
    data = path.read_bytes()
    if expected_sha256 is not None:
        if not HEX.fullmatch(expected_sha256) or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("independently_recorded_result_digest_mismatch")
    return json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def read_answer_rows(path, expected_sha256):
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000
            or filehash(path) != expected_sha256):
        raise ValueError("bound_original_answer_archive_required")
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        raise ValueError("incomplete_answer_journal_tail")
    return [json.loads(line, object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))
            for line in data.splitlines() if line.strip()]


def validate_reader_binding(report, protocol, run, completion, answers):
    expected = {"fixture_only": False, "maximum_global_generations": 64,
                "maximum_input_tokens": 4096, "max_new_tokens": 64,
                "do_sample": False, "use_cache": True, "external_inference_calls": 0,
                "secret_access": False, "new_human_reviews": 0, "scoring_targets_read": False}
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("fixed_reader_settings_required")
    model = run.get("model", {})
    if (model.get("model_id") != "Qwen/Qwen2.5-1.5B-Instruct"
            or model.get("model_revision") != "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"):
        raise ValueError("fixed_reader_model_identity_required")
    for key in ("generation_config_sha256", "chat_template_sha256", "load_protocol_sha256",
                "model_files_sha256", "load_receipt_sha256"):
        if not HEX.fullmatch(model.get(key, "")):
            raise ValueError("missing_reader_model_binding")
    if (run.get("prepared_protocol_sha256") != report.get("protocol_file_sha256")
            or run.get("requests_file_sha256") != protocol.get("requests_file_sha256")
            or run.get("gate_identity") != protocol.get("code_identity")
            or not HEX.fullmatch(run.get("reader_code_sha256", ""))):
        raise ValueError("reader_preparation_binding_mismatch")
    run_hash = json_digest(run)
    if (completion.get("complete") is not True or completion.get("fixture_only") is not False
            or completion.get("logical_planned") != 48 or completion.get("logical_completed") != 48
            or completion.get("run_protocol_sha256") != run_hash
            or completion.get("answers_sha256") != report.get("answers_file_sha256")
            or completion.get("new_provider_calls") != 0 or completion.get("new_human_reviews") != 0
            or completion.get("secret_access") is not False):
        raise ValueError("completed_reader_archive_binding_mismatch")
    grid = {(row["case_id"], row["arm"]): row for row in report["per_question"]}
    seen = set()
    if type(answers) is not list or len(answers) != 48:
        raise ValueError("all_original_reader_answers_required")
    for row in answers:
        key = (row.get("case_id"), row.get("arm"))
        if key not in grid or key in seen:
            raise ValueError("duplicate_or_unplanned_original_answer")
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        response = row.get("response_text")
        if (row.get("record_sha256") != json_digest(body) or type(response) is not str
                or row.get("response_sha256") != hashlib.sha256(response.encode()).hexdigest()
                or row.get("run_protocol_sha256") != run_hash
                or row.get("status") != "completed" or row.get("source_checks_passed") is not True
                or row.get("scored") is not False or not HEX.fullmatch(row.get("canonical_key", ""))
                or any(row.get(field) != grid[key][field] for field in ("request_sha256", "response_sha256"))):
            raise ValueError("original_answer_receipt_or_score_binding_mismatch")
        seen.add(key)
    return {"model": model, "reader_code_sha256": run["reader_code_sha256"],
            "run_protocol_sha256": run_hash,
            "logical_completed": completion["logical_completed"],
            "global_generated_attempts": completion.get("global_generated_attempts"),
            "generation_cost_owning_run": completion.get("generation_cost_owning_run"),
            "reused_canonical_receipts_from_other_runs": completion.get("reused_canonical_receipts_from_other_runs")}


def export_complete_evidence(args, result):
    """Archive complete original bytes; no count-only replacement for evidence.

    This writes a gzip file and prints metadata only.  It does not transfer it,
    construct a data URL, or bypass browser/file-service restrictions.  A caller
    may preserve the file through an authorized native cloud storage operation.
    """
    try:
        hosted = (bool(os.environ.get("COLAB_RELEASE_TAG")) and Path("/content").is_dir()
                  and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        hosted = False
    if not hosted:
        raise RuntimeError("genuine_hosted_colab_required_for_result_export")
    destination = Path(args.export_evidence).resolve()
    try:
        destination.relative_to(Path("/content"))
    except ValueError as error:
        raise ValueError("evidence_export_must_stay_in_hosted_content") from error
    paths = {"before_report": args.before_report, "before_protocol": args.before_protocol,
             "after_report": args.after_report, "after_protocol": args.after_protocol}
    for role in ("before_reader_run", "before_reader_completion", "before_answers",
                 "after_reader_run", "after_reader_completion", "after_answers"):
        paths[role] = getattr(args, role)
    if args.professional_summary:
        paths["professional_summary"] = args.professional_summary
    if args.calibration_manifest:
        paths["calibration_manifest"] = args.calibration_manifest
        paths["calibration_audit"] = args.calibration_audit
    allowed_extras = {"before_reader_run", "before_reader_completion", "before_answers",
                      "after_reader_run", "after_reader_completion", "after_answers",
                      "reader_global_ledger", "reader_canonical_answers", "reader_cost_summary"}
    for supplied in args.evidence_extra:
        role, separator, filename = supplied.partition("=")
        if not separator or role not in allowed_extras or role in paths:
            raise ValueError("unexpected_or_duplicate_evidence_role")
        paths[role] = filename
    entries = []
    for role, filename in paths.items():
        path = Path(filename)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000:
            raise ValueError("bounded_regular_evidence_file_required")
        data = path.read_bytes()
        entries.append({"role": role, "file_name": path.name, "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "data_base64": base64.b64encode(data).decode("ascii")})
    bundle = {"version": EVIDENCE_VERSION, "complete_original_bytes": True,
              "source": "genuine_hosted_colab_completed_controlled_reports",
              "before_report_sha256": args.before_report_sha256,
              "after_report_sha256": args.after_report_sha256,
              "professional_summary_sha256": args.professional_summary_sha256,
              "calibration_manifest_sha256": args.calibration_manifest_sha256,
              "calibration_audit_sha256": args.calibration_audit_sha256,
              "validated_controlled_results": result, "entries": entries,
              "no_reference_content_printed": True, "new_model_calls": 0}
    body = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if len(body) > MAX_EVIDENCE_BYTES:
        raise ValueError("complete_evidence_bundle_exceeds_bound")
    compressed = gzip.compress(body, mtime=0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(compressed)
        stream.flush()
        os.fsync(stream.fileno())
    return {"evidence_path": str(destination), "bytes": len(compressed),
            "sha256": hashlib.sha256(compressed).hexdigest(),
            "entries": [{key: value for key, value in entry.items() if key != "data_base64"}
                        for entry in entries], "new_model_calls": 0, "docx_written": False}


def import_complete_evidence(bundle_path, expected_sha256, directory):
    """Verify an independently recorded archive digest and every original file."""
    bundle_path, directory = Path(bundle_path), Path(directory)
    if (not HEX.fullmatch(expected_sha256) or bundle_path.is_symlink()
            or not bundle_path.is_file() or bundle_path.stat().st_size > MAX_EVIDENCE_BYTES
            or filehash(bundle_path) != expected_sha256):
        raise ValueError("complete_evidence_digest_mismatch")
    with gzip.open(bundle_path, "rb") as stream:
        data = stream.read(MAX_EVIDENCE_BYTES + 1)
    if len(data) > MAX_EVIDENCE_BYTES:
        raise ValueError("decompressed_evidence_exceeds_bound")
    bundle = json.loads(data.decode(), object_pairs_hook=_unique)
    if bundle.get("version") != EVIDENCE_VERSION or bundle.get("complete_original_bytes") is not True:
        raise ValueError("complete_original_evidence_required")
    required = {"before_report", "before_protocol", "after_report", "after_protocol"}
    optional = {"professional_summary", "calibration_manifest", "calibration_audit",
                "before_reader_run", "before_reader_completion", "before_answers",
                "after_reader_run", "after_reader_completion", "after_answers",
                "reader_global_ledger", "reader_canonical_answers", "reader_cost_summary"}
    entries, verified = bundle.get("entries"), {}
    if type(entries) is not list or not 4 <= len(entries) <= len(required | optional):
        raise ValueError("invalid_complete_evidence_entries")
    for entry in entries:
        role = entry.get("role")
        if role not in required | optional or role in verified:
            raise ValueError("unexpected_complete_evidence_role")
        raw = base64.b64decode(entry.get("data_base64", ""), validate=True)
        if (len(raw) != entry.get("bytes") or len(raw) > 10_000_000
                or hashlib.sha256(raw).hexdigest() != entry.get("sha256")):
            raise ValueError("complete_evidence_entry_identity_mismatch")
        verified[role] = raw
    if not required.issubset(verified):
        raise ValueError("missing_complete_evidence_file")
    for role in ("before_report", "after_report", "professional_summary", "calibration_manifest", "calibration_audit"):
        if role in verified and hashlib.sha256(verified[role]).hexdigest() != bundle.get(role + "_sha256"):
            raise ValueError("complete_evidence_report_identity_mismatch")
    directory.mkdir(parents=True, exist_ok=False)
    paths = {}
    for role, raw in verified.items():
        # Never use supplied file_name as a path, even with a trusted digest.
        path = directory / (role + ".json")
        with path.open("xb") as stream:
            stream.write(raw)
        paths[role] = str(path)
    return paths, bundle


def validate_scored_report(report, protocol):
    if (report.get("generation_receipts_independently_reconciled") is not True
            or report.get("reader_run_fixture_only") is not False
            or report.get("human_reviews") != 0
            or report.get("clinical_quality_claim") is not False
            or report.get("material") != MATERIAL
            or report.get("case_count") != 24
            or report.get("template_family_count") != 4):
        raise ValueError("authenticated_complete_controlled_report_required")
    required_protocol = {
        "material": MATERIAL, "case_count": 24, "template_family_count": 4,
        "recent_messages": 6, "retrieved_source_cap_each_arm": 9,
        "max_outbound_chars": 6000, "max_new_tokens": 64,
        "raw_and_managed_share_original_source_ranking": True,
        "recent_sources_do_not_consume_raw_retrieval_slots": True,
        "independent_human_review": False, "clinical_quality_claim": False,
    }
    if any(protocol.get(key) != expected for key, expected in required_protocol.items()):
        raise ValueError("controlled_comparison_settings_changed")
    if set(protocol.get("arms", [])) != set(ARMS):
        raise ValueError("controlled_arm_identity_mismatch")
    expected_cases = {
        f"controlled-{kind}-{language}-{variant}": ("template-" + kind, language)
        for kind in FAMILIES for language in ("zh", "en") for variant in (1, 2, 3)}
    rows = report.get("per_question")
    if type(rows) is not list or len(rows) != 48:
        raise ValueError("all_forty_eight_scored_rows_required")
    grid = {}
    for row in rows:
        if type(row) is not dict or type(row.get("correct")) is not bool:
            raise ValueError("invalid_scored_row")
        cid, arm = row.get("case_id"), row.get("arm")
        if cid not in expected_cases or arm not in ARMS or (cid, arm) in grid:
            raise ValueError("duplicate_or_unplanned_scored_row")
        family, language = expected_cases[cid]
        if (row.get("source_family_id"), row.get("language")) != (family, language):
            raise ValueError("source_family_or_language_mismatch")
        for key in ("request_sha256", "response_sha256"):
            if not HEX.fullmatch(row.get(key, "")):
                raise ValueError("missing_scored_row_identity")
        if (row.get("reason") == "correct") != row["correct"]:
            raise ValueError("score_reason_disagrees")
        grid[(cid, arm)] = row
    if set(grid) != {(cid, arm) for cid in expected_cases for arm in ARMS}:
        raise ValueError("missing_scored_case_arm")
    for arm in ARMS:
        actual = sum(row["correct"] for (_, a), row in grid.items() if a == arm)
        recorded = report.get("arms", {}).get(arm, {})
        if recorded.get("planned") != 24 or recorded.get("correct") != actual:
            raise ValueError("reported_arm_totals_disagree")
    expected_pairs = {}
    for cid, (family, language) in expected_cases.items():
        raw, managed = (grid[(cid, arm)]["correct"] for arm in ARMS)
        expected_pairs[cid] = {
            "case_id": cid, "source_family_id": family, "language": language,
            "raw_correct": raw, "managed_correct": managed,
            "outcome": "managed_win" if managed and not raw else "raw_win" if raw and not managed else "tie"}
    pairs = report.get("pairs")
    if (type(pairs) is not list or len(pairs) != 24
            or len({p.get("case_id") for p in pairs}) != 24
            or any(p != expected_pairs.get(p.get("case_id")) for p in pairs)):
        raise ValueError("paired_records_disagree")
    if report.get("paired_outcomes") != dict(Counter(p["outcome"] for p in pairs)):
        raise ValueError("paired_totals_disagree")
    return grid


def validate_comparison(before, before_protocol, after, after_protocol):
    previous = validate_scored_report(before, before_protocol)
    current = validate_scored_report(after, after_protocol)
    for key in previous:
        if key[1] == "raw_history_rag":
            fields = ("request_sha256", "response_sha256", "correct", "reason")
            if any(previous[key][field] != current[key][field] for field in fields):
                raise ValueError("raw_control_changed_between_versions")
    changes = [key for key in previous if
               previous[key]["request_sha256"] != current[key]["request_sha256"]]
    return {
        "raw_correct": after["arms"]["raw_history_rag"]["correct"],
        "managed_before": before["arms"]["managed_memory_rag"]["correct"],
        "managed_after": after["arms"]["managed_memory_rag"]["correct"],
        "changed_request_rows": len(changes),
        "changed_request_arms": sorted({arm for _, arm in changes}),
        "families": [{
            "id": "template-" + kind, "label": label,
            "raw": sum(row["correct"] for (_, arm), row in current.items()
                       if arm == "raw_history_rag" and row["source_family_id"] == "template-" + kind),
            "before": sum(row["correct"] for (_, arm), row in previous.items()
                          if arm == "managed_memory_rag" and row["source_family_id"] == "template-" + kind),
            "after": sum(row["correct"] for (_, arm), row in current.items()
                         if arm == "managed_memory_rag" and row["source_family_id"] == "template-" + kind),
            "denominator": 6} for kind, label in FAMILIES.items()],
        "paired_outcomes": after["paired_outcomes"],
        "managed_errors": after["arms"]["managed_memory_rag"].get("errors", {}),
    }


def authenticated_inputs(args):
    before = read_json(args.before_report, args.before_report_sha256)
    after = read_json(args.after_report, args.after_report_sha256)
    before_protocol = read_json(args.before_protocol, before["protocol_file_sha256"])
    after_protocol = read_json(args.after_protocol, after["protocol_file_sha256"])
    result = validate_comparison(before, before_protocol, after, after_protocol)
    before_run = read_json(args.before_reader_run)
    before_completion = read_json(args.before_reader_completion)
    after_run = read_json(args.after_reader_run)
    after_completion = read_json(args.after_reader_completion)
    before_reader = validate_reader_binding(
        before, before_protocol, before_run, before_completion,
        read_answer_rows(args.before_answers, before["answers_file_sha256"]))
    after_reader = validate_reader_binding(
        after, after_protocol, after_run, after_completion,
        read_answer_rows(args.after_answers, after["answers_file_sha256"]))
    if any(before_reader[key] != after_reader[key] for key in ("model", "reader_code_sha256")):
        raise ValueError("reader_model_or_execution_code_changed_between_versions")
    result["reader_bindings"] = {"before": before_reader, "after": after_reader}
    professional = None
    if args.professional_summary:
        if not args.professional_summary_sha256:
            raise ValueError("professional_summary_independent_digest_required")
        professional = read_json(args.professional_summary, args.professional_summary_sha256)
        expected = {"dataset": "Psychotherapy-LLM/CBT-Bench",
                    "revision": "bbd37514ba2e4db46bd9633d9207a57828c2ec9d",
                    "license": "cc-by-nc-4.0", "fixture_only": False,
                    "quality_anchor_role": True, "memory_effect_eligible": False,
                    "history_records": 0, "new_model_calls": 0,
                    "new_human_reviews": 0}
        if any(professional.get(key) != value for key, value in expected.items()):
            raise ValueError("professional_material_identity_mismatch")
        n = professional.get("counts", {}).get("reference_exercises")
        if type(n) is not int or not 1 <= n <= 1000:
            raise ValueError("professional_reference_count_missing")
    calibration = None
    if args.calibration_manifest:
        if not all((args.calibration_manifest_sha256, args.calibration_audit, args.calibration_audit_sha256)):
            raise ValueError("calibration_manifest_and_audit_independent_digests_required")
        manifest = read_json(args.calibration_manifest, args.calibration_manifest_sha256)
        audit = read_json(args.calibration_audit, args.calibration_audit_sha256)
        expected = {"repository": "therapygym/therapyjudgebench",
                    "revision": "fba2e5ad29356417cde50479d823c6cf0b316754",
                    "dataset_version": "2.0.0", "license": "CC-BY-4.0",
                    "dialogue_origin": "SIMULATED_LLM_patient_and_therapist_conversations",
                    "rating_origin": "published_consolidated_labels_by_two_licensed_CBT_trained_humans",
                    "complete_sources_and_ratings_assessor_only": True,
                    "generation_inputs_created": False,
                    "new_oursystem_human_quality_scores_created": False,
                    "model_calls": 0,
                    "role": "judge_calibration_source_bank_not_longitudinal_memory_effect_result"}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("published_clinician_calibration_identity_mismatch")
        if (audit.get("dialogues") != 116 or audit.get("patient_profiles") != 18
                or audit.get("therapist_models") != 8
                or len(audit.get("ctrs_by_dimension", {})) != 11
                or len(audit.get("safety_by_dimension", {})) != 4):
            raise ValueError("published_clinician_calibration_counts_mismatch")
        calibration = {"manifest": manifest, "audit": audit}
    return result, before, after, professional, calibration


def _text(paragraph, value):
    if paragraph.runs:
        paragraph.runs[0].text = value
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(value)


def _paragraph_before(anchor, value, template):
    from docx.oxml import OxmlElement
    from docx.text.paragraph import Paragraph
    node = OxmlElement("w:p")
    if template._p.pPr is not None:
        node.append(deepcopy(template._p.pPr))
    anchor._p.addprevious(node)
    paragraph = Paragraph(node, anchor._parent)
    paragraph.style = template.style
    if template.runs and template.runs[0]._r.rPr is not None:
        run = paragraph.add_run(value)
        run._r.insert(0, deepcopy(template.runs[0]._r.rPr))
    else:
        paragraph.add_run(value)
    return paragraph


def _table_style(table, widths):
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
    table.autofit = False
    for index, width in enumerate(widths):
        table.columns[index].width = Cm(width)
    for row_index, row in enumerate(table.rows):
        properties = row._tr.get_or_add_trPr()
        if row_index == 0 and properties.find(qn("w:tblHeader")) is None:
            properties.append(OxmlElement("w:tblHeader"))
        if properties.find(qn("w:cantSplit")) is None:
            properties.append(OxmlElement("w:cantSplit"))
        for index, cell in enumerate(row.cells):
            cell.width = Cm(widths[index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            for old in list(tcpr.findall(qn("w:shd"))):
                tcpr.remove(old)
            shade = OxmlElement("w:shd")
            shade.set(qn("w:fill"), "E7E6E6" if row_index == 0 else "FFFFFF")
            tcpr.append(shade)
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right"):
                item = OxmlElement("w:" + edge)
                for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
                    item.set(qn("w:" + key), value)
                borders.append(item)
            for old in list(tcpr.findall(qn("w:tcBorders"))):
                tcpr.remove(old)
            tcpr.append(borders)
            margins = OxmlElement("w:tcMar")
            for edge in ("top", "left", "bottom", "right"):
                item = OxmlElement("w:" + edge)
                item.set(qn("w:w"), "100")
                item.set(qn("w:type"), "dxa")
                margins.append(item)
            for old in list(tcpr.findall(qn("w:tcMar"))):
                tcpr.remove(old)
            tcpr.append(margins)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.line_spacing = 1.05
                paragraph.paragraph_format.space_after = Pt(3)
                paragraph.paragraph_format.space_before = Pt(3)
                paragraph.paragraph_format.keep_with_next = False
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(10)
                    run.font.bold = row_index == 0
                    run.font.color.rgb = RGBColor(0, 0, 0)


def _comparison_table(document, anchor, rows):
    table = document.add_table(rows=1, cols=4)
    for cell, value in zip(table.rows[0].cells,
                           ("Task family", "Raw history", "Managed before", "Managed after")):
        cell.text = value
    for values in rows:
        for cell, value in zip(table.add_row().cells, values):
            cell.text = value
    _table_style(table, (6.1, 3.1, 3.1, 3.1))
    anchor._p.addprevious(table._tbl)
    return table


def _fraction(n, total=24):
    return f"{n}/{total} ({100*n/total:.1f}%)"


def build_document(args, result, before, after, professional, calibration=None):
    from docx import Document
    from docx.shared import Pt, RGBColor
    source, output = Path(args.source), Path(args.output)
    if filehash(source) != SOURCE_V8_SHA256:
        raise ValueError("exact_library_v8_source_required")
    if output.exists() or output.resolve() == source.resolve():
        raise FileExistsError("new_working_output_required")
    document = Document(source)
    paragraphs = list(document.paragraphs)
    if (len(paragraphs) != 86 or len(document.tables) != 4
            or len(document.inline_shapes) != 3 or paragraphs[0].text != TITLE):
        raise ValueError("source_document_structure_changed")
    raw, old, new = (result[key] for key in ("raw_correct", "managed_before", "managed_after"))
    next_reference, professional_citations, calibration_citations = 12, None, None
    if professional:
        professional_citations = f"[{next_reference},{next_reference + 1}]"
        next_reference += 2
    if calibration:
        calibration_citations = f"[{next_reference},{next_reference + 1}]"
        next_reference += 2
    net = new - raw
    effect = (f"{net} more correct answers than raw-history retrieval" if net > 0 else
              "the same number of correct answers as raw-history retrieval" if net == 0 else
              f"{-net} fewer correct answers than raw-history retrieval")
    # Local changes preserve the original author, images, references and styles.
    updates = {
        3: ("This project develops a hierarchical memory system for psychotherapy reflection and care tracking, "
            "using cognitive behavioural therapy (CBT) as its application setting. It combines directly supplied recent "
            "messages with source-linked long-term memory and separates current task state from earlier reports. "
            f"On 24 controlled task-state questions from four bilingual template families, the revised system answered {_fraction(new)} "
            f"correctly, compared with {_fraction(raw)} for raw-history retrieval and {_fraction(old)} before the revision. "
            "The tasks test corrections, cumulative progress, agreement and reported benefit. A separate historical "
            "MemoryAgentBench comparison improved fact-consolidation accuracy from 52.38% to 60.38%. "
            "CACTUS examples illustrate storage and source-linked recall."),
        11: ("This project makes three contributions. First, it designs a hierarchical memory system for psychotherapy "
             "reflection and care tracking, combining direct recent-dialogue context with source-grounded long-term memory "
             "and keeping professional knowledge separate. Second, it implements inspectable storage and separate fields "
             "for agreement, progress and reported outcome. Third, it evaluates these mechanisms against raw-history "
             "retrieval through controlled task-state questions, alongside historical public memory benchmarks and "
             "executed synthetic CBT quotation-recall examples."),
        13: ("The main design question is how a hierarchical memory system should form, store, retrieve, update and stop "
             "using information to support psychotherapy reflection and care tracking over time. The empirical comparisons "
             "ask whether memory organisation improves factual continuity under the same permitted history, response "
             "model and context budget. Controlled task-state cases examine errors that affect follow-up; CACTUS probes "
             "examine original-statement recall. Professional CBT response quality is a separate evaluation question."),
        16: ("The project combines system development with memory evaluation. Table 1 separates the materials, units "
             "and scoring tasks. The primary comparison is managed memory versus retrieval of the same permitted "
             "raw history. A recent-only condition is used where a secondary comparison is informative."),
        17: "Table 1. Evaluation materials and scoring tasks.",
        18: ("The materials comprise public memory questions, simulated user-profile histories, synthetic CACTUS "
             "dialogue [10,11] and separately constructed task-state cases. The controlled cases are mechanism tests, "
             "not natural patient conversations. No patients or volunteers were recruited for these experiments."),
        44: ("The frozen CACTUS preparation contains 600 planned questions, of which 590 satisfy preparation criteria. "
             "Its initial run completed one probe using a four-turn recent window. A later development pilot tested "
             "three positions from one source family with six recent messages supplied directly to every condition. "
             "These overlapping pilot materials are reported separately, rather than pooled as independent cases. "
             "Source dates are unavailable, so recall follows dialogue order without invented dates or treatment visits."),
        52: "CBT quotation recall",
        53: ("Table 4 preserves the original executed CACTUS storage and recall example. Recent-only retrieval could "
             "not recover the earlier statement. Raw-history retrieval and managed memory returned the same complete "
             "statement and correct source through one shared model response."),
        54: ("In the later three-question development pilot, the original joint quotation-and-source scores were "
             "1/3 for recent-only, 2/3 for raw retrieval and 2/3 for managed memory. "
             "Both retrieval conditions copied the recent quotation correctly but attached another message's ID. "
             "A common validator, using only uniquely matching complete quotations already visible in each request, "
             "repaired these two IDs. Retrospective accuracy became 1/3, 3/3 and 3/3, respectively, without new "
             "model calls. This is a shared source-binding repair, not a managed-only advantage."),
        57: ("The initial CACTUS pilot used 16 model calls and 18,908 tokens. The later three-question pilot added "
             "nine answer calls and 11,239 tokens; its cumulative API time was 18.52 seconds. The common source "
             "validator required no additional model calls. These costs belong to separate pilot runs. The controlled "
             "task-state experiment used an open reader on hosted Colab CPU; its versioned execution records are "
             "retained in the experiment source index."),
        60: (f"The revised state representation produced {effect} on the 24 controlled questions. "
             "The comparison checks whether the reader uses current targets, cumulative totals, agreement and "
             "reported benefit consistently. Its raw-history control remained unchanged across versions. "
             "The separate historical fact-consolidation comparison produced 64 additional correct answers, "
             "with most of that gain arising in multi-hop questions."),
        66: ("The controlled task-state set contains 24 variants of four related templates, so it supports a mechanism "
             "comparison rather than an independent population estimate. The MemoryAgentBench material also contains "
             "related variants and was used during development. The CACTUS pilots test quotation recall with explicit "
             "locator cues; they do not measure natural CBT follow-up quality."),
        67: ("The current task-state answers are factual JSON outputs. CBT response quality and user effort require "
             "separate evaluation through professionally grounded response tasks, blinded review and user testing."),
        69: ("The next evaluation should retain the same histories, recent context, model and budgets while testing "
             "natural follow-up from professional CBT dialogue. It should measure correct use of necessary earlier "
             "facts, omissions, unsupported claims and adaptation to explicitly changed goals or practice states."),
        72: ("This project presents a hierarchical memory system for psychotherapy reflection and care tracking. "
             "It combines directly supplied recent conversation with source-linked long-term memory and separates "
             "goals, agreement, reported practice and outcome. "
             f"The revised system answered {new}/24 controlled task-state questions correctly versus {raw}/24 for "
             f"raw-history retrieval and {old}/24 before the revision. The historical fact-consolidation comparison "
             "also showed an 8.00 percentage-point gain. The CACTUS examples demonstrate storage and original-source "
             "recall. Together, these findings connect memory organisation to specific factual requirements of "
             "continued reflection and follow-up."),
    }
    for index, value in updates.items():
        _text(paragraphs[index], value)
    # Make the overview compact and avoid repeating result paragraphs in cells.
    overview = document.tables[0]
    while len(overview.rows) > 1:
        overview._tbl.remove(overview.rows[-1]._tr)
    overview_rows = [
        ("Evaluation", "Materials", "Comparison", "Measure"),
        ("MemoryAgentBench", "800 fact-consolidation questions", "Managed versus raw retrieval", "Official answer accuracy"),
        ("PersonaMem", "589 paired profile questions", "Full management versus raw retrieval", "Released option score"),
        ("CACTUS initial pilot [10,11]", "One completed quotation probe", "Recent-only and shared raw/managed output", "Complete quote and source ID"),
        ("CACTUS later pilot", "Three related development probes", "Recent-only, raw and managed", "Original and common-validator recall"),
        ("Controlled task state", "24 bilingual variants; four templates", "Raw control and two managed versions", "Strict factual answer accuracy"),
    ]
    if professional:
        overview_rows.append(("CBT-DP professional references", str(professional["counts"]["reference_exercises"]) + " practice exercises",
                              "Quality anchor; separate from memory test", "Published professional criteria " + professional_citations))
    if calibration:
        overview_rows.append(("TherapyJudgeBench", "116 simulated dialogues with clinician ratings",
                              "Judge-calibration materials only", "11 CTRS and four safety dimensions " + calibration_citations))
    for index, row in enumerate(overview_rows):
        cells = overview.rows[0].cells if index == 0 else overview.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = text
    _table_style(overview, (3.5, 4.1, 4.3, 3.5))
    # Insert a numbered method subsection, inheriting the existing Word list.
    method_anchor = paragraphs[45]
    _paragraph_before(method_anchor, "Controlled task state evaluation", paragraphs[42])
    _paragraph_before(method_anchor,
        "This experiment tests whether source-linked state organisation reduces factual follow-up errors. "
        "Four constructed families cover target corrections, repeated cumulative counts, proposals without agreement, "
        "and benefit distinct from completion. Each has three numerical or outcome variants in Chinese and English, "
        "giving 24 questions. Explicit dates belong to these constructed cases and are not added to CACTUS dialogue.", paragraphs[43])
    _paragraph_before(method_anchor,
        "Both conditions receive the same six recent messages directly, the same question and instructions, "
        "up to nine older sources, a 6,000-character complete-request ceiling and a 64-token output ceiling. "
        "Raw retrieval preserves the original reports; managed memory additionally supplies state derived from "
        "those permitted reports. The revised rules recognise supported English task reports alongside Chinese "
        "reports. Reference values are fixed separately before generation; exact answer value and JSON type "
        "determine correctness. The raw condition is unchanged in the before-and-after comparison.", paragraphs[43])
    if professional or calibration:
        _paragraph_before(method_anchor, "Professional CBT quality references", paragraphs[42])
    if professional:
        _paragraph_before(method_anchor,
            f"CBT-DP supplies {professional['counts']['reference_exercises']} professionally developed practice "
            "exercises and reference responses, together with published expert comparisons " + professional_citations + ". "
            "Its criteria address appropriate goals, collaborative agendas, homework, cognitive and behavioural "
            "work, emotions and the therapeutic relationship. References remain in the assessor channel. "
            "The exercises contain no longitudinal history and are used as a quality anchor, not as evidence "
            "that memory management improved clinical responses.", paragraphs[43])
    if calibration:
        _paragraph_before(method_anchor,
            "TherapyJudgeBench provides a complementary source of published ratings from two licensed, CBT-trained "
            "practitioners " + calibration_citations + ". Its 116 simulated dialogues are rated on 11 Cognitive Therapy "
            "Rating Scale dimensions and four safety categories. Dialogue generation and human rating are distinct: "
            "these are simulated conversations with clinician labels, not human therapist transcripts. The frozen "
            "bank is reserved for auditing future automated judges; no new professional ratings of this system "
            "are inferred from acquiring it. Missing ratings remain missing.", paragraphs[43])
    # A compact result table follows the historical quotation example.
    result_anchor = paragraphs[56]
    _paragraph_before(result_anchor, "Controlled task state results", paragraphs[52])
    _paragraph_before(result_anchor,
        f"The revised managed condition answered {_fraction(new)} questions correctly, compared with "
        f"{_fraction(raw)} for raw-history retrieval and {_fraction(old)} for the earlier managed condition "
        "(Table 5). All versions use the same 24 questions and fixed reader settings.", paragraphs[53])
    caption = _paragraph_before(result_anchor,
        "Table 5. Correct factual answers on the controlled task-state cases. Each family contains six related "
        "Chinese- and English-source variants. Raw answers are identical across versions.", paragraphs[55])
    caption.paragraph_format.keep_with_next = True
    rows = [(family["label"], f"{family['raw']}/6", f"{family['before']}/6", f"{family['after']}/6")
            for family in result["families"]]
    rows.append(("Total", f"{raw}/24", f"{old}/24", f"{new}/24"))
    _comparison_table(document, result_anchor, rows)
    wins = result["paired_outcomes"].get("managed_win", 0)
    losses = result["paired_outcomes"].get("raw_win", 0)
    ties = result["paired_outcomes"].get("tie", 0)
    _paragraph_before(result_anchor,
        f"Against raw retrieval, the revised condition won {wins} paired questions, lost {losses} and tied {ties}. "
        f"The revision changed {result['changed_request_rows']} managed request rows; the other requests and the "
        "raw control were preserved. The result isolates these task-state mechanisms on constructed cases; "
        "it does not provide a professional rating of free-form CBT replies.", paragraphs[53])
    # Preserve numbering and simple diagrams; prevent existing orphan headings.
    for name in ("Title", "Heading 1", "Heading 2"):
        document.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    for name in ("Heading 1", "Heading 2"):
        document.styles[name].paragraph_format.keep_with_next = True
    if professional or calibration:
        _text(paragraphs[70], paragraphs[70].text.replace("[12]", f"[{next_reference}]"))
        _text(paragraphs[85], paragraphs[85].text.replace("12. ", f"{next_reference}. ", 1))
        references, reference_number = [], 12
        if professional:
            references.extend([
            str(reference_number) + ". Zhang M, Yang X, Zhang X, Labrum T, Chiu JC, Eack SM, et al. CBT-Bench: evaluating large language "
            "models on assisting cognitive behavior therapy. In: Proceedings of NAACL 2025. Association for "
            "Computational Linguistics; 2025. p. 3864-3900. doi:10.18653/v1/2025.naacl-long.196. Available from: "
            "https://aclanthology.org/2025.naacl-long.196/",
            str(reference_number + 1) + ". Psychotherapy-LLM. CBT-Bench [dataset on the Internet]. Hugging Face; [cited 2026 Nov 12]. "
            "Available from: https://huggingface.co/datasets/Psychotherapy-LLM/CBT-Bench",
            ])
            reference_number += 2
        if calibration:
            references.extend([
                str(reference_number) + ". Huang F, Chbeir S, Khatua A, Wang S, Tan S, Ye K, et al. TherapyGym: evaluating "
                "and aligning clinical fidelity and safety in therapy chatbots [preprint]. arXiv. 2026 [revised 2026 Nov 3; "
                "cited 2026 Nov 12]:2603.18008v2. Available from: https://arxiv.org/abs/2603.18008v2",
                str(reference_number + 1) + ". TherapyGym. TherapyJudgeBench [dataset on the Internet]. Version 2.0.0. "
                "Hugging Face; 2026 Nov 3 [cited 2026 Nov 12]. Available from: "
                "https://huggingface.co/datasets/therapygym/therapyjudgebench",
            ])
        for value in references:
            _paragraph_before(paragraphs[85], value, paragraphs[85])
    appendix = document.add_paragraph(style="Heading 1")
    appendix.add_run("Appendix A Experiment source index")
    appendix.paragraph_format.page_break_before = True
    appendix.paragraph_format.keep_with_next = True
    index_values = [
        "The experiment records retain the complete source and request identities. These are materials of this "
        "study, not independent publications. Historical benchmarks, the CACTUS quotation pilots and the "
        "controlled task-state comparison retain separate protocols and costs.",
        "Controlled reader: Qwen/Qwen2.5-1.5B-Instruct; fixed revision "
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306; hosted Colab CPU; deterministic generation; "
        "six direct recent messages, nine-source retrieval cap, 6,000 complete-request characters and 64 output tokens.",
        "Before report SHA256: " + args.before_report_sha256,
        "After report SHA256: " + args.after_report_sha256,
        "Before prepared protocol SHA256: " + before["protocol_file_sha256"],
        "After prepared protocol SHA256: " + after["protocol_file_sha256"],
        "Before answer archive SHA256: " + before["answers_file_sha256"],
        "After answer archive SHA256: " + after["answers_file_sha256"],
        "CACTUS later-pilot protocol SHA256: " + SOURCE_PILOT["protocol_sha256"],
        "CACTUS common-validator transformed answers SHA256: " + SOURCE_PILOT["transformed_rows_sha256"],
    ]
    if professional:
        index_values.extend([
            "Professional reference dataset: Psychotherapy-LLM/CBT-Bench; revision " + professional["revision"]
            + "; dataset-card licence CC-BY-NC-4.0; academic noncommercial research.",
            "Professional reference preparation SHA256: " + args.professional_summary_sha256,
        ])
    if calibration:
        index_values.extend([
            "Clinical-judge calibration source: therapygym/therapyjudgebench; release 2.0.0; revision "
            + calibration["manifest"]["revision"] + "; CC-BY-4.0; sources and labels remain assessor-only.",
            "Calibration source manifest SHA256: " + args.calibration_manifest_sha256,
            "Calibration schema audit SHA256: " + args.calibration_audit_sha256,
        ])
    for value in index_values:
        paragraph = document.add_paragraph(style="Normal")
        paragraph.add_run(value)
        paragraph.paragraph_format.line_spacing = 1.1
        paragraph.paragraph_format.space_after = Pt(7)
        for run in paragraph.runs:
            run.font.size = Pt(9)
    all_text = "\n".join(p.text for p in document.paragraphs)
    if any(marker in all_text for marker in ("TBD", "{{", "PLACEHOLDER", "deepseek-v4-1-flash-260910")):
        raise ValueError("placeholder_or_request_alias_in_paper")
    if document.paragraphs[0].text != TITLE or len(document.inline_shapes) != 3:
        raise ValueError("title_or_architecture_changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output creation; preserve the original Library-backed source.
    with output.open("xb") as stream:
        document.save(stream)
    audit = {"version": "controlled-result-thesis-revision-v1",
             "source_library_file_id": SOURCE_LIBRARY_ID, "source_library_version": SOURCE_VERSION,
             "source_sha256": SOURCE_V8_SHA256, "output_sha256": filehash(output),
             "before_report_sha256": args.before_report_sha256,
             "after_report_sha256": args.after_report_sha256,
             "controlled_results": result, "source_quote_pilot": SOURCE_PILOT,
             "professional_summary_sha256": args.professional_summary_sha256,
             "calibration_manifest_sha256": args.calibration_manifest_sha256,
             "calibration_audit_sha256": args.calibration_audit_sha256,
             "clinical_quality_claim": False, "new_human_reviews": 0,
             "new_model_calls_by_authoring_script": 0,
             "render_and_visual_review_required_before_publish": True,
             "library_or_github_written": False}
    audit_path = output.with_suffix(".revision.json")
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    return {"working_docx": str(output.resolve()), "audit": str(audit_path.resolve()),
            "output_sha256": audit["output_sha256"], "ready_for_delivery": False,
            "required_next_step": "render and inspect all page PNGs"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--before-report")
    parser.add_argument("--before-report-sha256")
    parser.add_argument("--before-protocol")
    parser.add_argument("--after-report")
    parser.add_argument("--after-report-sha256")
    parser.add_argument("--after-protocol")
    for role in ("before-reader-run", "before-reader-completion", "before-answers",
                 "after-reader-run", "after-reader-completion", "after-answers"):
        parser.add_argument("--" + role)
    parser.add_argument("--professional-summary")
    parser.add_argument("--professional-summary-sha256")
    parser.add_argument("--calibration-manifest")
    parser.add_argument("--calibration-manifest-sha256")
    parser.add_argument("--calibration-audit")
    parser.add_argument("--calibration-audit-sha256")
    parser.add_argument("--output")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--export-evidence")
    parser.add_argument("--evidence-extra", action="append", default=[], metavar="ROLE=PATH")
    parser.add_argument("--evidence-bundle")
    parser.add_argument("--evidence-bundle-sha256")
    parser.add_argument("--evidence-directory")
    args = parser.parse_args()
    if args.export_evidence:
        # Fail before reading any actual source-result file outside hosted Colab.
        try:
            hosted = (bool(os.environ.get("COLAB_RELEASE_TAG")) and Path("/content").is_dir()
                      and importlib.util.find_spec("google.colab") is not None)
        except (ImportError, ValueError, AttributeError):
            hosted = False
        if not hosted:
            raise RuntimeError("genuine_hosted_colab_required_for_result_export")
    if args.evidence_bundle:
        if args.export_evidence or not args.evidence_bundle_sha256 or not args.evidence_directory:
            raise ValueError("independent_bundle_digest_and_new_import_directory_required")
        paths, bundle = import_complete_evidence(args.evidence_bundle, args.evidence_bundle_sha256,
                                                args.evidence_directory)
        for role in ("before_report", "before_protocol", "after_report", "after_protocol", "professional_summary",
                     "calibration_manifest", "calibration_audit", "before_reader_run", "before_reader_completion",
                     "before_answers", "after_reader_run", "after_reader_completion", "after_answers"):
            if role in paths:
                setattr(args, role, paths[role])
        for role in ("before_report", "after_report", "professional_summary", "calibration_manifest", "calibration_audit"):
            setattr(args, role + "_sha256", bundle.get(role + "_sha256"))
    for name in ("before_report", "before_report_sha256", "before_protocol",
                 "after_report", "after_report_sha256", "after_protocol", "before_reader_run", "before_reader_completion",
                 "before_answers", "after_reader_run", "after_reader_completion", "after_answers"):
        if not getattr(args, name):
            raise ValueError("complete_result_inputs_required")
    result, before, after, professional, calibration = authenticated_inputs(args)
    if args.export_evidence:
        print(json.dumps(export_complete_evidence(args, result), sort_keys=True))
        return
    if args.validate_only:
        print(json.dumps({"validated": True, "results": result, "docx_written": False}, sort_keys=True))
        return
    if not args.source or not args.output:
        raise ValueError("source_and_new_output_required_for_docx_revision")
    print(json.dumps(build_document(args, result, before, after, professional, calibration), sort_keys=True))


if __name__ == "__main__":
    main()
