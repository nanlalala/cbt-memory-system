"""Invented metadata fixtures; never generate a paper with fabricated scores."""
import copy
import base64
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_revision_contract", ROOT / "tools/revise_capstone_from_controlled_results.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture():
    protocol = {"material": M.MATERIAL, "case_count": 24, "template_family_count": 4,
                "recent_messages": 6, "retrieved_source_cap_each_arm": 9,
                "max_outbound_chars": 6000, "max_new_tokens": 64,
                "raw_and_managed_share_original_source_ranking": True,
                "recent_sources_do_not_consume_raw_retrieval_slots": True,
                "independent_human_review": False, "clinical_quality_claim": False,
                "arms": list(M.ARMS)}
    rows, pairs = [], []
    for kind in M.FAMILIES:
        for language in ("zh", "en"):
            for variant in (1, 2, 3):
                cid = f"controlled-{kind}-{language}-{variant}"
                for arm in M.ARMS:
                    rows.append({"case_id": cid, "arm": arm, "language": language,
                                 "source_family_id": "template-" + kind, "correct": True,
                                 "reason": "correct", "request_sha256": "a" * 64,
                                 "response_sha256": "b" * 64})
                pairs.append({"case_id": cid, "source_family_id": "template-" + kind,
                              "language": language, "raw_correct": True,
                              "managed_correct": True, "outcome": "tie"})
    report = {"generation_receipts_independently_reconciled": True,
              "reader_run_fixture_only": False, "human_reviews": 0,
              "clinical_quality_claim": False, "material": M.MATERIAL,
              "case_count": 24, "template_family_count": 4, "per_question": rows,
              "pairs": pairs, "paired_outcomes": dict(Counter(p["outcome"] for p in pairs)),
              "arms": {arm: {"correct": 24, "planned": 24} for arm in M.ARMS}}
    return report, protocol


class PaperRevisionContractTests(unittest.TestCase):
    def reader_fixture(self):
        report, protocol = fixture()
        protocol["requests_file_sha256"] = "d" * 64
        protocol["code_identity"] = {"fixture_contract": "invented_metadata_only"}
        report["protocol_file_sha256"] = "e" * 64
        model = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                 "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}
        for key in ("generation_config_sha256", "chat_template_sha256", "load_protocol_sha256",
                    "model_files_sha256", "load_receipt_sha256"):
            model[key] = "f" * 64
        run = {"fixture_only": False, "maximum_global_generations": 64,
               "maximum_input_tokens": 4096, "max_new_tokens": 64,
               "do_sample": False, "use_cache": True, "external_inference_calls": 0,
               "secret_access": False, "new_human_reviews": 0, "scoring_targets_read": False,
               "model": model, "prepared_protocol_sha256": report["protocol_file_sha256"],
               "requests_file_sha256": protocol["requests_file_sha256"],
               "gate_identity": protocol["code_identity"], "reader_code_sha256": "c" * 64}
        run_hash = M.json_digest(run)
        answers = []
        for scored in report["per_question"]:
            response = '{"answer":1}'
            response_hash = hashlib.sha256(response.encode()).hexdigest()
            scored["response_sha256"] = response_hash
            answer = {"case_id": scored["case_id"], "arm": scored["arm"],
                      "request_sha256": scored["request_sha256"], "canonical_key": "a" * 64,
                      "response_text": response, "response_sha256": response_hash,
                      "run_protocol_sha256": run_hash, "status": "completed",
                      "source_checks_passed": True, "scored": False}
            answer["record_sha256"] = M.json_digest(answer)
            answers.append(answer)
        report["answers_file_sha256"] = "b" * 64
        completion = {"complete": True, "fixture_only": False,
                      "logical_planned": 48, "logical_completed": 48,
                      "run_protocol_sha256": run_hash,
                      "answers_sha256": report["answers_file_sha256"],
                      "new_provider_calls": 0, "new_human_reviews": 0, "secret_access": False}
        return report, protocol, run, completion, answers

    def test_complete_metadata_contract_and_no_docx_side_effects(self):
        report, protocol = fixture()
        result = M.validate_comparison(report, protocol, report, protocol)
        self.assertEqual(result["raw_correct"], 24)
        self.assertEqual(result["changed_request_rows"], 0)
        self.assertEqual(len(result["families"]), 4)
        self.assertTrue(all(row["denominator"] == 6 for row in result["families"]))

    def test_missing_receipts_fixtures_and_incomplete_denominators_rejected(self):
        for key, value in [("generation_receipts_independently_reconciled", False),
                           ("reader_run_fixture_only", True), ("case_count", 23),
                           ("clinical_quality_claim", True), ("human_reviews", 1)]:
            report, protocol = fixture(); report[key] = value
            with self.assertRaises(ValueError):
                M.validate_scored_report(report, protocol)

    def test_complete_controls_required(self):
        for key, value in [("recent_messages", 4), ("retrieved_source_cap_each_arm", 10),
                           ("max_outbound_chars", 6001), ("max_new_tokens", 128),
                           ("raw_and_managed_share_original_source_ranking", False)]:
            report, protocol = fixture(); protocol[key] = value
            with self.assertRaisesRegex(ValueError, "settings_changed"):
                M.validate_scored_report(report, protocol)

    def test_changed_raw_response_rejected_even_when_correct(self):
        before, protocol = fixture(); after = copy.deepcopy(before)
        after["per_question"][0]["response_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "raw_control_changed"):
            M.validate_comparison(before, protocol, after, protocol)

    def test_duplicate_rows_and_false_aggregate_rejected(self):
        report, protocol = fixture()
        report["per_question"][-1] = copy.deepcopy(report["per_question"][0])
        with self.assertRaises(ValueError):
            M.validate_scored_report(report, protocol)
        report, protocol = fixture(); report["arms"][M.ARMS[1]]["correct"] = 23
        with self.assertRaisesRegex(ValueError, "totals_disagree"):
            M.validate_scored_report(report, protocol)

    def test_wrong_pairwise_summary_rejected(self):
        report, protocol = fixture(); report["paired_outcomes"] = {"managed_win": 24}
        with self.assertRaisesRegex(ValueError, "paired_totals"):
            M.validate_scored_report(report, protocol)

    def test_reader_model_settings_and_original_rows_are_bound(self):
        report, protocol, run, completion, answers = self.reader_fixture()
        result = M.validate_reader_binding(report, protocol, run, completion, answers)
        self.assertEqual(result["model"]["model_id"], "Qwen/Qwen2.5-1.5B-Instruct")
        changed = copy.deepcopy(run); changed["do_sample"] = True
        with self.assertRaisesRegex(ValueError, "reader_settings"):
            M.validate_reader_binding(report, protocol, changed, completion, answers)
        changed = copy.deepcopy(run); changed["model"]["model_revision"] = "another-model"
        with self.assertRaisesRegex(ValueError, "model_identity"):
            M.validate_reader_binding(report, protocol, changed, completion, answers)

    def test_reader_completion_and_published_answer_tampering_rejected(self):
        report, protocol, run, completion, answers = self.reader_fixture()
        completion["answers_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "archive_binding"):
            M.validate_reader_binding(report, protocol, run, completion, answers)
        report, protocol, run, completion, answers = self.reader_fixture()
        answers[0]["response_text"] = "INVENTED_TAMPERING"
        with self.assertRaisesRegex(ValueError, "receipt_or_score_binding"):
            M.validate_reader_binding(report, protocol, run, completion, answers)
        report, protocol, run, completion, answers = self.reader_fixture()
        answers[0]["source_checks_passed"] = False
        answers[0]["record_sha256"] = M.json_digest({k: v for k, v in answers[0].items()
                                                   if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "receipt_or_score_binding"):
            M.validate_reader_binding(report, protocol, run, completion, answers)

    def evidence_fixture(self):
        report, protocol = fixture()
        inputs = {"before_report": (json.dumps(report) + "\n").encode(),
                  "before_protocol": (json.dumps(protocol) + "\n").encode(),
                  "after_report": (json.dumps(report) + "\n").encode(),
                  "after_protocol": (json.dumps(protocol) + "\n").encode()}
        bundle = {"version": M.EVIDENCE_VERSION, "complete_original_bytes": True,
                  "before_report_sha256": hashlib.sha256(inputs["before_report"]).hexdigest(),
                  "after_report_sha256": hashlib.sha256(inputs["after_report"]).hexdigest(),
                  "entries": [{"role": role, "file_name": "../../ignored.json",
                               "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                               "data_base64": base64.b64encode(data).decode()}
                              for role, data in inputs.items()]}
        return bundle, inputs

    def test_complete_bundle_preserves_exact_bytes_and_ignores_supplied_paths(self):
        bundle, originals = self.evidence_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json.gz"
            path.write_bytes(gzip.compress(json.dumps(bundle).encode(), mtime=0))
            output = Path(folder) / "new"
            paths, loaded = M.import_complete_evidence(path, M.filehash(path), output)
            for role, original in originals.items():
                self.assertEqual(Path(paths[role]).read_bytes(), original)
                self.assertEqual(Path(paths[role]).parent, output)
            self.assertEqual(loaded["before_report_sha256"], bundle["before_report_sha256"])
            with self.assertRaises(FileExistsError):
                M.import_complete_evidence(path, M.filehash(path), output)

    def test_bundle_outer_and_inner_identity_fail_before_extraction(self):
        bundle, _ = self.evidence_fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.json.gz"
            path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
            output = Path(folder) / "not-created"
            with self.assertRaisesRegex(ValueError, "digest_mismatch"):
                M.import_complete_evidence(path, "0" * 64, output)
            self.assertFalse(output.exists())
            bundle["entries"][0]["sha256"] = "f" * 64
            path.write_bytes(gzip.compress(json.dumps(bundle).encode()))
            with self.assertRaisesRegex(ValueError, "entry_identity"):
                M.import_complete_evidence(path, M.filehash(path), output)
            self.assertFalse(output.exists())

    def test_result_export_guard_precedes_any_source_file_access(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                M.export_complete_evidence(SimpleNamespace(export_evidence="/content/never.gz"), {})


if __name__ == "__main__":
    unittest.main()
