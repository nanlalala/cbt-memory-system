"""Invented metadata and fake gates only; no model, tokenizer, data or Colab."""
from contextlib import ExitStack, nullcontext
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evaluation/cbt-memory-effect/gguf_state_comparison_v7_colab.py"
SPEC = importlib.util.spec_from_file_location("new_gguf_stage_fixture_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MainStageFreezeTests(unittest.TestCase):
    def setUp(self):
        guard_patch = patch.object(M, "guard")
        guard_patch.start(); self.addCleanup(guard_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "new-study"
        self.root.mkdir()
        M.reader._write_new(self.root / "study_protocol.json", {"invented": "new-study"})
        self.stage_sha = M.reader.file_hash(PATH)

    def side_fixture(self, side):
        prepared = self.root / "prepared" / side
        prepared.mkdir(parents=True)
        rows = []
        for index in range(48):
            payload_index = 48 + (index - 31) // 2 if side == "after" and index >= 31 and index % 2 else index
            payload = {"messages": [{"role": "user", "content": "Invented " + str(payload_index)}]}
            rows.append({"case_id": "fixture-" + str(index // 2), "arm": M.ARMS[index % 2],
                "request_payload": payload, "request_sha256": M.reader.digest(payload),
                "database_path": str(self.root / "source_prepared" / side / ("db-" + str(index // 2) + ".sqlite"))})
        with (prepared / "requests.jsonl").open("x") as stream:
            for row in rows:
                stream.write(M.reader.canonical(row) + "\n")
        protocol = {"code_identity": {"invented_source_version": side}, "case_count": 24,
            "template_family_count": 4, "case_file_sha256": "same-invented-cases",
            "targets_file_sha256": "same-invented-labels",
            "requests_file_sha256": M.reader.file_hash(prepared / "requests.jsonl"),
            "source_protocol_sha256": "invented-original-protocol", "source_requests_file_sha256": "invented-original-requests",
            "renderer_code_sha256": M.common.renderer_sha256(), "common_policy_sha256": M.reader.digest(M.common.policy()),
            "request_mapping_sha256": "invented-mapping"}
        M.reader._write_new(prepared / "protocol.json", protocol)
        return prepared, rows, protocol

    def fake_gates(self, rows, identity, *, fail_at=None):
        seen = []
        def preflight(row):
            seen.append(row)
            if len(seen) == fail_at:
                raise PermissionError("invented source withdrawal")
            return {"invented_source": row["case_id"], "arm": row["arm"]}
        gates = SimpleNamespace(identity=lambda: identity, preflight=Mock(side_effect=preflight),
            begin=Mock(), finish=Mock(), seen=seen)
        return gates

    def test_changed_stage_rejected_before_adapter_authentication(self):
        M.reader._write_new(self.root / "main_freeze.json", {"main_stage_code_sha256": "changed"})
        with patch.object(M, "study"), patch.object(M, "schema_module") as pilot:
            with self.assertRaisesRegex(ValueError, "main_stage_code_changed"):
                M.frozen(self.root)
            pilot.assert_not_called()

    def test_changed_renderer_rejected_before_adapter_authentication(self):
        M.reader._write_new(self.root / "main_freeze.json", {"main_stage_code_sha256": self.stage_sha,
            "common_retrieval_policy": {"changed": "renderer"}})
        with patch.object(M, "study"), patch.object(M, "schema_module") as pilot:
            with self.assertRaisesRegex(ValueError, "common_retrieval_policy_changed"):
                M.frozen(self.root)
            pilot.assert_not_called()

    def test_complete_ordered_source_audit_has_no_admissions_and_bound_receipt(self):
        prepared, rows, protocol = self.side_fixture("before")
        checkout = self.root.parent / "invented-before-checkout"
        output = self.root / "source_gate_audits/before.json"
        gates = self.fake_gates(rows, protocol["code_identity"])
        with patch.object(M, "guard"), patch.object(M, "study"), patch.object(M, "fresh_database") as fresh:
            with patch.object(M, "source_gates", return_value=gates) as load, patch.object(M.common, "validate_prepared", return_value=(protocol, rows)):
                receipt = M.audit_side(self.root, "before", checkout, prepared, output, self.stage_sha, M.common.renderer_sha256())
        self.assertEqual(gates.seen, rows)
        gates.begin.assert_not_called()
        gates.finish.assert_not_called()
        self.assertEqual(fresh.call_count, 48)
        load.assert_called_once_with({"checkout": str(checkout), "fresh_prepared": str(prepared),
            "source_code_identity": protocol["code_identity"]}, expected_stage_sha256=self.stage_sha)
        section = {key: receipt[key] for key in ("checkout", "fresh_prepared", "fresh_protocol_sha256",
            "requests_sha256", "source_code_identity", "new_database_count", "source_prepared", "source_protocol_sha256",
            "source_requests_sha256", "renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256")}
        section.update(source_gate_audit_path=str(output), source_gate_audit_sha256=M.reader.file_hash(output))
        self.assertEqual(M.verify_audit_receipt(self.root, "before", section, self.stage_sha), receipt)
        receipt["ordered_checks"][0]["request_sha256"] = "changed"
        output.write_text(M.reader.canonical(receipt))
        with self.assertRaisesRegex(ValueError, "audit_changed"):
            M.verify_audit_receipt(self.root, "before", section, self.stage_sha)
        section["source_gate_audit_sha256"] = M.reader.file_hash(output)
        with self.assertRaisesRegex(ValueError, "audit_binding_changed"):
            M.verify_audit_receipt(self.root, "before", section, self.stage_sha)

    def test_failed_source_audit_never_writes_success_or_admits_generation(self):
        prepared, rows, protocol = self.side_fixture("after")
        output = self.root / "source_gate_audits/after.json"
        gates = self.fake_gates(rows, protocol["code_identity"], fail_at=2)
        with patch.object(M, "guard"), patch.object(M, "study"), patch.object(M, "fresh_database"):
            with patch.object(M, "source_gates", return_value=gates), patch.object(M.common, "validate_prepared", return_value=(protocol, rows)):
                with self.assertRaises(PermissionError):
                    M.audit_side(self.root, "after", self.root.parent / "after", prepared, output, self.stage_sha, M.common.renderer_sha256())
        self.assertFalse(output.exists())
        gates.begin.assert_not_called()
        gates.finish.assert_not_called()

    def test_changed_stage_after_preflight_rejected_before_model_loader(self):
        section = {"fresh_prepared": "invented-preparation"}
        plan = {"main_stage_code_sha256": self.stage_sha, "sections": {"before": section}}
        with patch.object(M, "frozen", return_value=plan), patch.object(M.reader, "read_rows", return_value=[{}]):
            with patch.object(M, "source_gates", return_value=SimpleNamespace(preflight=lambda row: None)):
                with patch.object(M.reader, "_locked", side_effect=lambda path: nullcontext()):
                    with patch.object(M, "ledger", return_value=({str(i): {} for i in range(0)}, {})):
                        with patch.object(M, "require_main_identity", side_effect=[None, ValueError("changed")]):
                            with patch.object(M, "load_backend_for_phase") as loader:
                                with self.assertRaisesRegex(ValueError, "changed"):
                                    M.run_side(self.root, "before", constructor_log_path=self.root / "invented.log")
                                loader.assert_not_called()

    def test_changed_stage_rejected_before_source_module_import(self):
        with patch.object(M.importlib.util, "spec_from_file_location") as load:
            with self.assertRaisesRegex(ValueError, "main_stage_code_changed"):
                M.source_gates({}, expected_stage_sha256="changed")
            load.assert_not_called()

    def test_changed_stage_after_authentication_rejected_before_labels(self):
        sections = {side: {"fresh_prepared": "invented-" + side} for side in M.SIDES}
        plan = {"main_stage_code_sha256": self.stage_sha, "sections": sections, "main_unique_payloads": 54}
        with patch.object(M, "frozen", return_value=plan):
            with patch.object(M, "ledger", return_value=({str(i): {} for i in range(54)}, {})):
                with patch.object(M.reader, "read_rows", return_value=[]), patch.object(M, "authenticate_main", return_value=([], {})), patch.object(M, "load_backend_for_phase", return_value=(object(), {})):
                    with patch.object(M, "require_main_identity", side_effect=ValueError("changed")):
                        with patch.object(M.reader, "read_json") as labels:
                            with self.assertRaisesRegex(ValueError, "changed"):
                                M.score_main(self.root, constructor_log_path=self.root / "invented.log")
                            labels.assert_not_called()

    def test_main_authentication_binds_effective_renderer_and_actual_gate(self):
        M.reader._write_new(self.root / "main_freeze.json", {"invented": "neutral-freeze"})
        section = {"fresh_prepared": "invented-preparation", "fresh_protocol_sha256": "fresh-protocol",
            "renderer_code_sha256": "renderer", "common_policy_sha256": "common-policy",
            "request_mapping_sha256": "source-effective-mapping", "source_code_identity": {"actual": "product"}}
        metadata = {"side": "before", "main_freeze_sha256": M.reader.file_hash(self.root / "main_freeze.json"),
            "fresh_prepared_protocol_sha256": "fresh-protocol", "renderer_code_sha256": "renderer",
            "common_policy_sha256": "common-policy", "request_mapping_sha256": "source-effective-mapping",
            "all_cases_exposed_development_only": True}
        run = {"metadata": metadata, "gate_identity": section["source_code_identity"]}
        with patch.object(M.reader, "read_rows", return_value=[]), patch.object(M, "authenticate", return_value=([], run)):
            self.assertEqual(M.authenticate_main(self.root, "before", section, object()), ([], run))
            metadata["common_policy_sha256"] = "another-policy"
            with self.assertRaisesRegex(ValueError, "renderer_or_actual_gate_identity_changed"):
                M.authenticate_main(self.root, "before", section, object())

    def test_exact_frozen_unique_grid_not_cap_is_required_for_scoring(self):
        plan = {"main_unique_payloads": 54}
        with patch.object(M, "frozen", return_value=plan), patch.object(M, "ledger", return_value=({str(i): {} for i in range(53)}, {})):
            with patch.object(M, "authenticate_main") as authentication:
                with self.assertRaisesRegex(RuntimeError, "all_frozen_unique"):
                    M.score_main(self.root, constructor_log_path=self.root / "invented.log")
                authentication.assert_not_called()

    def test_strict_typed_metric_never_repairs_prose_or_wrong_type(self):
        self.assertTrue(M.strict_result('{"answer":12}', 12)["correct"])
        self.assertFalse(M.strict_result('{"answer":"12"}', 12)["correct"])
        self.assertFalse(M.strict_result('answer: 12', 12)["correct"])
        self.assertFalse(M.strict_result('{"answer":12,"status":"confirmed"}', 12)["correct"])
        self.assertTrue(M.strict_result('{"answer":null}', None)["correct"])

    def test_after_backend_load_precedes_before_archive_authentication(self):
        M.reader._write_new(self.root / "main_freeze.json", {"invented": "frozen-grid"})
        sections = {"before": {"unique_request_count": 48},
            "after": {"fresh_prepared": "invented-after", "fresh_protocol_sha256": "protocol",
                "renderer_code_sha256": "renderer", "common_policy_sha256": "policy",
                "request_mapping_sha256": "mapping"}}
        plan = {"sections": sections, "main_stage_code_sha256": self.stage_sha}
        events, backend = [], object()
        def load(root, phase, **kwargs):
            events.append(("load", phase)); return backend, {}
        def auth(root, side, section, actual_backend):
            self.assertIs(actual_backend, backend)
            events.append(("authenticate", side)); return [], {}
        def generate(*args, **kwargs):
            self.assertIs(args[3], backend)
            events.append(("generate", "after")); return {"fixture": True}
        with patch.object(M, "frozen", return_value=plan), patch.object(M, "require_main_identity"), \
             patch.object(M.reader, "read_rows", return_value=[{}]), \
             patch.object(M, "source_gates", return_value=SimpleNamespace(preflight=lambda row: None)), \
             patch.object(M.reader, "_locked", side_effect=lambda path: nullcontext()), \
             patch.object(M, "ledger", return_value=({str(i): {} for i in range(48)}, {})), \
             patch.object(M, "load_backend_for_phase", side_effect=load), \
             patch.object(M, "authenticate_main", side_effect=auth), \
             patch.object(M, "generate_rows", side_effect=generate):
            self.assertEqual(M.run_side(self.root, "after", constructor_log_path=self.root / "invented.log"), {"fixture": True})
        self.assertEqual(events, [("load", "after"), ("authenticate", "before"), ("generate", "after")])

    def test_scoring_loads_once_and_authenticates_both_sides_before_any_main_labels(self):
        sections = {side: {"fresh_prepared": "invented-" + side} for side in M.SIDES}
        plan = {"main_unique_payloads": 54, "sections": sections}
        backend = object()
        with patch.object(M, "frozen", return_value=plan), \
             patch.object(M, "ledger", return_value=({str(i): {} for i in range(54)}, {})), \
             patch.object(M, "load_backend_for_phase", return_value=(backend, {})) as load, \
             patch.object(M, "authenticate_main", side_effect=[([], {}), ValueError("after archive incomplete")]) as auth, \
             patch.object(M.reader, "read_json") as labels:
            with self.assertRaisesRegex(ValueError, "after archive incomplete"):
                M.score_main(self.root, constructor_log_path=self.root / "invented.log")
        load.assert_called_once_with(self.root, "score-authentication", constructor_log_path=self.root / "invented.log")
        self.assertEqual(auth.call_count, 2)
        labels.assert_not_called()

    def test_failed_preparation_child_records_duration_without_any_generation(self):
        child = SimpleNamespace(returncode=3, stdout="invented failure")
        with patch.object(M.subprocess, "run", return_value=child), \
             patch.object(M.time, "monotonic", side_effect=[10.0, 12.5]):
            self.assertIs(M.preparation_child(self.root, "before", "source_prepare", ["invented"]), child)
        receipt = M.reader.read_json(self.root / "preparation_timings/before_source_prepare.json")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["return_code"], 3)
        self.assertEqual(receipt["wall_seconds"], 2.5)
        self.assertEqual(receipt["new_model_generations"], 0)
        self.assertFalse(receipt["automatic_retry"])

    def test_preparation_launch_exception_preserves_unknown_duration(self):
        with patch.object(M.subprocess, "run", side_effect=OSError("invented spawn failure")), \
             patch.object(M.time, "monotonic", side_effect=[10.0, 13.0]):
            with self.assertRaisesRegex(OSError, "invented spawn failure"):
                M.preparation_child(self.root, "after", "render", ["invented"])
        receipt = M.reader.read_json(self.root / "preparation_timings/after_render.json")
        self.assertEqual(receipt["status"], "failed_or_unknown")
        self.assertIsNone(receipt["return_code"])
        self.assertEqual(receipt["wall_seconds"], 3.0)
        self.assertEqual(receipt["new_model_generations"], 0)


if __name__ == "__main__": unittest.main()
