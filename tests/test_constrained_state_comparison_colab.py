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
PATH = ROOT / "evaluation/cbt-memory-effect/constrained_state_comparison_colab.py"
SPEC = importlib.util.spec_from_file_location("new_full_stage_fixture_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MainStageFreezeTests(unittest.TestCase):
    def setUp(self):
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

    def test_changed_stage_rejected_before_pilot_authentication(self):
        M.reader._write_new(self.root / "main_freeze.json", {"main_stage_code_sha256": "changed"})
        with patch.object(M, "study"), patch.object(M, "pilot_passed") as pilot:
            with self.assertRaisesRegex(ValueError, "main_stage_code_changed"):
                M.frozen(self.root)
            pilot.assert_not_called()

    def test_changed_renderer_rejected_before_pilot_authentication(self):
        M.reader._write_new(self.root / "main_freeze.json", {"main_stage_code_sha256": self.stage_sha,
            "common_retrieval_policy": {"changed": "renderer"}})
        with patch.object(M, "study"), patch.object(M, "pilot_passed") as pilot:
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
                    with patch.object(M, "ledger", return_value=({str(i): {} for i in range(4)}, {})):
                        with patch.object(M, "require_main_identity", side_effect=[None, ValueError("changed")]):
                            with patch.object(M.reader, "_load_cached_model") as loader:
                                with self.assertRaisesRegex(ValueError, "changed"):
                                    M.run_side(self.root, "before")
                                loader.assert_not_called()

    def test_changed_stage_rejected_before_source_module_import(self):
        with patch.object(M.importlib.util, "spec_from_file_location") as load:
            with self.assertRaisesRegex(ValueError, "main_stage_code_changed"):
                M.source_gates({}, expected_stage_sha256="changed")
            load.assert_not_called()

    def test_changed_stage_after_authentication_rejected_before_labels(self):
        sections = {side: {"fresh_prepared": "invented-" + side} for side in M.SIDES}
        plan = {"main_stage_code_sha256": self.stage_sha, "sections": sections}
        with patch.object(M, "frozen", return_value=plan):
            with patch.object(M, "ledger", return_value=({str(i): {} for i in range(61)}, {})):
                with patch.object(M.reader, "read_rows", return_value=[]), patch.object(M, "authenticate_main", return_value=([], {})):
                    with patch.object(M, "require_main_identity", side_effect=ValueError("changed")):
                        with patch.object(M.reader, "read_json") as labels:
                            with self.assertRaisesRegex(ValueError, "changed"):
                                M.score_main(self.root)
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
            self.assertEqual(M.authenticate_main(self.root, "before", section), ([], run))
            metadata["common_policy_sha256"] = "another-policy"
            with self.assertRaisesRegex(ValueError, "renderer_or_actual_gate_identity_changed"):
                M.authenticate_main(self.root, "before", section)

    def test_freeze_uses_separate_audit_commands_before_writing_freeze(self):
        checkouts, references, fixture_rows, protocols = {}, {}, {}, {}
        for side in M.SIDES:
            prepared, rows, protocol = self.side_fixture(side)
            reference = self.root.parent / (side + "-reference")
            prepared.rename(reference)
            rows = [{**row, "database_path": str(reference / Path(row["database_path"]).name)} for row in rows]
            (reference / "requests.jsonl").write_text("".join(M.reader.canonical(row) + "\n" for row in rows))
            protocol["requests_file_sha256"] = M.reader.file_hash(reference / "requests.jsonl")
            (reference / "protocol.json").write_text(M.reader.canonical(protocol))
            references[side], fixture_rows[side], protocols[side] = reference, rows, protocol
            checkouts[side] = self.root.parent / (side + "-checkout")
        commands = []
        def process(command, **kwargs):
            commands.append(command)
            self.assertFalse((self.root / "main_freeze.json").exists())
            if command[2] == "prepare":
                side = Path(command[1]).parents[2].name.split("-")[0]
                fresh = Path(command[command.index("--output") + 1])
                fresh.mkdir()
                rows = [{**row, "database_path": str(fresh / Path(row["database_path"]).name)}
                    for row in fixture_rows[side]]
                with (fresh / "requests.jsonl").open("x") as stream:
                    for row in rows:
                        stream.write(M.reader.canonical(row) + "\n")
                protocol = {**protocols[side], "requests_file_sha256": M.reader.file_hash(fresh / "requests.jsonl")}
                M.reader._write_new(fresh / "protocol.json", protocol)
            elif "render-side" in command:
                self.assertEqual(command[1], str(PATH))
                side = command[command.index("--side") + 1]
                render(self.root / "source_prepared" / side, self.root / "prepared" / side)
            else:
                self.assertEqual(command[1], str(PATH))
                self.assertIn("audit-side", command)
                options = {key: command[command.index(key) + 1] for key in
                    ("--side", "--checkout", "--prepared", "--output", "--expected-stage-sha256", "--expected-renderer-sha256")}
                side = options["--side"]
                rows = M.reader.read_rows(Path(options["--prepared"]) / "requests.jsonl")
                gates = self.fake_gates(rows, protocols[side]["code_identity"])
                parent_pid = os.getpid()
                with patch.object(M, "source_gates", return_value=gates):
                    with patch.object(M.os, "getpid", return_value=parent_pid + 100):
                        with patch.object(M.os, "getppid", return_value=parent_pid):
                            receipt = M.audit_side(self.root, side, options["--checkout"], options["--prepared"],
                                options["--output"], options["--expected-stage-sha256"], options["--expected-renderer-sha256"])
            return SimpleNamespace(returncode=0, stdout="invented child result\n")
        def render(source, output):
            output.mkdir(parents=True)
            rows = M.reader.read_rows(source / "requests.jsonl")
            (output / "requests.jsonl").write_text("".join(M.reader.canonical(row) + "\n" for row in rows))
            protocol = {**M.reader.read_json(source / "protocol.json"),
                "source_protocol_sha256": M.reader.file_hash(source / "protocol.json"),
                "source_requests_file_sha256": M.reader.file_hash(source / "requests.jsonl"),
                "requests_file_sha256": M.reader.file_hash(output / "requests.jsonl")}
            M.reader._write_new(output / "protocol.json", protocol)
        def validate(output, source_prepared=None):
            return M.reader.read_json(Path(output) / "protocol.json"), M.reader.read_rows(Path(output) / "requests.jsonl")
        with ExitStack() as patches:
            for name in ("guard", "study", "fresh_database"):
                patches.enter_context(patch.object(M, name))
            patches.enter_context(patch.object(M, "pilot_passed", return_value={"invented": "passed"}))
            patches.enter_context(patch.object(M, "ledger", return_value=({str(i): {} for i in range(4)}, {})))
            patches.enter_context(patch.object(M, "verify_checkout", side_effect=lambda checkout, ref: protocols[Path(ref).name.split("-")[0]]))
            patches.enter_context(patch.object(M.subprocess, "run", side_effect=process))
            patches.enter_context(patch.object(M.common, "render_prepared", side_effect=render))
            patches.enter_context(patch.object(M.common, "validate_prepared", side_effect=validate))
            patches.enter_context(patch.object(M.common, "assert_payload_mapping"))
            result = M.freeze_main(self.root, checkouts, references)
        self.assertEqual(result["main_unique"], 57)
        self.assertEqual(len(commands), 6)
        renders = [command for command in commands if "render-side" in command]
        self.assertEqual([command[command.index("--side") + 1] for command in renders], list(M.SIDES))
        audits = [command for command in commands if "audit-side" in command]
        self.assertEqual([command[command.index("--side") + 1] for command in audits], list(M.SIDES))
        plan = M.reader.read_json(self.root / "main_freeze.json")
        self.assertEqual(plan["main_stage_code_sha256"], self.stage_sha)
        for side in M.SIDES:
            M.verify_audit_receipt(self.root, side, plan["sections"][side], self.stage_sha)


if __name__ == "__main__":
    unittest.main()
