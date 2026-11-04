"""Bounded synthetic pipeline tests; no network, downloaded model or full benchmark."""
from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
import run_formal_pipeline as pipeline
from benchmark_adapters import RetrievalExample
from execution_lock import EvaluationDirectoryBusy, exclusive_directories, exclusive_directory
from prepare_formal_evaluation import DEFAULT_PLAN, load_plan, prepare_examples
from run_three_arm import stable_digest, write_jsonl


class FormalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "pipeline"
        self.plan = load_plan(DEFAULT_PLAN)
        self.plan["statistics"]["precision_plan"]["resamples"] = 1000
        self.plan_path = self.base / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        example = RetrievalExample(
            example_id="synthetic-only", category="single-session-user", query="Which notebook?", answer="blue",
            evidence_ids=("old-1",), query_date="2026-10-10T00:00:00Z",
            sessions=({"session_id": "old", "date": "2026-10-01T00:00:00Z", "turns": [
                {"role": "user", "content": "I use a blue notebook."}]},
                {"session_id": "recent", "date": "2026-10-09T00:00:00Z", "turns": [
                    {"role": "user", "content": "I went walking today."}]}))
        prepare_examples([example], "longmemeval_s_cleaned", self.plan, {"sha256": "0" * 64, "bytes": 1},
                         self.root / "cohort", role="exploratory", fixture_only=True)
        self.prepare()

    def prepare(self, **kwargs):
        return pipeline.run_stage("prepare", self.root, dataset_key="longmemeval_s_cleaned",
                                  plan_path=self.plan_path, _fixture_only=True, **kwargs)

    def run_stage(self, stage, **kwargs):
        return pipeline.run_stage(stage, self.root, _fixture_only=True, **kwargs)

    def complete_reader(self):
        protocol, prompts, _, _ = pipeline.reader.load_run(self.root / "reader")
        rows = [{**{key: request[key] for key in ("example_id", "arm", "prompt_sha256")},
                 "protocol_sha256": stable_digest(protocol), "hypothesis": "blue", "finish_reason": "stop",
                 "provider_model": protocol["settings"]["model"], "latency_seconds": 0.1,
                 "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
                for request in prompts]
        write_jsonl(self.root / "reader/responses.jsonl", rows)

    def test_production_guard_runs_before_creating_any_lock_or_output(self):
        target = self.base / "must-not-exist"
        with patch.object(pipeline.reader, "require_colab", side_effect=RuntimeError("Colab required")):
            with self.assertRaises(RuntimeError):
                pipeline.run_stage("prepare", target, dataset_key="longmemeval_s_cleaned")
        self.assertFalse(target.exists())

    def test_partial_status_is_never_a_final_effectiveness_result(self):
        status = self.run_stage("status")
        self.assertFalse(status["final"])
        self.assertEqual(status["reader"]["requests"], 3)
        self.assertEqual(status["reader"]["pending"], 3)
        self.assertNotIn("evaluation", status)
        self.assertFalse(status["confirmatory_claim"])

    def test_prepare_resume_validates_without_rebuilding_candidates_or_models(self):
        with patch.object(pipeline.reader, "prepare_structured_run") as operation:
            result = self.prepare()
        operation.assert_not_called()
        self.assertIn("already_prepared", result["status"])

    def test_resume_rejects_backend_or_plan_changes(self):
        with self.assertRaises(ValueError):
            self.prepare(backend="hybrid")
        self.plan["shared_response_configuration"]["max_response_tokens"] = 2048
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_reader_and_judge_require_explicit_finite_positive_budget(self):
        for stage in ("reader", "judge"):
            for budget in (None, 0, -1, True, float("inf")):
                with self.subTest(stage=stage, budget=budget), self.assertRaises(ValueError):
                    self.run_stage(stage, max_calls=budget)

    def test_reader_passes_budget_once_and_never_loops_or_changes_model(self):
        with patch.object(pipeline.reader, "generate_run", return_value={"pending": 3}) as operation:
            self.assertEqual(self.run_stage("reader", max_calls=7), {"pending": 3})
        operation.assert_called_once_with(self.root / "reader", max_calls=7)

    def test_provider_identity_or_authentication_failure_stops_without_retry_loop(self):
        for error in (ValueError("identity failure"), RuntimeError("authentication rejected")):
            with patch.object(pipeline.reader, "generate_run", side_effect=error) as operation:
                with self.assertRaises(type(error)):
                    self.run_stage("reader", max_calls=8)
            self.assertEqual(operation.call_count, 1)

    def test_failure_finalization_cannot_skip_unattempted_reader_items(self):
        with patch.object(pipeline.judges, "require_colab"), \
             patch.object(pipeline.judges, "require_generation_cohort"), \
             patch.object(pipeline.judges, "execute_reference_judge") as operation:
            for finalize in (False, True):
                with self.subTest(finalize=finalize), self.assertRaises(ValueError):
                    self.run_stage("judge", max_calls=1, finalize_reader_failures=finalize)
        operation.assert_not_called()

    def test_pipeline_root_lock_excludes_second_executor_before_any_provider_call(self):
        with exclusive_directory(self.root / ".pipeline"), patch.object(pipeline.reader, "generate_run") as operation:
            with self.assertRaises(EvaluationDirectoryBusy):
                self.run_stage("reader", max_calls=1)
        operation.assert_not_called()

    def test_tampered_checkpoint_is_not_reported_as_partial(self):
        protocol_path = self.root / "reader/protocol.json"
        protocol = json.loads(protocol_path.read_text())
        protocol["settings"]["model"] = "unapproved-other-model"
        protocol_path.write_text(json.dumps(protocol))
        with self.assertRaises(ValueError):
            self.run_stage("status")

    def test_judge_is_bounded_and_fixture_result_stays_software_only(self):
        self.complete_reader()
        def response(*args, **kwargs):
            return {"hypothesis": "yes", "provider_model": self.plan["scoring"]["judge"]["model"],
                    "finish_reason": "stop", "latency_seconds": 0.1, "usage": {}}
        with patch.object(pipeline.judges, "require_colab"), \
             patch.object(pipeline.judges, "require_generation_cohort"), \
             patch.object(pipeline.judges, "generate_response", side_effect=response) as provider, \
             patch.dict("os.environ", {"CBT_EVAL_API_KEY": "synthetic-placeholder-never-export"}):
            first = self.run_stage("judge", max_calls=1)
            self.assertEqual(provider.call_count, 1)
            self.assertFalse(first["pipeline_status"]["final"])
            self.assertNotIn("evaluation", first["pipeline_status"])
            self.assertEqual(first["pipeline_status"]["judge"]["successful_labels"], 1)
            with patch.object(pipeline.judges, "prepare_reference_judge", side_effect=AssertionError("must not reprepare")):
                second = self.run_stage("judge", max_calls=2)
            self.assertEqual(provider.call_count, 3)
            status = second["pipeline_status"]
            self.assertIn("evaluation", status)
            self.assertFalse(status["final"])
            self.assertEqual(status["status"], "synthetic_software_smoke_not_effectiveness")
        with self.assertRaises(ValueError):
            self.run_stage("reader", max_calls=1)

    def test_export_is_checksummed_allowlist_and_retains_partial_boundary(self):
        for name in (".env", "locomo10.json", "private.sqlite"):
            (self.root / name).write_text("SECRET_NOT_FOR_ARCHIVE")
        result = self.run_stage("export")
        self.assertFalse(result["final"])
        with zipfile.ZipFile(result["archive"]) as archive:
            names = set(archive.namelist())
            self.assertIn("reader/prompts.jsonl", names)
            self.assertIn("reader/references.jsonl", names)
            self.assertIn("reader/candidate_memories.jsonl", names)
            self.assertIn("AUDIT_DECLARATION.json", names)
            self.assertFalse(any(name.endswith((".env", ".sqlite", "locomo10.json")) for name in names))
            sums = dict((line[66:], line[:64]) for line in archive.read("SHA256SUMS").decode().splitlines())
            self.assertEqual(set(sums), names - {"SHA256SUMS"})
            for name, expected in sums.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
                self.assertNotIn(b"SECRET_NOT_FOR_ARCHIVE", archive.read(name))
            destination = self.base / "standalone"
            archive.extractall(destination)
        command = destination / "code/evaluation/end-to-end-dialogue/reference_judge.py"
        output = subprocess.run([sys.executable, str(command), "--help"], cwd=destination,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(output.returncode, 0, output.stderr)

    def test_symlink_artifact_or_archive_output_is_rejected(self):
        target = self.root / "reader/responses.jsonl"
        external = self.base / "must-not-copy"
        external.write_text("")
        target.symlink_to(external)
        with self.assertRaises(ValueError):
            self.run_stage("export")
        target.unlink()
        archive = self.root / "audit.zip"
        archive.symlink_to(external)
        with self.assertRaises(ValueError):
            self.run_stage("export")

    def test_notebook_invalid_backup_preserves_raw_bytes_without_claiming_a_result(self):
        spec = importlib.util.spec_from_file_location("pipeline_notebook_builder", ROOT / "tools/build_structured_notebook.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        notebook = builder.build("0" * 40)  # Build/parse only; never execute any notebook cell.
        source = next(cell["source"] for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "def backup_invalid_unscored" in cell["source"])
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "backup_invalid_unscored")
        namespace = {"RUN_ROOT": self.root, "CODE_REVISION": "0" * 40,
                     "COHORT_FILES": pipeline.COHORT_FILES, "READER_FILES": pipeline.READER_FILES,
                     "JUDGE_FILES": pipeline.JUDGE_FILES, "exclusive_directory": exclusive_directory,
                     "exclusive_directories": exclusive_directories, "Path": Path, "tempfile": tempfile,
                     "os": os, "zipfile": zipfile, "hashlib": hashlib, "json": json,
                     "datetime": datetime, "timezone": timezone}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "synthetic-backup-helper", "exec"), namespace)
        raw = b'{"invalid_raw_protocol":true}\n'
        (self.root / "reader/protocol.json").write_bytes(raw)
        (self.root / "judge").mkdir(exist_ok=True)
        (self.root / "judge/reference_judge_summary.json").write_text("stale-result-must-not-escape")
        (self.root / ".env").write_text("secret-must-not-escape")
        archive_path = namespace["backup_invalid_unscored"]()
        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(archive.read("reader/protocol.json"), raw)
            self.assertNotIn("judge/reference_judge_summary.json", archive.namelist())
            self.assertNotIn(".env", archive.namelist())
            marker = json.loads(archive.read("INVALID_UNSCORED.json"))
            self.assertFalse(marker["final"])
            self.assertEqual(marker["status"], "invalid_unscored")
            sums = dict((line[66:], line[:64]) for line in archive.read("SHA256SUMS").decode().splitlines())
            self.assertEqual(set(sums), set(archive.namelist()) - {"SHA256SUMS"})
            for name, digest in sums.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
