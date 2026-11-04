"""Tiny notebook/export fixtures; never download data, load models or use an API."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))
import objective_audit as audit

spec = importlib.util.spec_from_file_location("build_objective_notebook", ROOT / "tools/build_objective_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ObjectiveNotebookTests(unittest.TestCase):
    def test_requires_immutable_nonzero_code_revision(self):
        for invalid in ("main", "abc", "0" * 40, "G" * 40):
            with self.assertRaises(ValueError):
                builder.build(invalid)

    def test_cells_parse_and_cannot_start_a_paid_batch_by_default(self):
        revision = "a" * 40
        notebook = builder.build(revision)
        self.assertEqual(notebook["metadata"]["code_revision"], revision)
        self.assertNotIn("accelerator", notebook["metadata"])
        code = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                code.append(cell["source"])
        combined = "\n".join(code)
        self.assertIn("RUN_MODEL_BATCH = False", combined)
        self.assertIn("PREPARE_RUN = False", combined)
        self.assertIn("EXPORT_CHECKPOINT = False", combined)
        self.assertIn("1 <= MAX_CALLS_THIS_BATCH <= 60", combined)
        self.assertIn("with objective_secret():", combined)
        self.assertIn("'Capstone'", combined)
        self.assertNotIn("drive.mount", combined)
        self.assertNotIn("pip install", combined)
        self.assertNotIn("while True", combined)

    def test_fresh_hosted_runtime_and_separate_run_root_are_explicit(self):
        notebook = builder.build("b" * 40)
        combined = "\n".join(cell["source"] for cell in notebook["cells"])
        self.assertIn("COLAB_RELEASE_TAG", combined)
        self.assertIn("fresh hosted", combined)
        self.assertIn("cbt-objective-runs-", combined)
        self.assertIn("managed quotation memory", combined)
        self.assertIn("not automatic semantic goal extraction", combined)
        self.assertIn("older 1,500-response", combined)

    def test_default_generation_cell_does_not_read_a_secret_or_call_provider(self):
        notebook = builder.build("e" * 40)
        source = next(cell["source"] for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "RUN_MODEL_BATCH = False" in cell["source"])
        def unexpected_call(*args, **kwargs):
            raise AssertionError("Default cell must not invoke generation")
        scope = {"os": os, "objective_cli": unexpected_call}
        # google.colab is deliberately unavailable here: the lazy Secret import
        # would fail if it were reached, making the default gate executable proof.
        with patch("builtins.print"):
            exec(compile(source, "<objective-generation-cell>", "exec"), scope)
        self.assertFalse(scope["RUN_MODEL_BATCH"])


class ObjectiveAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.run = self.base / "run"
        self.run.mkdir()
        for name in audit.RUN_REQUIRED:
            (self.run / name).write_text('{"dataset_key":"synthetic"}\n' if name == "protocol.json" else "{}\n")
        self.events = []

    def tearDown(self):
        self.temporary.cleanup()

    def dependencies(self, *, complete=True, fail=False):
        @contextmanager
        def lock(path):
            self.events.append("lock")
            try:
                yield
            finally:
                self.events.append("unlock")

        def load(path):
            self.events.append("validate")
            if fail:
                raise ValueError("invalid frozen input")

        def score(path, *, write_outputs):
            self.events.append("score")
            self.assertTrue(write_outputs)
            names = audit.SCORE_FILES if complete else (audit.PROGRESS_FILE,)
            for name in names:
                (path / name).write_text('{"status":"' + ("complete" if complete else "partial") + '"}\n')
            return {"status": "complete" if complete else "partial"}

        return lock, load, score

    def exported(self, *, complete=True):
        output = self.base / "checkpoint.zip"
        with patch.object(audit, "_dependencies", return_value=self.dependencies(complete=complete)), \
             patch.object(audit, "_source_snapshot", return_value=({"code/tiny.py": b"# code\n"}, "c" * 40)):
            receipt = audit.export_audit_zip(self.run, output)
        return output, receipt

    def test_complete_archive_is_allowlisted_and_every_member_is_hashed(self):
        (self.run / ".env").write_text("SECRET_SENTINEL_MUST_NOT_EXPORT")
        (self.run / "full-original-dataset.json").write_text("ORIGINAL_SENTINEL_MUST_NOT_EXPORT")
        (self.run / "memories.sqlite").write_bytes(b"sqlite fixture")
        output, receipt = self.exported()
        self.assertTrue(receipt["final"])
        self.assertEqual(self.events, ["lock", "validate", "score", "unlock"])
        self.assertEqual(receipt["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
            self.assertIn("run/objective_summary.json", names)
            self.assertIn("run/prompts.jsonl", names)
            self.assertIn("code/tiny.py", names)
            self.assertNotIn("run/.env", names)
            self.assertNotIn("run/full-original-dataset.json", names)
            for line in archive.read("SHA256SUMS").decode().splitlines():
                expected, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            content = b"".join(archive.read(name) for name in names)
            self.assertNotIn(b"SECRET_SENTINEL", content)
            self.assertNotIn(b"ORIGINAL_SENTINEL", content)

    def test_partial_archive_cannot_publish_stale_final_files(self):
        for name in audit.SCORE_FILES:
            (self.run / name).write_text("STALE_FINAL")
        output, receipt = self.exported(complete=False)
        self.assertFalse(receipt["final"])
        with zipfile.ZipFile(output) as archive:
            self.assertIn("run/objective_progress.json", archive.namelist())
            self.assertNotIn("run/objective_summary.json", archive.namelist())
            declaration = json.loads(archive.read("AUDIT_DECLARATION.json"))
            self.assertEqual(declaration["status"], "partial_recovery_checkpoint")

    def test_symlink_is_rejected_before_loader_or_scorer(self):
        (self.run / "responses.jsonl").symlink_to(self.base / "outside.txt")
        with patch.object(audit, "_dependencies", return_value=self.dependencies()):
            with self.assertRaises(ValueError):
                audit.export_audit_zip(self.run, self.base / "checkpoint.zip")
        self.assertEqual(self.events, ["lock", "unlock"])

    def test_invalid_frozen_run_does_not_leave_an_archive(self):
        output = self.base / "checkpoint.zip"
        with patch.object(audit, "_dependencies", return_value=self.dependencies(fail=True)):
            with self.assertRaises(ValueError):
                audit.export_audit_zip(self.run, output)
        self.assertFalse(output.exists())

    def test_existing_archive_and_live_directory_outputs_are_rejected(self):
        existing = self.base / "checkpoint.zip"
        existing.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            audit.export_audit_zip(self.run, existing)
        self.assertEqual(existing.read_bytes(), b"keep me")
        with self.assertRaises(ValueError):
            audit.export_audit_zip(self.run, self.run / "checkpoint.zip")

    def test_atomic_publication_cannot_overwrite_a_concurrent_output(self):
        output = self.base / "checkpoint.zip"
        original_link = os.link

        def raced_link(source, target):
            Path(target).write_bytes(b"concurrent archive")
            return original_link(source, target)

        with patch.object(audit, "_dependencies", return_value=self.dependencies()), \
             patch.object(audit, "_source_snapshot", return_value=({"code/tiny.py": b"pass\n"}, "c" * 40)), \
             patch.object(audit.os, "link", side_effect=raced_link):
            with self.assertRaises(FileExistsError):
                audit.export_audit_zip(self.run, output)
        self.assertEqual(output.read_bytes(), b"concurrent archive")
        self.assertEqual(list(self.base.glob(".objective-audit-*")), [])

    def test_snapshot_rejects_modified_code_against_committed_bytes(self):
        (self.base / "tiny.py").write_bytes(b"changed\n")
        with patch.object(audit, "CODE_FILES", ("tiny.py",)), \
             patch.object(audit.subprocess, "check_output", side_effect=["a" * 40 + "\n", b"committed\n"]):
            with self.assertRaisesRegex(ValueError, "differs"):
                audit._source_snapshot(self.base)

    def test_archive_refuses_to_omit_a_new_frozen_code_dependency(self):
        (self.run / "protocol.json").write_text(json.dumps({
            "code_sha256": {"new_dependency.py": hashlib.sha256(b"must include\n").hexdigest()}}))
        with self.assertRaisesRegex(ValueError, "Frozen code dependency"):
            self.exported()
        self.assertFalse((self.base / "checkpoint.zip").exists())


class ObjectiveAuditStandaloneTests(unittest.TestCase):
    def test_export_can_be_rescored_outside_original_repository(self):
        # Synthetic single-case fixture; construct genuine frozen artifacts and
        # valid mock responses, then execute only the ZIP's own scorer/code.
        import run_objective_comparison as runner
        from run_three_arm import Settings

        case = {"case_id": "fixture:q1", "group_id": "fixture:g1", "subject_id": "fixture:u1",
                "dataset_key": "memoryagentbench_conflict", "task": "factconsolidation_sh_6k",
                "query": "When is the appointment?", "options": [], "events": [],
                "history": [{"turn_id": "t1", "session_id": "s1", "role": "user",
                             "text": "My appointment is Thursday.", "order": 0}],
                "history_scope_id": "fixture:h1", "ordering_kind": "source_order", "as_of": None}
        reference = {"case_id": case["case_id"], "group_id": case["group_id"],
                     "dataset_key": case["dataset_key"], "task": case["task"],
                     "metric": "mab_substring_exact_match", "accepted_answers": ["Thursday"]}
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run = base / "fixture"
            runner.prepare_run([case], [reference], run, Settings(model="synthetic-model"),
                               "https://example.invalid/v1", {"dataset_key": case["dataset_key"]},
                               fixture_only=True)
            protocol, prompts, _, _ = runner.load_run(run)
            for key, prompt in runner._canonical(prompts).items():
                intent = {"request_key": key, "canonical_request_key": key, "attempt_index": 1,
                          "protocol_sha256": protocol["protocol_sha256"],
                          "prompt_sha256": prompt["prompt_sha256"], "started_at_unix": 1.}
                row = {**intent, "provider_model": "synthetic-model", "hypothesis": "Thursday",
                       "finish_reason": "stop", "latency_seconds": 0.,
                       "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}}
                row["response_sha256"] = runner.stable_digest(row)
                runner._append(run / "response_attempts.jsonl", intent)
                runner._append(run / "responses.jsonl", row)
                runner._append(run / "response_outcomes.jsonl", {**intent, "status": "success", "retryable": False})

            # New source files have not been committed at test time; use their
            # actual bytes for this fixture while the production exporter checks
            # each byte against a committed immutable Git checkout.
            source = {"code/" + path: (ROOT / path).read_bytes() for path in audit.CODE_FILES}
            with patch.object(audit, "_source_snapshot", return_value=(source, "d" * 40)):
                receipt = audit.export_audit_zip(run, base / "audit.zip")
            self.assertTrue(receipt["final"])
            isolated = base / "isolated"
            with zipfile.ZipFile(base / "audit.zip") as archive:
                archive.extractall(isolated)
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"}
            command = [sys.executable, "-s", str(isolated / "code/evaluation/objective-memory/score_objective_comparison.py"),
                       "--run-dir", str(isolated / "run")]
            result = subprocess.run(command, cwd=isolated, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((isolated / "run/objective_summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertFalse(summary["clinical_or_user_benefit_claim"])


if __name__ == "__main__":
    unittest.main()
