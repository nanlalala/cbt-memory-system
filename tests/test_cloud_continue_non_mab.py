"""One-shot orchestration guards only; no runtime, credentials or model calls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evaluation/architecture-rerun/cloud_continue_non_mab.py"
SPEC = importlib.util.spec_from_file_location("non_mab_driver_unit", SOURCE)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class NonMabAdmissionTests(unittest.TestCase):
    def test_unhosted_stops_before_parent_source_or_checkout(self):
        with patch.object(driver, "_hosted", side_effect=RuntimeError("unit unhosted")), \
                patch.object(driver, "_parents") as parents, patch.object(driver, "_source_paths") as sources, \
                patch.object(driver, "_checkout") as checkout:
            with self.assertRaises(RuntimeError):
                driver.run({}, repo="unused", expected_commit="unused", sources={})
            parents.assert_not_called(); sources.assert_not_called(); checkout.assert_not_called()

    def test_repeat_reports_existing_nonce_without_any_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "entry.json").write_text(json.dumps({"nonce": "original", "status": "stopped_without_retry"}))
            with patch.object(driver, "_hosted"), patch.object(driver, "_content_path", side_effect=lambda p, **kw: Path(p)), \
                    patch.object(driver, "_parents") as parents, patch.object(driver, "_checkout") as checkout:
                result = driver.run({}, root=root, repo="unused", expected_commit="unused", sources={})
                self.assertEqual(result["nonce"], "original")
                self.assertEqual(result["repeat_run_action"], "report_existing_invocation_only")
                parents.assert_not_called(); checkout.assert_not_called()

    def test_missing_parent_never_accesses_a_secret_or_restores_a_wrapper(self):
        with self.assertRaises(RuntimeError):
            driver._parents({})
        answer, judge = Mock(identity=driver.ANSWER_IDENTITY), Mock(identity=driver.JUDGE_IDENTITY)
        old_wrapper = SimpleNamespace(delegate=answer)
        namespace = {"v5_answer": old_wrapper, "v5_judge": SimpleNamespace(delegate=judge)}
        self.assertIs(driver._parents(namespace)[driver.ANSWER_IDENTITY["model"]], answer)
        self.assertIs(namespace["v5_answer"], old_wrapper)
        answer.assert_not_called(); judge.assert_not_called()

    def test_old_live_thread_and_process_both_block_new_admission(self):
        answer, judge = Mock(identity=driver.ANSWER_IDENTITY), Mock(identity=driver.JUDGE_IDENTITY)
        namespace = {"v5_answer": SimpleNamespace(delegate=answer), "v5_judge": SimpleNamespace(delegate=judge)}
        namespace["v6_cbt_process"] = SimpleNamespace(poll=lambda: None)
        with self.assertRaises(RuntimeError): driver._parents(namespace)
        del namespace["v6_cbt_process"]
        thread = threading.Thread()
        with patch.object(thread, "is_alive", return_value=True):
            namespace["v5_threads"] = [thread]
            with self.assertRaises(RuntimeError): driver._parents(namespace)
        answer.assert_not_called(); judge.assert_not_called()

    def test_prior_new_quality_calls_stop_instead_of_spending_repair_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality = root / "professional7_quality_v2"
            quality.mkdir()
            (quality / "intents.jsonl").write_text('{"key":"prior"}\n')
            with self.assertRaises(RuntimeError):
                driver._prior_recovery({"professional7": {"reader": str(root / "professional7")}})

    def test_source_only_lme_default_cannot_implicitly_authorize_paid_calls(self):
        self.assertIs(driver.run.__kwdefaults__["allow_longmemeval_paid"], False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(driver, "_hosted"), \
                patch.object(driver, "_content_path", side_effect=lambda p, **kw: Path(p)), \
                patch.object(driver, "_checkout") as checkout:
            with self.assertRaises(ValueError):
                driver.run({}, repo="unused", expected_commit="unused", sources={}, root=tmp,
                           allow_longmemeval_paid=True)
            checkout.assert_not_called()

    def test_incomplete_or_item_error_result_is_not_a_finished_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            quality = output / "quality_v2"
            quality.mkdir()
            (quality / "last_invocation.json").write_text('{"errors":[]}')
            with self.assertRaises(RuntimeError):
                driver._phase_result("cbt-quality-run", "professional7", {"complete": False}, output)
            (quality / "last_invocation.json").write_text('{"errors":[{"error_type":"CallLimit"}]}')
            with self.assertRaises(RuntimeError):
                driver._phase_result("cbt-quality-run", "professional7", {"complete": True}, output)

    def test_professional_grid_requires_21_bound_answers_before_grading(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "reader").mkdir()
            (output / "reader/last_invocation.json").write_text('{"errors":[]}')
            summary = {"planned_cases":7, "completed_cases":7, "unknown_receipts":0,
                       "unbound_answer_files":[], "arms":{"no_rag":{"received":7},
                       "raw_history_rag":{"received":7}, "managed_memory_rag":{"received":3}}}
            with self.assertRaises(RuntimeError): driver._phase_result("cbt-run", "professional7", summary, output)

    def test_old_local_paid_intents_and_invalid_quality_are_counted(self):
        sources = {c:{"reader":c+"-reader", "memory":c+"-memory", "old_quality":c+"-quality"}
                   for c in ("professional7", "cactus600")}
        sources["locomo1540"] = {"reader":"locomo-reader", "old_quality":"locomo-quality"}
        values = {"professional7-reader":27, "professional7-memory":24, "professional7-quality":19,
                  "cactus600-reader":889, "cactus600-memory":872, "cactus600-quality":0,
                  "locomo-reader":3319, "locomo-quality":0}
        prepared = {"professional7":{"maximum_new_generation_requests":8, "maximum_new_assertion_requests":4,
                         "quality_after_complete_grid":{"with_independent_10_percent_repeats":24}},
                    "cactus600":{"maximum_new_generation_requests":2111, "maximum_new_assertion_requests":2128,
                         "quality_after_complete_grid":{"with_independent_10_percent_repeats":1980}},
                    "locomo1540":{"maximum_attempts":1638, "maximum_independent_judge_calls":5082}}
        with patch.object(driver, "_paid_census", side_effect=lambda folder, *a, **kw: {"intents":values[folder]}):
            plan = driver._budget(sources, prepared)
            self.assertEqual(plan["cbt_actual_local_paid_intents"][driver.JUDGE_IDENTITY["model"]], 915)
            self.assertEqual(plan["cbt_new_upper_bounds"][driver.JUDGE_IDENTITY["model"]], 4136)
            self.assertEqual(plan["cbt_remaining_after_paid_intents"][driver.JUDGE_IDENTITY["model"]], 4141)
            prepared["locomo1540"]["maximum_attempts"] = 1640
            with self.assertRaises(ValueError): driver._budget(sources, prepared)

    def test_optional_missing_old_quality_has_zero_recorded_local_intents(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = driver._paid_census(Path(tmp) / "absent-quality", driver.JUDGE_IDENTITY, optional=True)
            self.assertEqual(result["intents"], 0)
            with self.assertRaises(FileNotFoundError):
                driver._paid_census(Path(tmp) / "absent-reader", driver.ANSWER_IDENTITY)

    def test_archive_only_traverses_own_allowed_outputs_not_mab_or_keys(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text('{"nonce":"unit","status":"finished"}')
            (root / "unexpected-secret.txt").write_text("must-not-be-exported")
            (root / "mab").mkdir(); (root / "mab/data.json").write_text("must-not-be-exported")
            (root / "studies/professional7").mkdir(parents=True)
            (root / "studies/professional7/summary.json").write_text('{"complete":true}')
            with patch.object(driver, "_content_path", side_effect=lambda p, **kw: Path(p)):
                receipt = driver.export_checkpoint(root, complete=True)
            with zipfile.ZipFile(receipt["path"]) as archive:
                self.assertIn("studies/professional7/summary.json", archive.namelist())
                self.assertNotIn("unexpected-secret.txt", archive.namelist())
                self.assertNotIn("mab/data.json", archive.namelist())

    def test_fresh_unrelated_interpreter_can_import_formal_continuation(self):
        # A copied helper is outside the selected checkout. Match its explicit
        # import paths; do not rely on pytest's cwd or already-imported modules.
        code = "import sys; " + "; ".join("sys.path.insert(0, " + repr(str(REPO / p)) + ")" for p in (
            "modules/agent-integration", "evaluation/cbt-memory-effect", "evaluation/architecture-rerun"))
        code += "; import formal_continuation; assert callable(formal_continuation.inspect_source_only)"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
