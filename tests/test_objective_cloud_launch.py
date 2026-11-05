"""Cloud-entry identity and finite scheduler checks; no Colab/API/data execution."""
import ast
from contextlib import redirect_stdout
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "evaluation/objective-memory"
PIN = "1511706b3a4b0ac71416998b5b4d752a13c4e313"
OLD = DIRECTORY / "Objective_Memory_Full_V3_Colab.ipynb"
NEW = DIRECTORY / "Objective_Memory_Authorized_RunAll_Colab.ipynb"


def notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_cells(path):
    return ["".join(cell["source"]) for cell in notebook(path)["cells"] if cell["cell_type"] == "code"]


def frozen_driver(directory):
    """Load the actual pinned driver bytes, rather than trusting today's helper."""
    relative = "evaluation/objective-memory/v3_suite_driver.py"
    source = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PIN}:{relative}"])
    path = Path(directory) / "pinned_suite_driver.py"
    path.write_bytes(source)
    sys.path.insert(0, str(DIRECTORY))
    try:
        spec = importlib.util.spec_from_file_location("frozen_suite_driver_151170", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(DIRECTORY))
    return module


class ObjectiveCloudLaunchTests(unittest.TestCase):
    def test_exact_bootstrap_is_accepted_by_actual_pinned_exposure_reader(self):
        for path in (OLD, NEW):
            with self.subTest(path=path.name), tempfile.TemporaryDirectory() as directory:
                cells = code_cells(path)
                exposure_cell = next(source for source in cells if "EXPOSURE_PROTOCOL_TEXT=" in source)
                namespace = {"WORK": Path(directory), "hashlib": hashlib}
                with redirect_stdout(io.StringIO()):
                    exec(compile(exposure_cell, path.name, "exec"), namespace)
                driver = frozen_driver(directory)
                cases, identity = driver._exposure(namespace["EXPOSURE"])
                self.assertEqual(len(cases), 16)
                self.assertEqual(namespace["EXPOSURE"].name, "cases.jsonl")
                self.assertEqual(identity["source_protocol_sha256"], "9b798544eaf0f7b4355eb7b2772e199b45f80420dc648154942db54f0751d4fd")
                self.assertEqual(identity["cases_sha256"], "d9cd969e9f9d22d3f9b35626433d6fd809e9ad3c4cf60616ddfd89651e6b29e9")
                self.assertTrue(all("history" not in case and "accepted_answers" not in case and "answer" not in case for case in cases))
                # A rerun is idempotent, while contradictory old artifacts stay unchanged.
                with redirect_stdout(io.StringIO()):
                    exec(compile(exposure_cell, path.name, "exec"), namespace)
                namespace["EXPOSURE"].write_text("tampered\n")
                with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
                    exec(compile(exposure_cell, path.name, "exec"), namespace)
                self.assertEqual(namespace["EXPOSURE"].read_text(), "tampered\n")

    def test_original_exposure_protocol_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            source = next(value for value in code_cells(NEW) if "EXPOSURE_PROTOCOL_TEXT=" in value)
            namespace = {"WORK": Path(directory), "hashlib": hashlib}
            with redirect_stdout(io.StringIO()):
                exec(source, namespace)
            saved = namespace["EXPOSURE_PROTOCOL"]
            saved.write_text("{}")
            with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
                exec(source, namespace)
            self.assertEqual(saved.read_text(), "{}")

    def test_new_entry_is_unexecuted_cloud_only_foreground_and_explicitly_authorized(self):
        value = notebook(NEW)
        self.assertEqual(value["metadata"]["capstone"]["executable_commit"], PIN)
        self.assertTrue(value["metadata"]["capstone"]["paid_execution_on_run_all"])
        cells = code_cells(NEW)
        self.assertEqual(len(cells), 3)
        for cell in value["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                ast.parse("".join(cell["source"]))
        self.assertIn("COLAB_RELEASE_TAG", cells[0])
        self.assertIn("google.colab", cells[0])
        combined = "\n".join(cells)
        self.assertNotIn("Popen", combined)
        self.assertNotIn("files.download", combined)
        self.assertNotIn("drive.mount", combined)
        self.assertNotIn("while True", combined)
        self.assertIn("userdata.get('Capstone')", combined)
        self.assertIn("maximum_attempts_per_invocation':7200", combined)
        self.assertIn("secret=None", combined)
        self.assertIn("from None", combined)
        self.assertIn("ephemeral", combined)

    def _scheduler(self):
        source = code_cells(NEW)[-1]
        tree = ast.parse(source)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_bounded")
        scope = {"MAX_PASSES": 10, "WORKERS": 4, "CALLS_PER_SHARD_PER_PASS": 60,
                 "SUITE": Path("unused-synthetic-suite"), "INVOCATION": "fixture", "json": json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture-scheduler", "exec"), scope)
        return scope["execute_bounded"]

    def test_cached_foreign_repository_module_is_rejected_before_secret_or_model(self):
        source = code_cells(NEW)[-1]
        tree = ast.parse(source)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "assert_pinned_module_origins")
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture-origin-check", "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "pinned"
            repo.mkdir()
            correct = repo / "v3_suite_driver.py"
            correct.write_text("# tiny source-only fixture\n")
            wrong = Path(directory) / "old" / "v3_suite_driver.py"
            checker = scope["assert_pinned_module_origins"]
            checker(repo, {"v3_suite_driver": SimpleNamespace(__file__=str(correct))}, ["v3_suite_driver.py"])
            with self.assertRaisesRegex(RuntimeError, "cached from another executable"):
                checker(repo, {"v3_suite_driver": SimpleNamespace(__file__=str(wrong))}, ["v3_suite_driver.py"])
            with self.assertRaises(RuntimeError):
                checker(repo, {"v3_suite_driver": SimpleNamespace()}, ["v3_suite_driver.py"])
            # An untracked lookalike inside the repository is not an approved origin.
            untracked = repo / "untracked" / "v3_suite_driver.py"
            untracked.parent.mkdir()
            untracked.write_text("# untracked stale helper\n")
            with self.assertRaises(RuntimeError):
                checker(repo, {"v3_suite_driver": SimpleNamespace(__file__=str(untracked))}, ["v3_suite_driver.py"])
        self.assertLess(source.index("assert_pinned_module_origins(REPO,sys.modules)"), source.index("import v3_suite_driver"))
        self.assertLess(source.index("assert_pinned_module_origins(REPO,sys.modules)"), source.index("userdata.get('Capstone')"))

    def test_scheduler_never_exceeds_finite_call_bound_or_scores_partial(self):
        class Fake:
            def __init__(self): self.calls = []
            def execute_suite(self, path, **kwargs):
                self.calls.append(kwargs)
                return {"all_shards_complete": False, "results": [{"stop_reason": "finite_round_cap"}] * 12}
            def score_suite(self, path): raise AssertionError("partial suite must not produce final scores")
        fake = Fake()
        checkpoints = []
        with redirect_stdout(io.StringIO()):
            result = self._scheduler()(fake, lambda s, label: checkpoints.append(label))
        self.assertFalse(result["complete"])
        self.assertEqual(len(fake.calls), 10)
        self.assertEqual(len(checkpoints), 10)
        self.assertTrue(all(call == {"workers": 4, "max_rounds": 1, "max_calls_per_batch": 60} for call in fake.calls))
        self.assertEqual(sum(12 * call["max_rounds"] * call["max_calls_per_batch"] for call in fake.calls), 7200)

    def test_whole_checkpoint_holds_the_suite_execution_lock(self):
        source = code_cells(NEW)[-1]
        tree = ast.parse(source)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "checkpoint_pass")
        held = []
        @contextmanager
        def lock(path):
            held.append(path)
            try:
                yield
            finally:
                held.pop()
        def checkpoint(suite, label):
            self.assertEqual(held, [Path("synthetic-suite")])
            return {"fixture": label}
        scope = {"SUITE": Path("synthetic-suite"), "checkpoint_pass_locked": checkpoint}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "fixture-checkpoint", "exec"), scope)
        fake = SimpleNamespace(runner=SimpleNamespace(exclusive_directory=lock))
        self.assertEqual(scope["checkpoint_pass"](fake, "pass-01"), {"fixture": "pass-01"})
        self.assertEqual(held, [])

    def test_scheduler_stops_on_inflight_and_waits_for_complete_before_final_score(self):
        class Fake:
            def __init__(self, complete=False): self.calls, self.scores, self.complete = 0, 0, complete
            def execute_suite(self, path, **kwargs):
                self.calls += 1
                return {"all_shards_complete": self.complete,
                        "results": [{"stop_reason": "complete" if self.complete else "unresolved_inflight"}]}
            def score_suite(self, path):
                self.scores += 1
                return {"status": "complete", "fixture": True}
        for complete in (False, True):
            with self.subTest(complete=complete):
                fake = Fake(complete)
                with redirect_stdout(io.StringIO()):
                    result = self._scheduler()(fake, lambda s, label: {"fixture_checkpoint": label})
                self.assertEqual(fake.calls, 1)
                self.assertEqual(fake.scores, int(complete))
                self.assertEqual(result["complete"], complete)


if __name__ == "__main__":
    unittest.main()
