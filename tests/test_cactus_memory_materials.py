"""Small invented cloud fixtures, never a public-data/model experiment."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cactus_materials", ROOT / "evaluation/cbt-memory-effect/cactus_materials.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class CactusMaterialTests(unittest.TestCase):
    def rows(self):
        return M.invented_fixture_rows()

    def test_source_spans_and_cutoffs_never_include_future(self):
        rows = self.rows()
        package = M.compile_rows(rows, family_limit=2, fixture_only=True)
        self.assertEqual(len(package["generation"]), 6)
        for case, control in zip(package["generation"], package["controller"]):
            raw = rows[control["source_row_index"]]["dialogue"]
            turns = M.parse_dialogue(raw)
            cutoff = control["cutoff"]
            self.assertEqual(case["input"], turns[cutoff]["content"])
            self.assertEqual(case["history"], [{key: t[key] for key in
                             ("message_id", "role", "content", "date", "source_order")} for t in turns[:cutoff]])
            self.assertTrue(all(t["date"] is None for t in case["history"]))
            for span in control["visible_source_spans"]:
                turn = next(t for t in turns if t["message_id"] == span["message_id"])
                self.assertEqual(raw[span["source_start"]:span["source_end"]], turn["content"])
            for future in turns[cutoff+1:]:
                self.assertNotIn(future["content"], json.dumps(case))

    def test_hidden_metadata_and_future_answer_never_enter_generation(self):
        rows = self.rows()
        rows[0].update(cbt_plan="SECRET_FUTURE_PLAN", patterns=["SECRET_LABEL"], intake_form="SECRET_INTAKE")
        package = M.compile_rows(rows, family_limit=2, fixture_only=True)
        text = json.dumps(package["generation"])
        for excluded in ["SECRET_FUTURE_PLAN", "SECRET_LABEL", "SECRET_INTAKE", "cbt_plan", "patterns", "intake_form"]:
            self.assertNotIn(excluded, text)
        self.assertTrue(all(r["memory_targets"] is None for r in package["references"]))
        self.assertFalse(package["summary"]["semantic_memory_effect_scoring_ready"])

    def test_shared_thought_or_intake_connects_transitively(self):
        row = self.rows()[0]
        rows = [copy.deepcopy(row) for _ in range(3)]
        rows[1].update(thought="a second thought", cbt_technique="other")
        rows[2].update(thought="a second thought", intake_form="a second intake", attitude="negative")
        for i, r in enumerate(rows):
            r["dialogue"] = r["dialogue"].replace("report 0-", f"report {i}-")
        groups, rejected = M.group_rows(rows)
        self.assertEqual(rejected, [])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["members"]), 3)
        result = M.compile_rows(rows, family_limit=1, fixture_only=True)
        self.assertEqual(result["summary"]["planned_cases"], 3)

    def test_row_reordering_preserves_selected_visible_cases(self):
        rows = self.rows()
        duplicate = copy.deepcopy(rows[0]); duplicate["attitude"] = "positive"
        rows.append(duplicate)
        a = M.compile_rows(rows, family_limit=2, fixture_only=True)
        b = M.compile_rows(list(reversed(rows)), family_limit=2, fixture_only=True)
        self.assertEqual(a["generation"], b["generation"])

    def test_ambiguous_inline_labels_fail_instead_of_guessed_roles(self):
        dialogue = self.rows()[0]["dialogue"]
        with self.assertRaisesRegex(ValueError, "ambiguous_inline"):
            M.parse_dialogue(dialogue.replace("\nClient:", " Client:", 1))
        with self.assertRaisesRegex(ValueError, "nonalternating"):
            M.parse_dialogue(dialogue.replace("\nClient:", "\nCounselor:", 1))

    def test_rejected_source_schema_keeps_denominator_and_blocks_completeness(self):
        rows = self.rows()
        rows[1]["intake_form"] = ""
        result = M.compile_rows(rows, family_limit=2, fixture_only=True)
        self.assertEqual(result["summary"]["source_rows"], 2)
        self.assertEqual(result["summary"]["source_errors"], 1)
        self.assertFalse(result["summary"]["grouping_complete"])
        self.assertFalse(result["summary"]["source_preparation_complete"])
        self.assertFalse(result["summary"]["generation_ready"])

    def test_full_output_manifest_is_bound_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "new"
            result = M.compile_rows(self.rows(), family_limit=2, fixture_only=True)
            M.write_package(target, result, source_identity={"fixture_only": True})
            manifest = json.loads((target / "manifest.json").read_text())
            expected = manifest.pop("manifest_sha256")
            self.assertEqual(expected, M.digest(manifest))
            for relative, expected_hash in manifest["files_sha256"].items():
                self.assertEqual(M.file_hash(target / relative), expected_hash)
            with self.assertRaises(FileExistsError):
                M.write_package(target, result, source_identity={"fixture_only": True})

    def test_family_and_cutoff_limits_are_frozen(self):
        for limit in [0, 201, True]:
            with self.assertRaises(ValueError):
                M.compile_rows(self.rows(), family_limit=limit, fixture_only=True)
        with self.assertRaises(ValueError):
            M.choose_cutoffs(M.parse_dialogue(self.rows()[0]["dialogue"]), 4)


if __name__ == "__main__":
    unittest.main()
