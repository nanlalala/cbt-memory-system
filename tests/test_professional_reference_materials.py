"""Tiny invented fixtures only; no released CBT data or model requests."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "professional_reference_materials", ROOT / "evaluation/cbt-memory-effect/professional_reference_materials.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def encoded(value):
    return json.dumps(value).encode()


def reference(item_id="4_b_1", level="beginner"):
    return {"id": item_id, "level": level,
            "client_statement": "INVENTED_CLIENT: I tried a small practice.",
            "response": "INVENTED_REFERENCE_ONLY: Explore that attempt together."}


def pairwise():
    row = {"Input.client_statement": "INVENTED_CLIENT",
           "Input.id": "4_b_1-invented-model", "Input.level": "beginner",
           "Input.order": "A_is_ref", "Input.res_a": "INVENTED_REFERENCE",
           "Input.res_b": "INVENTED_MODEL", "Input.exe": "INVENTED_EXERCISE"}
    for i in range(1, 5):
        row[f"Input.cr_{i}"] = f"INVENTED_CRITERION_{i}"
    for criterion in ("all", "cr_1", "cr_2", "cr_3", "cr_4"):
        for suffix in M.LABEL_SUFFIXES:
            key = f"Answer.{criterion}_{suffix}"
            row[key] = key.removeprefix("Answer.") if suffix == "a_e_b" else ""
    return row


class ProfessionalReferenceTests(unittest.TestCase):
    def test_inputs_have_no_reference_labels_or_invented_history(self):
        package = M.compile_references({"CBT-DP/dp-reference-exe-4.json": encoded([reference()])})
        public = json.dumps(package["inputs"])
        self.assertNotIn("INVENTED_REFERENCE_ONLY", public)
        self.assertNotIn("reference_response", public)
        row = package["inputs"][0]
        self.assertTrue(row["quality_anchor_role"])
        self.assertFalse(row["memory_effect_eligible"])
        self.assertEqual(row["history"], [])
        self.assertEqual(row["history_count"], 0)
        self.assertEqual(row["category_id"], 4)
        self.assertEqual(row["case_id"], "cbt-dp-4_b_1")
        self.assertEqual(package["references"][0]["reference_response"], reference()["response"])

    def test_schema_extras_and_missing_fields_rejected(self):
        extra = reference(); extra["hidden_future"] = "invented"
        missing = reference(); del missing["response"]
        for row in (extra, missing):
            with self.assertRaisesRegex(ValueError, "schema"):
                M.parse_reference_bytes(encoded([row]), category_id=4)

    def test_manual_exposure_is_registered_and_propagates_without_holdout_claim(self):
        known = reference("4_m_6", "intermediate")
        linked = reference("4_b_1")
        category = reference("9_b_2"); category["client_statement"] = "INVENTED_CATEGORY_NINE"
        unseen = reference("4_b_2"); unseen["client_statement"] = "INVENTED_NOT_INSPECTED"
        package = M.compile_references({
            "CBT-DP/dp-reference-exe-4.json": encoded([known, linked, unseen]),
            "CBT-DP/dp-reference-exe-9.json": encoded([category])})
        statuses = {row["source_item_id"]: row["exposure_status"] for row in package["inputs"]}
        for item in ("4_m_6", "4_b_1", "9_b_2"):
            self.assertEqual(statuses[item], "development_exposed")
        self.assertEqual(statuses["4_b_2"],
                         "not_manually_inspected_in_this_run_not_verified_holdout")
        self.assertEqual(len(M.MANUALLY_EXPOSED_ITEM_IDS), 11)
        self.assertFalse(package["exposure_register"]["unlisted_rows_are_verified_holdout"])
        self.assertEqual(package["counts"]["by_exposure_status"]["development_exposed"], 3)

    def test_category_difficulty_and_duplicate_ids_rejected(self):
        for row, category, reason in [(reference(), 3, "category"),
                                      (reference(level="advanced"), 4, "level")]:
            with self.assertRaisesRegex(ValueError, reason):
                M.parse_reference_bytes(encoded([row]), category_id=category)
        with self.assertRaisesRegex(ValueError, "duplicate_reference_id"):
            M.parse_reference_bytes(encoded([reference(), reference()]), category_id=4)

    def test_actual_counts_not_clerical_table_counts(self):
        rows = [reference("4_b_1"), reference("4_m_1", "intermediate"),
                reference("4_m_2", "intermediate")]
        package = M.compile_references({"CBT-DP/dp-reference-exe-4.json": encoded(rows)})
        self.assertEqual(package["counts"]["reference_exercises"], 3)
        self.assertEqual(package["counts"]["by_category_id"], {"4": 3})
        self.assertEqual(package["counts"]["by_difficulty"], {"beginner": 1, "intermediate": 2})
        self.assertEqual(len({r["group_id"] for r in package["inputs"]}), 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            M.compile_references({"CBT-DP/dp-reference-exe-4.json": encoded(rows)},
                                 require_all_categories=True)

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for payload in (b'{"id":"x","id":"y"}', b'[NaN]'):
            with self.assertRaises(ValueError):
                M.parse_json(payload)

    def test_card_license_must_be_frontmatter_not_prose(self):
        self.assertEqual(M.validate_license_card(b"---\nlicense: cc-by-nc-4.0\n---\nText"), M.LICENSE)
        for payload in (b"license: cc-by-nc-4.0", b"---\nlicense: mit\n---\nlicense: cc-by-nc-4.0",
                        b"---\nlicense: cc-by-nc-4.0\nlicense: mit\n---"):
            with self.assertRaises(ValueError):
                M.validate_license_card(payload)

    def test_pairwise_preserves_labels_and_does_not_create_scores(self):
        row = pairwise(); before = copy.deepcopy(row)
        result = M.inspect_pairwise_bytes(encoded([row]))
        self.assertEqual(row, before)
        self.assertEqual(result["source_rows"], 1)
        self.assertEqual(result["structurally_valid_rows"], 1)
        self.assertFalse(result["new_scores_created"])
        self.assertFalse(result["schema_inspection_is_semantic_validation"])
        self.assertEqual(result["new_model_judgments"], 0)
        self.assertNotIn("INVENTED_REFERENCE", json.dumps(result))

    def test_pairwise_ambiguous_or_unrecognized_labels_diagnosed(self):
        a = pairwise(); a["Answer.all_a_mb_b"] = "all_a_mb_b"
        b = pairwise(); b["Answer.all_a_e_b"] = "unsupported"
        c = pairwise(); c["Input.order"] = "invented_order"
        result = M.inspect_pairwise_bytes(encoded([a, b, c]))
        self.assertEqual(result["source_rows"], 3)
        self.assertEqual(result["structurally_valid_rows"], 0)
        self.assertEqual(result["invalid_row_reasons"], {
            "invalid_or_missing_one_hot_label": 2, "invalid_reference_order": 1})

    def test_write_is_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "original.json"
            M._write_json(path, {"fixture_only": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                M._write_json(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), before)

    def test_guard_runs_before_filesystem_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "not-created"
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(M, "urlopen") as network:
                with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                    M.acquire(output)
                self.assertFalse(output.exists())
                network.assert_not_called()

    def test_download_limit_rejects_large_or_empty_fixture(self):
        for content in (b"", b"x" * (M.MAX_SOURCE_BYTES + 1)):
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.geturl.return_value = "https://huggingface.co/example"
            response.headers = {}
            response.read.return_value = content
            with mock.patch.object(M, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "bound_or_empty"):
                    M._download_bytes("https://huggingface.co/fixture")


if __name__ == "__main__":
    unittest.main()
