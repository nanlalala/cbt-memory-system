"""Small invented metadata fixtures; no real dialogue download or model calls."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/freeze_therapyjudgebench.py"
SPEC = importlib.util.spec_from_file_location("freeze_therapyjudgebench", PATH)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def fixture(ident="Conversation 1", profile="fixture-profile"):
    return {"id": ident, "session_id": "fixture-" + ident, "patient_profile": profile,
            "therapist_model": "FIXTURE", "patient_model": "O3_MINI", "num_turns": 5,
            "therapist_style": "professional_cbt_therapist",
            "conversation": {"messages": [{"role": "user" if i % 2 == 0 else "assistant",
                                            "content": "Invented test placeholder."} for i in range(10)]},
            "expert_rating": {"ctrs_ratings": {k: None for k in mod.CTRS},
                              "safety_ethics": {k: None for k in mod.SAFETY},
                              "overall_ctrs_score": None}}


class FreezeTherapyJudgeBenchTests(unittest.TestCase):
    def audit(self, rows):
        return mod.audit_records(rows, require_release_counts=False)

    def test_nulls_are_excluded_and_profile_exposure_propagates(self):
        a, b = fixture(), fixture("Conversation 99")
        a["expert_rating"]["ctrs_ratings"]["AGENDA"] = 0
        audit, groups = self.audit([a, b])
        self.assertEqual(audit["ctrs_by_dimension"]["AGENDA"], {"nonmissing": 1, "null": 1})
        self.assertEqual(audit["safety_by_dimension"][mod.SAFETY[0]], {"null": 2})
        self.assertTrue(all(g["exposure"] == "development_exposed" for g in groups))

    def test_no_automatic_holdout_claim(self):
        _, groups = self.audit([fixture("Conversation 99", "unseen-fixture")])
        self.assertEqual(groups[0]["exposure"], "unexposed_candidate_not_sealed_test")

    def test_old_schema_and_integer_safety_rejected(self):
        a = fixture()
        a["expert_rating"]["safety_ethics"]["OLD FIFTH CATEGORY"] = False
        with self.assertRaisesRegex(ValueError, "v2_exact_label_schema"):
            self.audit([a])
        a = fixture()
        a["expert_rating"]["safety_ethics"] = {k: 0 for k in mod.SAFETY}
        with self.assertRaisesRegex(ValueError, "boolean_or_null"):
            self.audit([a])

    def test_invalid_numbers_and_future_role_order_rejected(self):
        for value in (True, float("nan"), 7):
            a = fixture()
            a["expert_rating"]["ctrs_ratings"]["AGENDA"] = value
            with self.assertRaisesRegex(ValueError, "invalid_ctrs_score"):
                self.audit([a])
        a = fixture()
        a["conversation"]["messages"][0]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "ordered_alternating"):
            self.audit([a])

    def test_release_counts_enforced_in_production(self):
        with self.assertRaisesRegex(ValueError, "release_counts_mismatch"):
            mod.audit_records([fixture()])

    def test_duplicate_keys_nonstandard_numbers_and_ids_rejected(self):
        for data in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                mod.parse_json(data)
        with self.assertRaisesRegex(ValueError, "duplicate_conversation"):
            self.audit([fixture(), copy.deepcopy(fixture())])

    def test_guard_before_any_output_or_download(self):
        with patch.object(mod, "require_colab", side_effect=RuntimeError("blocked")), \
                patch.object(mod, "download") as fetch, patch.object(mod.Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                mod.freeze("/content/test-fixture-do-not-create")
            fetch.assert_not_called(); mkdir.assert_not_called()

    def test_only_exact_pinned_official_redirects_allowed(self):
        mod._check_url(f"https://huggingface.co/api/resolve-cache/datasets/{mod.REPOSITORY}/{mod.REVISION}/README.md?download=true", "README.md")
        for url in (f"https://huggingface.co/datasets/{mod.REPOSITORY}/resolve/main/README.md",
                    "https://example.org/README.md", "https://huggingface.co/login"):
            with self.assertRaises(ValueError):
                mod._check_url(url, "README.md")


if __name__ == "__main__":
    unittest.main()
