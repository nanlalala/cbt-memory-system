"""F14: historical identity bridges cannot create false independent groups."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
from v2_followup_contract import freeze_materials


def case(cid, split="development", *, sources=None, mothers=None, exposed=False):
    return {"case_id": cid, "split": split,
            "source_group_ids": sources or ["source-" + cid],
            "mother_ids": mothers or ["mother-" + cid],
            "parent_case_ids": [], "exposed": exposed}


def row(*, cases=(), sources=(), mothers=(), reason="Earlier material was inspected."):
    return {"case_ids": list(cases), "source_group_ids": list(sources),
            "mother_ids": list(mothers), "reason": reason}


class HistoricalLedgerGraphTests(unittest.TestCase):
    def test_two_rows_bridge_current_cases_via_external_mother(self):
        result = freeze_materials([case("A"), case("B")], exposure_ledger=[
            row(cases=["A"], mothers=["external"]),
            row(cases=["B"], mothers=["external"]),
        ], exposure_ledger_complete=True)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["case_ids"], ["A", "B"])
        self.assertTrue(result["groups"][0]["exposed"])

    def test_external_bridge_rejects_development_validation_split(self):
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            freeze_materials([case("A"), case("B", "validation")], exposure_ledger=[
                row(cases=["A"], mothers=["external"]),
                row(cases=["B"], mothers=["external"]),
            ], exposure_ledger_complete=True)

    def test_multihop_external_case_source_and_mother_bridge(self):
        ledger = [row(cases=["A", "old-A"]),
                  row(cases=["old-A"], sources=["old-source"]),
                  row(sources=["old-source"], mothers=["old-mother"]),
                  row(cases=["B"], mothers=["old-mother"])]
        result = freeze_materials([case("A"), case("B")], exposure_ledger=ledger)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["case_ids"], ["A", "B"])

    def test_exposure_reason_reaches_current_group_from_external_only_row(self):
        result = freeze_materials([case("A")], exposure_ledger=[
            row(cases=["A"], mothers=["external"], reason="Link seen."),
            row(mothers=["external"], sources=["old-source"], reason="Old answer inspected."),
        ], exposure_ledger_complete=True)
        self.assertEqual(result["groups"][0]["exposure_reasons"],
                         ["Link seen.", "Old answer inspected."])
        with self.assertRaisesRegex(ValueError, "Blind test"):
            freeze_materials([case("A", "blind_test")], exposure_ledger=[
                row(mothers=["mother-A"], sources=["old-source"]),
                row(sources=["old-source"]),
            ], exposure_ledger_complete=True)

    def test_identity_namespaces_do_not_connect_equal_strings(self):
        # A's mother named X is unrelated to B's source named X. An explicit
        # ledger edge is needed before differently typed identities can merge.
        cases = [case("A", mothers=["X"]), case("B", sources=["X"])]
        clean = freeze_materials(cases, exposure_ledger=[])
        self.assertEqual(len(clean["groups"]), 2)
        linked = freeze_materials(cases, exposure_ledger=[row(sources=["X"], mothers=["X"])])
        self.assertEqual(len(linked["groups"]), 1)

    def test_row_and_identity_order_do_not_change_frozen_manifest(self):
        cases = [case("A"), case("B")]
        ledger = [row(cases=["A", "old-case"], mothers=["external", "other"]),
                  row(cases=["B"], mothers=["external"])]
        first = freeze_materials(cases, exposure_ledger=ledger)
        reverse = deepcopy(ledger)
        for entry in reverse:
            for field in ("case_ids", "source_group_ids", "mother_ids"):
                entry[field].reverse()
        second = freeze_materials(list(reversed(cases)), exposure_ledger=list(reversed(reverse)))
        self.assertEqual(first, second)

    def test_equivalent_bridge_encodings_have_equal_current_groups(self):
        cases = [case("A"), case("B")]
        separate = freeze_materials(cases, exposure_ledger=[
            row(cases=["A"], mothers=["external"]),
            row(cases=["B"], mothers=["external"]),
        ])
        combined = freeze_materials(cases, exposure_ledger=[
            row(cases=["A", "B"], mothers=["external"]),
        ])
        self.assertEqual(separate["groups"], combined["groups"])
        # Different historical records remain different provenance bytes.
        self.assertNotEqual(separate["materials_sha256"], combined["materials_sha256"])


if __name__ == "__main__":
    unittest.main()
