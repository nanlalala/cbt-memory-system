"""Source-only version views: preserve unsupported facts and access boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
from v3_fact_versions import Assertion, consolidate_rows, parse_assertion


def source(text, order, turn=None, session="s", **extra):
    return {"session_id": session, "turn_id": turn or f"t{order}", "order": order, "text": text,
            "role": "context", **extra}


class FactVersionTests(unittest.TestCase):
    def test_declared_templates_parse_without_using_real_world_truth(self):
        self.assertEqual(parse_assertion("12. The official language of Fictionland is Basque."),
                         Assertion("Fictionland", "official language", "Basque"))
        self.assertEqual(parse_assertion("Dr. A. Person is a citizen of Fictionland."),
                         Assertion("Dr. A. Person", "citizenship", "Fictionland"))
        self.assertEqual(parse_assertion("A Story was written by Noam Writer."),
                         Assertion("A Story", "author", "Noam Writer"))
        self.assertEqual(parse_assertion("The country of citizenship of W. G. Example is Fictionland."),
                         Assertion("W. G. Example", "citizenship", "Fictionland"))

    def test_subject_names_containing_of_do_not_change_slot_delimiter(self):
        self.assertEqual(parse_assertion("The director of University of Fiction is Mira."),
                         Assertion("University of Fiction", "director", "Mira"))

    def test_largest_source_order_wins_and_original_source_is_preserved(self):
        old = source("1. The official language of Fictionland is Danish.", 10)
        new = source("2. The official language of Fictionland is Basque.", 20)
        rows = [new, old]; before = deepcopy(rows)
        selected, trace = consolidate_rows(rows)
        self.assertEqual(selected, [new])
        self.assertIs(selected[0], new)
        self.assertEqual(rows, before)
        self.assertEqual(trace["superseded_rows"], 1)
        self.assertEqual(trace["entries"][1]["status"], "superseded")
        self.assertEqual(trace["entries"][1]["latest_source"]["turn_id"], "t20")

    def test_unsupported_qualifications_and_negations_are_not_dropped(self):
        texts = ["The director of Fiction Institute is not Mira.",
                 "The director of Fiction Institute might be Mira.",
                 "The director of Fiction Institute is Mira and the owner is Noor.",
                 "Several directors appeared at the meeting."]
        rows = [source(text, i) for i, text in enumerate(texts)]
        for text in texts:
            self.assertIsNone(parse_assertion(text))
        selected, trace = consolidate_rows(rows)
        self.assertEqual(selected, rows)
        self.assertEqual(trace["unparsed_rows"], 4)

    def test_same_subject_different_relation_and_different_scope_are_separate(self):
        rows = [source("The director of Fiction Institute is Mira.", 1),
                source("The owner of Fiction Institute is Noor.", 2),
                source("The director of Fiction Institute is Zara.", 9, session="other")]
        selected, trace = consolidate_rows(rows)
        self.assertEqual(len(selected), 3)
        self.assertEqual(trace["version_groups"], 3)

    def test_equal_latest_order_does_not_silently_choose_one_conflict(self):
        rows = [source("The director of Fiction Institute is Mira.", 4, "first"),
                source("The director of Fiction Institute is Noor.", 4, "second")]
        selected, trace = consolidate_rows(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual(trace["unresolved_latest_tie_groups"], 1)
        self.assertTrue(all(e["status"] == "unresolved_latest_tie" for e in trace["entries"]))

    def test_alias_surface_forms_share_a_relation_only_when_declared(self):
        rows = [source("A Story was written by Mira.", 1),
                source("The author of A Story is Noor.", 5)]
        selected, trace = consolidate_rows(rows)
        self.assertEqual(selected, [rows[1]])
        self.assertEqual(trace["superseded_rows"], 1)

    def test_inactive_restricted_and_duplicate_sources_rejected(self):
        for changes in ({"lifecycle_status": "deleted"}, {"lifecycle_status": "expired"},
                        {"lifecycle_status": "superseded"}, {"sensitivity": "restricted"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "eligible"):
                consolidate_rows([source("The director of Fiction Institute is Mira.", 1, **changes)])
        row = source("The director of Fiction Institute is Mira.", 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            consolidate_rows([row, dict(row)])


class DistinctRelationBoundaryTests(unittest.TestCase):
    def test_partner_and_spouse_are_not_silently_merged(self):
        a = parse_assertion("The partner of Orion is Alpha.")
        b = parse_assertion("The spouse of Orion is Beta.")
        self.assertNotEqual(a.relation, b.relation)


if __name__ == "__main__":
    unittest.main()
