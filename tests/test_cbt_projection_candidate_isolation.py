"""Fictional, no-network checks for versioned per-candidate source binding.

These distinguish recoverable extraction item defects from a corrupt envelope;
they do not treat literal quotation binding as semantic support.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
import cbt_projection_contract as contract


class CandidateBindingIsolation(unittest.TestCase):
    def setUp(self):
        self.sources = [
            {"source_id":"u0", "role":"user", "content":"Before lunch I completed one thought record. It was tiring."},
            {"source_id":"a0", "role":"assistant", "content":"We could try a shorter record if you would like."},
            {"source_id":"u1", "role":"user", "content":"Not today; I want to pause that exercise."},
            {"source_id":"repeat", "role":"user", "content":"again again"},
        ]
        self.item = {"item_id":"m0", "section":"intervention_learning",
            "value":{"completed_count":1, "reported_experience":"tiring", "scope":"before lunch"},
            "origin":"explicit_user_statement", "evidence":[
                {"source_id":"u0", "quote":"Before lunch I completed one thought record. It was tiring."}]}

    def bind(self, items):
        return contract.bind_candidates_with_status(json.dumps({"items":items}), self.sources)

    def assert_statuses(self, statuses, retained):
        self.assertEqual([s["candidate_index"] for s in statuses], list(range(len(statuses))))
        self.assertEqual([s["binding_status"] for s in statuses],
                         ["retained" if okay else "rejected" for okay in retained])
        for status, okay in zip(statuses, retained):
            if okay:
                self.assertIsNone(status["reason"])
            else:
                self.assertIsInstance(status["reason"], str)
                self.assertTrue(status["reason"].strip())

    def test_new_version_and_valid_binding_preserve_original_representation(self):
        self.assertEqual(contract.BINDING_VERSION, "cbt-source-projection-binding-v2")
        original = deepcopy(self.item)
        bound, statuses = self.bind([self.item])
        self.assertEqual(self.item, original)
        self.assertEqual(bound, contract.bind_candidates(json.dumps({"items":[self.item]}), self.sources))
        self.assertEqual(bound[0]["value"], original["value"])
        self.assertEqual(bound[0]["evidence"], [{"source_id":"u0", "text":self.sources[0]["content"],
            "start":0, "end":len(self.sources[0]["content"])}])
        self.assertNotIn("support_status", bound[0])
        self.assert_statuses(statuses, [True])

    def test_one_quote_error_does_not_discard_valid_peer_and_strict_api_still_fails(self):
        invalid = {**deepcopy(self.item), "item_id":"bad"}
        invalid["evidence"][0]["quote"] = "I finished five records."
        last = {**deepcopy(self.item), "item_id":"last"}
        bound, statuses = self.bind([self.item, invalid, last])
        self.assertEqual([i["item_id"] for i in bound], ["m0", "last"])
        self.assertEqual([s["item_id"] for s in statuses], ["m0", "bad", "last"])
        self.assert_statuses(statuses, [True, False, True])
        with self.assertRaises(ValueError):
            contract.bind_candidates(json.dumps({"items":[self.item, invalid, last]}), self.sources)

    def test_duplicate_ids_reject_every_occurrence_without_first_wins(self):
        peer = {**deepcopy(self.item), "item_id":"unique"}
        duplicate = {**deepcopy(self.item), "value":"A second interpretation with the same ID."}
        bound, statuses = self.bind([self.item, peer, duplicate])
        self.assertEqual([i["item_id"] for i in bound], ["unique"])
        self.assert_statuses(statuses, [False, True, False])

    def test_literal_binding_never_repairs_case_whitespace_or_repeated_quote(self):
        for source_id, quote in (("u0", "before lunch I completed one thought record."),
                                 ("u0", "I completed  one thought record."), ("repeat", "again")):
            with self.subTest(source_id=source_id, quote=quote):
                invalid = {**deepcopy(self.item), "item_id":"bad",
                           "evidence":[{"source_id":source_id, "quote":quote}]}
                bound, statuses = self.bind([invalid, self.item])
                self.assertEqual([i["item_id"] for i in bound], ["m0"])
                self.assert_statuses(statuses, [False, True])

    def test_speaker_roles_are_not_relabelled_to_rescue_an_item(self):
        invalid_user = {**deepcopy(self.item), "item_id":"bad-user",
            "evidence":[{"source_id":"a0", "quote":self.sources[1]["content"]}]}
        invalid_proposal = {**deepcopy(self.item), "item_id":"bad-assistant", "origin":"assistant_proposal"}
        valid_proposal = {**deepcopy(invalid_user), "item_id":"proposal", "origin":"assistant_proposal"}
        bound, statuses = self.bind([invalid_user, invalid_proposal, valid_proposal, self.item])
        self.assertEqual([i["item_id"] for i in bound], ["proposal", "m0"])
        self.assert_statuses(statuses, [False, False, True, True])

    def test_inference_and_factual_sections_remain_separate(self):
        bad_inference = {**deepcopy(self.item), "item_id":"bad-inference", "origin":"agent_inference"}
        bad_fact = {**deepcopy(self.item), "item_id":"bad-fact", "section":"working_formulation"}
        tentative = {**deepcopy(self.item), "item_id":"hypothesis", "section":"working_formulation",
                     "origin":"agent_inference", "value":"Possibly the task felt effortful."}
        bound, statuses = self.bind([bad_inference, bad_fact, tentative])
        self.assertEqual([i["item_id"] for i in bound], ["hypothesis"])
        self.assertEqual(bound[0]["origin"], "agent_inference")
        self.assertNotIn("support_status", bound[0])
        self.assert_statuses(statuses, [False, False, True])

    def test_unhashable_or_wrong_candidate_types_are_local_rejections(self):
        for key, value in (("item_id", []), ("section", ["case_state"]),
                           ("origin", {"value":"explicit_user_statement"}),
                           ("evidence", "u0"), ("value", ["summary"])):
            with self.subTest(field=key):
                invalid = {**deepcopy(self.item), key:value}
                bound, statuses = self.bind([invalid, {**self.item, "item_id":"safe"}])
                self.assertEqual([i["item_id"] for i in bound], ["safe"])
                self.assert_statuses(statuses, [False, True])
        for invalid in (None, [], "candidate", 1, True):
            with self.subTest(candidate=invalid):
                bound, statuses = self.bind([invalid, self.item])
                self.assertEqual([i["item_id"] for i in bound], ["m0"])
                self.assert_statuses(statuses, [False, True])

    def test_invalid_evidence_pointer_types_do_not_crash_or_mutate_peer(self):
        for pointer in ({"source_id":[], "quote":"x"}, {"source_id":{"id":"u0"}, "quote":"x"},
                        {"source_id":"u0", "quote":[]}, {"source_id":"missing", "quote":"x"},
                        {"source_id":"u0", "quote":""}, {"source_id":"u0", "quote":"x", "start":0},
                        [], None):
            with self.subTest(pointer=pointer):
                invalid = {**deepcopy(self.item), "item_id":"bad", "evidence":[pointer]}
                bound, statuses = self.bind([invalid, self.item])
                self.assertEqual([i["item_id"] for i in bound], ["m0"])
                self.assert_statuses(statuses, [False, True])

    def test_candidate_bounds_and_extra_fields_are_not_silently_trimmed(self):
        for change in ({"value":"x" * 1401}, {"item_id":"x" * 101}, {"evidence":[]},
                       {"evidence":deepcopy(self.item["evidence"]) * 9}, {"extra":"ignored?"}):
            with self.subTest(change=change):
                invalid = {**deepcopy(self.item), "item_id":"bad", **change}
                bound, statuses = self.bind([invalid, self.item])
                self.assertEqual([i["item_id"] for i in bound], ["m0"])
                self.assert_statuses(statuses, [False, True])

    def test_empty_well_formed_batch_is_distinct_from_all_rejected(self):
        self.assertEqual(self.bind([]), ([], []))
        bound, statuses = self.bind([{**self.item, "evidence":[]}])
        self.assertEqual(bound, [])
        self.assert_statuses(statuses, [False])

    def test_corrupt_or_unbounded_envelope_remains_fatal(self):
        raws = ["{", "[]", '{"items":{},"extra":1}', '{"items":[] ,"extra":1}',
                '{"items":[],"items":[]}', '{"items":[NaN]}',
                json.dumps({"items":[{**self.item, "item_id":str(i)} for i in range(17)]})]
        for raw in raws:
            with self.subTest(raw=raw[:80]):
                with self.assertRaises(ValueError):
                    contract.bind_candidates_with_status(raw, self.sources)


if __name__ == "__main__":
    unittest.main()
