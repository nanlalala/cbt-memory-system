"""Small invented output strings only; no model, data, provider or old results."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/common_answer_adapter_v3.py"
SPEC = importlib.util.spec_from_file_location("common_answer_adapter_v3_fixture", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class CommonAnswerAdapterTests(unittest.TestCase):
    def accepted(self, raw, value, rule):
        result = A.adapt(raw)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["rule"], rule)
        self.assertEqual(result["reason"], "accepted")
        self.assertEqual(result["raw_response_text"], raw)
        self.assertEqual(result["raw_response_sha256"], hashlib.sha256(raw.encode("utf-8")).hexdigest())
        canonical = json.dumps({"answer": value}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self.assertEqual(result["canonical_response_text"], canonical)
        self.assertEqual(result["canonical_response_sha256"], hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        return result

    def rejected(self, raw):
        result = A.adapt(raw)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["rule"], "adapter_rejected")
        self.assertIsNone(result["canonical_response_text"])
        self.assertIsNone(result["canonical_response_sha256"])
        self.assertTrue(result["reason"])
        self.assertEqual(result["raw_response_text"], raw)
        self.assertEqual(result["raw_response_sha256"], hashlib.sha256(raw.encode("utf-8")).hexdigest())

    def test_complete_single_key_json_with_exact_types(self):
        self.accepted(' { "answer" : 27 }\n', 27, "single_answer_json")
        self.accepted('{"answer":"mixed"}', "mixed", "single_answer_json")
        self.accepted('{"answer":null}', None, "single_answer_json")
        self.accepted('{"ans\\u0077er":"appro\\u0076ed"}', "approved", "single_answer_json")

    def test_bare_and_exact_prefix_forms(self):
        for number in (0, 3, 27, 9999):
            self.accepted(str(number), number, "bare_integer")
            self.accepted(" \nanswer:\t" + str(number) + "\r\n", number, "answer_prefix_integer")
        self.accepted("approved", "approved", "bare_status")
        self.accepted('"approved"', "approved", "json_quoted_status")
        self.accepted("answer: approved", "approved", "answer_prefix_status")
        self.accepted('answer: "approved"', "approved", "answer_prefix_json_quoted_status")
        self.accepted(" null \n", None, "bare_null")
        self.accepted("answer: null", None, "answer_prefix_null")

    def test_all_fixed_statuses_and_null_are_distinct_from_unknown_string(self):
        self.assertEqual(len(A.STATUSES), 37)
        self.assertEqual(len(set(A.STATUSES)), 37)
        for status in A.STATUSES:
            self.accepted(status, status, "bare_status")
            self.accepted(json.dumps(status), status, "json_quoted_status")
        self.accepted("unknown", "unknown", "bare_status")
        self.rejected('"null"')
        self.rejected('answer: "null"')
        self.rejected('{"answer":"null"}')

    def test_reject_numeric_lookalikes_including_json_negative_zero(self):
        for literal in ("-0", "-1", "+0", "+3", "00", "03", "10000", "1.0", "1e2", "1E+2", "NaN", "Infinity", "-Infinity", "true", "false", "1_000", "０", "٣"):
            for text in (literal, "answer: " + literal, '{"answer":' + literal + '}'):
                with self.subTest(text=text):
                    self.rejected(text)
        self.rejected('"27"')
        self.rejected('{"answer":"27"}')

    def test_reject_ambiguous_extra_duplicate_and_non_scalar_values(self):
        for raw in ('{"answer":3,"extra":0}', '{"answer":3,"answer":3}', '{"answer":3,"ans\\u0077er":3}',
            '{}', '{"value":3}', '{"answer":[3]}', '{"answer":{"answer":3}}', '[3]',
            '{"answer":3}{"answer":4}', '{"answer":3}\n4', '3 4', 'approved, mixed', 'answer: 3\nanswer: 4',
            'answer: {"answer":3}', 'answer: answer: 3', 'answer: [3]', 'answer:', ''):
            with self.subTest(raw=raw):
                self.rejected(raw)

    def test_reject_prose_fences_case_changes_and_incomplete_literals(self):
        for raw in ('The answer is 3.', 'I think approved', '3 because it was reported.', 'answer: 3.',
            '```json\n{"answer":3}\n```', 'prefix answer: 3', 'Answer: 3', 'answer : 3',
            'APPROVED', 'answer: Approved', '"approv', '{"answer":3', '\n\t  '):
            with self.subTest(raw=raw):
                self.rejected(raw)

    def test_raw_sha_preserves_whitespace_and_rejection_never_becomes_null(self):
        first = self.accepted("3", 3, "bare_integer")
        second = self.accepted(" 3\n", 3, "bare_integer")
        self.assertNotEqual(first["raw_response_sha256"], second["raw_response_sha256"])
        self.assertEqual(first["canonical_response_sha256"], second["canonical_response_sha256"])
        rejected = A.adapt("There is no supported single value here.")
        accepted_null = self.accepted("null", None, "bare_null")
        self.assertIsNone(rejected["canonical_response_text"])
        self.assertEqual(accepted_null["canonical_response_text"], '{"answer":null}')

    def test_output_only_contract_and_identity(self):
        self.assertEqual(list(inspect.signature(A.adapt).parameters), ["text"])
        with self.assertRaises(TypeError):
            A.adapt("3", question="invented")
        with self.assertRaises(TypeError):
            A.adapt(3)
        identity = A.identity()
        self.assertEqual(identity["version"], A.VERSION)
        self.assertEqual(identity["code_sha256"], hashlib.sha256(PATH.read_bytes()).hexdigest())
        serialized = json.dumps(A.policy(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.assertEqual(identity["policy_sha256"], hashlib.sha256(serialized.encode()).hexdigest())
        self.assertEqual(identity, A.identity())
        self.assertFalse(identity["policy"]["model_or_logits_modified"])
        self.assertFalse(identity["policy"]["factual_correctness_established"])


if __name__ == "__main__":
    unittest.main()
