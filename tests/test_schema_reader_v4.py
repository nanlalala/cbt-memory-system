"""Invented character tokens and questions only; no real model/data/tokenizer."""
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect"
sys.path.insert(0, str(HERE))
import schema_reader_v4 as G


EOS = 100000
COUNT = "What is my current agreed target count for thought records this week?"
STATUS = "What outcome have I explicitly reported for thought record practice?"
AMBIGUOUS = "Please recap the earlier conversation."


class Tensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))

    def tolist(self):
        return [self.ids]


class CharacterTokenizer:
    chat_template = "invented-v4-character-template"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join("" if token == EOS and skip_special_tokens else
                       "<eos>" if token == EOS else chr(token) for token in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": True,
                          "return_dict": True, "return_tensors": "pt"}
        return {"input_ids": Tensor(self.encode("FIXTURE:" + json.dumps(messages)))}


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CharacterTokenizer()
        cls.count = G.AnswerTrie(cls.tokenizer, [EOS], question=COUNT, fixture_only=True)
        cls.status = G.AnswerTrie(cls.tokenizer, [EOS], question=STATUS, fixture_only=True)
        cls.union = G.AnswerTrie(cls.tokenizer, [EOS], question=AMBIGUOUS, fixture_only=True)

    def test_public_contract_selects_type_without_answer_values(self):
        c, s, u = (trie.schema for trie in (self.count, self.status, self.union))
        self.assertEqual((c["field"], c["value_kind"]), ("target_count", "integer"))
        self.assertEqual((s["field"], s["value_kind"]), ("outcome", "status"))
        self.assertEqual((u["field"], u["value_kind"]), (None, "unrestricted_scalar"))
        self.assertEqual(self.count.identity["candidate_count"], 10001)
        self.assertEqual(self.status.identity["candidate_count"], 1 + len(G.STATUSES))
        self.assertEqual(self.union.identity["candidate_count"], 10001 + len(G.STATUSES))
        self.assertEqual(set(inspect.signature(G.schema_for_question).parameters), {"question"})
        for schema in (c, s, u):
            self.assertNotIn("expected_answer", schema)
            self.assertNotIn("arm", schema)
            self.assertTrue(schema["null_allowed"])

    def test_all_values_of_selected_type_are_available(self):
        for trie, values in ((self.count, [0, 5, 17, 9999, None]),
                             (self.status, [*G.STATUSES, None]),
                             (self.union, [0, 5007, "helpful", "unchanged", None])):
            for value in values:
                ids = self.tokenizer.encode(G.canonical({"answer": value}))
                for i, token in enumerate(ids):
                    self.assertIn(token, trie.allowed(ids[:i]))
                self.assertEqual(trie.allowed(ids), [EOS])
                self.assertEqual(trie.finish(ids + [EOS])["chosen_answer"], value)

    def test_count_and_status_domains_never_cross(self):
        for trie, value in ((self.count, "mixed"), (self.status, 3)):
            text = G.canonical({"answer": value})
            self.assertFalse(G.validate_response(text, trie.schema)["accepted"])
            with self.assertRaises(ValueError):
                trie.finish(self.tokenizer.encode(text) + [EOS])

    def test_eos_never_precedes_complete_whole_object(self):
        for trie in (self.count, self.status):
            for text in ("", "{", '{"answer":'):
                self.assertNotIn(EOS, trie.allowed(self.tokenizer.encode(text)))
            with self.assertRaises(ValueError):
                trie.finish(self.tokenizer.encode('{"answer":null}'))
            with self.assertRaises(ValueError):
                trie.finish(self.tokenizer.encode('{"answer":null}') + [EOS, EOS])

    def test_validator_rejects_output_extraction_coercion_or_repair(self):
        invalid = ["null", "3", '"mixed"', "```json\n{\"answer\":3}\n```",
                   'Text {"answer":3}', '{"answer":3} trailing',
                   '{"answer":3,"extra":false}', '{"answer":3,"answer":4}',
                   '{"answer":true}', '{"answer":3.0}', '{"answer":-1}',
                   '{"answer":10000}', '{"answer":03}', '{"answer":-0}',
                   '{"answer":"3"}', '{"answer":NaN}', '{"answer":Infinity}',
                   '{"answer":{"x":3}}', '{"answer":[3]}', '{"answer":']
        for text in invalid:
            self.assertFalse(G.validate_response(text, self.union.schema)["accepted"], text)
        self.assertTrue(G.validate_response(' { "answer" : 5 } ', self.count.schema)["accepted"])

    def test_null_stays_distinct_from_invalid_output(self):
        valid = G.validate_response('{"answer":null}', self.count.schema)
        invalid = G.validate_response("No evidence", self.count.schema)
        self.assertTrue(valid["accepted"])
        self.assertFalse(invalid["accepted"])
        self.assertIsNone(valid["answer"])
        self.assertNotEqual(valid["reason"], invalid["reason"])

    def test_actual_input_has_no_prefill_and_binds_exact_question(self):
        messages = [{"role": "user", "content": COUNT}]
        pre = G.input_preflight(self.tokenizer, messages, self.count, fixture_only=True)
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt")["input_ids"].tolist()
        self.assertEqual(pre["input_tensor_sha256"], G.digest(ids))
        self.assertEqual(pre["input_tokens"], len(ids[0]))
        self.assertEqual(pre["prefilled_token_count"], 0)
        self.assertEqual(pre["actual_input_token_count"], pre["base_input_token_count"])
        with self.assertRaisesRegex(ValueError, "last_user_question"):
            G.input_preflight(self.tokenizer, [{"role": "user", "content": STATUS}],
                              self.count, fixture_only=True)

    def test_input_limit_rejects_whole_request_without_truncation(self):
        messages = [{"role": "system", "content": "x" * 4096},
                    {"role": "user", "content": COUNT}]
        with self.assertRaisesRegex(ValueError, "actual_input_token_limit"):
            G.input_preflight(self.tokenizer, messages, self.count, fixture_only=True)

    def test_hosted_guard_happens_before_tokenizer_or_model_work(self):
        with patch.object(G, "require_hosted_colab", side_effect=RuntimeError("hosted_first")):
            with self.assertRaisesRegex(RuntimeError, "hosted_first"):
                G.AnswerTrie(object(), [EOS], question=COUNT)
            with self.assertRaisesRegex(RuntimeError, "hosted_first"):
                G.input_preflight(object(), [], self.count)
            with self.assertRaisesRegex(RuntimeError, "hosted_first"):
                G.generate(object(), object(), [], self.count)

    def test_no_forged_decoding_or_cutoff_repair(self):
        ids = self.tokenizer.encode('{"answer":5}')
        answer = self.count.finish(ids + [EOS])
        self.assertEqual(answer["raw_generated_token_ids"], ids + [EOS])
        self.assertEqual(answer["decoding_identity_sha256"], G.digest(self.count.identity))
        for prefix in ([True], ["123"], [-1]):
            with self.assertRaises(ValueError):
                self.count.allowed(prefix)

    def test_changed_contract_domain_is_rejected(self):
        for key, value in (("integer_maximum", 5), ("integer_allowed", False),
                           ("status_allowed", True), ("null_allowed", False)):
            altered = dict(self.count.schema, **{key: value})
            with self.assertRaisesRegex(ValueError, "schema_domain_changed"):
                G.candidate_texts(altered)


if __name__ == "__main__":
    unittest.main()
