"""Tiny invented token fixtures, no tensor/model/corpus/network execution."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]/'evaluation/cbt-memory-effect'
sys.path.insert(0,str(HERE))
import constrained_json_grammar as G
import constrained_open_reader as R

class CharacterTokenizer:
    chat_template='invented-char-template'
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i) for i in ids)

class GrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tokenizer=CharacterTokenizer();cls.trie=G.AnswerTrie(cls.tokenizer,[100000])
    def test_full_generic_domain_has_no_gold_or_arm_selection(self):
        for value in (0,11,9999,None,'helpful','unhelpful','mixed','unchanged','approved','ongoing'):
            text=G.canonical({'answer':value});ids=self.tokenizer.encode(text)
            for i,token in enumerate(ids):self.assertIn(token,self.trie.allowed(ids[:i]))
            self.assertEqual(self.trie.allowed(ids),[100000])
            self.assertEqual(self.trie.finish(ids+[100000])['chosen_answer'],value)
        self.assertEqual(self.trie.identity['candidate_count'],10001+len(G.STATUSES))
    def test_eos_never_allowed_before_complete_object(self):
        for text in ('','{','{"answer":','{"answer":11'):
            self.assertNotIn(100000,self.trie.allowed(self.tokenizer.encode(text)))
        for text in ('{"answer":-1}','{"answer":10000}','{"answer":true}','{"answer":11,"extra":0}'):
            with self.assertRaises(ValueError):self.trie.finish(self.tokenizer.encode(text)+[100000])
    def test_no_cutoff_repair_or_unknown_inference(self):
        with self.assertRaises(ValueError):self.trie.finish(self.tokenizer.encode('{"answer":11}'))
        with self.assertRaises(ValueError):self.trie.allowed(self.tokenizer.encode('not JSON'))
        answer=self.trie.finish(self.tokenizer.encode('{"answer":null}')+[100000])
        self.assertIsNone(answer['chosen_answer']);self.assertIn('raw_generated_token_ids',answer)
    def test_canonical_key_is_separate_from_old_unconstrained_key(self):
        payload=R.requests()[0]['request_payload'];model={'id':'invented'}
        old=R.reader.digest({'model':model,'payload':payload,'max_new_tokens':64,'do_sample':False})
        new=R.reader.digest({'model':model,'decoding':self.trie.identity,'payload':payload,'max_new_tokens':64,'do_sample':False})
        self.assertNotEqual(old,new)
    def test_pilots_are_new_and_generation_rows_do_not_contain_expected_values(self):
        rows=R.requests();self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(set(row),{'case_id','request_payload','request_sha256'})
            self.assertNotIn('controlled-',row['case_id']);self.assertNotIn('CBT',json.dumps(row))
        self.assertEqual([r[2] for r in R.FIXTURES],[11,'unchanged'])
    def test_hosted_guard_before_any_path_io(self):
        with patch.object(R.reader,'require_colab',side_effect=RuntimeError('hosted_required')):
            with self.assertRaisesRegex(RuntimeError,'hosted_required'):R.run('/missing','/missing')
            with self.assertRaisesRegex(RuntimeError,'hosted_required'):R.prepare('/missing')

if __name__=='__main__':unittest.main()
