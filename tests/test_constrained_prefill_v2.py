"""Invented character tokens/metadata only; no real tokenizer, weights or API."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]/'evaluation/cbt-memory-effect'
sys.path.insert(0,str(HERE))
import constrained_json_prefill_v2 as G
import constrained_prefill_pilot_v2 as P

class Row:
    def __init__(self,ids):self.ids=ids
    def tolist(self):return self.ids
class Tensor:
    def __init__(self,ids):self.ids=ids;self.shape=(1,len(ids))
    def __getitem__(self,index):assert index==0;return Row(self.ids)
class Tokenizer:
    chat_template='invented-prefix-token-fixture'
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return ''.join('' if x==100000 and skip_special_tokens else '<eos>' if x==100000 else chr(x) for x in ids)
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs['add_generation_prompt'] is True
        return {'input_ids':Tensor(self.encode('FIXTURE_HEADER:'+json.dumps(messages)))}

class PrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.tokenizer=Tokenizer();cls.trie=G.AnswerTrie(cls.tokenizer,[100000])
    def test_generic_domain_and_fixed_space_prefix(self):
        self.assertEqual(G.PREFIX,'{"answer": ')
        for value in (None,0,11,9999,'helpful','unhelpful','mixed','approved','ongoing','unchanged'):
            suffix=G.canonical(value)+'}';ids=self.tokenizer.encode(suffix)
            for i,token in enumerate(ids):self.assertIn(token,self.trie.allowed(ids[:i]))
            out=self.trie.finish(ids+[100000])
            self.assertEqual(json.loads(out['response_text']),{'answer':value})
            self.assertEqual(out['prefilled_token_count'],len(G.PREFIX))
        self.assertEqual(self.trie.identity['candidate_count'],10001+len(G.STATUSES))
    def test_eos_only_after_full_suffix_and_truncation_never_repaired(self):
        self.assertNotIn(100000,self.trie.allowed([]))
        self.assertNotIn(100000,self.trie.allowed(self.tokenizer.encode('11')))
        self.assertEqual(self.trie.allowed(self.tokenizer.encode('11}')),[100000])
        with self.assertRaises(ValueError):self.trie.finish(self.tokenizer.encode('11}'))
        with self.assertRaises(ValueError):self.trie.finish(self.tokenizer.encode('true}')+[100000])
    def test_actual_input_preflight_includes_fixed_prefix_and_budget(self):
        messages=[{'role':'user','content':'Invented statement.'}]
        pre=G.input_preflight(self.tokenizer,messages,self.trie)
        base=self.tokenizer.apply_chat_template(messages,add_generation_prompt=True)['input_ids'].ids
        self.assertEqual(pre['input_tensor_sha256'],G.digest([base+self.trie.prefix_ids]))
        self.assertEqual(pre['input_tokens'],len(base)+len(self.trie.prefix_ids))
        with self.assertRaises(ValueError):G.append_prefill_ids([1]*4096,[2])
    def test_four_prospective_rows_use_identical_old_questions(self):
        rows=P.requests();old=P.v1.requests()
        self.assertEqual(len(rows),4)
        self.assertEqual([r['decode_mode'] for r in rows],['unconstrained']*2+['prefill_v2']*2)
        self.assertEqual([r['request_payload'] for r in rows],2*[r['request_payload'] for r in old])
        self.assertEqual(len({r['logical_id'] for r in rows}),4)
    def fixture(self,root):
        rows=P.requests();P.reader._write_new(root/'requests.json',rows)
        targets=root/'assessor_only/targets.json'
        P.reader._write_new(targets,{cid:value for cid,_,value in P.FROZEN_CASES})
        P.reader._write_new(root/'protocol.json',{'version':P.VERSION,'code_sha256':P.code_identity(),
            'logical_rows':4,'v2_policy':G.policy(),'requests_sha256':P.reader.digest(rows),
            'targets_file_sha256':P.reader.file_hash(targets)})
        return [{**r,'status':'completed','response_text':('The clerk said eleven.' if r['case_id'].endswith('books') else 'unchanged')
            if r['decode_mode']=='unconstrained' else '{"answer": '+('11' if r['case_id'].endswith('books') else '"unchanged"')+'}'} for r in rows]
    def test_readability_gate_and_reference_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);answers=self.fixture(root)
            result=P.score_authenticated(answers,prepared_dir=root)
            self.assertTrue(result['ready_for_main']);self.assertEqual(result['prefill_basic_readability_correct'],2)
            self.assertIsNone(result['schema_valid_by_mode']['unconstrained'])
            answers[2]['response_text']='{"answer": null}'
            result=P.score_authenticated(answers,prepared_dir=root)
            self.assertFalse(result['ready_for_main']);self.assertEqual(result['prefill_schema_valid'],2)
            self.assertEqual(result['prefill_basic_readability_correct'],1)
    def test_all_four_and_frozen_labels_required_before_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);answers=self.fixture(root)
            with self.assertRaises(ValueError):P.score_authenticated(answers[:-1],prepared_dir=root)
            (root/'assessor_only/targets.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'targets_changed'):P.score_authenticated(answers,prepared_dir=root)

if __name__=='__main__':unittest.main()
