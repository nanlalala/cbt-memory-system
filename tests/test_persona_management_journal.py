"""Tiny fictional provider outputs only; never benchmark/model/Colab execution."""
import importlib.util
import json
from pathlib import Path
import sys

import unittest
import tempfile

DIRECTORY=Path(__file__).resolve().parents[1]/'evaluation/objective-memory'
sys.path.insert(0,str(DIRECTORY))
from persona_management_journal import Journal,BudgetExhausted,append,request_body,stable,ALIAS

def protocol():return {'protocol_sha256':'f'*64,'maximum_provider_attempts':4,'output_limits':{'extraction':4096,'response':64}}
def messages(text='Tiny fictional source.'):return [{'role':'user','content':text}]
def response(latency=1):return {'hypothesis':'(a)','provider_model':ALIAS,'finish_reason':'stop','usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7},'latency_seconds':latency}

class JournalTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup);self.run=Path(directory.name)

    def test_durable_intent_and_exact_reuse_with_purpose_isolation(self):
        tmp_path=self.run
        observed=[]
        def transport(body,purpose):
            intents=[json.loads(s) for s in (tmp_path/'attempts.jsonl').read_text().splitlines()]
            assert intents[-1]['purpose']==purpose
            assert not (tmp_path/'outcomes.jsonl').exists() or len(intents)>len((tmp_path/'outcomes.jsonl').read_text().splitlines())
            observed.append(purpose);return response()
        journal=Journal(tmp_path,protocol(),max_new_calls=3,transport=transport)
        first=journal.call(messages(),'response');assert journal.call(messages(),'response')==first
        journal.call(messages(),'extraction')
        assert observed==['response','extraction'] and journal.cost()['provider_attempts']==2
        journal.reload();assert journal.cost()['provider_attempts']==2
        resumed=Journal(tmp_path,protocol(),transport=transport)
        assert resumed.call(messages(),'response')==first and observed==['response','extraction']

    def test_budget_is_total_actual_attempts_not_logical_rows(self):
        tmp_path=self.run
        journal=Journal(tmp_path,protocol(),max_new_calls=1,transport=lambda *_:response())
        journal.call(messages(),'response')
        with self.assertRaises(BudgetExhausted):journal.call(messages('different'),'response')
        assert journal.cost()['provider_attempts']==1

    def test_unresolved_intent_refuses_resume(self):
        tmp_path=self.run
        p=protocol();request=request_body(messages(),'response',64);key=stable(request)
        append(tmp_path/'requests.jsonl',{'protocol_sha256':p['protocol_sha256'],'request_key':key,'request':request})
        intent={'protocol_sha256':p['protocol_sha256'],'request_key':key,'attempt_index':1,'purpose':'response'};intent['record_sha256']=stable(intent);append(tmp_path/'attempts.jsonl',intent)
        with self.assertRaisesRegex(RuntimeError,'in-flight'):Journal(tmp_path,p)

    def test_wrong_model_halts_and_preserves_supplied_cost(self):
        tmp_path=self.run
        journal=Journal(tmp_path,protocol(),max_new_calls=3,transport=lambda *_:{**response(2),'provider_model':'other-model'})
        _,out=journal.call(messages(),'response')
        assert out['status']=='protocol_validation_failure' and out['reported_usage']['total_tokens']==7
        with self.assertRaisesRegex(RuntimeError,'Fatal'):journal.call(messages('other'),'response')
        assert journal.cost()['reported_usage']['total_tokens']==7 and journal.cost()['reported_latency_seconds_median']==2
        assert Journal(tmp_path,protocol()).aborted

    def test_failure_unknown_cost_and_separate_purpose_latency(self):
        tmp_path=self.run
        def transport(_,purpose):
            if purpose=='extraction':raise RuntimeError('DO_NOT_PERSIST_SECRET_OR_HTTP_BODY')
            return response(3)
        journal=Journal(tmp_path,protocol(),max_new_calls=3,transport=transport)
        journal.call(messages(),'extraction');journal.call(messages(),'response')
        cost=journal.cost()
        assert cost['by_purpose']['extraction']['billable_usage_unknown_calls']==1
        assert cost['by_purpose']['response']['reported_latency_seconds_median']==3
        assert 'DO_NOT_PERSIST' not in (tmp_path/'outcomes.jsonl').read_text()

    def test_journal_tamper_rejected(self):
        tmp_path=self.run
        journal=Journal(tmp_path,protocol(),max_new_calls=1,transport=lambda *_:response());journal.call(messages(),'response')
        path=tmp_path/'outcomes.jsonl';value=json.loads(path.read_text());value['response']['hypothesis']='changed';path.write_text(json.dumps(value)+'\n')
        with self.assertRaisesRegex(ValueError,'seal'):Journal(tmp_path,protocol())

    def test_global_latency_median_is_not_median_of_purpose_medians(self):
        tmp_path=self.run
        latencies=iter([1,2,100,1000]);journal=Journal(tmp_path,protocol(),max_new_calls=4,transport=lambda *_:response(next(latencies)))
        for n in range(4):journal.call(messages(str(n)),'extraction' if n<3 else 'response')
        assert journal.cost()['reported_latency_seconds_median']==51

    def test_real_guard_before_missing_experimental_args(self):
        import persona_management_runner as runner
        from argparse import Namespace
        with self.assertRaisesRegex(RuntimeError,'Genuine hosted Colab'):runner.prepare(Namespace())

    def test_request_ceiling_and_model_policy_are_fixed(self):
        with self.assertRaisesRegex(ValueError,'ceiling'):request_body(messages('x'*24001),'response',64)
        assert request_body(messages(),'response',64)['thinking_mode']=='disabled'

    def test_fatal_dispatch_stops_queued_work_and_retains_authoritative_cost(self):
        from persona_management_runner import dispatch,atomic_json,read_json
        tmp_path=self.run;calls=[]
        def transport(body,purpose):
            calls.append(purpose);return {**response(),'provider_model':'wrong-model'}
        journal=Journal(tmp_path,protocol(),max_new_calls=4,transport=transport)
        atomic_json(tmp_path/'journal_cost.json',{'prior_phase_projection':True})
        assert dispatch(journal,[{'messages':messages(str(i))} for i in range(40)],'response')
        assert journal.aborted and 1<=len(calls)<=4
        assert all(r['status']=='protocol_validation_failure' for r in journal.outcomes.values())
        # The production phase finally writes this authoritative projection even
        # when queued calls are refused after the first fatal model identity.
        atomic_json(tmp_path/'journal_cost.json',journal.cost())
        assert read_json(tmp_path/'journal_cost.json')['provider_attempts']==len(calls)
        assert Journal(tmp_path,protocol()).cost()==journal.cost()

    def test_changed_frozen_inputs_halt_all_dispatch_before_provider_intent(self):
        from persona_management_runner import dispatch
        def changed():raise ValueError('Changed frozen fixture identity')
        journal=Journal(self.run,protocol(),max_new_calls=4,verify_identity=changed,transport=lambda *_:self.fail('Provider must not run'))
        assert dispatch(journal,[{'messages':messages(str(i))} for i in range(40)],'response')
        assert journal.abort_kind=='frozen_input_identity' and journal.cost()['provider_attempts']==0
        assert not (self.run/'attempts.jsonl').exists()

if __name__=="__main__":unittest.main()
