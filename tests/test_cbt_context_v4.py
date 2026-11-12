"""Readout regressions for source admission, meaning status and real messages."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from cbt_context import CBT_POLICY, ContextualItem, build_cbt_context
from evidence_bundles import EvidenceSpan, OutboundBudgetError
from controlled_dialogue import ControlledDialogueSession, PrototypeLimits
from durable_memory import DurableConflict


def registry():
    return {sid:dict(user_id='fictional', source_id=sid, text=text, role=role, source_order=i,
                    source_time=None, permissions=dict(save=True,long_term_use=True,response_external=True))
            for i,(sid,text,role) in enumerate([
                ('s1','Perhaps practise a thought record seven times.','assistant'),
                ('s2','I agree, but change the target to two practices.','user'),
                ('s3','I want to pause further thought records.','user')])}


def item(sources, *, support='source_supported', section='case_state', origin='explicit_user_statement'):
    evidence=tuple(EvidenceSpan(sid,sid,s['text'],0,len(s['text']),None,s['role'])
                   for sid,s in sources.items())
    return ContextualItem('latest-task',section,{'target':2,'participation':'paused'},evidence,
                          origin=origin,support_status=support,verifier_revision='fixture-verifier-v1')


class ReadoutTests(unittest.TestCase):
    def assemble(self, sources=None, **kwargs):
        sources=registry() if sources is None else sources
        settings=dict(source_registry=sources,subject_id='fictional',max_outbound_chars=9000,
            max_response_tokens=64,request_metadata={'model':'fixture-model'},
            contextual_items=[item(sources)])
        settings.update(kwargs)
        return build_cbt_context([{'role':'system','content':CBT_POLICY},
                                  {'role':'user','content':'What is the current arrangement?'}],**settings)

    def test_processed_projection_reaches_user_data_with_audited_evidence_dependencies(self):
        out=self.assemble(representation='processed')
        self.assertEqual([m['role'] for m in out.messages],['system','user','user'])
        data=json.loads(out.messages[-2]['content'].split('\n',1)[1])
        state=data['case_state'][0]
        self.assertEqual(state['value'],{'target':2,'participation':'paused'})
        self.assertNotIn('evidence',state)
        self.assertEqual(out.source_ids,('s1','s2','s3'))
        self.assertEqual(out.char_count,len(out.serialized_request))
        self.assertNotIn('source_hashes',out.serialized_request)

    def test_revoked_or_cross_subject_dependency_prevents_projection_readout(self):
        for field,value in [('permissions',dict(save=True,long_term_use=False,response_external=True)),
                            ('user_id','another-person')]:
            sources=registry(); sources['s2'][field]=value
            out=self.assemble(sources)
            self.assertIn('latest-task',out.dropped_claims)
            self.assertNotIn('paused',out.messages[-2]['content'])

    def test_hypothesis_never_enters_factual_case_section(self):
        sources=registry()
        bad=item(sources,origin='agent_inference',support='hypothesis')
        out=self.assemble(sources,contextual_items=[bad])
        self.assertIn('latest-task',out.dropped_claims)
        good=item(sources,origin='agent_inference',support='hypothesis',section='working_formulation')
        data=json.loads(self.assemble(sources,contextual_items=[good]).messages[-2]['content'].split('\n',1)[1])
        self.assertEqual(data['working_formulation'][0]['epistemic_status'],'unverified_hypothesis')

    def test_fixed_selection_preserves_context_role_order_and_fails_whole_grid_overflow(self):
        sources=registry()
        sources['s1']['role']='context'
        span=EvidenceSpan('s1','fact',sources['s1']['text'],role='context')
        fact=ContextualItem('fact','case_state','quoted benchmark fact',(span,),origin='benchmark_source',
                            support_status='rule_verified',verifier_revision='fixture-parser-v1')
        out=self.assemble(sources,representation='raw',contextual_items=[fact],selected_source_ids=['s2','s1'])
        data=json.loads(out.messages[-2]['content'].split('\n',1)[1])
        self.assertEqual([x['source_id'] for x in data['source_reports']],['s1','s2'])
        self.assertEqual(data['source_reports'][0]['role'],'context')
        with self.assertRaises(OutboundBudgetError):
            self.assemble(sources,contextual_items=[fact],selected_source_ids=['s1'],max_outbound_chars=300)

    def test_recent_binding_includes_recent_supported_projection(self):
        sources=registry()
        selected=['s1','s2']
        recent=[{'role':'user','content':sources['s3']['text']}]
        out=self.assemble(sources,selected_source_ids=selected,recent_messages=recent,recent_source_ids=['s3'])
        self.assertEqual(out.messages[1],recent[0])
        self.assertIn('latest-task',out.included_claim_ids)
        altered=copy.deepcopy(recent);altered[0]['content']='different source'
        with self.assertRaises(ValueError):
            self.assemble(sources,recent_messages=altered,recent_source_ids=['s3'])

    def test_benchmark_speaker_survives_raw_and_evidence_without_quote_rewriting(self):
        sources=registry()
        sources['s1'].update(role='context',source_speaker='Released Participant A')
        span=EvidenceSpan('s1','speaker-fact',sources['s1']['text'],role='context')
        fact=ContextualItem('speaker-fact','case_state',{'speaker':'Released Participant A','reported_fact':'fixture'},
            (span,),origin='benchmark_source',support_status='rule_verified',verifier_revision='fixture-parser-v1')
        for representation in ('raw','processed_with_evidence'):
            result=self.assemble(sources,representation=representation,contextual_items=[fact],selected_source_ids=['s1'])
            data=json.loads(result.messages[-2]['content'].split('\n',1)[1])
            view=data['source_reports'][0] if representation=='raw' else data['case_state'][0]['evidence'][0]
            self.assertEqual(view['source_speaker'],'Released Participant A')
            self.assertEqual(view['text'],sources['s1']['text'])

    def test_recent_metadata_preserves_original_date_and_speaker_and_omits_unknown_date(self):
        sources=registry()
        sources['s2'].update(source_time='2026-11-03T09:00:00Z',source_speaker='Fictional Participant')
        recent=[dict(role=sources[sid]['role'],content=sources[sid]['text']) for sid in ('s2','s3')]
        result=self.assemble(sources,recent_messages=recent,recent_source_ids=['s2','s3'])
        data=json.loads(result.messages[-2]['content'].split('\n',1)[1])
        metadata=data['recent_sources']
        self.assertEqual(metadata[0]['source_time'],'2026-11-03T09:00:00Z')
        self.assertEqual(metadata[0]['source_speaker'],'Fictional Participant')
        self.assertNotIn('source_time',metadata[1])

    def test_tight_budget_follows_retrieval_rank_before_item_identity(self):
        sources=registry()
        higher=replace(item(sources),item_id='z-high-relevance',retrieval_rank=0)
        lower=replace(item(sources),item_id='a-low-relevance',retrieval_rank=1)
        cap=self.assemble(sources,contextual_items=[higher]).char_count
        result=self.assemble(sources,contextual_items=[lower,higher],max_outbound_chars=cap)
        self.assertEqual(result.included_claim_ids,('z-high-relevance',))
        self.assertEqual(result.trace['item_retrieval_ranks'],{'a-low-relevance':1,'z-high-relevance':0})

    def test_product_default_uses_real_projected_pause_and_single_current_input(self):
        class Capture:
            def __init__(self):self.messages=[]
            def __call__(self,*,messages,purpose):
                from dialogue_service import ModelOutput
                self.messages.append(messages)
                return ModelOutput('This is a fictional response.')
        with tempfile.TemporaryDirectory() as folder:
            provider=Capture()
            session=ControlledDialogueSession.open(str(Path(folder)/'memory.sqlite'),approved_root=folder,
                user_id='fictional',provider=provider,background=False)
            self.addCleanup(session.close)
            for i,text in enumerate(['我同意每周做两次想法记录。','想法记录累计做了一次。','这周暂停']):
                result=session.send_message(text,input_id=str(i),as_of=f'2026-11-0{3+i}T09:00:00Z')
            messages=provider.messages[-1]
            data=json.loads(next(m['content'] for m in messages if m['content'].startswith('SOURCED MEMORY')).split('\n',1)[1])
            state={c['field']:c['value'] for c in data['claims'] if c['task_instance_id']}
            self.assertEqual((state['engagement'],state['target_count'],state['reported_count']),('paused',2,1))
            self.assertEqual(sum(m['content']=='这周暂停' for m in messages),1)
            self.assertEqual(next(m['role'] for m in messages if m['content'].startswith('SOURCED MEMORY')),'user')
            self.assertEqual(result['retrieval_trace']['context_assembly']['assembly_version'],'cbt-context-v4')
            self.assertIn('turn_guidance',data)

    def test_small_request_cap_never_silently_shortens_shared_counselor_policy(self):
        class Capture:
            def __init__(self):self.calls=[]
            def __call__(self,**kwargs):
                self.calls.append(kwargs)
                raise AssertionError('over-budget provider call')
        with tempfile.TemporaryDirectory() as folder:
            provider=Capture()
            session=ControlledDialogueSession.open(str(Path(folder)/'memory.sqlite'),approved_root=folder,
                user_id='fictional',provider=provider,background=False,
                limits=PrototypeLimits(max_outbound_chars=4000))
            self.addCleanup(session.close)
            result=session.send_message('虚构当前问题。',input_id='small-cap',as_of='2026-11-03T09:00:00Z')
            self.assertEqual(result['source_commit']['source_status'],'committed')
            self.assertEqual(result['answer_status'],'budget_not_sent')
            self.assertEqual(provider.calls,[])

    def test_derived_pause_plan_keeps_task_version_even_if_other_claims_drop(self):
        with tempfile.TemporaryDirectory() as folder:
            s=ControlledDialogueSession.open(str(Path(folder)/'memory.sqlite'),approved_root=folder,
                user_id='fictional',provider=None,background=False)
            self.addCleanup(s.close)
            s.send_message('我同意做两次想法记录。',input_id='agree',as_of='2026-11-03T09:00:00Z')
            paused=s.send_message('想法记录暂停。',input_id='pause',as_of='2026-11-04T09:00:00Z')
            task=s._objects('task')[0]
            source=s.durable.get_source('fictional',paused['source_commit']['source_id'])
            bundles,_,_=s._bundles('想法记录现在怎么安排？',as_of='2026-11-04T09:00:00Z',task=task)
            _,_,_,items,versions=s._context_readout(as_of='2026-11-04T09:00:00Z',
                current_source=source,bundles=bundles,task_snapshot=task)
            plan=next(item for item in items if item.section=='turn_guidance')
            self.assertEqual(versions[plan.item_id],(task['object_id'],task['version']))
            s.send_message('想法记录继续。',input_id='continue',as_of='2026-11-05T09:00:00Z')
            with self.assertRaises(DurableConflict):
                s.durable.begin_attempt('fictional','old-plan','response',[source['source_id']],
                    run_id=s.run_id,session_id=s.session_id,
                    expected_object_versions={task['object_id']:task['version']})

    def test_new_default_run_preserves_old_frozen_run_and_retained_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'memory.sqlite')
            old=ControlledDialogueSession.open(path,approved_root=folder,user_id='fictional',
                run_id='controlled-v2',provider=None,background=False)
            reply=old.send_message('虚构旧版保留的用户原话。',input_id='old-source',as_of='2026-11-03T09:00:00Z')
            with old.store.transaction(immediate=True):
                old.store.connection.execute("UPDATE cp_runs SET protocol_json=? WHERE run_id=?",
                                             ('{"version":"controlled-memory-v2.1"}','controlled-v2'))
            old.close()
            fresh=ControlledDialogueSession.open(path,approved_root=folder,user_id='fictional',
                provider=None,background=False)
            self.addCleanup(fresh.close)
            self.assertEqual(fresh.run_id,'controlled-v4')
            self.assertIsNotNone(fresh.durable.get_source('fictional',reply['source_commit']['source_id']))
            preserved=fresh.store.connection.execute('SELECT protocol_json FROM cp_runs WHERE run_id=?',
                                                      ('controlled-v2',)).fetchone()[0]
            self.assertEqual(json.loads(preserved)['version'],'controlled-memory-v2.1')

    def test_product_searches_supported_processed_value_without_raw_word_match(self):
        with tempfile.TemporaryDirectory() as folder:
            s=ControlledDialogueSession.open(str(Path(folder)/'memory.sqlite'),approved_root=folder,
                user_id='fictional',provider=None,background=False)
            self.addCleanup(s.close)
            text='躺下后脑子就停不下来。'
            reply=s.send_message(text,input_id='source',as_of='2026-11-03T09:00:00Z')
            sid=reply['source_commit']['source_id']
            payload={'kind':'supported_projection','section':'case_state',
                'value':{'historical_user_report':'用户报告入睡前反复思考。'},
                'origin':'explicit_user_statement','source_ids':[sid],
                'evidence':[dict(source_id=sid,start=0,end=len(text),text=text,role='user')],
                'verifier_revision':'fixture-review-v1','verification':{'decision':'supported'},
                'source_scope':{'kind':'historical_user_report','source_time':'2026-11-03T09:00:00Z'}}
            s.durable.put_effect('fictional','fixture-projection','projection:fixture',expected_version=0,
                payload=payload,dependencies=[dict(source_id=sid,start=0,end=len(text))])
            query=dict(user_id='fictional',source_id='ephemeral-query',role='user',text='入睡',
                source_time='2026-11-03T10:00:00Z',permissions=dict(save=False,long_term_use=False,response_external=True))
            _,_,_,items,_=s._context_readout(as_of=query['source_time'],current_source=query,bundles=[])
            self.assertEqual([item.item_id for item in items],['projection:fixture'])
            unrelated={**query,'text':'qzxnomatch'}
            _,_,_,items,_=s._context_readout(as_of=query['source_time'],current_source=unrelated,bundles=[])
            self.assertEqual(items,[])


if __name__=='__main__':unittest.main()
