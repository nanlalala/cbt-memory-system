"""Small fictional checks for production semantic support and durable stages."""
import json
import sqlite3
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
sys.path.insert(0, str(ROOT / 'app'))
from controlled_dialogue import ControlledDialogueSession
from dialogue_service import ModelOutput
from durable_memory import AttemptUnknown, DurableConflict, DurableMemoryStore
from cbt_projection_contract import EXTRACT, VERIFY, bind_candidates
from cbt_retained_dialogue import retained_assistant_sources, expanded_source_dependencies
import colab_provider


class ProjectionProvider:
    supports_cbt_projection = True
    def __init__(self, verdict='supported', on_verify=None):
        self.calls = []
        self.verdict = verdict
        self.on_verify = on_verify

    def __call__(self, *, messages, purpose):
        self.calls.append({'messages':messages, 'purpose':purpose})
        if purpose == 'response':
            return ModelOutput('可以从这次组会的经历聊起。')
        request = json.loads(messages[-1]['content'])
        if purpose == 'projection':
            turn = request['history'][-1]
            return ModelOutput(json.dumps({'items':[{
                'item_id':'fixture-m0', 'section':'case_state',
                'value':'用户表示想到组会发言会紧张。', 'origin':'explicit_user_statement',
                'evidence':[{'source_id':turn['source_id'], 'quote':turn['content']}]}]}))
        if self.on_verify:
            self.on_verify()
        if self.verdict == 'timeout':
            raise TimeoutError('fixture lost response')
        return ModelOutput(json.dumps({'verdicts':[{'item_id':p['item_id'],
            'supported':self.verdict=='supported', 'reason':'fixture assessment of supplied text'} for p in request['items']]}),
            provider_model='fixture-verifier', usage={'prompt_tokens':10, 'completion_tokens':5})


class AssistantProjectionProvider(ProjectionProvider):
    def __call__(self, *, messages, purpose):
        if purpose == 'projection':
            history = json.loads(messages[-1]['content'])['history']
            assistants = [turn for turn in history if turn['role']=='assistant']
            if assistants:
                self.calls.append({'messages':messages,'purpose':purpose})
                turn=assistants[-1]
                return ModelOutput(json.dumps({'items':[{
                    'item_id':'assistant-proposal','section':'intervention_learning',
                    'value':{'proposal':turn['content'],'user_acceptance':'not_established'},
                    'origin':'assistant_proposal',
                    'evidence':[{'source_id':turn['source_id'],'quote':turn['content']}]}]}))
        return super().__call__(messages=messages,purpose=purpose)


class ProductionProjectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def session(self, provider):
        s = ControlledDialogueSession.open(str(Path(self.folder.name)/'memory.sqlite'),
            approved_root=self.folder.name, user_id='fictional', provider=provider, background=False)
        self.addCleanup(s.close)
        return s

    def initial(self, session):
        return session.send_message('想到组会上发言，我就紧张。', input_id='first', as_of='2026-11-03T09:00:00Z')

    def test_supported_projection_enters_normal_product_messages_and_deletion_removes_it(self):
        provider = ProjectionProvider()
        session = self.session(provider)
        first = self.initial(session)
        worker = session.run_pending()
        self.assertEqual(worker[-1]['supported_projection_count'],1)
        objects = session._objects('supported_projection')
        self.assertEqual(len(objects),1)
        self.assertEqual(objects[0]['payload']['verification']['kind'],'model_support_check')
        self.assertEqual(objects[0]['payload']['source_scope']['kind'],'historical_source_reports')
        self.assertEqual(next(c for c in provider.calls if c['purpose']=='projection')['messages'][0]['content'],EXTRACT)
        self.assertEqual(next(c for c in provider.calls if c['purpose']=='verification')['messages'][0]['content'],VERIFY)
        self.assertTrue(all(x.confirmation_state=='unconfirmed' for x in session.store.list_for_user('fictional')))
        session.send_message('我想继续聊组会。',input_id='next',as_of='2026-11-03T10:00:00Z')
        memory=next(m['content'] for m in provider.calls[-1]['messages'] if m['content'].startswith('SOURCED MEMORY'))
        self.assertIn('用户表示想到组会发言会紧张',memory)
        session.durable.revoke('fictional','source',first['source_commit']['source_id'])
        self.assertEqual(session.durable.get_object_state('fictional',objects[0]['object_id'])['status'],'blocked')
        session.send_message('现在有什么信息？',input_id='after-delete',as_of='2026-11-03T11:00:00Z')
        self.assertNotIn('用户表示想到组会发言会紧张',json.dumps(provider.calls[-1]['messages'],ensure_ascii=False))

    def test_rejected_verdict_does_not_promote_candidate(self):
        session = self.session(ProjectionProvider(verdict='unsupported'))
        self.initial(session)
        worker=session.run_pending()
        self.assertEqual(worker[-1]['supported_projection_count'],0)
        self.assertEqual(session._objects('supported_projection'),[])
        audit=session._objects('projection_support_audit')[0]['payload']
        self.assertFalse(audit['checks'][0]['supported'])
        self.assertEqual(len(session._objects('candidate')),1)

    def test_unknown_verification_is_preserved_and_not_resent(self):
        provider=ProjectionProvider(verdict='timeout')
        session=self.session(provider)
        self.initial(session)
        result=session.run_pending()
        self.assertEqual(result[-1]['status'],'attempt_unknown')
        before=len(provider.calls)
        session.run_pending()
        self.assertEqual(len(provider.calls),before)
        rows=session.store.connection.execute('SELECT purpose,status FROM dm_attempts ORDER BY created_at').fetchall()
        self.assertIn(('verification','attempt_unknown'),[tuple(r) for r in rows])
        self.assertEqual(session._objects('supported_projection'),[])

    def test_revocation_during_verifier_fences_projection_publication(self):
        provider=ProjectionProvider()
        session=self.session(provider)
        first=self.initial(session)
        provider.on_verify=lambda:session.durable.revoke('fictional','source',first['source_commit']['source_id'])
        result=session.run_pending()
        self.assertEqual(result[-1]['status'],'stale')
        self.assertEqual(session._objects('supported_projection'),[])

    def test_job_stages_allow_completed_predecessor_and_block_active_or_failed_predecessor(self):
        session=self.session(None)
        self.initial(session)
        durable=session.durable
        job=durable.claim_job(user_id='fictional',run_id=session.run_id)
        common=dict(run_id=session.run_id,session_id=job['session_id'],expected_generation=job['generation'],
                    job_id=job['job_id'],claim_token=job['claim_token'])
        first=durable.begin_attempt('fictional','stage-one','extraction',[job['source_id']],**common)
        with self.assertRaises(AttemptUnknown):
            durable.begin_attempt('fictional','stage-two','verification',[job['source_id']],**common)
        durable.finish_attempt(first['attempt_id'],outcome='success',response='fixture',retain_response=False)
        second=durable.begin_attempt('fictional','stage-two','verification',[job['source_id']],**common)
        durable.finish_attempt(second['attempt_id'],outcome='failed',retain_response=False)
        with self.assertRaises((AttemptUnknown,DurableConflict)):
            durable.begin_attempt('fictional','stage-three','verification',[job['source_id']],**common)

    def test_verification_provider_preview_keeps_its_own_output_contract(self):
        with patch.object(colab_provider,'require_colab'):
            provider=colab_provider.ColabChatProvider(key='fixture-key')
        messages=[{'role':'system','content':'Check support and return checks.'},
                  {'role':'user','content':'fictional support input'}]
        preview=provider.preview_request(messages,purpose='verification')
        self.assertEqual(preview['messages'],messages)
        self.assertNotIn('candidates',preview['serialized_request'])

    def test_queued_future_input_does_not_enter_earlier_projection_job(self):
        provider=ProjectionProvider()
        session=self.session(provider)
        first=self.initial(session)
        future=session.send_message('未来记录只用于后一个队列任务。',input_id='later',as_of='2026-11-03T10:00:00Z')
        session.run_pending()
        projection=next(c for c in provider.calls if c['purpose']=='projection')
        history=json.loads(projection['messages'][-1]['content'])['history']
        self.assertEqual([s['source_id'] for s in history],[first['source_commit']['source_id']])
        self.assertNotIn(future['source_commit']['source_id'],json.dumps(history))


    def test_retained_assistant_reply_enters_later_projection_as_proposal_with_original_dependencies(self):
        provider=AssistantProjectionProvider()
        session=self.session(provider)
        first=self.initial(session)
        session.run_pending()
        second=session.send_message('我还没决定是否做练习。',input_id='second',as_of='2026-11-03T10:00:00Z')
        sources=retained_assistant_sources(session.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],as_of='2026-11-03T10:00:00Z',purpose='projection')
        self.assertEqual(sources['assistant:first']['role'],'assistant')
        self.assertIsNone(sources['assistant:first']['source_time'])
        self.assertEqual(expanded_source_dependencies(list(sources.values())),[first['source_commit']['source_id']])
        outcome=session.run_pending()
        self.assertEqual(outcome[-1]['status'],'completed')
        history=json.loads([call for call in provider.calls if call['purpose']=='projection'][-1]['messages'][-1]['content'])['history']
        self.assertEqual([turn['source_id'] for turn in history],['source:first','assistant:first','source:second'])
        proposal=next(obj['payload'] for obj in session._objects('supported_projection')
            if obj['payload']['origin']=='assistant_proposal')
        self.assertEqual(proposal['value']['user_acceptance'],'not_established')
        self.assertEqual(proposal['source_ids'],['assistant:first'])
        with self.assertRaises(ValueError):
            bind_candidates(json.dumps({'items':[{'item_id':'bad','section':'case_state',
                'value':'用户已同意','origin':'explicit_user_statement',
                'evidence':[{'source_id':'assistant:first','quote':sources['assistant:first']['text']}]}]}),
                [{'source_id':'assistant:first','role':'assistant','content':sources['assistant:first']['text']}])
        rows=session.store.connection.execute("SELECT source_ids_json FROM dm_attempts WHERE purpose IN ('projection','verification')").fetchall()
        self.assertTrue(all('assistant:' not in row['source_ids_json'] for row in rows))

    def test_processed_assistant_projection_rehydrates_into_actual_messages_after_restart(self):
        provider=AssistantProjectionProvider()
        session=self.session(provider)
        self.initial(session);session.run_pending()
        session.send_message('我还没有决定。',input_id='second',as_of='2026-11-03T10:00:00Z')
        session.run_pending();session.close()
        fresh=AssistantProjectionProvider()
        restored=self.session(fresh)
        self.assertEqual(restored._turns,[])
        restored.send_message('之前建议从组会的经历聊起吗？',input_id='third',as_of='2026-11-03T11:00:00Z')
        request=next(call['messages'] for call in reversed(fresh.calls) if call['purpose']=='response')
        dynamic='\n'.join(message['content'] for message in request if message['content'].startswith('SOURCED MEMORY'))
        self.assertIn('assistant_proposal',dynamic)
        self.assertIn('not_established',dynamic)
        attempt=restored.durable.get_attempt('fictional','response:third')
        self.assertTrue(attempt['publishable'])
        ids=restored.store.connection.execute("SELECT source_ids_json FROM dm_attempts WHERE request_key='response:third'").fetchone()[0]
        self.assertNotIn('assistant:',ids)

    def test_reply_rehydrates_after_restart_and_source_withdrawal_removes_it(self):
        provider=ProjectionProvider()
        session=self.session(provider)
        first=self.initial(session)
        second=session.send_message('稍后继续聊。',input_id='second',as_of='2026-11-03T10:00:00Z')
        session.close()
        restored=self.session(ProjectionProvider())
        self.assertEqual(restored._turns,[])
        aliases=retained_assistant_sources(restored.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection')
        self.assertIn('assistant:first',aliases)
        restored.durable.revoke('fictional','source',first['source_commit']['source_id'])
        aliases=retained_assistant_sources(restored.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection')
        self.assertNotIn('assistant:first',aliases)

    def test_reply_completed_after_later_input_commit_cannot_enter_that_jobs_prefix(self):
        provider=ProjectionProvider()
        session=self.session(provider)
        self.initial(session)
        second=session.send_message('第二条已经提交。',input_id='second',as_of='2026-11-03T10:00:00Z')
        # Authored host-metadata fixture: a pending earlier reply arrives only
        # after the second source was committed. No clinical date is inferred.
        with session.store.transaction(immediate=True):
            session.store.connection.execute("UPDATE dm_attempts SET completed_at='9999-01-01T00:00:00+00:00' WHERE request_key='response:first'")
        self.assertEqual(retained_assistant_sources(session.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection'),{})
        session.run_pending();session.run_pending()
        histories=[json.loads(call['messages'][-1]['content'])['history'] for call in provider.calls if call['purpose']=='projection']
        self.assertTrue(all(turn['role']=='user' for history in histories for turn in history))

    def test_delayed_assistant_completion_is_ordered_after_intervening_user_in_later_prefix(self):
        provider=ProjectionProvider()
        session=self.session(provider)
        self.initial(session)
        session.send_message('我还没有同意任何提议。',input_id='second',as_of='2026-11-03T10:00:00Z')
        session.send_message('现在继续。',input_id='third',as_of='2026-11-03T11:00:00Z')
        with session.store.transaction(immediate=True):
            for sid,minute in [('source:first',1),('source:second',2),('source:third',5)]:
                session.store.connection.execute('UPDATE dm_sources SET committed_at=? WHERE source_id=?',
                    (f'2027-01-01T00:0{minute}:00+00:00',sid))
            for iid,minute in [('first',3),('second',4),('third',6)]:
                session.store.connection.execute('UPDATE dm_attempts SET completed_at=? WHERE request_key=?',
                    (f'2027-01-01T00:0{minute}:00+00:00','response:'+iid))
        for _ in range(3):session.run_pending()
        histories=[json.loads(call['messages'][-1]['content'])['history'] for call in provider.calls if call['purpose']=='projection']
        self.assertEqual([turn['source_id'] for turn in histories[1]],['source:first','source:second'])
        self.assertEqual([turn['source_id'] for turn in histories[2]],
            ['source:first','source:second','assistant:first','assistant:second','source:third'])
        self.assertIsNone(histories[2][2]['source_time'])

    def test_assistant_extraction_permission_is_inherited_from_all_original_sources(self):
        session=self.session(ProjectionProvider())
        first=session.send_message('这条只允许回答，暂不允许后台提取。',input_id='first',as_of='2026-11-03T09:00:00Z',
            permissions={'save':True,'long_term_use':True,'response_external':True,'extraction_external':False})
        second=session.send_message('继续。',input_id='second',as_of='2026-11-03T10:00:00Z')
        common=dict(cutoff_source_id=second['source_commit']['source_id'])
        self.assertIn('assistant:first',retained_assistant_sources(session.durable,'fictional',purpose='response',**common))
        self.assertNotIn('assistant:first',retained_assistant_sources(session.durable,'fictional',purpose='projection',**common))
        self.assertIsNotNone(session.durable.get_source('fictional',first['source_commit']['source_id']))

    def test_retained_assistant_object_update_during_support_check_blocks_publication(self):
        provider=AssistantProjectionProvider()
        session=self.session(provider)
        first=self.initial(session)
        sid=first['source_commit']['source_id']
        session.durable.put_effect('fictional','fixture-state-v1','fixture-state',expected_version=0,
            payload={'fixture':1},dependencies=[{'source_id':sid}])
        # Authored retained-response fixture with its exact originating mutable
        # object version. This is host metadata, never model-supplied evidence.
        with session.store.transaction(immediate=True):
            session.store.connection.execute("UPDATE dm_attempts SET object_versions_json=? WHERE request_key='response:first'",
                (json.dumps({'fixture-state':1}),))
        session.run_pending()
        second=session.send_message('还没有答应练习。',input_id='second',as_of='2026-11-03T10:00:00Z')
        aliases=retained_assistant_sources(session.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection')
        self.assertEqual(aliases['assistant:first']['object_versions'],{'fixture-state':1})
        provider.on_verify=lambda:session.durable.put_effect('fictional','fixture-state-v2','fixture-state',expected_version=1,
            payload={'fixture':2},dependencies=[{'source_id':sid}])
        result=session.run_pending()
        self.assertEqual(result[-1]['status'],'stale')
        self.assertTrue(all(obj['payload']['origin']!='assistant_proposal' for obj in session._objects('supported_projection')))
        self.assertNotIn('assistant:first',retained_assistant_sources(session.durable,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection'))

    def test_old_schema_migration_keeps_original_sources_but_does_not_guess_reply_completion(self):
        session=self.session(ProjectionProvider())
        first=self.initial(session)
        second=session.send_message('继续。',input_id='second',as_of='2026-11-03T10:00:00Z')
        session.close()
        path=Path(self.folder.name)/'memory.sqlite'
        connection=sqlite3.connect(path)
        connection.execute('ALTER TABLE dm_sources DROP COLUMN committed_at')
        connection.execute('ALTER TABLE dm_attempts DROP COLUMN completed_at')
        connection.commit();connection.close()
        migrated=DurableMemoryStore.open(path,approved_root=self.folder.name)
        self.addCleanup(migrated.close)
        self.assertIsNotNone(migrated.get_source('fictional',first['source_commit']['source_id']))
        self.assertEqual(retained_assistant_sources(migrated,'fictional',
            cutoff_source_id=second['source_commit']['source_id'],purpose='projection'),{})
        self.assertIn('completed_at',{row[1] for row in migrated.connection.execute('PRAGMA table_info(dm_attempts)')})
        self.assertIn('committed_at',{row[1] for row in migrated.connection.execute('PRAGMA table_info(dm_sources)')})


if __name__=='__main__':
    unittest.main()
