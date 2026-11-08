from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from controlled_dialogue import ControlledDialogueSession, PrototypeLimits
from dialogue_service import ModelOutput
from durable_memory import DurableLimits
from knowledge_adapter_v2 import KnowledgeAdapter

TIMES = ['2026-11-03T09:00:00Z', '2026-11-04T09:00:00Z', '2026-11-05T09:00:00Z', '2026-11-06T09:00:00Z']


class OfflineProvider:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    def __call__(self, *, messages, purpose):
        self.calls.append({'messages': messages, 'purpose': purpose})
        if self.fail:
            raise TimeoutError('transport outcome may be unknown')
        if purpose == 'extraction':
            request = json.loads(messages[-1]['content'])
            turn = request['turns'][0]
            return ModelOutput(json.dumps({'candidates': [{'memory_type': 'explicit_fact', 'content': turn['text'],
                'source_turn_ids': [turn['turn_id']], 'supporting_text': turn['text']}]}))
        return ModelOutput('我们可以按本轮明确报告来跟进。')


class ControlledDialogueTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.sessions = []
    def session(self, *, provider=None, name='memory.sqlite', **kwargs):
        s = ControlledDialogueSession.open(str(Path(self.folder.name) / name), approved_root=self.folder.name,
            user_id='fictional-user', provider=provider, background=False, **kwargs)
        self.sessions.append(s)
        self.addCleanup(s.close)
        return s
    def send(self, s, text, i, **kwargs):
        return s.send_message(text, input_id='input-' + str(i), as_of=TIMES[min(i, 3)], **kwargs)

    def test_source_first_acceptance_progress_pause_outcome_axes(self):
        s = self.session()
        a = self.send(s, '我同意每周做两次想法记录。', 0)
        tid = a['task_state']['task_instance_id']
        self.send(s, '想法记录累计做了一次。', 1)
        paused = self.send(s, '这周暂停', 2)['task_state']
        self.assertEqual(paused['task_instance_id'], tid)
        self.assertEqual(paused['acceptance'], 'agreed')
        self.assertEqual(paused['progress']['reported_count'], 1)
        self.assertEqual(paused['progress']['target_count'], 2)
        self.assertEqual(paused['participation'], 'paused')
        self.assertEqual(paused['outcome'], 'unknown')
        self.assertTrue(all(x.confirmation_state == 'unconfirmed' for x in s.store.list_for_user(s.user_id)))

    def test_negated_predicates_never_become_opposite_durable_task_state(self):
        for index, text in enumerate(['想法记录不要暂停。', '想法记录不要取消。', '我没有拒绝想法记录。']):
            with self.subTest(text=text):
                s = self.session(name='negated-' + str(index) + '.sqlite')
                first = self.send(s, '我同意每周做两次想法记录。', 0)
                state_before = first['task_state']
                reply = self.send(s, text, 1)
                self.assertEqual(reply['source_commit']['source_status'], 'committed')
                self.assertEqual(reply['task_state']['acceptance'], 'agreed')
                self.assertEqual(reply['task_state']['participation'], state_before['participation'])
                persisted = s.task_states()[0]
                self.assertEqual(persisted['acceptance'], 'agreed')
                self.assertEqual(persisted['participation'], state_before['participation'])
                later = self.send(s, '想法记录现在怎么安排？', 2)
                self.assertEqual(later['task_state']['acceptance'], 'agreed')
                self.assertEqual(later['task_state']['participation'], state_before['participation'])

    def test_task_questions_do_not_update_the_state_they_are_asking_about(self):
        s = self.session()
        self.send(s, '我同意做两次想法记录。', 0)
        for index, text in enumerate(['想法记录暂停了吗？', '想法记录取消了吗？', '我是否拒绝了想法记录？']):
            reply = self.send(s, text, index + 1)
            self.assertEqual(reply['source_commit']['source_status'], 'committed')
            self.assertEqual(s.task_states()[0]['acceptance'], 'agreed')
            self.assertEqual(s.task_states()[0]['participation'], 'unknown')

    def test_negated_cancellation_does_not_infer_reactivation_of_existing_pause(self):
        s = self.session()
        self.send(s, '我同意做两次想法记录。', 0)
        self.send(s, '想法记录暂停。', 1)
        reply = self.send(s, '想法记录不要取消。', 2)
        self.assertEqual(reply['source_commit']['source_status'], 'committed')
        self.assertEqual(reply['task_state']['participation'], 'paused')
        self.assertEqual(s.task_states()[0]['participation'], 'paused')

    def test_no_save_pause_is_current_only_and_cannot_return_through_next_context(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        self.send(s, '我同意每周做两次想法记录。', 0)
        reply = self.send(s, '这周暂停，这条别保存', 1)
        self.assertEqual(reply['source_commit']['source_status'], 'not_saved')
        self.assertEqual(reply['organizing_status'], 'not_queued')
        self.assertEqual(reply['task_state']['participation'], 'paused')
        self.assertTrue(reply['task_state']['not_retained'])
        dump = '\n'.join(s.store.connection.iterdump())
        self.assertNotIn('这条别保存', dump)
        self.assertNotIn('这周暂停', json.dumps(s.snapshot(), ensure_ascii=False))
        self.send(s, '想法记录现在怎么安排？', 2)
        self.assertNotIn('这条别保存', json.dumps(provider.calls[-1], ensure_ascii=False))
        state = s.durable.get_object_state(s.user_id, s._objects()[0]['object_id'])
        self.assertEqual(state['status'], 'blocked')

    def test_answer_unknown_keeps_source_and_control_committed_and_does_not_recharge(self):
        provider = OfflineProvider(fail=True); s = self.session(provider=provider, session_id='stable-session')
        result = self.send(s, '我同意做两次想法记录。', 0)
        self.assertEqual(result['source_commit']['source_status'], 'committed')
        self.assertEqual(result['answer_status'], 'failed_or_unknown')
        self.assertEqual(s.task_states()[0]['acceptance'], 'agreed')
        self.assertEqual(len(provider.calls), 1)
        same = self.send(s, '我同意做两次想法记录。', 0)
        self.assertTrue(same['replayed'])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(s.durable.list_sources(s.user_id)), 1)
        self.assertEqual(same['answer_status'], 'attempt_unknown')

    def test_response_success_does_not_claim_a_failed_source_write(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        with patch.object(s.durable, 'submit_source', side_effect=OSError('fake storage failure')):
            reply = self.send(s, '这是一个虚构的反思。', 0)
        self.assertEqual(reply['answer_status'], 'succeeded')
        self.assertEqual(reply['source_commit']['source_status'], 'failed')
        self.assertEqual(reply['memory_processing']['stored_count'], 0)
        self.assertIsNone(reply['usage'])

    def test_shared_risk_pause_closes_old_session_and_background_worker_admission(self):
        provider = OfflineProvider()
        first = self.session(provider=provider)
        second = self.session(provider=provider)
        self.send(first, '我同意做两次想法记录。', 0)
        self.send(first, '我想自杀', 1)
        before = len(provider.calls)
        blocked = self.send(second, '旧会话的一条虚构普通输入', 2)
        self.assertTrue(blocked['safety_bypass'])
        self.assertEqual(blocked['source_commit']['source_status'], 'not_saved')
        self.assertEqual(second.run_pending(), [{'status': 'safety_paused'}])
        self.assertEqual(len(provider.calls), before)
        self.assertNotIn('旧会话的一条虚构普通输入', '\n'.join(first.store.connection.iterdump()))

    def test_check_burden_records_answer_separately_without_confirming_old_memory(self):
        s = self.session()
        self.send(s, '我同意做想法记录。', 0)
        asked = self.send(s, '想法记录安排几次？', 1)
        self.assertEqual(len(asked['memory_checks']), 1)
        self.send(s, '我同意每周两次想法记录。', 2)
        ledger = s.snapshot()['clarification_events']
        self.assertEqual([x['event'] for x in ledger], ['check_shown', 'check_answered'])
        self.assertTrue(all(x.confirmation_state != 'user_confirmed' for x in s.store.list_for_user(s.user_id)))

    def test_today_topic_pause_blocks_task_context_until_local_day_end(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        self.send(s, '我同意做两次想法记录。', 0)
        calls = len(provider.calls)
        paused = self.send(s, '想法记录今天不谈', 1)
        self.assertEqual(paused['answer_status'], 'local_preference_control')
        self.assertEqual(len(provider.calls), calls)
        query = s.send_message('想法记录现在怎么安排？', input_id='same-day', as_of='2026-11-04T10:00:00Z')
        self.assertIsNone(query['task_state'])
        self.assertEqual(query['answer_status'], 'local_preference_control')
        self.assertEqual(len(provider.calls), calls)
        query = self.send(s, '想法记录现在怎么安排？', 2)
        self.assertEqual(query['task_state']['progress']['target_count'], 2)

    def test_latest_short_total_survives_irrelevant_long_superseded_totals(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        self.send(s, '我同意每周做五次想法记录。', 0)
        for i in range(3):
            s.send_message('虚构长背景' * 180 + '。想法记录累计做了一次。', input_id='long-total-' + str(i),
                           as_of='2026-11-04T0' + str(i + 1) + ':00:00Z')
        s.send_message('想法记录累计做了两次。', input_id='latest-total', as_of='2026-11-05T09:00:00Z')
        result = s.send_message('想法记录目前完成了多少次？', input_id='ask-total', as_of='2026-11-06T09:00:00Z')
        self.assertEqual(result['task_state']['progress']['reported_count'], 2)
        context = next(m['content'] for m in provider.calls[-1]['messages'] if m['content'].startswith('SOURCED MEMORY'))
        data = json.loads(context.split('\n', 1)[1])
        count_claim = next(c for c in data['claims'] if c['field'] == 'reported_count')
        self.assertEqual(count_claim['value'], 2)
        self.assertEqual({e['source_id'] for e in count_claim['evidence']}, {'source:latest-total'})
        self.assertNotIn('虚构长背景', json.dumps(count_claim, ensure_ascii=False))

    def test_later_explicit_object_does_not_rebind_earlier_implicit_control_to_future_source(self):
        s = self.session()
        self.send(s, '我同意做两次想法记录。', 0)
        self.send(s, '这周暂停', 1)
        resumed = self.send(s, '想法记录恢复。', 2)
        self.assertEqual(resumed['source_commit']['source_status'], 'committed')
        self.assertEqual(s.task_states()[0]['participation'], 'active')
        self.assertEqual(s.task_states()[0]['progress']['target_count'], 2)

    def test_historical_or_quoted_agreement_is_not_current_self_report(self):
        for i, text in enumerate(['去年我同意每周做两次想法记录。',
                                  'Last year I agreed to do two thought records.',
                                  '老师说：“我同意做两次想法记录。”',
                                  '2025年我同意每周做两次想法记录。',
                                  '2025-11-03，我同意每周做两次想法记录。']):
            s = self.session(name='historical-' + str(i) + '.sqlite')
            reply = self.send(s, text, 0)
            self.assertFalse(reply['task_state'] and reply['task_state']['acceptance'] == 'agreed')
            self.assertEqual(s.task_states(), [])
            self.assertEqual(reply['source_commit']['source_status'], 'committed')

    def test_topic_pause_filters_same_topic_unstructured_quotes_from_other_queries(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        self.send(s, '我同意做两次想法记录。', 0)
        s.send_message('想法记录让我想到一个虚构的旧经历，细节代号青色山谷。',
                       input_id='topic-quote', as_of='2026-11-04T09:00:00Z')
        s.send_message('想法记录今天不谈。', input_id='topic-pause', as_of='2026-11-04T10:00:00Z')
        result = s.send_message('青色山谷是什么意思？', input_id='different-query', as_of='2026-11-04T11:00:00Z')
        self.assertEqual(result['answer_status'], 'succeeded')
        memory = [m['content'] for m in provider.calls[-1]['messages'] if m['role'] == 'system']
        self.assertNotIn('细节代号青色山谷', json.dumps(memory, ensure_ascii=False))

    def test_multi_object_pause_evidence_contains_paused_predicate_for_same_object(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        self.send(s, '我同意做两次想法记录。', 0)
        result = self.send(s, '散步继续，想法记录暂停。', 1)
        self.assertEqual(result['task_state']['participation'], 'paused')
        context = next(m['content'] for m in provider.calls[-1]['messages'] if m['content'].startswith('SOURCED MEMORY'))
        data = json.loads(context.split('\n', 1)[1])
        paused = next(c for c in data['claims'] if c['field'] == 'engagement')
        evidence = ' '.join(e['text'] for e in paused['evidence'])
        self.assertIn('想法记录暂停', evidence)
        self.assertNotIn('散步继续', evidence)

    def test_response_attempt_after_failed_source_is_still_canonical_on_retry(self):
        provider = OfflineProvider(); s = self.session(provider=provider)
        with patch.object(s.durable, 'submit_source', side_effect=OSError('fake storage failure')):
            first = self.send(s, '虚构写入失败但回答完成', 0)
        self.assertEqual(first['answer_status'], 'succeeded')
        again = self.send(s, '虚构写入失败但回答完成', 0)
        self.assertTrue(again['replayed'])
        self.assertEqual(again['source_commit']['source_status'], 'failed')
        self.assertEqual(len(provider.calls), 1)

    def test_legacy_opt_out_equivalents_remain_unsaved_in_controlled_entry(self):
        s = self.session()
        for i, text in enumerate(['不用记录，这是一条虚构秘密。', '不必保存这条反思。',
                                  'Don’t remember this private detail.', "I don't want you to remember this."]):
            reply = self.send(s, text, i)
            self.assertEqual(reply['source_commit']['source_status'], 'not_saved')
            self.assertEqual(reply['organizing_status'], 'not_queued')
            self.assertNotIn(text, '\n'.join(s.store.connection.iterdump()))
            self.assertNotIn(text, json.dumps(s.snapshot(), ensure_ascii=False))

    def test_external_permissions_are_separate_and_unknown_never_allows(self):
        p = OfflineProvider(); s = self.session(provider=p)
        r = self.send(s, '不要外发，这是一条虚构记录。', 0)
        self.assertEqual(r['source_commit']['source_status'], 'committed')
        self.assertEqual(r['answer_status'], 'external_response_not_permitted')
        self.assertEqual(p.calls, [])
        self.assertEqual(s.run_pending(), [])
        s2 = self.session(provider=p, name='unknown.sqlite', permissions={k: None for k in
            ('save', 'long_term_use', 'response_external', 'extraction_external')})
        r2 = self.send(s2, '一条未授权记录', 0)
        self.assertEqual(r2['source_commit']['source_status'], 'not_saved')
        self.assertEqual(p.calls, [])

    def test_risk_route_has_no_normal_source_job_or_provider_and_delete_still_works(self):
        p = OfflineProvider(); s = self.session(provider=p)
        first = self.send(s, '我同意做两次想法记录。', 0)
        oid = 'task:' + first['task_state']['task_instance_id'] + ':' + first['task_state']['period_id']
        before = len(p.calls)
        risky = s.send_message('我想自杀', input_id='risk', as_of=TIMES[1],
            local_controls=[{'scope': 'object', 'target_id': oid}])
        self.assertTrue(risky['safety_bypass'])
        self.assertTrue(risky['local_delete']['logical_blocked'])
        self.assertEqual(s.durable.get_object_state(s.user_id, oid)['status'], 'blocked')
        self.assertEqual(len(p.calls), before)
        self.assertNotIn('我想自杀', '\n'.join(s.store.connection.iterdump()))
        s.new_session()
        again = self.send(s, '你好', 2)
        self.assertTrue(again['safety_bypass'])

    def test_current_correction_keeps_identity_and_uses_its_own_source(self):
        p = OfflineProvider(); s = self.session(provider=p)
        a = self.send(s, '我同意做三次想法记录。', 0)
        b = self.send(s, '这周改成两次', 1)
        self.assertEqual(a['task_state']['task_instance_id'], b['task_state']['task_instance_id'])
        self.assertEqual(b['task_state']['progress']['target_count'], 2)
        data = json.dumps(p.calls[-1]['messages'], ensure_ascii=False)
        self.assertIn('source:input-1', data)
        task = s._objects('task')[0]['payload']
        self.assertEqual(task['events'][0]['payload']['target_count'], 3)
        self.assertEqual(task['events'][-1]['event_type'], 'corrected')

    def test_unrelated_question_has_no_task_state_injection(self):
        p = OfflineProvider(); s = self.session(provider=p)
        self.send(s, '我同意做两次想法记录。', 0)
        out = self.send(s, 'Explain recursion in Python.', 1)
        self.assertIsNone(out['task_state'])
        self.assertNotIn('target_count', json.dumps(p.calls[-1]['messages']))

    def test_only_unique_actual_proposal_accepts_next_yes_not_memory_confirmation(self):
        s = self.session()
        proposed = s.propose_task('想法记录', 2, as_of=TIMES[0])
        result = s.send_message('yes', input_id='yes', as_of='2026-11-03T10:00:00Z')
        self.assertEqual(result['task_state']['acceptance'], 'agreed')
        self.assertEqual(result['task_state']['progress']['target_count'], 2)
        self.assertNotIn('user_confirmed', json.dumps(result))
        s2 = self.session(name='bare.sqlite')
        self.assertIsNone(self.send(s2, 'yes', 0)['task_state'])
        s3 = self.session(name='multi.sqlite')
        s3.propose_task('想法记录', 2, as_of=TIMES[0])
        s3.propose_task('散步', 2, as_of=TIMES[0])
        self.assertIsNone(s3.send_message('yes', input_id='multi-yes', as_of='2026-11-03T10:00:00Z')['task_state'])

    def test_duplicate_id_cannot_bypass_risk_or_hide_different_source_text(self):
        s = self.session()
        self.send(s, '一条虚构反思。', 0)
        risk = s.send_message('我想自杀', input_id='input-0', as_of=TIMES[1])
        self.assertTrue(risk['safety_bypass'])
        self.assertTrue(s.snapshot()['safety_paused'])
        other = self.session(name='different.sqlite')
        self.send(other, '一条虚构反思。', 0)
        with self.assertRaises(ValueError):
            other.send_message('不同正文', input_id='input-0', as_of=TIMES[1])

    def test_revoke_between_context_build_and_admission_never_filters_away_dependency(self):
        class RevokingProvider(OfflineProvider):
            hook = None
            def preview_request(self, *, messages, purpose):
                if self.hook:
                    action, self.hook = self.hook, None
                    action()
                payload = {'messages': messages, 'max_tokens': 4096}
                return {'payload': payload, 'serialized_length': len(json.dumps(payload)), 'within_character_budget': True}
        provider = RevokingProvider(); s = self.session(provider=provider)
        self.send(s, '我的日记本是紫色。', 0)
        before = len(provider.calls)
        provider.hook = lambda: s.delete_content('source', 'source:input-0')
        r = self.send(s, '我的日记本是什么颜色？', 1)
        self.assertEqual(r['answer_status'], 'stale_not_sent')
        self.assertEqual(len(provider.calls), before)
        self.assertIsNone(r['task_state'])

    def test_offline_worker_uses_actual_canonical_manager_candidates(self):
        s = self.session()
        self.send(s, '我同意做两次想法记录。', 0)
        work = s.run_pending()
        self.assertEqual(work[-1]['semantic_status'], 'source_only_no_model')
        records = s.store.list_for_user(s.user_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].schema_version, '2.0')
        self.assertEqual(records[0].confirmation_state, 'unconfirmed')
        self.assertEqual(records[0].lifecycle_status, 'candidate')

    def test_new_session_and_reopen_do_not_reset_run_budget(self):
        p = OfflineProvider(); limits = DurableLimits(max_run_calls=1, max_session_calls=1)
        s = self.session(provider=p, durable_limits=limits)
        self.send(s, '我同意做两次想法记录。', 0)
        s.new_session()
        r = self.send(s, '想法记录怎么安排？', 1)
        self.assertEqual(r['answer_status'], 'budget_not_sent')
        self.assertEqual(len(p.calls), 1)

    def test_k1_without_approved_material_is_unavailable_not_a_real_knowledge_result(self):
        s = self.session(knowledge=KnowledgeAdapter(mode='K1'))
        r = self.send(s, '我同意做两次想法记录。', 0)
        self.assertEqual(r['knowledge_status'], 'unavailable')
        self.assertEqual(r['knowledge_ids'], [])


if __name__ == '__main__':
    unittest.main()
