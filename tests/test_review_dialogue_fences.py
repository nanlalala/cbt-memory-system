"""Adversarial review regressions; fictional inputs and an offline provider only."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from controlled_dialogue import ControlledDialogueSession
from dialogue_service import ModelOutput
from durable_memory import DurableConflict

T0 = '2026-11-03T09:00:00Z'
T1 = '2026-11-04T09:00:00Z'


class ReviewProvider:
    def __init__(self):
        self.calls = []
        self.hook = None

    def __call__(self, *, messages, purpose):
        self.calls.append((purpose, messages))
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        return ModelOutput('fictional-answer-from-original-input')


class DialogueReviewFences(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def session(self, name='review.sqlite', **kwargs):
        s = ControlledDialogueSession.open(str(Path(self.temp.name) / name),
            approved_root=self.temp.name, user_id='review-user', background=False, **kwargs)
        self.addCleanup(s.close)
        return s

    def test_global_bare_yes_uniqueness_cannot_be_narrowed_by_explicit_selectors(self):
        for selector in ('none', 'first', 'last', 'period'):
            with self.subTest(selector=selector):
                s = self.session(name=selector+'.sqlite')
                first = s.propose_task('想法记录', 2, as_of=T0)['proposal']
                last = s.propose_task('散步', 2, as_of=T0)['proposal']
                opts = {}
                if selector in {'first', 'last'}:
                    selected = first if selector == 'first' else last
                    opts = {'task_instance_id': selected['task_instance_id'], 'period_id': selected['period_id']}
                elif selector == 'period':
                    opts = {'period_id': last['period_id']}
                s.send_message('yes', input_id='ambiguous-yes', as_of='2026-11-03T10:00:00Z', **opts)
                self.assertTrue(all(t['acceptance'] != 'agreed' for t in s.task_states()))

    def test_single_bare_yes_still_binds_its_actual_proposal(self):
        s = self.session()
        p = s.propose_task('想法记录', 2, as_of=T0)['proposal']
        result = s.send_message('yes', input_id='single-yes', as_of='2026-11-03T10:00:00Z',
            task_instance_id=p['task_instance_id'], period_id=p['period_id'])
        self.assertEqual(result['task_state']['acceptance'], 'agreed')
        self.assertEqual(result['task_state']['progress']['target_count'], 2)

    def test_task_correction_after_handoff_discards_body_and_cache(self):
        provider = ReviewProvider()
        first = self.session(provider=provider, run_id='answer-run')
        first.send_message('我同意做三次想法记录。', input_id='agree', as_of=T0)
        other = self.session(run_id='correction-run')
        provider.hook = lambda: other.send_message('想法记录改成两次。', input_id='correct', as_of=T1)
        result = first.send_message('想法记录现在安排几次？', input_id='question', as_of=T1)
        self.assertEqual(other.task_states()[0]['progress']['target_count'], 2)
        self.assertEqual(result['answer_status'], 'stale_discarded')
        attempt = first.durable.get_attempt(first.user_id, 'response:question')
        self.assertIsNone(attempt['response'])
        self.assertIsNone(first.store.connection.execute(
            "SELECT response_json FROM dm_attempts WHERE request_key='response:question'").fetchone()[0])
        count = len(provider.calls)
        retry = first.send_message('想法记录现在安排几次？', input_id='question', as_of=T1)
        self.assertNotIn('fictional-answer-from-original-input', retry['text'])
        self.assertEqual(len(provider.calls), count)

    def test_task_correction_after_assembly_prevents_send(self):
        provider = ReviewProvider()
        first = self.session(provider=provider, run_id='answer-run')
        first.send_message('我同意做三次想法记录。', input_id='agree', as_of=T0)
        other = self.session(run_id='correction-run')
        original_begin = first.durable.begin_attempt
        def before_admission(*args, **kwargs):
            other.send_message('想法记录改成两次。', input_id='correct', as_of=T1)
            return original_begin(*args, **kwargs)
        count = len(provider.calls)
        with patch.object(first.durable, 'begin_attempt', side_effect=before_admission):
            result = first.send_message('想法记录现在安排几次？', input_id='question', as_of=T1)
        self.assertEqual(result['answer_status'], 'stale_not_sent')
        self.assertEqual(len(provider.calls), count)

    def test_unrelated_object_update_does_not_invalidate_used_task(self):
        provider = ReviewProvider()
        first = self.session(provider=provider, run_id='answer-run')
        first.send_message('我同意做三次想法记录。', input_id='agree', as_of=T0)
        other = self.session(run_id='correction-run')
        provider.hook = lambda: other.send_message('我同意散步两次。', input_id='walking', as_of=T1)
        result = first.send_message('想法记录现在安排几次？', input_id='question', as_of=T1)
        self.assertEqual(result['answer_status'], 'succeeded')

    def test_nonlongterm_answer_cache_is_current_session_only(self):
        provider = ReviewProvider()
        s = self.session(provider=provider, session_id='original-session')
        perms = dict(save=True, long_term_use=False, response_external=True, extraction_external=True)
        s.send_message('一条虚构的仅当轮报告。', input_id='limited-input', as_of=T0, permissions=perms)
        same = s.send_message('一条虚构的仅当轮报告。', input_id='limited-input', as_of=T0, permissions=perms)
        self.assertIn('fictional-answer-from-original-input', same['text'])
        original_session = s.session_id
        s.new_session()
        result = s.send_message('一条虚构的仅当轮报告。', input_id='limited-input', as_of=T0, permissions=perms)
        self.assertNotIn('fictional-answer-from-original-input', result['text'])
        self.assertTrue(result['replayed'])
        self.assertEqual(result['source_commit']['session_id'], original_session)
        self.assertEqual(len(provider.calls), 1)

    def test_nonlongterm_cache_is_not_released_after_reopen_in_other_session(self):
        provider = ReviewProvider()
        s = self.session(provider=provider, session_id='original-session')
        perms = dict(save=True, long_term_use=False, response_external=True, extraction_external=True)
        s.send_message('虚构的受限报告。', input_id='limited', as_of=T0, permissions=perms)
        other = self.session(provider=provider, session_id='other-session')
        result = other.send_message('虚构的受限报告。', input_id='limited', as_of=T0, permissions=perms)
        self.assertNotIn('fictional-answer-from-original-input', result['text'])
        self.assertEqual(len(provider.calls), 1)

    def test_retry_cannot_change_permissions_or_explicit_source_time(self):
        provider = ReviewProvider()
        s = self.session(provider=provider)
        s.send_message('虚构的固定输入。', input_id='fixed', as_of=T0)
        for options in ({'permissions': dict.fromkeys(s.default_permissions, False), 'as_of': T0},
                        {'as_of': T1}):
            with self.subTest(options=options), self.assertRaises(DurableConflict):
                s.send_message('虚构的固定输入。', input_id='fixed', **options)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(s.durable.list_sources(s.user_id)), 1)

    def test_ui_retry_without_explicit_time_inherits_original_operation_clock(self):
        provider = ReviewProvider()
        s = self.session(provider=provider)
        s.send_message('虚构的固定输入。', input_id='fixed')
        result = s.send_message('虚构的固定输入。', input_id='fixed')
        self.assertTrue(result['replayed'])
        self.assertEqual(result['answer_replay_status'], 'body_available')
        self.assertEqual(len(provider.calls), 1)


if __name__ == '__main__':
    unittest.main()
