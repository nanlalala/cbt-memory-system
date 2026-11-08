from datetime import datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from natural_memory_policy import CheckRequest, bound_task_choice, decide_check, permissions_for_input, refusal_for_input


class NaturalPolicyTests(unittest.TestCase):
    def test_no_save_does_not_grant_organizer_or_remove_current_response_permission(self):
        permissions = permissions_for_input('这周暂停，这条别保存', {k: True for k in
            ('save', 'long_term_use', 'response_external', 'extraction_external')})
        self.assertEqual(permissions, {'save': False, 'long_term_use': False,
                                      'response_external': True, 'extraction_external': False})

    def test_unknown_permission_is_not_coerced(self):
        self.assertIsNone(permissions_for_input('hello', {k: None for k in
            ('save', 'long_term_use', 'response_external', 'extraction_external')})['save'])

    def test_same_field_not_reasked_after_unrelated_progress_change(self):
        r = CheckRequest('u', 'task', 'target_count', None, 'unknown_quantity', '几次？', True)
        asked = {'identity': r.identity, 'event': 'check_shown', 'occurred_at': '2026-11-08T00:00:00Z'}
        self.assertEqual(decide_check(r, as_of='2026-11-08T01:00:00Z', events=[asked])['action'], 'suppress')
        other = CheckRequest('u', 'other-task', 'target_count', None, 'unknown_quantity', '几次？', True)
        self.assertEqual(decide_check(other, as_of='2026-11-08T01:00:00Z', events=[asked])['action'], 'ask')

    def test_explicit_report_and_immaterial_uncertainty_never_interrupt(self):
        for necessary, explicit in [(True, True), (False, False)]:
            r = CheckRequest('u', 'preference:writings', 'engagement', 'paused', 'current', '继续？', necessary, explicit)
            self.assertEqual(decide_check(r, as_of='2026-11-08T00:00:00Z')['action'], 'suppress')

    def test_refusal_scope_survives_turns_but_not_next_local_day(self):
        refusal = refusal_for_input('今天不谈', subject_id='t', as_of='2026-11-08T15:00:00Z', timezone_name='Asia/Shanghai')
        r = CheckRequest('u', 't', 'count', None, 'unknown', '几次？', True)
        self.assertEqual(decide_check(r, as_of='2026-11-08T15:30:00Z', refusals=[refusal])['action'], 'suppress')
        self.assertEqual(decide_check(r, as_of='2026-11-08T16:01:00Z', refusals=[refusal])['action'], 'ask')
        forever = refusal_for_input('以后别问', subject_id='t', as_of='2026-11-08T15:00:00Z')
        self.assertIsNone(forever['expires_at'])
        self.assertEqual(forever['kind'], 'check_refusal')

    def test_yes_requires_exact_unique_action_and_next_turn_binding(self):
        p = {'proposal_id': 'p', 'task_instance_id': 't', 'period_id': 'week', 'user_id': 'u',
             'session_id': 's', 'turn_id': 'a', 'action_count': 1, 'question_count': 1,
             'created_at': '2026-11-08T00:00:00Z', 'expires_at': '2026-11-09T00:00:00Z'}
        kwargs = dict(user_id='u', session_id='s', reply_to_turn_id='a', as_of='2026-11-08T01:00:00Z')
        self.assertEqual(bound_task_choice('yes', [p], **kwargs)['choice'], 'agreed')
        for proposals in [[], [p, {**p, 'proposal_id': 'p2'}],
                          [p, {**p, 'proposal_id': 'compound', 'action_count': 2}],
                          [{**p, 'question_count': 2}], [{**p, 'action_count': 2}]]:
            self.assertIsNone(bound_task_choice('yes', proposals, **kwargs))
        self.assertIsNone(bound_task_choice('yes', [p], **{**kwargs, 'reply_to_turn_id': 'unrelated'}))
        self.assertIsNone(bound_task_choice('yes', [p], **{**kwargs, 'as_of': '2026-11-09T00:00:00Z'}))


if __name__ == '__main__':
    unittest.main()
