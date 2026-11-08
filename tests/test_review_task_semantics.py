"""Review counterexamples: bounded source rules, task identity and coverage.

All materials are fictional. No model, provider or network is used.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from controlled_dialogue import ControlledDialogueSession
from task_projector import project_task, validate_task_event, verify_claim_binding
from task_input_rules import resolve_attempt_reference

MON = '2026-11-02T09:00:00Z'
WED = '2026-11-04T10:00:00Z'
THU = '2026-11-05T10:00:00Z'
FRI = '2026-11-06T10:00:00Z'


def binding(text, kind, value, field=None):
    claim = {'kind': kind, 'value': value}
    if field:
        claim['field'] = field
    return {'start': 0, 'end': len(text), 'claim': claim}


def event_for(text, choice, *, ident='e', reported=WED, occurred=None, count=None, cutoff=None):
    label = 'thought record' if 'thought record' in text else '想法记录'
    payload = {'object_label': label}
    kind = ('participation_changed' if choice in {'paused', 'cancelled', 'active'} else
            'outcome_report' if choice in {'unhelpful', 'helpful', 'mixed', 'unknown'} else
            'progress_report' if choice == 'cumulative' else choice)
    if kind == 'participation_changed':
        payload['participation'] = choice
    elif kind == 'outcome_report':
        payload['outcome'] = choice
    elif kind == 'progress_report':
        payload.update(report_mode='cumulative', count=count, covers_through=cutoff or occurred or reported)
    obj = {'event_id': ident, 'source_id': ident, 'user_id': 'u', 'goal_id': 'g',
           'task_instance_id': 'task', 'period_id': '2026-11-02', 'event_type': kind,
           'occurred_at': occurred or reported, 'reported_at': reported,
           'origin': 'explicit_user_statement', 'payload': payload,
           'claim_bindings': [binding(text, 'object', label), binding(text, 'choice', choice)]}
    if count is not None:
        obj['claim_bindings'].append(binding(text, 'quantity', count, 'count'))
    source = {'user_id': 'u', 'text': text, 'role': 'user', 'source_time': reported}
    return obj, source


class ReviewTaskRuleTests(unittest.TestCase):
    BAD_POLARITY = [
        ('想法记录暂停不了。', 'paused'),
        ('想法记录取消不掉。', 'cancelled'),
        ('我拒绝不做想法记录。', 'declined'),
        ('I refuse not to do thought record.', 'declined'),
        ('想法记录不再没有帮助。', 'unhelpful'),
        ('thought record is not mixed.', 'mixed'),
    ]

    def test_complete_predicate_polarity_is_required_by_public_entrypoints(self):
        for text, choice in self.BAD_POLARITY:
            with self.subTest(text=text):
                self.assertFalse(verify_claim_binding(text, 0, len(text), {'kind': 'choice', 'value': choice}))
                obj, source = event_for(text, choice)
                with self.assertRaises(ValueError):
                    validate_task_event(obj, sources={'e': source})
                with self.assertRaises(ValueError):
                    project_task([obj], user_id='u', task_instance_id='task', period_id='2026-11-02', sources={'e': source})

    def test_outer_negation_cannot_be_omitted_by_narrow_bindings(self):
        for text, choice, excerpt in [
            ('想法记录暂停不了。', 'paused', '暂停'),
            ('想法记录取消不掉。', 'cancelled', '取消'),
            ('我拒绝不做想法记录。', 'declined', '拒绝'),
            ('I refuse not to do thought record.', 'declined', 'refuse'),
            ('想法记录不再没有帮助。', 'unhelpful', '没有帮助'),
            ('thought record is not mixed.', 'mixed', 'mixed'),
        ]:
            start = text.index(excerpt)
            with self.subTest(text=text):
                self.assertFalse(verify_claim_binding(text, start, start + len(excerpt), {'kind': 'choice', 'value': choice}))

    def test_assertions_and_genuine_negative_choices_remain_usable(self):
        for text, choice in [
            ('想法记录暂停。', 'paused'), ('想法记录这周不做。', 'paused'),
            ('我不同意想法记录。', 'declined'), ('我拒绝做想法记录。', 'declined'),
            ('想法记录没帮助。', 'unhelpful'), ('想法记录没有帮助。', 'unhelpful'),
            ('thought record is mixed.', 'mixed'), ('thought record is not helpful.', 'unhelpful'),
            ('I paused thought record.', 'paused'), ('想法记录不知道有没有用。', 'unknown'),
            ('想法记录暂停，我不想散步。', 'paused'),
        ]:
            with self.subTest(text=text):
                self.assertTrue(verify_claim_binding(text, 0, len(text), {'kind': 'choice', 'value': choice}))
                obj, source = event_for(text, choice)
                self.assertEqual(validate_task_event(obj, sources={'e': source})['event_type'], obj['event_type'])

    def test_unpunctuated_questions_do_not_assert_current_state(self):
        for text, choice in [
            ('Have I paused thought record', 'paused'),
            ('Is thought record paused', 'paused'),
            ('Was thought record cancelled', 'cancelled'),
            ('Did I agree to thought record', 'agreed'),
            ('想法记录暂停行不行', 'paused'),
            ('想法记录取消好不好', 'cancelled'),
            ('想法记录能不能暂停', 'paused'),
            ('想法记录暂停了没', 'paused'),
            ('想法记录会不会暂停', 'paused'),
            ('Have I paused thought record?', 'paused'),
        ]:
            with self.subTest(text=text):
                obj, source = event_for(text, choice)
                with self.assertRaises(ValueError):
                    validate_task_event(obj, sources={'e': source})

    def test_unique_day_reference_reuses_existing_second_attempt(self):
        second = resolve_attempt_reference('今天做了第二次想法记录。', source_time=WED,
            task_instance_id='task', period_id='2026-11-02')
        existing = [{'event_type': 'progress_report', 'task_instance_id': 'task',
                     'period_id': '2026-11-02', 'event_date': '2026-11-04',
                     'payload': {'report_mode': 'attempt', 'attempt_id': second['attempt_id'], 'count': 1}}]
        again = resolve_attempt_reference('周三那次做过一次想法记录。', source_time=FRI,
            task_instance_id='task', period_id='2026-11-02', existing_events=existing)
        self.assertEqual(again['attempt_id'], second['attempt_id'])
        first = resolve_attempt_reference('今天做了第一次想法记录。', source_time=WED,
            task_instance_id='task', period_id='2026-11-02', existing_events=existing)
        self.assertNotEqual(first['attempt_id'], second['attempt_id'])
        ambiguous = existing + [{**existing[0], 'payload': {**existing[0]['payload'], 'attempt_id': first['attempt_id']}}]
        self.assertIsNone(resolve_attempt_reference('周三那次做过一次想法记录。', source_time=FRI,
            task_instance_id='task', period_id='2026-11-02', existing_events=ambiguous))

    def test_cumulative_coverage_not_report_order_controls_progress(self):
        current, s1 = event_for('想法记录累计做了3次。', 'cumulative', ident='current', reported=THU, count=3)
        for prior, conflict in [(2, False), (4, True)]:
            earlier, s2 = event_for(f'想法记录累计做了{prior}次。', 'cumulative', ident='late', reported=FRI,
                                   occurred=WED, cutoff=WED, count=prior)
            with self.subTest(prior=prior):
                state = project_task([current, earlier], user_id='u', task_instance_id='task',
                                     period_id='2026-11-02', sources={'current': s1, 'late': s2})
                self.assertEqual(state['progress']['reported_total'], 3)
                self.assertEqual(state['progress']['count_conflict'], conflict)
                self.assertEqual(state['progress']['reported_count'], None if conflict else 3)
                self.assertEqual(state['field_support_event_ids']['reported_total'], ['current'])

    def test_conflicting_same_coverage_does_not_use_receipt_order_as_truth(self):
        first, source1 = event_for('想法记录累计做了3次。', 'cumulative', ident='first', reported=THU, occurred=WED, cutoff=WED, count=3)
        second, source2 = event_for('想法记录累计做了2次。', 'cumulative', ident='second', reported=FRI, occurred=WED, cutoff=WED, count=2)
        state = project_task([first, second], user_id='u', task_instance_id='task', period_id='2026-11-02',
                             sources={'first': source1, 'second': source2})
        self.assertTrue(state['progress']['count_conflict'])
        self.assertIsNone(state['progress']['reported_count'])
        # The late source must not affect an earlier visibility cutoff.
        earlier = project_task([first, second], user_id='u', task_instance_id='task', period_id='2026-11-02',
                               sources={'first': source1, 'second': source2}, source_cutoff=THU)
        self.assertEqual(earlier['progress']['reported_count'], 3)


class ReviewTaskChatTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.serial = 0

    def open(self, name='memory.sqlite'):
        session = ControlledDialogueSession.open(str(Path(self.folder.name) / name), approved_root=self.folder.name,
            user_id='fictional-review-task', provider=None, background=False, timezone_name='UTC')
        self.addCleanup(session.close)
        return session

    def send(self, session, text, stamp):
        self.serial += 1
        return session.send_message(text, input_id=f'review-task-{self.serial}', as_of=stamp)

    def test_bad_polarity_and_questions_do_not_persist_opposite_axes(self):
        inputs = [t for t, _ in ReviewTaskRuleTests.BAD_POLARITY] + [
            'Have I paused thought record', '想法记录暂停行不行', '想法记录取消好不好']
        for index, text in enumerate(inputs):
            with self.subTest(text=text):
                name = f'bad-{index}.sqlite'; session = self.open(name)
                label = 'thought record' if 'thought record' in text else '想法记录'
                self.send(session, f'I agreed to do {label} 2 times.' if label.isascii() else '我同意做两次想法记录。', MON)
                before = deepcopy(session.task_states()[0])
                reply = self.send(session, text, WED)
                self.assertEqual(reply['source_commit']['source_status'], 'committed')
                session.close(); reopened = self.open(name)
                after = reopened.task_states()[0]
                for axis in ['acceptance', 'participation', 'outcome', 'state_version']:
                    self.assertEqual(after[axis], before[axis], axis)

    def test_second_then_unqualified_report_counts_once_after_reopen(self):
        session = self.open()
        self.send(session, '我同意做两次想法记录。', MON)
        self.send(session, '今天做了第二次想法记录。', WED)
        state = self.send(session, '周三那次做过一次想法记录。', FRI)['task_state']
        self.assertEqual(state['progress']['reported_count'], 1)
        self.assertEqual(state['progress']['completion'], 'partial')
        self.assertEqual(state['progress']['known_attempt_count'], 1)
        session.close()
        self.assertEqual(self.open().task_states()[0]['progress']['reported_count'], 1)

    def test_late_dated_cumulative_does_not_replace_current_snapshot(self):
        for prior, conflict in [('两', False), ('四', True)]:
            with self.subTest(prior=prior):
                name = f'total-{prior}.sqlite'; session = self.open(name)
                self.send(session, '我同意做三次想法记录。', MON)
                self.send(session, '想法记录累计做了三次。', THU)
                self.send(session, f'周三想法记录累计做了{prior}次。', FRI)
                session.close(); state = self.open(name).task_states()[0]
                self.assertEqual(state['progress']['reported_total'], 3)
                self.assertEqual(state['progress']['reported_count'], None if conflict else 3)
                self.assertEqual(state['progress']['count_conflict'], conflict)

    def test_correction_targets_latest_coverage_after_a_late_earlier_report(self):
        session = self.open()
        self.send(session, '我同意做四次想法记录。', MON)
        self.send(session, '想法记录累计做了三次。', THU)
        current_id = session._objects('task')[0]['payload']['events'][-1]['event_id']
        self.send(session, '周三想法记录累计做了两次。', FRI)
        result = self.send(session, '想法记录累计更正为四次。', '2026-11-07T10:00:00Z')['task_state']
        self.assertEqual(result['progress']['reported_count'], 4)
        events = session._objects('task')[0]['payload']['events']
        self.assertEqual(events[-1]['event_type'], 'corrected')
        self.assertEqual(events[-1]['payload']['target_event_id'], current_id)
        self.assertEqual(next(e for e in events if e['event_id'] == current_id)['payload']['count'], 3)

    def test_multi_ordinal_day_falls_back_without_merging_real_attempts(self):
        session = self.open()
        self.send(session, '我同意做三次想法记录。', MON)
        self.send(session, '今天做了第一次想法记录。', WED)
        self.send(session, '今天做了第二次想法记录。', '2026-11-04T16:00:00Z')
        before = session.task_states()[0]
        state = self.send(session, '周三那次做过一次想法记录。', FRI)['task_state']
        self.assertEqual(state['progress']['reported_count'], 2)
        self.assertEqual(state['progress']['attempt_ids'], before['progress']['attempt_ids'])
        self.assertEqual(state['state_version'], before['state_version'])


if __name__ == '__main__':
    unittest.main()
