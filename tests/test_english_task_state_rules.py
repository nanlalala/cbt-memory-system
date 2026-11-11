"""Invented local grammar fixtures; no model, benchmark data, or API calls."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from controlled_dialogue import ControlledDialogueSession
from task_projector import supporting_spans_for_event


class EnglishTaskStateRulesTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.session = ControlledDialogueSession.open(
            str(Path(self.folder.name) / 'memory.sqlite'), approved_root=self.folder.name,
            user_id='fictional-user', background=False)
        self.addCleanup(self.session.close)
        self.index = 0

    def send(self, text):
        i = self.index
        self.index += 1
        return self.session.send_message(text, input_id=f'en-{i}',
            as_of=f'2026-11-03T09:{i:02d}:00Z')

    def agree(self):
        state = self.send('I agreed to do thought record 7 times.')['task_state']
        self.assertEqual(state['progress']['target_count'], 7)

    def test_explicit_english_corrections_replace_current_target_and_keep_history(self):
        self.agree()
        for count in (2, 3, 4):
            state = self.send(f'The thought record target is corrected to {count} times.')['task_state']
            self.assertEqual(state['progress']['target_count'], count)
            self.assertEqual(state['acceptance'], 'agreed')
        events = self.session._objects('task')[0]['payload']['events']
        self.assertEqual(events[0]['payload']['target_count'], 7)
        self.assertEqual(len([e for e in events if e['event_type'] == 'corrected']), 3)

    def test_english_comma_cumulative_repeats_are_not_added(self):
        self.agree()
        for count in (1, 1, 2, 2):
            state = self.send(f'I completed thought record, cumulative total {count} times.')['task_state']
            self.assertEqual(state['progress']['reported_count'], count)
            self.assertEqual(state['progress']['target_count'], 7)

    def test_cumulative_witness_retains_completion_subject_and_literal_total(self):
        self.agree()
        original = 'I completed thought record, cumulative total 2 times.'
        self.send(original)
        payload = self.session._objects('task')[0]['payload']
        event = next(e for e in payload['events'] if e['event_type'] == 'progress_report')
        sources = {s['source_id']: s for s in self.session.durable.list_sources(self.session.user_id)}
        spans = supporting_spans_for_event(event, sources, context_objects=payload['context_objects'])
        witness = ' '.join(sources[s['source_id']]['text'][s['start']:s['end']] for s in spans)
        self.assertIn('I completed thought record, cumulative total 2 times', witness)

    def test_cumulative_correction_is_not_also_new_progress_and_replay_is_once(self):
        self.agree()
        self.send('I completed thought record, cumulative total 3 times.')
        text = 'The thought record cumulative total is corrected to 2 times.'
        state = self.send(text)['task_state']
        self.assertEqual(state['progress']['reported_count'], 2)
        replay = self.session.send_message(text, input_id='en-2', as_of='2026-11-03T09:02:00Z')
        self.assertTrue(replay['replayed'])
        events = self.session._objects('task')[0]['payload']['events']
        self.assertEqual([e['event_type'] for e in events], ['agreed', 'progress_report', 'corrected'])
        # A later identical total correction may have its own source, but must
        # still neither add counts nor create a second ordinary progress report.
        self.assertEqual(self.send(text)['task_state']['progress']['reported_count'], 2)
        events = self.session._objects('task')[0]['payload']['events']
        self.assertEqual(len([e for e in events if e['event_type'] == 'progress_report']), 1)

    def test_negative_hypothetical_future_or_third_party_totals_do_not_update(self):
        self.agree()
        texts = [
            'I did not complete thought record, cumulative total 2 times.',
            'If I completed thought record, cumulative total 2 times.',
            'I will have completed thought record, cumulative total 2 times.',
            'I plan to complete thought record, cumulative total 2 times.',
            'She completed thought record, cumulative total 2 times.',
            'I completed walking, cumulative total 2 times.',
            'I completed thought record; cumulative total 2 times.',
            'I completed thought record, cumulative total 2 times?',
        ]
        for text in texts:
            with self.subTest(text=text):
                self.send(text)
                self.assertIsNone(self.session.task_states()[0]['progress']['reported_count'])

    def test_english_negated_or_hypothetical_corrections_do_not_replace_target(self):
        self.agree()
        for text in [
            'The thought record target is not corrected to 2 times.',
            'If the thought record target is corrected to 2 times.',
            'The thought record target might be corrected to 2 times.',
            'The thought record target will be corrected to 2 times.',
            'I plan to have the thought record target corrected to 2 times.',
            'The thought record target is corrected to 2 times?',
        ]:
            with self.subTest(text=text):
                self.send(text)
                self.assertEqual(self.session.task_states()[0]['progress']['target_count'], 7)

    def test_plan_and_try_do_not_become_agreed_target(self):
        state = self.send('I plan to try thought record 2 times.')['task_state']
        self.assertNotEqual(state['acceptance'], 'agreed')


if __name__ == '__main__':
    unittest.main()
