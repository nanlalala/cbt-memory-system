import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
sys.path.insert(0, str(ROOT / 'evaluation/cbt-memory-effect'))
from controlled_dialogue import ControlledDialogueSession
from auto_eval_v2.dated_task_readout import prepare_task_requests


class DatedTaskReadoutTest(unittest.TestCase):
    def test_real_projector_supplies_versioned_fields_instead_of_authored_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            session = ControlledDialogueSession.open(str(Path(folder) / 'replay.sqlite'),
                approved_root=folder, user_id='fictional', provider=None, background=False)
            self.addCleanup(session.close)
            ids = []
            for i, text in enumerate(('我同意每周做两次想法记录。','想法记录累计做了一次。','这周暂停')):
                reply = session.send_message(text, input_id=f'fictional-{i}',
                                             as_of=f'2026-11-0{3+i}T09:00:00Z')
                ids.append(reply['source_commit']['source_id'])
            args = dict(current_input='想法记录现在怎么安排？', as_of='2026-11-06T09:00:00Z',
                        selected_ids=ids, recent_ids=[], max_request_chars=9000,
                        request_options={'model':'fixture-model','max_tokens':64})
            with self.assertRaisesRegex(ValueError,'dated'):
                prepare_task_requests(session, clock_mode='ordinal-only-v1', **args)
            out = prepare_task_requests(session, clock_mode='dated-sources-v1', **args)
            blocks = {k:json.loads(v['request']['messages'][-2]['content'].split('\n',1)[1])
                      for k,v in out.items()}
            self.assertEqual(len(blocks['raw']['sources']), 3)
            fields = {m['summary'] for m in blocks['processed']['memories']}
            self.assertEqual(fields, {'engagement = "paused"', 'acceptance = "agreed"',
                                      'target_count = 2', 'reported_count = 1'})
            self.assertTrue(all('evidence' not in m for m in blocks['processed']['memories']))
            self.assertTrue(all(m['evidence'] for m in blocks['processed_with_evidence']['memories']))
            cut = prepare_task_requests(session, clock_mode='dated-sources-v1', **{**args,'selected_ids':ids[:2]})
            self.assertNotIn('paused',cut['processed']['request']['messages'][-2]['content'])
