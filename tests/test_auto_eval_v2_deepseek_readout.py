import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect"))
from deepseek_context_mechanism import ALIAS, MODEL, correct, digest, plan


class DeepSeekReplicationTest(unittest.TestCase):
    def test_complete_paired_wire_grid(self):
        p = plan()
        self.assertEqual((p['model'], p['required_returned_model']), (MODEL, ALIAS))
        self.assertEqual(len(set(r['wire_sha256'] for r in p['rows'])), 9)
        for row in p['rows']:
            encoded = json.dumps(row['request'])
            self.assertEqual(row['wire_sha256'], digest(encoded))
            self.assertEqual(row['wire_chars'], len(encoded))
            self.assertLessEqual(len(encoded), 6000)
            self.assertEqual(row['request']['thinking'], {'type':'disabled'})

    def test_strict_types_and_unevaluated_extra_content(self):
        self.assertTrue(correct('{"answer":2}', 2))
        self.assertFalse(correct('{"answer":1}', True))
        self.assertFalse(correct('{"answer":true,"answer":false}', False))
        self.assertFalse(correct('{"answer":false,"why":"unknown"}', False))
