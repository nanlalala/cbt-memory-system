"""Priority behaviour of the draft v5 readout. Fictional data only."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from cbt_context_v5_draft import CBT_POLICY, VERSION, ContextualItem, build_cbt_context
from evidence_bundles import EvidenceSpan


def registry():
    rows = [('s1', 'My homework used to be ten minutes a day.', 'user'),
            ('s2', 'Actually make it five minutes, ten is too much.', 'user'),
            ('s3', 'I like walking in the evening.', 'user')]
    return {sid: dict(user_id='fictional', source_id=sid, text=t, role=r, source_order=i, source_time=None,
                      permissions=dict(save=True, long_term_use=True, response_external=True))
            for i, (sid, t, r) in enumerate(rows)}


def make(sources, item_id, sid, section, origin, support, value):
    span = EvidenceSpan(sid, item_id, sources[sid]['text'], 0, len(sources[sid]['text']), None, 'user')
    return ContextualItem(item_id, section, value, (span,), origin=origin, support_status=support,
                          verifier_revision=None if support == 'hypothesis' else 'fixture-v1')


class PriorityTests(unittest.TestCase):
    def build(self, items, cap):
        s = registry()
        return build_cbt_context([{'role': 'system', 'content': CBT_POLICY},
                                  {'role': 'user', 'content': 'What should I do this week?'}],
                                 source_registry=s, subject_id='fictional', max_outbound_chars=cap,
                                 max_response_tokens=64, request_metadata={'model': 'fixture'},
                                 contextual_items=items)

    def items(self):
        s = registry()
        return [make(s, 'guide', 's3', 'turn_guidance', 'agent_inference', 'hypothesis', 'try a long walk'),
                make(s, 'fact', 's1', 'case_state', 'explicit_user_statement', 'source_supported', 'ten minutes'),
                make(s, 'fix', 's2', 'case_state', 'user_correction', 'source_supported', 'five minutes')]

    def tight_cap(self):
        base = len(self.build([], 99999).serialized_request)
        one = len(self.build([self.items()[2]], 99999).serialized_request)
        return one + (one - base) // 4        # room for the correction only

    def test_version_is_distinct_from_pinned_v4(self):
        self.assertEqual(VERSION, 'cbt-context-v5-draft')

    def test_correction_survives_when_only_one_item_fits(self):
        out = self.build(self.items(), self.tight_cap())
        self.assertEqual(out.trace['included_item_ids'], ['fix'])
        self.assertEqual(out.trace['pinned_item_ids'], ['fix'])

    def test_guidance_no_longer_outranks_user_fact(self):
        s = registry()
        pair = [make(s, 'guide', 's3', 'turn_guidance', 'agent_inference', 'hypothesis', 'try a long walk'),
                make(s, 'fact', 's1', 'case_state', 'explicit_user_statement', 'source_supported', 'ten minutes')]
        base = len(self.build([], 99999).serialized_request)
        one = len(self.build([pair[1]], 99999).serialized_request)
        out = self.build(pair, one + (one - base) // 4)
        self.assertEqual(out.trace['included_item_ids'], ['fact'])

    def test_everything_fits_when_budget_allows(self):
        out = self.build(self.items(), 99999)
        self.assertEqual(set(out.trace['included_item_ids']), {'guide', 'fact', 'fix'})
        self.assertEqual(out.trace['included_item_ids'][0], 'fix')


if __name__ == '__main__':
    unittest.main()
