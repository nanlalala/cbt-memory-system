"""Fictional date provenance at the final memory readout boundary; no model I/O."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from cbt_context import ContextualItem, build_cbt_context
from evidence_bundles import EvidenceBundle, EvidenceClaim, EvidenceSpan


SOURCE_TIME = '2026-11-01T09:00:00Z'
UNBOUND_TIME = '2099-01-01T00:00:00Z'


class SourceTimeReadoutTests(unittest.TestCase):
    def assemble(self, *, source_time=SOURCE_TIME, span_time=SOURCE_TIME,
                 representation='processed_with_evidence', alternatives=False):
        text = 'Fictional report: one practice completed.'
        sources = {'source': dict(user_id='fictional', source_id='source', role='user',
            text=text, source_order=7, source_time=source_time,
            permissions=dict(save=True, long_term_use=True, response_external=True))}
        before = deepcopy(sources)
        span = EvidenceSpan('source', 'fictional-span', text, source_time=span_time)
        evidence_options = ((span,),)
        if alternatives:
            actual = EvidenceSpan('source', 'source-bound-alternative', text,
                                  source_time=source_time)
            evidence_options += ((actual,),)
        claim = EvidenceClaim('dated-claim', 'reported_count', 1,
            evidence_options=evidence_options, support_status='rule_verified')
        item = ContextualItem('dated-projection', 'case_state', {'reported_count': 1},
            (span,), support_status='rule_verified', verifier_revision='fictional-verifier-v1')
        result = build_cbt_context(
            [{'role': 'system', 'content': 'Fictional test policy.'},
             {'role': 'user', 'content': 'Recall the supplied report.'}],
            [EvidenceBundle('fictional-bundle', (claim,))], source_registry=sources,
            subject_id='fictional', max_outbound_chars=10000,
            representation=representation, contextual_items=[item])
        self.assertEqual(sources, before)
        data = json.loads(result.messages[-2]['content'].split('\n', 1)[1])
        return result, data

    def test_unbound_date_cannot_enter_claim_or_processed_projection(self):
        for representation in ('processed', 'processed_with_evidence'):
            with self.subTest(representation=representation):
                result, data = self.assemble(span_time=UNBOUND_TIME,
                                             representation=representation)
                self.assertEqual(result.included_claim_ids, ())
                self.assertEqual(set(result.dropped_claims), {'dated-claim', 'dated-projection'})
                self.assertEqual(data['claims'], [])
                self.assertNotIn(UNBOUND_TIME, result.serialized_request)

    def test_ordinal_only_source_does_not_acquire_a_calendar_date(self):
        result, _ = self.assemble(source_time=None, span_time=UNBOUND_TIME)
        self.assertEqual(result.included_claim_ids, ())
        self.assertNotIn(UNBOUND_TIME, result.serialized_request)
        result, data = self.assemble(source_time=None, span_time=None)
        self.assertEqual(set(result.included_claim_ids), {'dated-claim', 'dated-projection'})
        self.assertIsNone(data['claims'][0]['evidence'][0]['source_time'])
        self.assertIsNone(data['case_state'][0]['evidence'][0]['source_time'])
        self.assertNotIn(SOURCE_TIME, result.serialized_request)

    def test_matching_date_is_retained_and_an_omitted_date_stays_unknown(self):
        result, data = self.assemble()
        self.assertEqual(set(result.included_claim_ids), {'dated-claim', 'dated-projection'})
        self.assertEqual(data['claims'][0]['evidence'][0]['source_time'], SOURCE_TIME)
        self.assertEqual(data['case_state'][0]['evidence'][0]['source_time'], SOURCE_TIME)
        result, data = self.assemble(span_time=None)
        self.assertEqual(set(result.included_claim_ids), {'dated-claim', 'dated-projection'})
        self.assertIsNone(data['claims'][0]['evidence'][0]['source_time'])
        self.assertIsNone(data['case_state'][0]['evidence'][0]['source_time'])
        self.assertNotIn(SOURCE_TIME, result.serialized_request)

    def test_valid_alternative_survives_rejection_of_a_wrong_date(self):
        result, data = self.assemble(span_time=UNBOUND_TIME, alternatives=True)
        self.assertEqual(result.included_claim_ids, ('dated-claim',))
        self.assertEqual(data['claims'][0]['evidence'][0]['span_id'], 'source-bound-alternative')
        self.assertEqual(data['claims'][0]['evidence'][0]['source_time'], SOURCE_TIME)
        self.assertIn('dated-projection', result.dropped_claims)
        self.assertNotIn(UNBOUND_TIME, result.serialized_request)


if __name__ == '__main__':
    unittest.main()
