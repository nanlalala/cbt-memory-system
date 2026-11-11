"""Invented source-ranking fixtures; no corpus, model, gold or network calls."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
import controlled_dialogue as product
from source_candidate_selection import select_source_supplements, VERSION
from evidence_bundles import build_outbound_messages
from stable_retrieval import SourceSpanHit

STAMP = '2026-11-01T08:00:00Z'
AFTER = '2026-11-01T09:00:00Z'
PERMISSIONS = {key: True for key in ('save', 'long_term_use', 'response_external', 'extraction_external')}
TEXTS = {'A': 'I feel anxious before speaking at work.',
         'B': 'My goal is to ask one question in the meeting.',
         'C': 'The last thought record felt overwhelming.',
         'D': 'I prefer to practise in a smaller group.',
         'Z': 'I want to start with a smaller meeting.'}


def source(letter):
    return {'source_id': 's-' + letter, 'user_id': 'u', 'role': 'user',
            'text': TEXTS[letter], 'source_time': STAMP, 'input_id': letter,
            'permissions': dict(PERMISSIONS)}


def hit(ident, ids, *, pointer=None, **changes):
    fields = dict(memory_id=ident, user_id='u', source_role='user', lifecycle_status='candidate',
                  sensitivity='standard', permissions=dict(PERMISSIONS),
                  source_evidence=SimpleNamespace(turn_ids=ids,
                      supporting_text=pointer if pointer is not None else TEXTS[ids[0][-1]]))
    fields.update(changes)
    return SimpleNamespace(record=SimpleNamespace(**fields), score=0.8)


def raw_hit(s, score=1):
    return SourceSpanHit(s['source_id'], s['source_id'] + ':full', 0, len(s['text']),
        s['text'], s['source_time'], 'user', None, score, {'fixture': score})


class SourceCandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sources = {'s-' + k: source(k) for k in TEXTS}

    def select(self, hits, **kwargs):
        return select_source_supplements(hits, sources=self.sources, user_id='u', **kwargs)

    def test_top_three_duplicate_raw_candidates_do_not_starve_three_new_sources(self):
        hits = [hit('m1', ['s-A']), hit('m2', ['s-A']), hit('m3', ['s-A']),
                hit('m4', ['s-B']), hit('m5', ['s-C']), hit('m6', ['s-D'])]
        # The former top-three-then-deduplicate path has zero supplemental sources.
        former = {h.record.source_evidence.turn_ids[0] for h in hits[:3]} - {'s-A'}
        self.assertEqual(former, set())
        chosen, trace = self.select(hits, already_present_source_ids={'s-A'})
        self.assertEqual([x.span.source_id for x in chosen], ['s-B', 's-C', 's-D'])
        self.assertEqual([x.candidate_rank for x in chosen], [4, 5, 6])
        self.assertEqual(trace['selected_source_count'], 3)

    def test_duplicate_new_source_uses_one_source_slot(self):
        chosen, _ = self.select([hit('m1', ['s-A']), hit('m2', ['s-A']), hit('m3', ['s-B'])],
                                max_new_sources=2)
        self.assertEqual([x.span.source_id for x in chosen], ['s-A', 's-B'])

    def test_invalid_pointer_does_not_consume_source_slot(self):
        chosen, trace = self.select([hit('m1', ['s-A'], pointer='unsupported partial claim'),
                                    hit('m2', ['s-B'])], max_new_sources=1)
        self.assertEqual([x.span.source_id for x in chosen], ['s-B'])
        self.assertEqual(trace['decisions'][0]['reason'], 'no_new_exact_source_locator')

    def test_returns_complete_qualifying_source_not_index_text_or_substring(self):
        self.sources['s-A']['text'] = 'I have not agreed to continue the exercise.'
        chosen, trace = self.select([hit('m1', ['s-A'], pointer='agreed to continue')])
        self.assertEqual(chosen[0].span.text, self.sources['s-A']['text'])
        self.assertFalse(trace['candidate_index_text_is_response_evidence'])

    def test_all_multi_source_dependencies_must_be_currently_permitted(self):
        self.sources['s-B']['permissions']['response_external'] = False
        chosen, trace = self.select([hit('m1', ['s-A', 's-B']), hit('m2', ['s-C'])])
        self.assertEqual([x.span.source_id for x in chosen], ['s-C'])
        self.assertEqual(trace['decisions'][0]['reason'], 'unavailable_source_dependency')

    def test_multi_source_dependency_is_retained_when_only_one_locator_resolves(self):
        chosen, _ = self.select([hit('m1', ['s-A', 's-B'])])
        self.assertEqual([x.span.source_id for x in chosen], ['s-A'])
        self.assertEqual(chosen[0].candidate_source_dependencies, ('s-A', 's-B'))

    def test_candidate_permission_or_assistant_source_cannot_qualify(self):
        denied = dict(PERMISSIONS, response_external=False)
        self.sources['s-B']['role'] = 'assistant'
        chosen, trace = self.select([hit('m1', ['s-A'], permissions=denied), hit('m2', ['s-B'])])
        self.assertEqual(chosen, ())
        self.assertEqual([d['reason'] for d in trace['decisions']],
                         ['candidate_permission', 'unavailable_source_dependency'])

    def test_other_subject_ids_do_not_leak_into_trace(self):
        chosen, trace = self.select([hit('foreign-memory', ['private-foreign-source'], user_id='other',
                                       pointer='private')])
        self.assertEqual(chosen, ())
        self.assertNotIn('foreign-memory', str(trace))
        self.assertNotIn('private-foreign-source', str(trace))

    def test_inputs_are_unchanged_and_source_budget_does_not_become_record_budget(self):
        before = deepcopy(self.sources)
        chosen, trace = self.select([hit('m1', ['s-A']), hit('m2', ['s-B']), hit('m3', ['s-C'])],
                                    max_new_sources=1)
        self.assertEqual(len(chosen), 1)
        self.assertEqual(trace['decisions'][-1]['reason'], 'distinct_source_budget')
        self.assertEqual(self.sources, before)


class SourceCandidateLiveIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.session = product.ControlledDialogueSession.open(
            str(Path(self.folder.name) / 'fixture.sqlite'), approved_root=self.folder.name,
            user_id='u', run_id='new-source-selection-fixture', background=False)
        self.addCleanup(self.session.close)
        self.sources = {'s-' + k: source(k) for k in TEXTS}
        for s in self.sources.values():
            self.session.durable.submit_source(s, enqueue=False,
                run_id=self.session.run_id, session_id=self.session.session_id)

    def records(self, bindings):
        objects = []
        for i, ids in enumerate(bindings):
            record = product._source_record(self.sources[ids[0]], self.session.session_id,
                memory_id='m-' + str(i).zfill(2), content='work')
            record.source_evidence.turn_ids = list(ids)
            objects.append({'status': 'available', 'payload': {'kind': 'candidate', 'record': record.to_dict()}})
        return objects

    def search_patch(self, letters):
        return patch.object(product, 'StableSourceRetriever', return_value=SimpleNamespace(
            search=lambda *a, **kw: SimpleNamespace(hits=tuple(raw_hit(self.sources['s-' + k])
                                                               for k in letters), trace={})))

    def test_live_bundles_reaches_later_distinct_sources_and_records_new_protocol(self):
        records = self.records([['s-A'], ['s-A'], ['s-A'], ['s-B'], ['s-C'], ['s-D']])
        with patch.object(self.session, '_objects', return_value=records), self.search_patch(['A']):
            bundles, trace, _ = self.session._bundles('work', as_of=AFTER)
        self.assertEqual([b.claims[0].spans[0].source_id for b in bundles], ['s-A', 's-B', 's-C', 's-D'])
        self.assertEqual([b.claims[0].priority for b in bundles], [0, -1, -2, -3])
        self.assertEqual(trace['candidate_source_selection']['ranked_candidate_count'], 6)
        self.assertEqual(self.session.protocol['candidate_source_selection'], VERSION)

    def test_budget_keeps_best_original_rank_not_lexically_smallest_source_id(self):
        with patch.object(self.session, '_objects', return_value=[]), self.search_patch(['Z', 'A']):
            bundles, _, _ = self.session._bundles('work', as_of=AFTER)
        base = [{'role': 'system', 'content': 'fixture'}, {'role': 'user', 'content': 'work'}]
        one = build_outbound_messages(base, bundles[:1], max_outbound_chars=10000, max_response_tokens=32)
        limited = build_outbound_messages(base, bundles, max_outbound_chars=one.char_count, max_response_tokens=32)
        self.assertEqual(limited.source_ids, ('s-Z',))

    def test_live_send_fences_unquoted_multi_source_index_dependency(self):
        class Provider:
            messages = None
            def __call__(inner, *, messages, purpose):
                inner.messages = messages
                # A dependency that was not itself quoted changes after send.
                self.session.delete_content('source', 's-C')
                return product.ModelOutput('fixture response')
        provider = Provider(); self.session.provider = provider
        records = self.records([['s-B', 's-C']])
        def objects(kind=None):
            return records if kind == 'candidate' else []
        with patch.object(self.session, '_objects', side_effect=objects), self.search_patch([]), \
             patch.object(self.session, '_schedule_worker'), \
             patch.object(self.session.durable, 'begin_attempt', wraps=self.session.durable.begin_attempt) as admission:
            reply = self.session.send_message('work', input_id='current', as_of=AFTER)
        self.assertEqual(reply['answer_status'], 'stale_discarded')
        self.assertIn('s-C', admission.call_args.args[3])
        self.assertIn(TEXTS['B'], str(provider.messages))
        self.assertNotIn(TEXTS['C'], str(provider.messages))


if __name__ == '__main__':
    unittest.main()
