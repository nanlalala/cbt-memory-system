"""Offline regressions for source-pointer and pinned-knowledge integrity.

All sources are authored fixtures. No model, external API or clinical data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from controlled_dialogue import ControlledDialogueSession, PrototypeLimits
from dialogue_service import ModelOutput
from evidence_bundles import EvidenceBundle, EvidenceClaim, build_outbound_messages, restore_source_quote
from knowledge_adapter_v2 import KnowledgeAdapter
from source_registry import KnowledgeConfig, SourceRegistry


class FixtureRetriever:
    def __init__(self, registry, chunks):
        self.registry = registry
        self.chunks = deepcopy(chunks)
        self.knowledge_config = KnowledgeConfig.from_payload(
            {'sources': [{'id': 'fixture'}]}, registry, chunker_config={'fixture': 'review'})
        self.manifest = self.knowledge_config.corpus_manifest(self.chunks, registry)
        self.corpus_fingerprint = self.manifest['corpus_fingerprint']
        self.after_rank = None
    def validate_corpus(self):
        self.knowledge_config.validate_manifest(self.chunks, self.manifest, self.registry)
    def rank(self, query, mode, k=10):
        if self.after_rank:
            self.after_rank()
        return list(range(min(k, len(self.chunks)))), 0.0


class KnowledgeSnapshotTests(unittest.TestCase):
    def fixture(self, *, pdf_path=False):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        folder = Path(directory.name)
        data = b'%PDF-1.4\nOriginal project-authored engineering fixture' if pdf_path else b'Original project-authored engineering fixture'
        role, coverage = ('reference', 'authored_mechanical_pdf_fixture') if pdf_path else ('test_only', 'fixture_only')
        path = folder / 'fixture.pdf'; path.write_bytes(data)
        registry = SourceRegistry({'schema_version': 1, 'registry_version': 'review-fixture', 'sources': [{
            'id': 'fixture', 'title': 'Review fixture', 'file': 'fixture.pdf', 'role': role,
            'access': 'project_authored', 'coverage': coverage, 'source_url': 'https://example.invalid/fixture',
            'version': '1', 'content_sha256': hashlib.sha256(data).hexdigest(),
            'retrieval_enabled': True, 'page_count': 1,
            'license': {'id': 'fixture', 'status': 'project_authored',
                'evidence_url': 'https://example.invalid/review', 'reviewed_on': '2026-11-08',
                'approved_by': 'fixture only', 'approval_reference': 'original authored fixture',
                'approved_uses': ['retrieval', 'external_model_transmission'], 'usage_limits': ['Test only']}}]})
        chunk = registry.stamp_chunk({'chunk_id': 'fixture:0', 'source_id': 'fixture', 'title': 'Review fixture',
            'section': 'Authored fixture', 'page_start': 1, 'page_end': 1, 'role': role,
            'access': 'project_authored', 'coverage': coverage, 'cbt_topics': ['homework_review'],
            'risk_level': 'low', 'text': 'Review homework practice and barriers.', 'citation': 'Fixture, p. 1'})
        retriever = FixtureRetriever(registry, [chunk])
        adapter = KnowledgeAdapter(retriever, registry=registry, source_root=folder, mode='K1', fixture_only=not pdf_path)
        return adapter, retriever, path

    def test_source_replaced_during_rank_yields_empty_same_request(self):
        adapter, retriever, path = self.fixture()
        retriever.after_rank = lambda: path.write_bytes(b'Replaced while ranking')
        selection = adapter.retrieve('homework practice', task_kind='homework_review')
        self.assertEqual(selection.status, 'unavailable')
        self.assertFalse(selection.has_support)
        self.assertEqual(selection.items, ())

    def test_source_deleted_during_rank_yields_empty_same_request(self):
        adapter, retriever, path = self.fixture()
        retriever.after_rank = path.unlink
        selection = adapter.retrieve('homework practice', task_kind='homework_review')
        self.assertEqual(selection.status, 'unavailable')
        self.assertEqual(selection.items, ())

    def test_replaced_source_cannot_enter_actual_response_messages(self):
        adapter, retriever, path = self.fixture()
        retriever.after_rank = lambda: path.write_bytes(b'Replaced while ranking')
        with tempfile.TemporaryDirectory() as folder:
            provider = RecordingProvider()
            session = ControlledDialogueSession.open(Path(folder)/'memory.sqlite', approved_root=folder,
                user_id='fictional-review', provider=provider, background=False, knowledge=adapter)
            self.addCleanup(session.close)
            result = session.send_message('Review CBT homework practice', input_id='knowledge-query',
                                          as_of='2026-11-04T10:00:00Z')
            self.assertEqual(result['answer_status'], 'succeeded')
            self.assertEqual(result['knowledge_status'], 'unavailable')
            self.assertEqual(result['knowledge_ids'], [])
            self.assertNotIn('fixture:0', json.dumps(provider.calls[-1], ensure_ascii=False))
            self.assertNotIn('Review homework practice and barriers.', json.dumps(provider.calls[-1]))

    def test_unchanged_pinned_fixture_still_provides_labelled_support(self):
        adapter, _, path = self.fixture()
        selection = adapter.retrieve('homework practice', task_kind='homework_review')
        self.assertTrue(selection.has_support)
        self.assertEqual(selection.items[0]['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(selection.items[0]['evidence_status'], 'fixture_not_real_k1')

    def test_source_update_during_card_construction_also_fails_closed(self):
        adapter, _, path = self.fixture()
        original = adapter.registry.evidence_card
        def changed_after_card(*args, **kwargs):
            card = original(*args, **kwargs)
            path.write_bytes(b'Updated while evidence packet was being built')
            return card
        adapter.registry.evidence_card = changed_after_card
        selection = adapter.retrieve('homework practice', task_kind='homework_review')
        self.assertEqual(selection.status, 'unavailable')
        self.assertEqual(selection.items, ())

    def test_pdf_parser_uses_the_same_frozen_bytes_as_source_hash(self):
        # The parser is a local mock. This does not validate a real CBT source.
        adapter, _, path = self.fixture(pdf_path=True)
        original_bytes = path.read_bytes(); parsed = []
        class OfflinePDFReader:
            def __init__(self, stream):
                parsed.append(stream.read())
                path.write_bytes(b'%PDF-1.4\nA different file after parsing began')
                self.pages = [object()]
        with patch.dict(sys.modules, {'pypdf': SimpleNamespace(PdfReader=OfflinePDFReader)}):
            selection = adapter.retrieve('homework practice', task_kind='homework_review')
        self.assertEqual(parsed, [original_bytes])
        self.assertEqual(selection.status, 'unavailable')
        self.assertEqual(selection.items, ())


class RecordingProvider:
    def __init__(self, source=None, pointer=None):
        self.calls = []; self.source = source; self.pointer = pointer
    def __call__(self, *, messages, purpose):
        self.calls.append({'messages': deepcopy(messages), 'purpose': purpose})
        if purpose == 'extraction':
            turn = json.loads(messages[-1]['content'])['turns'][0]
            return ModelOutput(json.dumps({'candidates': [{'memory_type': 'explicit_fact',
                'content': 'routine commitment', 'supporting_text': self.pointer or turn['text'],
                'source_turn_ids': [turn['turn_id']]}]}))
        return ModelOutput('Offline authored reply')


class SemanticPointerIntegrationTests(unittest.TestCase):
    def test_semantic_only_hit_keeps_negation_attribution_and_qualifiers(self):
        examples = [
            ('我没答应每天做想法记录。', '答应每天做想法记录'),
            ('老师说他答应每天做想法记录，不是我。', '答应每天做想法记录'),
            ('如果时间允许，我才会做想法记录。', '我才会做想法记录'),
            ('我只答应一周做两次想法记录，并非每天。', '做两次想法记录'),
        ]
        for index, (source, pointer) in enumerate(examples):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as folder:
                provider = RecordingProvider(source, pointer)
                session = ControlledDialogueSession.open(Path(folder)/'memory.sqlite', approved_root=folder,
                    user_id='fictional-review', provider=provider, background=False)
                try:
                    session.send_message(source, input_id='source', as_of='2026-11-04T09:00:00Z')
                    work = session.run_pending()
                    self.assertEqual(work[-1]['status'], 'completed')
                    result = session.send_message('routine commitment', input_id='query', as_of='2026-11-04T10:00:00Z')
                    self.assertEqual(result['answer_status'], 'succeeded')
                    self.assertEqual(result['retrieval_trace']['selected_span_count'], 0)
                    self.assertIn('source:source', result['retrieval_trace']['semantic_selected_source_ids'])
                    context = next(m['content'] for m in provider.calls[-1]['messages']
                                   if m['content'].startswith('SOURCED MEMORY'))
                    claims = json.loads(context.split('\n',1)[1])['claims']
                    quote = next(c for c in claims if c['field'] == 'history_quote')
                    self.assertEqual(quote['value'], source)
                    self.assertEqual(quote['evidence'][0]['text'], source)
                    self.assertEqual((quote['evidence'][0]['start'], quote['evidence'][0]['end']), (0,len(source)))
                    self.assertTrue(all(record.confirmation_state == 'unconfirmed'
                                        for record in session.store.list_for_user(session.user_id)))
                finally:
                    session.close()

    def test_common_raw_and_semantic_hit_keeps_one_complete_original_quote(self):
        source = '我没答应每天做想法记录。'; pointer = '答应每天做想法记录'
        with tempfile.TemporaryDirectory() as folder:
            provider = RecordingProvider(source, pointer)
            session = ControlledDialogueSession.open(Path(folder)/'memory.sqlite', approved_root=folder,
                user_id='fictional-review', provider=provider, background=False)
            try:
                session.send_message(source, input_id='source', as_of='2026-11-04T09:00:00Z')
                session.run_pending()
                result = session.send_message('想法记录 routine commitment', input_id='query',
                                              as_of='2026-11-04T10:00:00Z')
                self.assertEqual(result['answer_status'], 'succeeded')
                self.assertEqual(result['retrieval_trace']['unique_source_count'], 1)
                context = next(m['content'] for m in provider.calls[-1]['messages']
                               if m['content'].startswith('SOURCED MEMORY'))
                quotes = [c for c in json.loads(context.split('\n',1)[1])['claims']
                          if c['field'] == 'history_quote']
                self.assertEqual([c['value'] for c in quotes], [source])
            finally:
                session.close()

    def test_tight_actual_request_drops_whole_quote_instead_of_positive_substring(self):
        source = '虚构背景' * 300 + '。我没答应每天做想法记录。'; pointer = '答应每天做想法记录'
        with tempfile.TemporaryDirectory() as folder:
            provider = RecordingProvider(source, pointer)
            session = ControlledDialogueSession.open(Path(folder)/'memory.sqlite', approved_root=folder,
                user_id='fictional-review', provider=provider, background=False,
                limits=PrototypeLimits(max_outbound_chars=6000))
            try:
                first = session.send_message(source, input_id='source', as_of='2026-11-04T09:00:00Z')
                self.assertEqual(first['source_commit']['source_status'], 'committed')
                session.run_pending()
                result = session.send_message('routine commitment', input_id='query',
                                              as_of='2026-11-04T10:00:00Z')
                self.assertEqual(result['answer_status'], 'succeeded')
                self.assertIn('complete_claim_exceeds_outbound_budget', result['drop_reasons'].values())
                self.assertEqual(result['memory_ids'], [])
                context = json.dumps(provider.calls[-1]['messages'], ensure_ascii=False)
                self.assertNotIn(pointer, context)
                self.assertNotIn('"history_quote"', context)
            finally:
                session.close()


class SourcePointerBoundaryTests(unittest.TestCase):
    def source(self):
        return {'source_id': 's', 'text': '老师说他没答应每天做想法记录，而我还没决定。',
                'source_time': '2026-11-04T09:00:00Z', 'role': 'user'}

    def test_locator_restores_whole_source_with_exact_unicode_offsets(self):
        source = self.source()
        span = restore_source_quote(source, pointer_text='答应每天做想法记录', span_id='semantic:s')
        self.assertEqual(span.text, source['text'])
        self.assertEqual((span.start, span.end), (0,len(source['text'])))

    def test_missing_empty_or_fabricated_locator_has_no_evidence(self):
        for pointer in [None, '', '我每天都做', '另一个人的来源']:
            with self.subTest(pointer=pointer):
                self.assertIsNone(restore_source_quote(self.source(), pointer_text=pointer, span_id='semantic:s'))

    def test_restored_quote_is_atomic_under_escaped_request_budget(self):
        source = self.source();source['text'] = '虚构背景'*100 + source['text']
        span = restore_source_quote(source, pointer_text='答应每天做想法记录', span_id='semantic:s')
        quote = EvidenceClaim('semantic:s', 'history_quote', span.text, (span,), support_status='source_quote')
        package = build_outbound_messages([{'role':'user','content':'routine commitment'}],
            [EvidenceBundle('semantic:s',(quote,))],max_outbound_chars=1000)
        self.assertEqual(package.included_claim_ids, ())
        self.assertEqual(package.dropped_claims['semantic:s'], 'complete_claim_exceeds_outbound_budget')
        self.assertNotIn('答应每天做想法记录', json.dumps(package.messages,ensure_ascii=False))


if __name__ == '__main__':
    unittest.main()
