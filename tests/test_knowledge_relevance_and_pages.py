from __future__ import annotations

import json
import hashlib
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/cbt-knowledge-rag'))
sys.path.insert(0, str(ROOT / 'tests'))
from cbt_rag_v1 import (CORPUS_CHUNKER_CONFIG, RELEVANCE_RULE_VERSION, Retriever,
                        build_corpus, evaluate, load_corpus, make_chunks, relevant)
from source_registry import KnowledgeConfig, SourcePolicyError, SourceRegistry
from test_source_registry import fixture_config, fixture_payload, raw_fixture_chunk


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    for lines in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        commands = ['BT /F1 10 Tf 20 750 Td']
        for line in lines:
            line = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            commands.extend([f'({line}) Tj', '0 -14 Td'])
        stream.set_data(('\n'.join(commands) + '\nET').encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    with path.open('wb') as handle:
        writer.write(handle)


class KnowledgeRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.chunk = raw_fixture_chunk()

    def test_only_actual_content_conditions_participate(self):
        for item, expected in [
            ({'expected_topics': ['anxiety']}, False),
            ({'gold_keywords': ['red pencil']}, False),
            ({'expected_topics': ['anxiety'], 'gold_keywords': ['red pencil']}, False),
            ({'expected_topics': ['anxiety'], 'gold_keywords': ['blue notebook']}, False),
            ({'expected_topics': ['anxiety'], 'gold_keywords': ['blue test notebook']}, True),
            ({'expected_sources': ['synthetic']}, True),
            ({'expected_sources': ['other']}, False),
            ({}, True),
        ]:
            with self.subTest(item=item):
                self.assertEqual(relevant(self.chunk, item), expected)
        topical = {**self.chunk, 'cbt_topics': ['anxiety']}
        self.assertTrue(relevant(topical, {'expected_topics': ['anxiety']}))
        self.assertTrue(relevant(topical, {'expected_topics': ['anxiety'], 'gold_keywords': ['missing']}))
        self.assertFalse(relevant(topical, {'expected_sources': ['other'], 'expected_topics': ['anxiety']}))

    def test_single_condition_evaluation_scores_nonmatch_zero_and_records_version(self):
        registry = SourceRegistry(fixture_payload())
        config = fixture_config(registry)
        chunks = [registry.stamp_chunk(self.chunk)]
        retriever = Retriever(chunks, download_models=False, registry=registry,
                              knowledge_config=config, manifest=config.corpus_manifest(chunks, registry))
        questions = [
            {'question_id': 'topic-only', 'category': 'fixture', 'query': 'blue test notebook', 'expected_topics': ['anxiety']},
            {'question_id': 'keyword-only', 'category': 'fixture', 'query': 'blue test notebook', 'gold_keywords': ['red pencil']},
        ]
        with tempfile.TemporaryDirectory() as directory, patch('cbt_rag_v1.RESULTS', Path(directory)):
            summary = evaluate(retriever, questions)
            for key in ('recall_at_5', 'recall_at_10', 'mrr_at_10'):
                self.assertTrue((summary[key] == 0).all())
            manifest = json.loads((Path(directory) / 'evaluation_run_manifest.json').read_text())
            self.assertEqual(manifest['relevance_rule_version'], RELEVANCE_RULE_VERSION)
            self.assertEqual(manifest['chunker_config'], CORPUS_CHUNKER_CONFIG)

    def test_old_page_algorithm_manifest_is_not_reused(self):
        registry = SourceRegistry(fixture_payload())
        config = fixture_config(registry)
        chunks = [registry.stamp_chunk(self.chunk)]
        old_config = KnowledgeConfig.from_payload(
            json.loads(config.normalized_config_json), registry,
            chunker_config={**CORPUS_CHUNKER_CONFIG, 'version': 'pdf-cleaning-v2-full-metadata'})
        manifest = old_config.corpus_manifest(chunks, registry)
        with self.assertRaisesRegex(SourcePolicyError, 'chunker'):
            Retriever(chunks, download_models=False, registry=registry, knowledge_config=old_config, manifest=manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunks.jsonl'
            path.write_text(json.dumps(chunks[0]) + '\n')
            path.with_name('corpus_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(SourcePolicyError, 'chunker'):
                load_corpus(path, registry=registry, knowledge_config=old_config)


class KnowledgePageProvenanceTests(unittest.TestCase):
    def chunks(self, pages, *, target_chars=1400, overlap_chars=180):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / 'fixture.pdf', pages)
            source = {**fixture_payload()['sources'][0], 'file': 'fixture.pdf'}
            with patch('cbt_rag_v1.ROOT', root):
                return make_chunks(source, target_chars=target_chars, overlap_chars=overlap_chars)

    def test_short_previous_page_tail_is_not_attributed_to_new_heading(self):
        chunks = self.chunks([
            ['onlypageonemarker is a short original note.'],
            ['Second Page Heading', 'the new page has its own material. ' * 18],
        ])
        first = next(c for c in chunks if 'onlypageonemarker' in c['text'])
        self.assertEqual((first['page_start'], first['page_end']), (1, 1))
        self.assertEqual(first['section'], 'Front matter')
        self.assertIn('pp. 1-1', first['citation'])
        later = [c for c in chunks if c['section'] == 'Second Page Heading']
        self.assertTrue(later)
        self.assertTrue(all(c['page_start'] == c['page_end'] == 2 for c in later))
        self.assertTrue(all('onlypageonemarker' not in c['text'] for c in later))

    def test_same_page_heading_preserves_short_prior_section(self):
        chunks = self.chunks([['First Section Heading', 'shortpriorsectionmarker is here.',
                               'Second Section Heading', 'the second material remains separate. ' * 12]])
        first = next(c for c in chunks if 'shortpriorsectionmarker' in c['text'])
        self.assertEqual(first['section'], 'First Section Heading')
        self.assertEqual((first['page_start'], first['page_end']), (1, 1))
        self.assertFalse(any('shortpriorsectionmarker' in c['text'] and c['section'] == 'Second Section Heading' for c in chunks))

    def test_cross_page_overlap_retains_all_actual_page_origins(self):
        chunks = self.chunks([
            ['intro ' + 'word ' * 20 + 'onlypageonemarker ' + 'tail ' * 5],
            ['onlypagetwomarker ' + 'more ' * 24],
            ['onlypagethreemarker ' + 'last ' * 25],
        ], target_chars=140, overlap_chars=80)
        crossing = [c for c in chunks if 'onlypageonemarker' in c['text'] and 'onlypagetwomarker' in c['text']]
        self.assertTrue(crossing)
        self.assertTrue(all(c['page_start'] <= 1 and c['page_end'] >= 2 for c in crossing))
        for chunk in chunks:
            for marker, page in [('onlypageonemarker', 1), ('onlypagetwomarker', 2), ('onlypagethreemarker', 3)]:
                if marker in chunk['text']:
                    self.assertLessEqual(chunk['page_start'], page)
                    self.assertGreaterEqual(chunk['page_end'], page)
        self.assertTrue(any(c['page_start'] == 2 and c['page_end'] == 3 for c in chunks))

    def test_approved_build_and_evidence_card_keep_short_tail_on_page_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pdf(root / 'fixture.pdf', [
                ['onlypageonemarker is original fixture content.'],
                ['Second Page Heading', 'separate second-page material. ' * 15],
            ])
            payload = fixture_payload()
            payload['sources'][0]['content_sha256'] = hashlib.sha256((root / 'fixture.pdf').read_bytes()).hexdigest()
            registry = SourceRegistry(payload)
            config = fixture_config(registry)
            data = root / 'data'
            data.mkdir()
            with patch('cbt_rag_v1.ROOT', root), patch('cbt_rag_v1.DATA', data):
                built = build_corpus(root / 'unused.yaml', registry=registry, knowledge_config=config)
                loaded = load_corpus(data / 'chunks.jsonl', registry=registry, knowledge_config=config)
                retriever = Retriever(loaded, download_models=False, registry=registry,
                                      knowledge_config=config, manifest=config.corpus_manifest(loaded, registry))
                index = next(i for i, chunk in enumerate(built) if 'onlypageonemarker' in chunk['text'])
                card = retriever.evidence_cards([index])[0]
            self.assertEqual((card['page_start'], card['page_end']), (1, 1))
            self.assertEqual(card['section'], 'Front matter')
            self.assertIn('pp. 1-1', card['citation'])

    def test_near_target_overlap_still_advances_at_short_sentence_cut(self):
        # Run the formerly hanging input in a subprocess so a regression has a
        # deadline rather than hanging the whole test suite.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pdf'
            write_pdf(path, [['lowercase ' + 'word ' * 14 + 'sentence. ' + 'remaining ' * 18 + 'finaltailmarker.']])
            script = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from cbt_rag_v1 import make_chunks
source={'id':'synthetic','title':'Fixture','file':sys.argv[2],
        'role':'test_only','access':'project_authored','coverage':'fixture_only'}
chunks=make_chunks(source,target_chars=120,overlap_chars=110)
print(json.dumps({'count':len(chunks),'final_tail':any('finaltailmarker' in x['text'] for x in chunks),
                  'pages_valid':all(x['page_start']==x['page_end']==1 for x in chunks)}))
"""
            result = subprocess.run([sys.executable, '-c', script,
                                     str(ROOT / 'modules/cbt-knowledge-rag'), str(path)],
                                    capture_output=True, text=True, timeout=15, check=True)
            report = json.loads(result.stdout)
            self.assertGreater(report['count'], 0)
            self.assertTrue(report['final_tail'])
            self.assertTrue(report['pages_valid'])

    def test_invalid_overlap_cannot_loop_forever(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.chunks([['small text.']], target_chars=20, overlap_chars=20)


if __name__ == '__main__':
    unittest.main()
