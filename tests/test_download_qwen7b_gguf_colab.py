"""Tiny header/path/catalog fixtures; no downloads, real tokenizer or model."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'evaluation/cbt-memory-effect/download_qwen7b_gguf_colab.py'
spec = importlib.util.spec_from_file_location('qwen7b_download_test_fixture', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadContractTests(unittest.TestCase):
    def test_exact_partial_headers_accept_only_requested_offset_and_file_total(self):
        checked = module.validate_headers(206, {'Content-Range': 'bytes 7-13/100',
            'Content-Length': '7'}, start=7, end=13, total=100)
        self.assertEqual(checked['content_length'], 7)
        for status, headers in [
            (200, {'Content-Range': 'bytes 7-13/100', 'Content-Length': '7'}),
            (206, {'Content-Range': 'bytes 0-6/100', 'Content-Length': '7'}),
            (206, {'Content-Range': 'bytes 7-13/101', 'Content-Length': '7'}),
            (206, {'Content-Range': 'bytes 7-13/*', 'Content-Length': '7'}),
            (206, {'Content-Range': 'bytes 7-13/100', 'Content-Length': '8'}),
            (206, {'Content-Range': 'bytes 7-13/100', 'Content-Length': '7', 'Content-Encoding': 'gzip'}),
        ]:
            with self.subTest(status=status, headers=headers), self.assertRaises(ValueError):
                module.validate_headers(status, headers, start=7, end=13, total=100)

    def test_contiguous_range_plan_has_no_gap_overlap_or_past_end(self):
        for size in (1, module.BLOCK_BYTES, module.BLOCK_BYTES + 3, *[x['bytes'] for x in module.FILES]):
            plan = module.ranges(size)
            self.assertEqual(plan[0][0], 0)
            self.assertEqual(plan[-1][1], size - 1)
            self.assertEqual(sum(end - start + 1 for start, end in plan), size)
            self.assertTrue(all(b[0] == a[1] + 1 for a, b in zip(plan, plan[1:])))
            self.assertTrue(all(0 < end - start + 1 <= module.BLOCK_BYTES for start, end in plan))

    def test_filename_and_initial_url_are_fixed_not_arbitrary_paths(self):
        for item in module.FILES:
            url = module.fixed_url(item)
            self.assertIn('/resolve/' + module.REVISION + '/', url)
            self.assertTrue(url.startswith('https://huggingface.co/Qwen/'))
            self.assertNotIn('?', url)
        for name in ('../weights.gguf', '/tmp/weights.gguf', 'different.gguf'):
            with self.assertRaises(ValueError):
                module.fixed_url({'name': name, 'bytes': 10, 'sha256': '0' * 64})

    def test_official_catalog_must_match_both_full_shard_digests(self):
        value = {'id': module.REPO, 'sha': module.REVISION,
            'siblings': [{'rfilename': x['name'], 'size': x['bytes'],
                'lfs': {'size': x['bytes'], 'sha256': x['sha256']}} for x in module.FILES]}
        self.assertEqual(module.validate_metadata_value(value)['files'], list(module.FILES))
        changed = json.loads(json.dumps(value)); changed['siblings'][1]['lfs']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            module.validate_metadata_value(changed)
        changed = json.loads(json.dumps(value)); changed['sha'] = 'main'
        with self.assertRaises(ValueError):
            module.validate_metadata_value(changed)
        changed = json.loads(json.dumps(value)); changed['siblings'].append(changed['siblings'][0])
        with self.assertRaises(ValueError):
            module.validate_metadata_value(changed)

    def test_raw_metadata_identity_cannot_be_substituted(self):
        with self.assertRaisesRegex(ValueError, 'metadata_sha256'):
            module.validate_metadata(b'{"id":"invented fixture, not official metadata"}')

    def test_journal_claim_is_exclusive_and_records_chain_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.jsonl'
            journal = module.Journal(path)
            journal.append({'event': 'range_intent', 'attempt': 1, 'file': 'tiny-invented-fixture'})
            journal.append({'event': 'range_outcome', 'attempt': 1, 'status': 'failed_or_unknown'})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertIsNone(rows[0]['previous_record_sha256'])
            self.assertEqual(rows[1]['previous_record_sha256'], rows[0]['record_sha256'])
            for row in rows:
                recorded = row.pop('record_sha256')
                self.assertEqual(recorded, module.digest(row))
            with self.assertRaises(FileExistsError):
                module.Journal(path)

    def test_local_actual_entry_is_refused_before_artifact_or_network_io(self):
        with self.assertRaisesRegex(RuntimeError, 'genuine_hosted_colab'):
            module.initialize()
        with self.assertRaisesRegex(RuntimeError, 'genuine_hosted_colab'):
            module.download()


if __name__ == '__main__':
    unittest.main()
