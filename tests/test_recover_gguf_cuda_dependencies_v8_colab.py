"""Tiny invented metadata/file fixtures; no network, install or model run."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('cuda12_recovery_tiny_fixture',
    Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect/recover_gguf_cuda_dependencies_v8_colab.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Response(io.BytesIO):
    status = 200


class RecoveryTests(unittest.TestCase):
    def item(self, body=b'invented dependency'):
        return {'name': 'fixture', 'filename': 'fixture.whl', 'url': 'https://example.invalid/tiny',
                'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}

    def test_publisher_metadata_requires_exact_release_size_sha_and_linux_wheel(self):
        p = m.PACKAGES[0]
        data = {'info': {'name': p['name'], 'version': p['version']}, 'urls': [{
            'packagetype': 'bdist_wheel', 'filename': 'fixture-py3-none-manylinux2014_x86_64.whl',
            'size': p['bytes'], 'digests': {'sha256': p['sha256']},
            'url': 'https://files.pythonhosted.org/packages/tiny.whl'}]}
        self.assertEqual(m.wheel_entry_value(p, data)['bytes'], p['bytes'])
        data['urls'][0]['digests']['sha256'] = '0' * 64
        with self.assertRaises(ValueError): m.wheel_entry_value(p, data)
        with self.assertRaises(ValueError): m.wheel_entry(p, b'invented nonpublisher metadata')

    def test_exact_streamed_body_succeeds_once(self):
        b = b'invented dependency'
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                patch.object(m.urllib.request, 'urlopen', return_value=Response(b)) as request:
            self.assertEqual(m.download(self.item(b), time.monotonic() + 10).read_bytes(), b)
            self.assertEqual(request.call_count, 1)
            rows = [json.loads(x) for x in (Path(d) / 'recovery_journal.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'dependency_download_verified')

    def test_wrong_or_incomplete_body_stops_no_retry(self):
        for actual in (b'abc', b'abcdefghi'):
            with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                    patch.object(m.urllib.request, 'urlopen', return_value=Response(actual)) as request:
                with self.assertRaises(ValueError): m.download(self.item(b'abcdef'), time.monotonic() + 10)
                self.assertEqual(request.call_count, 1)
                r = json.loads((Path(d) / 'recovery_journal.jsonl').read_text().splitlines()[-1])
                self.assertEqual(r['event'], 'dependency_download_failed_or_unknown')
                self.assertEqual(r['bytes_received'], len(actual)); self.assertFalse(r['automatic_retry'])

    def test_existing_partial_and_expired_deadline_prevent_network(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                patch.object(m.urllib.request, 'urlopen') as request:
            p = Path(d) / 'fixture.whl'; p.write_bytes(b'partial')
            with self.assertRaises(FileExistsError): m.download(self.item(), time.monotonic() + 10)
            p.unlink()
            with self.assertRaises(TimeoutError): m.download(self.item(), time.monotonic() - 1)
            request.assert_not_called()

    @unittest.skipIf(bool(os.environ.get('COLAB_RELEASE_TAG')), 'local-only guard fixture')
    def test_local_main_refuses_before_network_or_directory_creation(self):
        with patch.object(m.urllib.request, 'urlopen') as request, patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(RuntimeError): m.main()
            request.assert_not_called(); mkdir.assert_not_called()


if __name__ == '__main__': unittest.main()
