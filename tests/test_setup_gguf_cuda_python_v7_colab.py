"""Tiny invented metadata/file fixtures; no package, network or model run."""
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

SOURCE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect/setup_gguf_cuda_python_v7_colab.py'
spec = importlib.util.spec_from_file_location('cuda_setup_v7_fixture', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Response(io.BytesIO):
    status = 200


class SetupTests(unittest.TestCase):
    def release(self):
        return {'tag_name': m.RELEASE_TAG, 'assets': [{'name': m.WHEEL, 'size': m.WHEEL_BYTES,
            'digest': 'sha256:' + m.WHEEL_SHA,
            'browser_download_url': 'https://github.com/abetlen/llama-cpp-python/releases/download/' + m.RELEASE_TAG + '/' + m.WHEEL}]}

    def item(self, body=b'invented tiny package bytes'):
        return {'filename': 'fixture.whl', 'url': 'https://example.invalid/fixture.whl',
                'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}

    def test_official_asset_requires_tag_size_digest_and_exact_url(self):
        self.assertEqual(m.validate_release_value(self.release())['name'], m.WHEEL)
        for key, value in [('size', 1), ('digest', 'sha256:' + '0' * 64),
                           ('browser_download_url', 'https://example.invalid/wheel')]:
            data = self.release(); data['assets'][0][key] = value
            with self.assertRaises(ValueError): m.validate_release_value(data)
        data = self.release(); data['tag_name'] = 'v0.3.36'
        with self.assertRaises(ValueError): m.validate_release_value(data)
        with self.assertRaises(ValueError): m.validate_release(b'invented metadata is not the publisher bytes')

    def test_exact_pypi_version_and_owned_universal_wheel(self):
        body = {'info': {'name': 'diskcache', 'version': '5.6.3'}, 'urls': [{
            'filename': 'diskcache-5.6.3-py3-none-any.whl', 'packagetype': 'bdist_wheel',
            'url': 'https://files.pythonhosted.org/packages/fixture.whl', 'size': 25,
            'digests': {'sha256': 'a' * 64}, 'yanked': False}]}
        self.assertEqual(m.diskcache_wheel(json.dumps(body).encode())['size'], 25)
        body['info']['version'] = '5.6.4'
        with self.assertRaises(ValueError): m.diskcache_wheel(json.dumps(body).encode())
        body['info']['version'] = '5.6.3'; body['urls'][0]['filename'] = '../bad-py3-none-any.whl'
        with self.assertRaises(ValueError): m.diskcache_wheel(json.dumps(body).encode())

    def test_streamed_bytes_and_publisher_hash_are_both_required(self):
        body = b'invented tiny package bytes'
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                patch.object(m.urllib.request, 'urlopen', return_value=Response(body)) as network:
            output = m.download(self.item(body), time.monotonic() + 10)
            self.assertEqual(output.read_bytes(), body)
            self.assertEqual(network.call_count, 1)
            rows = [json.loads(x) for x in (Path(d) / 'setup_journal.jsonl').read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows], ['wheel_download_intent', 'wheel_download_verified'])
            self.assertEqual(rows[-1]['bytes_received'], len(body))

    def test_short_oversized_and_wrong_hash_bodies_stop_after_one_get(self):
        for body, item in [(b'ab', self.item(b'abc')), (b'abcd', self.item(b'abc')),
                           (b'abc', {**self.item(b'abc'), 'sha256': '0' * 64})]:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                    patch.object(m.urllib.request, 'urlopen', return_value=Response(body)) as network:
                with self.assertRaises(ValueError): m.download(item, time.monotonic() + 10)
                self.assertEqual(network.call_count, 1)
                row = json.loads((Path(d) / 'setup_journal.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['event'], 'wheel_download_failed_or_unknown')
                self.assertEqual(row['bytes_received'], len(body))
                self.assertIs(row['automatic_retry'], False)

    def test_existing_or_escaping_wheel_cannot_make_a_request(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                patch.object(m.urllib.request, 'urlopen') as network:
            (Path(d) / 'fixture.whl').write_bytes(b'existing partial fixture')
            for item in (self.item(), {**self.item(), 'filename': '../escape.whl'}):
                with self.assertRaises(FileExistsError): m.download(item, time.monotonic() + 10)
            network.assert_not_called()

    def test_absolute_deadline_prevents_a_new_get(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m, 'ROOT', Path(d)), \
                patch.object(m.urllib.request, 'urlopen') as network:
            with self.assertRaises(TimeoutError): m.download(self.item(), time.monotonic() - 1)
            network.assert_not_called()
            self.assertEqual(json.loads((Path(d) / 'setup_journal.jsonl').read_text().splitlines()[-1])['bytes_received'], 0)

    def test_gpu_query_requires_a_real_successful_t4_result(self):
        class Result:
            returncode = 0
            stdout = 'Tesla T4, fixture-driver, 15360 MiB, 0 MiB'
        with patch.object(m.subprocess, 'run', return_value=Result()) as command:
            self.assertIn('T4', m.gpu_query(time.monotonic() + 10))
            self.assertEqual(command.call_count, 1)
        Result.stdout = 'CPU only'
        with patch.object(m.subprocess, 'run', return_value=Result()):
            with self.assertRaises(RuntimeError): m.gpu_query(time.monotonic() + 10)

    @unittest.skipIf(bool(os.environ.get('COLAB_RELEASE_TAG')), 'local guard fixture only')
    def test_non_colab_main_stops_before_network_or_directory_creation(self):
        with patch.object(m.urllib.request, 'urlopen') as network, patch.object(Path, 'mkdir') as mkdir:
            with self.assertRaises(RuntimeError): m.main()
            network.assert_not_called(); mkdir.assert_not_called()


if __name__ == '__main__':
    unittest.main()
