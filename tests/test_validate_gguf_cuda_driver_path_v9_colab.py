"""Three tiny pure/file fixtures; no driver/package/model or network run."""
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('driver_path_v9_tiny',
    Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect/validate_gguf_cuda_driver_path_v9_colab.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class DriverPathTests(unittest.TestCase):
    def test_prefix_contains_only_three_owned_directories_and_observed_driver(self):
        self.assertEqual(m.library_prefix().split(':'), [
            '/content/cbt-qwen7b-gpu-python-recovery-20261112-v8/site-packages/nvidia/cuda_runtime/lib',
            '/content/cbt-qwen7b-gpu-python-recovery-20261112-v8/site-packages/nvidia/cublas/lib',
            '/content/cbt-qwen7b-gpu-python-20261112-v7/site-packages/llama_cpp/lib',
            '/usr/lib64-nvidia'])

    def test_streamed_member_identity_matches_invented_bytes(self):
        body = b'tiny invented library bytes, not executable software'
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fixture.zip'
            with zipfile.ZipFile(p, 'w') as z: z.writestr('tiny.so', body)
            with zipfile.ZipFile(p) as z:
                self.assertEqual(m.stream_member_sha(z, z.getinfo('tiny.so')), hashlib.sha256(body).hexdigest())

    @unittest.skipIf(bool(os.environ.get('COLAB_RELEASE_TAG')), 'local guard fixture only')
    def test_local_entry_refuses_before_directory_creation_or_subprocess(self):
        with patch.object(Path, 'mkdir') as mkdir, patch.object(m.subprocess, 'run') as child:
            with self.assertRaises(RuntimeError): m.main()
            mkdir.assert_not_called(); child.assert_not_called()


if __name__ == '__main__': unittest.main()
