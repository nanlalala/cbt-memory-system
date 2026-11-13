"""Product audit metadata against the real provider's pure request preview.

All inputs, outputs and counters are invented fixtures. No hosted marker,
credential, provider construction or model/network call is used.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'app'), str(ROOT / 'modules/agent-integration')]
from colab_provider import ColabChatProvider
from controlled_dialogue import ControlledDialogueSession, PrototypeLimits
from dialogue_service import ModelOutput


USAGE = {'prompt_tokens': 13, 'completion_tokens': 3, 'total_tokens': 16}


class PreviewCapture:
    """Use the production serializer, with a local function as the backend."""

    def __init__(self, *, include_wire=True, fail=False):
        self.settings = SimpleNamespace(model='fixture-not-a-model',
            max_response_tokens=64, thinking_mode='disabled')
        self.max_prompt_chars = 50000
        self.include_wire, self.fail, self.calls = include_wire, fail, []

    def preview_request(self, messages, purpose):
        preview = ColabChatProvider.preview_request(self, messages, purpose)
        if not self.include_wire:
            preview.pop('serialized_request')
        return preview

    def __call__(self, *, messages, purpose):
        self.calls.append(self.preview_request(messages, purpose))
        if self.fail:
            raise TimeoutError('invented offline transport failure')
        return ModelOutput('A fictional response.', usage=dict(USAGE))


class WithoutPreview:
    def __init__(self):
        self.calls = []

    def __call__(self, *, messages, purpose):
        self.calls.append(messages)
        return ModelOutput('A fictional response with no usage receipt.')


class ProductRequestTraceTests(unittest.TestCase):
    def send(self, provider):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        session = ControlledDialogueSession.open(Path(directory.name) / 'memory.sqlite',
            approved_root=directory.name, user_id='fictional-audit', provider=provider,
            limits=PrototypeLimits(max_response_tokens=64), background=False)
        self.addCleanup(session.close)
        return session.send_message('This is a fictional audit input.',
            input_id='fixture-input', as_of='2026-11-03T09:00:00Z')

    def test_actual_preview_wire_hash_and_received_usage_are_separate_from_canonical_hash(self):
        provider = PreviewCapture()
        reply = self.send(provider)
        self.assertEqual(reply['answer_status'], 'succeeded')
        self.assertEqual(len(provider.calls), 1)
        sent = provider.calls[0]
        trace = reply['retrieval_trace']['context_assembly']
        canonical = json.dumps(sent['payload'], ensure_ascii=True, sort_keys=True, allow_nan=False)
        self.assertEqual(json.loads(sent['serialized_request']), json.loads(canonical))
        self.assertNotEqual(sent['serialized_request'], canonical)
        self.assertEqual(trace['request_hash_scope'], 'canonical_renderer_payload')
        self.assertEqual(trace['request_sha256'], sha256(canonical.encode()).hexdigest())
        self.assertEqual(trace['provider_request_sha256'], sha256(sent['serialized_request'].encode()).hexdigest())
        self.assertNotEqual(trace['request_sha256'], trace['provider_request_sha256'])
        self.assertEqual(trace['provider_request_chars'], sent['serialized_length'])
        self.assertEqual(trace['request_chars'], len(canonical))
        self.assertEqual(trace['provider_usage'], reply['usage'])
        self.assertEqual(trace['provider_usage'], USAGE)
        self.assertIsNone(trace['request_tokens'])
        self.assertEqual(sum(m['content'] == 'This is a fictional audit input.'
                             for m in sent['payload']['messages']), 1)

    def test_preview_without_wire_keeps_provider_hash_and_size_unknown(self):
        provider = PreviewCapture(include_wire=False)
        reply = self.send(provider)
        trace = reply['retrieval_trace']['context_assembly']
        self.assertEqual(reply['answer_status'], 'succeeded')
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(trace['provider_request_sha256'])
        self.assertIsNone(trace['provider_request_chars'])
        self.assertEqual(trace['provider_usage'], USAGE)

    def test_no_preview_or_usage_does_not_fabricate_a_wire_hash_or_zero_tokens(self):
        provider = WithoutPreview()
        reply = self.send(provider)
        trace = reply['retrieval_trace']['context_assembly']
        self.assertEqual(reply['answer_status'], 'succeeded')
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(trace['provider_request_sha256'])
        self.assertIsNone(trace['provider_request_chars'])
        self.assertIsNone(trace['provider_usage'])
        self.assertIsNone(reply['usage'])

    def test_failed_backend_does_not_turn_preview_metadata_into_reported_usage(self):
        provider = PreviewCapture(fail=True)
        reply = self.send(provider)
        trace = reply['retrieval_trace']['context_assembly']
        self.assertEqual(reply['answer_status'], 'failed_or_unknown')
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(trace['provider_usage'])
        self.assertIsNone(reply['usage'])


if __name__ == '__main__':
    unittest.main()
