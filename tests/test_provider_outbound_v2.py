from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import colab_provider as adapter


KEY = "fixture-not-a-real-key"
ALIAS = "deepseek-v4-1-flash-260910"
MESSAGES = [{"role": "system", "content": "Fictional source-only demo."},
            {"role": "user", "content": "这周暂停，次数未知。"}]


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.data).encode()


class CapturingOpener:
    """A local transport fixture; never opens a URL or reads a credential."""

    def __init__(self):
        self.requests = []

    def open(self, request, *, timeout):
        self.requests.append((request, timeout))
        return FakeResponse({"model": ALIAS, "choices": [{"finish_reason": "stop",
                                                           "message": {"content": "fixture answer"}}]})


class ProviderOutboundTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(adapter, "require_colab")
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.runner = adapter._runner()

    def provider(self, **kwargs):
        return adapter.ColabChatProvider(key=KEY, **kwargs)

    def test_preview_is_side_effect_free_and_excludes_secret_and_usage_guesses(self):
        provider = self.provider(extraction_schema_version="2.0")
        before = provider.metadata()
        original = json.dumps(MESSAGES, sort_keys=True)
        with patch.object(adapter, "require_colab", side_effect=AssertionError("preview requires no runtime I/O")), \
             patch.object(adapter, "_read_colab_secret", side_effect=AssertionError("no secret read")), \
             patch.object(self.runner, "generate_response", side_effect=AssertionError("no request")):
            preview = provider.preview_request(MESSAGES, "extraction")
        self.assertEqual(before, provider.metadata())
        self.assertEqual(original, json.dumps(MESSAGES, sort_keys=True))
        self.assertEqual(preview["serialized_length"], len(preview["serialized_request"]))
        self.assertNotIn(KEY, json.dumps(preview))
        self.assertNotIn("usage", preview)
        self.assertEqual(provider.calls_attempted, 0)
        self.assertEqual(provider.requested_output_tokens_reserved, 0)

    def test_preview_equals_actual_unchanged_runner_bytes_for_every_purpose_and_thinking_mode(self):
        for purpose in ("response", "extraction"):
            for mode in ("provider_default", "disabled", "enabled"):
                for version in ("1.0", "2.0"):
                    with self.subTest(purpose=purpose, mode=mode, version=version):
                        provider = self.provider(thinking_mode=mode, extraction_schema_version=version)
                        preview = provider.preview_request(MESSAGES, purpose)
                        opener = CapturingOpener()
                        with patch.object(self.runner, "build_opener", return_value=opener):
                            output = provider(messages=MESSAGES, purpose=purpose)
                        self.assertEqual(len(opener.requests), 1)
                        request, timeout = opener.requests[0]
                        self.assertEqual(request.data.decode(), preview["serialized_request"])
                        self.assertEqual(json.loads(request.data), preview["payload"])
                        self.assertEqual(preview["payload"]["messages"], preview["messages"])
                        self.assertEqual(timeout, 180)
                        self.assertEqual(output.usage, {})
                        self.assertEqual(provider.calls_attempted, 1)
                        self.assertEqual("thinking" in preview["payload"], mode != "provider_default")

    def test_unicode_escape_can_exceed_cap_when_raw_content_fits(self):
        messages = [{"role": "user", "content": "记" * 80}]
        baseline = self.provider().preview_request(messages, "response")
        cap = baseline["serialized_length"] - 1
        self.assertLess(sum(len(m["content"]) for m in messages), cap)
        provider = self.provider(max_prompt_chars=cap)
        self.assertFalse(provider.preview_request(messages, "response")["within_character_budget"])
        with patch.object(self.runner, "generate_response") as generate:
            with self.assertRaisesRegex(ValueError, "complete outbound JSON.*character budget"):
                provider(messages=messages, purpose="response")
        generate.assert_not_called()
        self.assertEqual(provider.calls_attempted, 0)
        self.assertEqual(provider.requested_output_tokens_reserved, 0)

    def test_cap_includes_schema_appended_after_original_messages(self):
        baseline = self.provider(extraction_schema_version="2.0")
        response_size = baseline.preview_request(MESSAGES, "response")["serialized_length"]
        extraction_size = baseline.preview_request(MESSAGES, "extraction")["serialized_length"]
        self.assertGreater(extraction_size, response_size)
        provider = self.provider(extraction_schema_version="2.0", max_prompt_chars=extraction_size - 1)
        with patch.object(self.runner, "generate_response") as generate:
            with self.assertRaisesRegex(ValueError, "character budget"):
                provider(messages=MESSAGES, purpose="extraction")
        generate.assert_not_called()
        self.assertEqual(provider.calls_attempted, 0)

    def test_exact_cap_admits_request_and_metadata_is_inside_body_budget(self):
        baseline = self.provider(thinking_mode="enabled", model="fixed-requested-model")
        preview = baseline.preview_request(MESSAGES, "response")
        provider = self.provider(thinking_mode="enabled", model="fixed-requested-model",
                                 max_prompt_chars=preview["serialized_length"])
        opener = CapturingOpener()
        with patch.object(self.runner, "build_opener", return_value=opener):
            provider(messages=MESSAGES, purpose="response")
        body = opener.requests[0][0].data.decode()
        self.assertEqual(len(body), provider.max_prompt_chars)
        self.assertEqual(json.loads(body)["model"], "fixed-requested-model")
        self.assertIn("temperature", body)
        self.assertIn("max_tokens", body)
        self.assertIn("thinking", body)

    def test_v2_candidate_instruction_keeps_product_qualification_out_of_model_control(self):
        provider = self.provider(extraction_schema_version="2.0")
        text = provider.preview_request(MESSAGES, "extraction")["messages"][0]["content"]
        for required in ("fixed_historical", "mutable_state", "F1", "F4", "V1", "V5", "claim_bindings",
                         "zero-based Unicode", "Do not claim rule verification", "unconfirmed candidate"):
            self.assertIn(required, text)
        self.assertIn("Do not emit schema_version, origin, permissions", text)
        self.assertIn("semantic_support, confirmation_state or lifecycle_status", text)
        self.assertIn("never a product state update", text)

    def test_default_v1_instruction_and_from_secret_version_forwarding_are_preserved(self):
        default = self.provider()
        self.assertEqual(default.extraction_schema_version, "1.0")
        text = default.preview_request(MESSAGES, "extraction")["messages"][0]["content"]
        self.assertIn(adapter.EXTRACTION_SCHEMA_INSTRUCTION, text)
        self.assertNotIn("Schema 2.0 candidate additions", text)
        with patch.object(adapter, "_read_colab_secret", return_value=KEY) as read:
            provider = adapter.ColabChatProvider.from_colab_secret("ChosenSecret", extraction_schema_version="2.0")
        read.assert_called_once_with("ChosenSecret")
        self.assertEqual(provider.extraction_schema_version, "2.0")

    def test_bad_version_fails_before_secret_access_and_mode_is_read_only(self):
        for version in ("3.0", None, True, []):
            with self.subTest(version=version), patch.object(adapter, "_read_colab_secret") as read:
                with self.assertRaises(ValueError):
                    adapter.ColabChatProvider.from_colab_secret(extraction_schema_version=version)
                read.assert_not_called()
        provider = self.provider()
        with self.assertRaises(AttributeError):
            provider.extraction_schema_version = "2.0"
        self.assertEqual(provider.metadata()["prompt_character_budget_scope"],
                         "complete_ascii_json_request_after_extraction_schema")

    def test_preview_inserts_schema_before_user_messages_without_mutating_them(self):
        messages = [{"role": "user", "content": "Source-only text"}]
        provider = self.provider()
        preview = provider.preview_request(messages, "extraction")
        self.assertEqual(messages, [{"role": "user", "content": "Source-only text"}])
        self.assertEqual(preview["messages"][0]["role"], "system")
        self.assertEqual(preview["messages"][1], messages[0])
        self.assertEqual(provider.calls_attempted, 0)

    def test_schema_insertion_cannot_exceed_actual_message_count_bound(self):
        provider = self.provider()
        with self.assertRaisesRegex(ValueError, "actual outbound messages"):
            provider.preview_request([{"role": "user", "content": "source"}] * 128, "extraction")


if __name__ == "__main__":
    unittest.main()
