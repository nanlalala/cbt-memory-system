from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import colab_provider as adapter  # noqa: E402


KEY = "fixture-not-a-real-api-key"
ALIAS = "deepseek-v4-1-flash-260910"
MESSAGES = [{"role": "system", "content": "Fictional demo."}, {"role": "user", "content": "My fictional goal?"}]


def response(**updates):
    result = {
        "hypothesis": "A fictional answer.", "provider_model": ALIAS,
        "finish_reason": "stop", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result.update(updates)
    return result


class ColabProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = patch.object(adapter, "require_colab")
        self.guard_mock = self.guard.start()
        self.addCleanup(self.guard.stop)
        self.runner = adapter._runner()

    def provider(self, **settings):
        return adapter.ColabChatProvider(key=KEY, **settings)

    def test_secret_access_is_explicit_and_not_exposed_by_repr_or_metadata(self) -> None:
        with patch.object(adapter, "_read_colab_secret", return_value=KEY) as read:
            provider = adapter.ColabChatProvider.from_colab_secret("ChosenSecret")
        read.assert_called_once_with("ChosenSecret")
        self.assertNotIn(KEY, repr(provider))
        self.assertNotIn(KEY, json.dumps(provider.metadata()))
        self.assertEqual(provider.remaining_calls, 12)
        self.assertTrue(provider.requires_colab)
        self.assertTrue(self.guard_mock.called)

    def test_outside_colab_fails_before_secret_read_or_transport(self) -> None:
        self.guard.stop()
        with patch.object(adapter.importlib.util, "find_spec", return_value=None), \
             patch.object(adapter, "_read_colab_secret") as read, \
             patch.object(self.runner, "generate_response") as generate:
            with self.assertRaisesRegex(RuntimeError, "only inside Google Colab"):
                adapter.ColabChatProvider.from_colab_secret()
            read.assert_not_called()
            generate.assert_not_called()

    def test_runtime_is_rechecked_before_each_request(self) -> None:
        provider = self.provider()
        with patch.object(adapter, "require_colab", side_effect=RuntimeError("outside Colab")), \
             patch.object(self.runner, "generate_response") as generate:
            with self.assertRaisesRegex(RuntimeError, "outside Colab"):
                provider(messages=MESSAGES, purpose="response")
            generate.assert_not_called()
        self.assertEqual(provider.calls_attempted, 0)

    def test_installing_colab_module_outside_hosted_runtime_does_not_unlock_adapter(self) -> None:
        self.guard.stop()
        with patch.dict(adapter.os.environ, {}, clear=True), \
             patch.object(adapter.Path, "is_dir", return_value=True), \
             patch.object(adapter.importlib.util, "find_spec", return_value=object()), \
             patch.object(adapter, "_read_colab_secret") as read:
            with self.assertRaisesRegex(RuntimeError, "only inside Google Colab"):
                adapter.ColabChatProvider.from_colab_secret()
            read.assert_not_called()

    def test_hosted_runtime_requires_module_working_directory_and_nonsecret_marker(self) -> None:
        self.guard.stop()
        for directory, module_exists, should_pass in ((True, True, True), (False, True, False), (True, False, False)):
            with self.subTest(directory=directory, module_exists=module_exists), \
                 patch.dict(adapter.os.environ, {"COLAB_RELEASE_TAG": "fixture-colab-runtime"}, clear=True), \
                 patch.object(adapter.Path, "is_dir", return_value=directory), \
                 patch.object(adapter.importlib.util, "find_spec", return_value=object() if module_exists else None), \
                 patch.object(adapter.importlib, "import_module") as imported:
                if should_pass:
                    adapter.require_colab()
                    imported.assert_called_once_with("google.colab")
                else:
                    with self.assertRaisesRegex(RuntimeError, "only inside Google Colab"):
                        adapter.require_colab()
                    imported.assert_not_called()

    def test_success_reuses_transport_and_returns_explicit_live_metadata(self) -> None:
        provider = self.provider(max_calls=2)
        with patch.object(self.runner, "generate_response", return_value=response()) as generate:
            output = provider(messages=MESSAGES, purpose="response")
        self.assertEqual(generate.call_count, 1)
        request, settings = generate.call_args.args
        self.assertEqual(request["messages"], MESSAGES)
        self.assertEqual(settings.thinking_mode, "disabled")
        self.assertEqual(settings.max_response_tokens, 4096)
        self.assertEqual(generate.call_args.kwargs["api_base"], "https://tokendance.space/gateway/v1")
        self.assertEqual(output.provider_model, ALIAS)
        self.assertTrue(output.live_model_or_network_used)
        self.assertEqual(provider.calls_attempted, 1)
        self.assertEqual(provider.requested_output_tokens_reserved, 4096)

    def test_failure_counts_once_and_suppresses_secret_and_exception_context(self) -> None:
        provider = self.provider(max_calls=1)
        with patch.object(self.runner, "generate_response", side_effect=RuntimeError(KEY)) as generate:
            with self.assertRaises(adapter.ProviderRequestError) as caught:
                provider(messages=MESSAGES, purpose="response")
            self.assertNotIn(KEY, str(caught.exception))
            self.assertIsNone(caught.exception.__context__)
            with self.assertRaises(adapter.ProviderBudgetExceeded):
                provider(messages=MESSAGES, purpose="response")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(provider.calls_attempted, 1)

    def test_output_reservation_limit_stops_call_before_transport(self) -> None:
        provider = self.provider(max_calls=12, max_requested_output_tokens=4096)
        with patch.object(self.runner, "generate_response", return_value=response()) as generate:
            provider(messages=MESSAGES, purpose="response")
            with self.assertRaises(adapter.ProviderBudgetExceeded):
                provider(messages=MESSAGES, purpose="response")
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(provider.calls_attempted, 1)

    def test_prompt_cap_includes_extraction_instruction_without_changing_inputs(self) -> None:
        provider = self.provider(max_prompt_chars=100)
        original = json.dumps(MESSAGES, sort_keys=True)
        with patch.object(self.runner, "generate_response") as generate:
            with self.assertRaisesRegex(ValueError, "character budget"):
                provider(messages=MESSAGES, purpose="extraction")
        generate.assert_not_called()
        self.assertEqual(provider.calls_attempted, 0)
        self.assertEqual(json.dumps(MESSAGES, sort_keys=True), original)

    def test_extraction_instruction_freezes_schema_and_confirmation_boundary(self) -> None:
        provider = self.provider()
        with patch.object(self.runner, "generate_response", return_value=response(hypothesis='{"candidates": []}')) as generate:
            provider(messages=MESSAGES, purpose="extraction")
        sent = generate.call_args.args[0]["messages"][0]["content"]
        self.assertIn(adapter.EXTRACTION_SCHEMA_INSTRUCTION, sent)
        self.assertIn("never assert user confirmation", sent)
        self.assertIn("supporting_text", sent)
        self.assertIn("intensity (integer 0-10)", sent)

    def test_wrong_missing_truncated_and_changed_model_names_fail(self) -> None:
        for row in (
            response(provider_model="different-model"), response(provider_model=""),
            response(finish_reason="length"), response(hypothesis=""),
        ):
            with self.subTest(row={key: row[key] for key in ("provider_model", "finish_reason")}):
                provider = self.provider()
                with patch.object(self.runner, "generate_response", return_value=row):
                    with self.assertRaises(adapter.ProviderRequestError):
                        provider(messages=MESSAGES, purpose="response")
                self.assertEqual(provider.calls_attempted, 1)
        provider = self.provider()
        with patch.object(self.runner, "generate_response", side_effect=[response(), response(provider_model="deepseek-v4.1-flash")]):
            provider(messages=MESSAGES, purpose="response")
            with self.assertRaises(adapter.ProviderRequestError):
                provider(messages=MESSAGES, purpose="response")

    def test_usage_is_a_whitelist_of_nonnegative_integer_counters(self) -> None:
        provider = self.provider()
        usage = {
            "prompt_tokens": True, "completion_tokens": -2, "total_tokens": 15,
            "reasoning_tokens": "secret text", "untrusted": KEY,
            "completion_tokens_details": {"reasoning_tokens": 3, "other": KEY},
        }
        with patch.object(self.runner, "generate_response", return_value=response(usage=usage)):
            output = provider(messages=MESSAGES, purpose="response")
        self.assertEqual(output.usage, {"total_tokens": 15, "reasoning_tokens": 3})

    def test_invalid_settings_fail_before_secret_access(self) -> None:
        for kwargs in ({"api_base": "http://not-https.invalid"}, {"max_calls": True}, {"max_calls": 0},
                       {"accepted_provider_models": "not-a-sequence-of-names"}):
            with self.subTest(kwargs=kwargs), patch.object(adapter, "_read_colab_secret") as read:
                with self.assertRaises(ValueError):
                    adapter.ColabChatProvider.from_colab_secret(**kwargs)
                read.assert_not_called()

    def test_public_identity_and_budget_configuration_are_read_only(self) -> None:
        provider = self.provider()
        for field, value in (
            ("max_calls", 10000), ("max_prompt_chars", 1000000),
            ("max_requested_output_tokens", 1000000),
            ("api_base", "https://another-host.invalid"), ("settings", None),
        ):
            with self.subTest(field=field), self.assertRaises(AttributeError):
                setattr(provider, field, value)

    def test_secret_read_failure_does_not_leak_error_or_context(self) -> None:
        with patch.object(adapter, "_read_colab_secret", side_effect=RuntimeError(KEY)):
            with self.assertRaises(RuntimeError) as caught:
                adapter.ColabChatProvider.from_colab_secret()
        self.assertNotIn(KEY, str(caught.exception))
        self.assertIsNone(caught.exception.__context__)


if __name__ == "__main__":
    unittest.main()
