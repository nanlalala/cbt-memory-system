"""Offline regression evidence for session-bound UI retries and replay mutations."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_controlled_dialogue_ui_v2 as ui_support

spec = importlib.util.spec_from_file_location("review_controlled_demo", ROOT / "app/controlled_memory_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
import controlled_dialogue as controlled
from stable_retrieval import SourceSearchResult


class SessionBoundUITests(unittest.TestCase):
    launch = ui_support.ControlledUITests.launch
    button = ui_support.ControlledUITests.button
    rendered = ui_support.ControlledUITests.rendered
    send = ui_support.ControlledUITests.send

    def session(self, folder):
        provider = demo.LocalMockProvider()
        session = ui_support.ui._new_session(database=Path(folder) / "ui.sqlite", approved_root=folder,
            user_id="fictional-ui-review", run_id="ui-review", session_id="original-session",
            provider=provider, background=False)
        self.addCleanup(session.close)
        return session, provider

    @staticmethod
    def interrupt_after_success(session):
        original = session.send_message
        identities = []
        def send(text, **kwargs):
            identities.append(kwargs["input_id"])
            result = original(text, **kwargs)
            if len(identities) == 1:
                raise OSError("fictional UI handoff lost after committed answer")
            return result
        session.send_message = send
        return identities

    def test_same_session_manual_retry_displays_original_receipt_and_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            session, provider = self.session(folder)
            identities = self.interrupt_after_success(session)
            root = self.launch(session)
            self.send(root, "fictional reflected detail")
            self.assertEqual(len(provider.calls), 1)
            self.button(root, "发送消息").click()
            self.assertEqual(identities[0], identities[1])
            self.assertEqual(len(provider.calls), 1)
            visible = self.rendered(root)
            self.assertIn("已恢复原操作回执", visible)
            self.assertIn(identities[0], visible)
            self.assertIn("本地虚构回声：fictional reflected detail", visible)

    def test_receipt_render_failure_preserves_retry_identity_after_provider_success(self):
        with tempfile.TemporaryDirectory() as folder:
            session, provider = self.session(folder)
            root = self.launch(session)
            original_send = session.send_message
            identities = []
            def capture(text, **kwargs):
                identities.append(kwargs["input_id"])
                return original_send(text, **kwargs)
            session.send_message = capture
            original_render = ui_support.ui.render_operation_receipts
            rendered = []
            def render_once(reply, **kwargs):
                rendered.append(reply)
                if len(rendered) == 1:
                    raise OSError("fictional receipt display failed after provider success")
                return original_render(reply, **kwargs)
            with patch.object(ui_support.ui, "render_operation_receipts", side_effect=render_once):
                self.send(root, "fictional rendering handoff")
                self.assertEqual(len(provider.calls), 1)
                self.button(root, "发送消息").click()
            self.assertEqual(identities[0], identities[1])
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(len(session.durable.list_sources(session.user_id)), 1)
            visible = self.rendered(root)
            self.assertIn("已恢复原操作回执", visible)
            self.assertIn("本地虚构回声：fictional rendering handoff", visible)

    def test_receipt_only_cross_session_uses_original_identity_and_never_displays_body(self):
        with tempfile.TemporaryDirectory() as folder:
            session, provider = self.session(folder)
            permissions = dict(session.default_permissions, long_term_use=False)
            text = "FICTIONAL_PRIVATE_REPLAY_BODY"
            session.send_message(text, input_id="prior-input", permissions=permissions)
            original_id = session.session_id
            session.new_session()
            current_id = session.session_id
            receipt = session.send_message(text, input_id="prior-input", permissions=permissions)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(receipt["answer_replay_status"], "receipt_only")
            self.assertEqual(receipt["original_session_id"], original_id)
            self.assertEqual(receipt["session_id"], current_id)
            self.assertNotIn(text, receipt["text"])
            root = self.launch(session)
            # Even a malformed service adapter must not display body text when
            # the authoritative receipt says that only metadata is reusable.
            with patch.object(session, "send_message", return_value={**receipt, "text": text}):
                self.send(root, "look up the prior operation receipt")
            visible = self.rendered(root)
            self.assertIn("原会话：" + original_id, visible)
            self.assertNotIn("原会话：" + current_id, visible)
            self.assertNotIn(text, visible)
            self.assertIn("没有可重放的回答正文", visible)

    def test_stale_answer_receipt_has_an_explicit_non_success_label(self):
        visible = ui_support.ui.render_operation_receipts({"answer_status": "stale"})
        self.assertIn("旧回答失效，仅回执", visible)
        self.assertNotIn("回答已完成", visible)

    def test_switching_sessions_clears_pending_identity_without_sending(self):
        with tempfile.TemporaryDirectory() as folder:
            session, provider = self.session(folder)
            identities = self.interrupt_after_success(session)
            root = self.launch(session)
            self.send(root, "same fictional text")
            old_session = session.session_id
            self.button(root, "开始下一次会话").click()
            self.assertNotEqual(old_session, session.session_id)
            self.assertEqual(len(provider.calls), 1)
            visible = self.rendered(root)
            self.button(root, "发送消息").click()
            self.assertNotEqual(identities[0], identities[1])
            self.assertEqual(len(provider.calls), 2)
            self.assertEqual(len(session.durable.list_sources(session.user_id)), 2)
            self.assertIn("上一会话有未决操作，未自动重试", visible)
            self.assertIn(identities[0], visible)
            self.assertIn(old_session, visible)
            self.assertIn("本地虚构回声：same fictional text", self.rendered(root))

    def test_changed_text_or_permissions_are_new_input_identities(self):
        for change in ("text", "permission"):
            with self.subTest(change=change):
                session = ui_support.ControlledStub()
                identities = []
                def interrupted(text, *, input_id, permissions):
                    identities.append(input_id)
                    raise OSError("fictional unresolved UI handoff")
                session.send_message = interrupted
                root = self.launch(session)
                self.send(root, "first fictional input")
                if change == "text":
                    self.send(root, "different fictional input")
                else:
                    field = next(w for w in ui_support.support.walk(root)
                                 if w.description == "允许长期使用")
                    field.value = False
                    self.button(root, "发送消息").click()
                self.assertNotEqual(identities[0], identities[1])

    def test_new_session_does_not_release_a_safety_pause_or_send(self):
        with tempfile.TemporaryDirectory() as folder:
            session, provider = self.session(folder)
            self.interrupt_after_success(session)
            root = self.launch(session)
            self.send(root, "我想自杀")
            self.button(root, "开始下一次会话").click()
            self.assertTrue(session.snapshot()["safety_paused"])
            self.assertTrue(self.button(root, "发送消息").disabled)
            self.assertEqual(provider.calls, [])


class ReplayMutationTests(unittest.TestCase):
    def replay(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        return demo.run_scripted(Path(folder.name) / "replay.sqlite", approved_root=folder.name,
                                 run_id="mutation-review")

    def test_baseline_checks_real_history_packet_and_source_counts(self):
        report = self.replay()
        stage = next(s for s in report["scenarios"] if s["scenario"] == "history_source_integrity")
        self.assertEqual(len(stage["assertions"]), 5)
        self.assertTrue(all(a["passed"] for a in stage["assertions"]))
        first = next(s for s in report["scenarios"] if s["scenario"] == "multi_axis_and_worker")
        self.assertNotIn("retrieval_follows_current_multi_axis_projection",
                         [a["assertion"] for a in first["assertions"]])
        state_check = next(a for a in first["assertions"]
                           if a["assertion"] == "task_projection_preserves_current_multi_axis_state")
        self.assertIn("included_claim_count", state_check["observed"])
        self.assertIn("history_quote_claim_count", state_check["observed"])

    def test_zero_hit_mutation_is_rejected(self):
        original = controlled.StableSourceRetriever.search
        def no_hits(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            return SourceSearchResult((), result.trace)
        with patch.object(controlled.StableSourceRetriever, "search", no_hits):
            with self.assertRaisesRegex(demo.DemoContractFailure, "history_query_retrieves_expected_sources"):
                self.replay()

    def test_ignore_query_mutation_is_rejected_by_zero_match_control(self):
        original = controlled.StableSourceRetriever.search
        def ignores_query(self, query, **kwargs):
            return original(self, "archivekey", **kwargs)
        with patch.object(controlled.StableSourceRetriever, "search", ignores_query):
            with self.assertRaisesRegex(demo.DemoContractFailure, "zero_hit_query"):
                self.replay()

    def test_ignore_top_k_mutation_is_rejected(self):
        original = controlled.StableSourceRetriever.search
        def returns_all_relevant(self, query, **kwargs):
            kwargs["top_k"] = 999
            return original(self, query, **kwargs)
        with patch.object(controlled.StableSourceRetriever, "search", returns_all_relevant):
            with self.assertRaisesRegex(demo.DemoContractFailure, "history_query_retrieves_expected_sources"):
                self.replay()

    def test_correct_ids_with_wrong_final_outbound_body_are_rejected(self):
        original = controlled.build_cbt_context
        def corrupt_body(*args, **kwargs):
            package = original(*args, **kwargs)
            messages = deepcopy(list(package.messages))
            for message in messages:
                if not message["content"].startswith("SOURCED MEMORY DATA ONLY."):
                    continue
                header, body = message["content"].split("\n", 1)
                packet = json.loads(body)
                for claim in packet["claims"]:
                    if claim["field"] == "history_quote":
                        claim["value"] = "X" * len(claim["value"])
                        for span in claim["evidence"]:
                            span["text"] = "X" * len(span["text"])
                message["content"] = header + "\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True)
            payload = {**package.request_payload, "messages": messages}
            serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
            return replace(package, messages=tuple(messages), request_payload=payload,
                           serialized_request=serialized, char_count=len(serialized))
        with patch.object(controlled, "build_cbt_context", corrupt_body):
            with self.assertRaisesRegex(demo.DemoContractFailure, "final_outbound_quotes_match_full_source"):
                self.replay()


if __name__ == "__main__":
    unittest.main()
