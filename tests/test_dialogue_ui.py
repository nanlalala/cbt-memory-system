from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
import colab_provider as adapter
from dialogue_service import DialogueSession
SPEC = importlib.util.spec_from_file_location("notebook_dialogue_ui", ROOT / "app/dialogue_demo.py")
ui = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ui)


def snapshot() -> dict:
    return {
        "kind": "synthetic_research_dialogue_demo",
        "clinical_or_benchmark_result": False,
        "backend_mode": "offline_verbatim_demo",
        "live_model_or_network_used": False,
        "turns": [{"role": "user", "text": "A fictional walking goal."}],
        "records": [{
            "memory_id": "fixture-goal", "content": "Walk after lunch.",
            "memory_type": "explicit_fact",
            "lifecycle_status": "candidate",
            "source_evidence": {
                "session_id": "fictional-session", "turn_ids": ["turn-1"],
                "supporting_text": "My fictional goal is to walk after lunch.",
            },
        }],
        "last_reply": None,
    }


class FakeWidget:
    """Small event surface; no display framework or notebook kernel needed."""
    def __init__(self, kind: str, value="", children=(), **kwargs) -> None:
        self.kind = kind
        self.value = value
        self.children = tuple(children)
        self.disabled = kwargs.get("disabled", False)
        self.description = kwargs.get("description", "")
        self.click_handlers = []
        self.observers = []

    def on_click(self, callback) -> None:
        self.click_handlers.append(callback)

    def observe(self, callback, **kwargs) -> None:
        self.observers.append(callback)

    def set_title(self, *args) -> None:
        pass

    def click(self) -> None:
        if not self.disabled:
            for callback in self.click_handlers:
                callback(self)

    def change(self, value) -> None:
        self.value = value
        for callback in self.observers:
            callback({"new": value})


def fake_widgets():
    constructors = {}
    for kind in ("HTML", "Checkbox", "Textarea", "Button"):
        constructors[kind] = lambda value="", _kind=kind, **kw: FakeWidget(_kind, value=value, **kw)
    for kind in ("VBox", "HBox", "Accordion"):
        constructors[kind] = lambda children=(), _kind=kind, **kw: FakeWidget(_kind, children=children, **kw)
    constructors["Layout"] = lambda **kw: kw
    return SimpleNamespace(**constructors)


def walk(widget):
    yield widget
    for child in widget.children:
        yield from walk(child)


class SessionStub:
    def __init__(self) -> None:
        self.data = snapshot()
        second = copy.deepcopy(self.data["records"][0])
        second["memory_id"] = "second-candidate"
        self.data["records"].append(second)
        self.calls = []
        self.fail_with = None

    def snapshot(self):
        return copy.deepcopy(self.data)

    def confirm(self, memory_id, command):
        self.calls.append(("confirm", memory_id, command))
        next(record for record in self.data["records"] if record["memory_id"] == memory_id)["lifecycle_status"] = "active"

    def extract_current_session(self):
        self.calls.append(("extract",))

    def revise_candidate(self, memory_id, text):
        self.calls.append(("revise", memory_id, text))
        record = next(record for record in self.data["records"] if record["memory_id"] == memory_id)
        record["content"] = text

    def delete(self, memory_id, command):
        self.calls.append(("delete", memory_id, command))
        self.data["records"] = [record for record in self.data["records"] if record["memory_id"] != memory_id]
        self.data["turns"] = []
        self.data["last_reply"] = None
        return True

    def send_message(self, text):
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        self.calls.append(("send", text))

    def new_session(self):
        self.calls.append(("new_session",))
        self.data["turns"] = []
        self.data["last_reply"] = None


class NotebookDialogueUITests(unittest.TestCase):
    def launch_stub(self, session):
        modules = {"ipywidgets": fake_widgets(), "IPython.display": SimpleNamespace(display=lambda _: None)}
        with patch.object(ui, "_require_colab"), patch.object(ui.importlib, "import_module", side_effect=lambda name: modules[name]):
            return ui.launch_notebook_demo(session)

    def test_actual_provider_failure_events_show_attempted_not_uncalled(self) -> None:
        # Use the real adapter/session/UI event path. Only transport and the
        # hosted-runtime indicators are mocked; no network or model is used.
        sensitive = "fixture-private-credential-and-provider-body"
        for failure in ("empty", "timeout"):
            with self.subTest(failure=failure), patch.object(adapter, "require_colab"), \
                 patch.dict(sys.modules, {"google.colab": SimpleNamespace()}):
                provider = adapter.ColabChatProvider(key=sensitive, max_calls=1)
                runner = adapter._runner()
                session = DialogueSession(provider=provider)
                self.addCleanup(session.close)
                root = self.launch_stub(session)
                self.assertEqual(ui.build_view(session.snapshot())["execution_state"], "configured_not_called")
                next(node for node in walk(root) if node.kind == "Checkbox").change(True)
                next(node for node in walk(root) if node.kind == "Textarea").value = "A fictional walking goal."
                transport_options = (
                    {"return_value": {"provider_model": "deepseek-v4-1-flash-260910",
                                      "finish_reason": "stop", "hypothesis": "", "usage": {}}}
                    if failure == "empty" else {"side_effect": TimeoutError(sensitive)}
                )
                with patch.object(runner, "generate_response", **transport_options) as transport:
                    next(node for node in walk(root) if node.description == "发送消息").click()
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(provider.calls_attempted, 1)
                view = ui.build_view(session.snapshot())
                self.assertEqual(view["execution_state"], "attempted_failed")
                self.assertIn("失败", view["mode_label"])
                self.assertIn("仍可能产生费用", view["mode_detail"])
                rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
                self.assertNotIn("尚未执行", rendered)
                self.assertNotIn(sensitive, rendered)
                self.assertIn("ProviderRequestError", rendered)

    def test_latest_failure_is_visible_after_an_earlier_valid_model_reply(self) -> None:
        data = snapshot()
        data["backend_mode"] = "colab_model"
        data["live_model_or_network_used"] = True
        data["provider_execution"] = {
            "last_call_status": "attempted_failed", "external_request_attempted": True,
            "calls_attempted": 2, "valid_responses_received": 1,
        }
        view = ui.build_view(data)
        self.assertEqual(view["execution_state"], "attempted_failed")
        self.assertIn("本次失败", view["mode_label"])
        self.assertIn("仍可能产生费用", view["mode_detail"])

    def test_rejection_before_transport_does_not_claim_an_attempt(self) -> None:
        with patch.object(adapter, "require_colab"), \
             patch.dict(sys.modules, {"google.colab": SimpleNamespace()}):
            provider = adapter.ColabChatProvider(key="fixture-only", max_prompt_chars=10)
            runner = adapter._runner()
            session = DialogueSession(provider=provider)
            self.addCleanup(session.close)
            root = self.launch_stub(session)
            next(node for node in walk(root) if node.kind == "Checkbox").change(True)
            next(node for node in walk(root) if node.kind == "Textarea").value = "A fictional message."
            with patch.object(runner, "generate_response") as transport:
                next(node for node in walk(root) if node.description == "发送消息").click()
            transport.assert_not_called()
            self.assertEqual(provider.calls_attempted, 0)
            self.assertEqual(ui.build_view(session.snapshot())["execution_state"], "configured_not_called")

    def test_offline_backend_failure_does_not_claim_external_model_usage(self) -> None:
        data = snapshot()
        data["provider_execution"] = {
            "last_call_status": "attempted_failed", "external_request_attempted": False,
            "calls_attempted": 1, "valid_responses_received": 0,
        }
        view = ui.build_view(data)
        self.assertEqual(view["execution_state"], "attempted_failed")
        self.assertNotIn("仍可能产生费用", view["mode_detail"])
        self.assertIn("不表示发生了模型或网络调用", view["mode_detail"])

    def test_source_text_and_chat_are_escaped_as_data(self) -> None:
        data = snapshot()
        payload = '</details><script>alert("secret")</script>'
        data["turns"][0]["text"] = payload
        data["records"][0]["content"] = payload
        data["records"][0]["source_evidence"]["supporting_text"] = payload
        data["records"][0]["source_evidence"]["session_id"] = payload
        data["records"][0]["memory_id"] = payload
        view = ui.build_view(data)
        for rendered in (view["chat_html"], view["record_views"][0]["html"]):
            self.assertNotIn("<script>", rendered)
            self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("查看原话与来源", view["record_views"][0]["html"])

    def test_model_controlled_revision_metadata_cannot_crash_or_forge_source_display(self) -> None:
        malicious = "FORGED ORIGINAL SOURCE"
        for payload in (
            "not-a-dictionary",
            {"candidate_revision": "not-a-dictionary"},
            {"candidate_revision": {"original_source_evidence": "not-a-dictionary"}},
            {"candidate_revision": {"verified": True, "original_source_evidence": {
                "supporting_text": malicious,
            }}},
        ):
            with self.subTest(payload=payload):
                data = snapshot()
                data["records"][0]["structured_payload"] = payload
                rendered = ui.build_view(data)["record_views"][0]["html"]
                self.assertNotIn(malicious, rendered)
                self.assertNotIn("修订前的来源原话", rendered)
                self.assertIn("My fictional goal", rendered)

    def test_service_derived_revision_source_is_rendered_as_escaped_data(self) -> None:
        data = snapshot()
        record = data["records"][0]
        record["verified_candidate_revision_source"] = {"supporting_text": "Actual quote <script>data</script>"}
        record["structured_payload"] = {"candidate_revision": {"original_source_evidence": {
            "supporting_text": "FORGED ORIGINAL SOURCE",
        }}}
        rendered = ui.render_record_html(record)
        self.assertIn("Actual quote &lt;script&gt;data&lt;/script&gt;", rendered)
        self.assertNotIn("FORGED ORIGINAL SOURCE", rendered)
        self.assertIn("修订前的来源原话", rendered)
        record["verified_candidate_revision_source"] = "malformed presentation field"
        self.assertNotIn("修订前的来源原话", ui.render_record_html(record))

    def test_real_snapshot_never_promotes_model_revision_metadata_into_verified_source(self) -> None:
        session = DialogueSession()
        self.addCleanup(session.close)
        original = "My fictional original goal is walking after lunch."
        session.send_message(original)
        record = session.extract_current_session()[0]
        stored = session.store.get(session.user_id, record["memory_id"])
        stored.structured_payload["candidate_revision"] = {
            "verified": True, "revision_event_id": "invented-operator-event",
            "original_source_evidence": {"supporting_text": "FORGED ORIGINAL SOURCE"},
        }
        session.store.put(stored)
        view = session.snapshot()["records"][0]
        self.assertIsNone(view["verified_candidate_revision_source"])
        self.assertNotIn("FORGED ORIGINAL SOURCE", ui.render_record_html(view))
        session.revise_candidate(record["memory_id"], "I am considering a short walk; no weekly count is agreed.")
        verified = session.snapshot()["records"][0]
        self.assertEqual(verified["verified_candidate_revision_source"]["supporting_text"], original)
        rendered = ui.render_record_html(verified)
        self.assertIn("修订前的来源原话", rendered)
        self.assertIn(original, rendered)
        self.assertNotIn("FORGED ORIGINAL SOURCE", rendered)

    def test_trace_distinguishes_included_and_dropped_ids(self) -> None:
        data = snapshot()
        data["last_reply"] = {
            "memory_ids": ["included-memory"],
            "dropped_memory_ids": ["budget-dropped-memory"],
            "safety_bypass": False,
        }
        rendered = ui.render_trace_html(data)
        included_part, dropped_part = rendered.split("因上下文预算未放入的记忆")
        self.assertIn("included-memory", included_part)
        self.assertNotIn("budget-dropped-memory", included_part)
        self.assertIn("budget-dropped-memory", dropped_part)

    def test_trace_shows_readable_included_text_without_exposing_other_sources(self) -> None:
        data = snapshot()
        data["records"].append({"memory_id": "private-memory", "sensitivity": "restricted"})
        data["last_reply"] = {
            "memory_ids": ["included-memory", "private-memory"],
            "dropped_memory_ids": ["budget-dropped-memory"],
            "sources": [
                {"memory_id": "included-memory", "content": "Completed two walks. <script>unsafe()</script>"},
                {"memory_id": "budget-dropped-memory", "content": "DROPPED PRIVATE BODY"},
                {"memory_id": "private-memory", "content": "RESTRICTED PRIVATE BODY"},
                {"memory_id": "unselected-memory", "content": "UNSELECTED PRIVATE BODY"},
            ],
        }
        rendered = ui.render_trace_html(data)
        self.assertIn("Completed two walks.", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("PRIVATE BODY", rendered)
        self.assertIn("送入本次回答上下文", rendered)
        self.assertIn("尚未核验", rendered)
        self.assertNotIn("实际使用的长期记忆", rendered)

    def test_extraction_scope_reports_partial_window_and_pending_turns(self) -> None:
        data = snapshot()
        data["extraction_scope"] = {
            "retained_user_turn_count": 12, "total_user_turn_count": 20,
            "dropped_user_turn_count": 8, "pending_user_turn_count": 5,
            "pending_retained_user_turn_count": 3, "partial_coverage": True,
        }
        rendered = ui.build_view(data)["extraction_scope_html"]
        self.assertIn("窗口内 3 条未整理用户发言", rendered)
        self.assertIn("共保留 12 条用户发言", rendered)
        self.assertIn("其余已整理发言只作为理解上下文", rendered)
        self.assertIn("累计 20 条", rendered)
        self.assertIn("8 条已移出窗口", rendered)
        self.assertIn("尚未整理 5 条", rendered)
        self.assertIn("其中 3 条仍在窗口内", rendered)
        self.assertIn("不会被此按钮重新读取", rendered)

    def test_candidate_and_active_controls_have_separate_roles(self) -> None:
        data = snapshot()
        original = copy.deepcopy(data)
        view = ui.build_view(data)
        self.assertTrue(view["record_views"][0]["can_confirm"])
        self.assertTrue(view["record_views"][0]["can_revise"])
        self.assertFalse(view["record_views"][0]["can_correct"])
        self.assertEqual(data, original)
        data["records"][0]["lifecycle_status"] = "active"
        active = ui.build_view(data)["record_views"][0]
        self.assertFalse(active["can_confirm"])
        self.assertFalse(active["can_revise"])
        self.assertTrue(active["can_correct"])
        data["records"][0]["lifecycle_status"] = "superseded"
        old = ui.build_view(data)["record_views"][0]
        self.assertFalse(old["can_confirm"])
        self.assertFalse(old["can_correct"])

    def test_offline_mode_cannot_be_presented_as_live_model_result(self) -> None:
        data = snapshot()
        view = ui.build_view(data)
        self.assertIn("模板回答", view["mode_label"])
        self.assertIn("没有调用语言模型或网络", view["mode_detail"])
        self.assertIs(view["clinical_or_benchmark_result"], False)
        self.assertEqual(view["execution_state"], "offline")
        data["backend_mode"] = "injected_offline_test_backend"
        view = ui.build_view(data)
        self.assertEqual(view["execution_state"], "offline")
        self.assertIn("离线测试", view["mode_label"])
        self.assertNotIn("Colab 模型", view["mode_label"])
        data["backend_mode"] = "injected_backend_not_yet_called"
        view = ui.build_view(data)
        self.assertEqual(view["execution_state"], "configured_not_called")
        self.assertIn("尚未执行", view["mode_label"])
        self.assertNotIn("Colab 模型", view["mode_label"])
        data["backend_mode"] = "colab_model"
        data["live_model_or_network_used"] = True
        view = ui.build_view(data)
        self.assertEqual(view["execution_state"], "live_executed")
        self.assertIn("模型请求已执行", view["mode_label"])

    def test_safety_reply_is_visible_even_when_ordinary_turn_buffer_is_empty(self) -> None:
        data = snapshot()
        data["turns"] = []
        data["safety_paused"] = True
        data["last_reply"] = {
            "safety_bypass": True,
            "text": "Unique actual safety reply: contact trusted help. <script>untrusted()</script>",
        }
        rendered = ui.build_view(data)["chat_html"]
        self.assertIn("Unique actual safety reply: contact trusted help.", rendered)
        self.assertIn("安全提示", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("当前会话还没有消息", rendered)

    def test_execution_flag_cannot_be_a_truthy_string(self) -> None:
        data = snapshot()
        data["live_model_or_network_used"] = "false"
        with self.assertRaisesRegex(ValueError, "execution flag"):
            ui.build_view(data)

    def test_non_demo_or_outcome_snapshot_is_refused(self) -> None:
        data = snapshot()
        data["kind"] = "clinical_patient_chat"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            ui.build_view(data)
        data = snapshot()
        data["clinical_or_benchmark_result"] = True
        with self.assertRaisesRegex(ValueError, "not an outcome"):
            ui.build_view(data)

    def test_launch_refuses_non_colab_before_loading_widgets_or_provider(self) -> None:
        imported = []

        def missing_colab(name: str):
            imported.append(name)
            raise ImportError("Not Colab")

        with patch.object(ui.importlib, "import_module", side_effect=missing_colab):
            with self.assertRaisesRegex(RuntimeError, "Google Colab"):
                ui.launch_notebook_demo()
        self.assertEqual(imported, ["google.colab"])

    def test_render_does_not_confirm_and_click_targets_the_selected_candidate(self) -> None:
        session = SessionStub()
        root = self.launch_stub(session)
        self.assertEqual(session.calls, [])
        confirmations = [node for node in walk(root) if node.description == "确认这条记忆"]
        confirmations[1].click()
        self.assertEqual(session.calls, [])
        acknowledgement = next(node for node in walk(root) if node.kind == "Checkbox")
        acknowledgement.change(True)
        confirmations[1].click()
        self.assertEqual(session.calls, [("confirm", "second-candidate", "CONFIRM second-candidate")])
        self.assertEqual(session.data["records"][0]["lifecycle_status"], "candidate")

    def test_candidate_can_be_revised_before_any_confirmation(self) -> None:
        session = SessionStub()
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        editors = [node for node in walk(root) if node.kind == "Textarea"]
        editors[2].value = "I might try walking; this is not a committed weekly target."
        confirmations = [node for node in walk(root) if node.description == "确认这条记忆"]
        confirmations[1].click()
        self.assertEqual(session.calls, [])
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("请先保存修订", rendered)
        revisions = [node for node in walk(root) if node.description == "保存修订，保持待确认"]
        revisions[1].click()
        self.assertEqual(session.calls, [("revise", "second-candidate", editors[2].value)])
        self.assertEqual(session.data["records"][1]["lifecycle_status"], "candidate")
        self.assertEqual(session.data["records"][0]["content"], "Walk after lunch.")

    def test_complex_slot_without_revision_support_explains_why_editing_is_unavailable(self) -> None:
        session = SessionStub()
        candidate = session.data["records"][0]
        candidate["memory_type"] = "emotion_state"
        candidate["candidate_revision_allowed"] = False
        candidate["candidate_revision_reason"] = "这条候选包含不能仅靠文字安全修改的结构字段。请删除后重新提供明确信息。"
        view = ui.build_view(session.snapshot())["record_views"][0]
        self.assertFalse(view["can_revise"])
        self.assertTrue(view["can_confirm"])
        root = self.launch_stub(session)
        revisions = [node for node in walk(root) if node.description == "保存修订，保持待确认"]
        self.assertEqual(len(revisions), 1)  # Only the second plain candidate.
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("不能仅靠文字安全修改", rendered)
        self.assertEqual(session.calls, [])
        candidate.pop("candidate_revision_allowed")
        candidate["structured_payload"] = {"candidate_revision_allowed": True, "slot_key": "emotion.current"}
        self.assertFalse(ui.build_view(session.snapshot())["record_views"][0]["can_revise"])

    def test_real_session_ui_revision_recall_and_delete_are_explicit(self) -> None:
        # Exercise the real service and widgets' event path with the offline
        # template backend. Passing this is software integration, not a model
        # quality result or a user study.
        session = DialogueSession()
        self.addCleanup(session.close)
        session.send_message("我这周想试试午饭后散步。")
        record = session.extract_current_session()[0]
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        revised = "我只是想试试午饭后散步，还没有决定每周次数。"
        [node for node in walk(root) if node.kind == "Textarea"][1].value = revised
        next(node for node in walk(root) if node.description == "保存修订，保持待确认").click()
        self.assertEqual(session.snapshot()["records"][0]["lifecycle_status"], "candidate")
        self.assertEqual(session.snapshot()["records"][0]["content"], revised)
        next(node for node in walk(root) if node.description == "确认这条记忆").click()
        next(node for node in walk(root) if node.description == "开始下一次会话").click()
        next(node for node in walk(root) if node.kind == "Textarea").value = "我午饭后散步的目标是什么？"
        next(node for node in walk(root) if node.description == "发送消息").click()
        reply = session.snapshot()["last_reply"]
        self.assertEqual(reply["memory_ids"], [record["memory_id"]])
        self.assertIn(revised, ui.render_trace_html(session.snapshot()))
        next(node for node in walk(root) if node.description == "删除此条记忆").click()
        self.assertEqual(len(session.snapshot()["records"]), 1)
        next(node for node in walk(root) if node.description == "取消，保留当前内容").click()
        self.assertEqual(session.snapshot()["last_reply"], reply)
        next(node for node in walk(root) if node.description == "删除此条记忆").click()
        next(node for node in walk(root) if node.description == "确认删除并重置当前聊天").click()
        self.assertEqual(session.snapshot()["records"], [])
        self.assertEqual(session.snapshot()["turns"], [])
        self.assertIsNone(session.snapshot()["last_reply"])

    def test_delete_explains_scope_and_allows_cancel_before_state_change(self) -> None:
        session = SessionStub()
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        before = copy.deepcopy(session.data)
        next(node for node in walk(root) if node.description == "删除此条记忆").click()
        self.assertEqual(session.calls, [])
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("重置当前聊天上下文和上一条回答", rendered)
        self.assertIn("其他长期记忆不受影响", rendered)
        self.assertTrue(next(node for node in walk(root) if node.description == "发送消息").disabled)
        next(node for node in walk(root) if node.description == "取消，保留当前内容").click()
        self.assertEqual(session.data, before)
        self.assertEqual(session.calls, [])
        next(node for node in walk(root) if node.description == "删除此条记忆").click()
        next(node for node in walk(root) if node.description == "确认删除并重置当前聊天").click()
        self.assertEqual(session.calls, [("delete", "fixture-goal", "DELETE fixture-goal")])
        self.assertEqual([record["memory_id"] for record in session.data["records"]], ["second-candidate"])

    def test_switch_with_pending_input_requires_a_separate_discard_choice(self) -> None:
        session = SessionStub()
        session.data["extraction_scope"] = {"pending_user_turn_count": 3}
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        next(node for node in walk(root) if node.description == "开始下一次会话").click()
        self.assertEqual(session.calls, [])
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("还有 3 条用户发言未整理", rendered)
        self.assertIn("不会自动保存", rendered)
        next(node for node in walk(root) if node.description == "取消，保留当前内容").click()
        self.assertTrue(session.data["turns"])
        self.assertEqual(session.calls, [])
        next(node for node in walk(root) if node.description == "开始下一次会话").click()
        next(node for node in walk(root) if node.description == "放弃未整理内容并切换会话").click()
        self.assertEqual(session.calls, [("new_session",)])

    def test_switch_without_pending_input_does_not_add_an_empty_warning(self) -> None:
        session = SessionStub()
        session.data["extraction_scope"] = {"pending_user_turn_count": 0}
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        next(node for node in walk(root) if node.description == "开始下一次会话").click()
        self.assertEqual(session.calls, [("new_session",)])

    def test_provider_error_details_are_not_echoed_into_notebook_output(self) -> None:
        session = SessionStub()
        session.fail_with = "private-api-key-value and raw-provider-body"
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        next(node for node in walk(root) if node.kind == "Textarea").value = "A fictional message."
        next(node for node in walk(root) if node.description == "发送消息").click()
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("RuntimeError", rendered)
        self.assertNotIn("private-api-key-value", rendered)
        self.assertNotIn("raw-provider-body", rendered)

    def test_safety_pause_disables_normal_dialogue_but_keeps_management_visible(self) -> None:
        session = SessionStub()
        session.data["safety_paused"] = True
        session.data["last_reply"] = {"memory_ids": [], "safety_bypass": True}
        root = self.launch_stub(session)
        next(node for node in walk(root) if node.kind == "Checkbox").change(True)
        self.assertTrue(next(node for node in walk(root) if node.description == "发送消息").disabled)
        self.assertTrue(next(node for node in walk(root) if node.description == "整理当前窗口的记忆候选").disabled)
        self.assertFalse(next(node for node in walk(root) if node.description == "开始下一次会话").disabled)
        self.assertFalse(next(node for node in walk(root) if node.description == "删除此条记忆").disabled)
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("不表示风险已解除", rendered)


if __name__ == "__main__":
    unittest.main()
