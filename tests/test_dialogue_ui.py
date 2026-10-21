from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
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

    def send_message(self, text):
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        self.calls.append(("send", text))

    def new_session(self):
        self.calls.append(("new_session",))


class NotebookDialogueUITests(unittest.TestCase):
    def launch_stub(self, session):
        modules = {"ipywidgets": fake_widgets(), "IPython.display": SimpleNamespace(display=lambda _: None)}
        with patch.object(ui, "_require_colab"), patch.object(ui.importlib, "import_module", side_effect=lambda name: modules[name]):
            return ui.launch_notebook_demo(session)

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

    def test_candidate_and_active_controls_have_separate_roles(self) -> None:
        data = snapshot()
        original = copy.deepcopy(data)
        view = ui.build_view(data)
        self.assertTrue(view["record_views"][0]["can_confirm"])
        self.assertFalse(view["record_views"][0]["can_correct"])
        self.assertEqual(data, original)
        data["records"][0]["lifecycle_status"] = "active"
        active = ui.build_view(data)["record_views"][0]
        self.assertFalse(active["can_confirm"])
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
        self.assertTrue(next(node for node in walk(root) if node.description == "整理本次记忆候选").disabled)
        self.assertFalse(next(node for node in walk(root) if node.description == "开始下一次会话").disabled)
        self.assertFalse(next(node for node in walk(root) if node.description == "删除此条记忆").disabled)
        rendered = "\n".join(str(node.value) for node in walk(root) if node.kind == "HTML")
        self.assertIn("不表示风险已解除", rendered)


if __name__ == "__main__":
    unittest.main()
