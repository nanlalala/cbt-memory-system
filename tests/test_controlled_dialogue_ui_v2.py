from __future__ import annotations

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dialogue_ui as support

ui = support.ui
ROOT = Path(__file__).resolve().parents[1]


class ControlledStub(support.SessionStub):
    def __init__(self):
        super().__init__()
        self.default_permissions = {key: True for key in ("save", "long_term_use", "response_external", "extraction_external")}
        self.data.update(prototype_version="controlled-memory-v2.0", task_states=[], turns=[],
                         memory_mode="conversational", automatic_memory_enabled=True)
        self.reply = {"text": "An actual fixture answer", "source_commit": {"source_status": "committed", "processing_status": "queued"},
                      "answer_status": "offline_software_demo", "organizing_status": "queued", "response_retained": True,
                      "usage": None}
        self.worker_result = [{"status": "completed"}]

    def send_message(self, text, *, permissions, input_id=None):
        self.calls.append(("send", permissions))
        result = copy.deepcopy(self.reply)
        if "别保存" in text or not permissions["save"]:
            result.update(text="本轮私密回复", response_retained=False,
                          source_commit={"source_status": "not_saved", "processing_status": "not_queued"})
            self.data["last_reply"] = {"text": "本轮消息与回复未保留。", "source_commit": result["source_commit"],
                                       "answer_status": result["answer_status"]}
        else:
            self.data["last_reply"] = copy.deepcopy(result)
        return result

    def run_pending(self):
        self.calls.append(("run_pending",))
        return copy.deepcopy(self.worker_result)

    def delete_content(self, scope, target_id, **kwargs):
        self.calls.append(("delete_content", scope, target_id, kwargs))
        return {"logical_blocked": True, "control_status": "withdrawn", "scope": scope,
                "replica_cleanup": "not_claimed_external_copies_not_controlled"}


class ControlledUITests(unittest.TestCase):
    def launch(self, session):
        modules = {"ipywidgets": support.fake_widgets(), "IPython.display": SimpleNamespace(display=lambda _: None)}
        with patch.object(ui, "_require_colab"), patch.object(ui.importlib, "import_module", side_effect=lambda name: modules[name]):
            root = ui.launch_notebook_demo(session=session)
        next(w for w in support.walk(root) if w.kind == "Checkbox").change(True)
        return root

    def button(self, root, name):
        return next(w for w in support.walk(root) if w.description == name)

    def rendered(self, root):
        return "\n".join(str(w.value) for w in support.walk(root) if w.kind == "HTML")

    def send(self, root, text):
        next(w for w in support.walk(root) if w.kind == "Textarea" and w.description == "消息").value = text
        self.button(root, "发送消息").click()

    def test_manual_retry_after_ui_exception_reuses_same_input_identity(self):
        session = ControlledStub()
        attempted = []
        real_send = session.send_message
        def failing_once(text, *, permissions, input_id):
            attempted.append(input_id)
            if len(attempted) == 1:
                raise RuntimeError('fictional interruption after admission')
            return real_send(text, permissions=permissions, input_id=input_id)
        session.send_message = failing_once
        root = self.launch(session)
        self.send(root, 'fictional identical input')
        self.button(root, '发送消息').click()
        self.assertEqual(len(attempted), 2)
        self.assertEqual(attempted[0], attempted[1])
        self.send(root, 'a new fictional input')
        self.assertNotEqual(attempted[1], attempted[2])

    def test_separate_receipts_do_not_turn_answer_failure_into_success(self):
        session = ControlledStub()
        session.reply["answer_status"] = "failed_or_unknown"
        root = self.launch(session)
        self.send(root, "A fictional task")
        html = self.rendered(root)
        self.assertIn("来源已提交", html)
        self.assertIn("回答失败或结果未知", html)
        self.assertIn("已排队，尚未取得完成回执", html)
        self.assertNotIn("回答已完成", html)
        self.assertNotIn("已回复。历史记录会自动处理", html)

    def test_source_failure_is_visible_even_when_answer_completed(self):
        session = ControlledStub()
        session.reply.update(answer_status="succeeded", source_commit={"source_status": "failed", "processing_status": "not_queued"})
        root = self.launch(session)
        self.send(root, "A fictional task")
        html = self.rendered(root)
        self.assertIn("来源提交失败", html)
        self.assertIn("回答已完成", html)
        self.assertNotIn("来源已提交", html)

    def test_no_save_text_is_only_transient_widget_and_cleared_on_next_operation(self):
        session = ControlledStub()
        root = self.launch(session)
        private = "本轮PRIVATE_TOKEN这条别保存"
        self.send(root, private)
        self.assertIn(private, self.rendered(root))
        self.assertIn("本轮私密回复", self.rendered(root))
        self.assertNotIn(private, json.dumps(session.snapshot(), ensure_ascii=False))
        self.button(root, "刷新状态").click()
        self.assertNotIn(private, self.rendered(root))
        self.assertNotIn("本轮私密回复", self.rendered(root))

    def test_finite_worker_failure_has_its_own_receipt_and_no_retry_loop(self):
        session = ControlledStub()
        session.worker_result = [{"status": "failed"}]
        session.data["last_reply"] = copy.deepcopy(session.reply)
        root = self.launch(session)
        self.button(root, "查看并运行有限整理").click()
        self.assertEqual(session.calls, [("run_pending",)])
        self.assertIn("整理失败", self.rendered(root))
        self.assertNotIn("整理批次完成", self.rendered(root))

    def test_risk_pause_blocks_normal_send_but_keeps_local_withdrawal(self):
        session = ControlledStub()
        session.data["safety_paused"] = True
        root = self.launch(session)
        self.assertTrue(self.button(root, "发送消息").disabled)
        self.assertTrue(self.button(root, "查看并运行有限整理").disabled)
        self.assertFalse(self.button(root, "执行本地撤回").disabled)
        next(w for w in support.walk(root) if w.description == "对象编号").value = "source:existing"
        self.button(root, "执行本地撤回").click()
        self.assertEqual(session.calls, [("delete_content", "source", "source:existing", {})])
        self.assertTrue(session.snapshot()["safety_paused"])

    def test_task_axes_do_not_replace_known_progress_with_pause_or_unknown_outcome(self):
        state = {"task_instance_id": "t", "period_id": "2026-11-02", "state_version": 4,
                 "acceptance": "agreed", "participation": "paused", "outcome": "unknown", "due_review": True,
                 "progress": {"reported_count": 1, "target_count": 2}}
        html = ui.render_task_states_html([state])
        for expected in ("agreed", "paused", "已报告次数：1", "目标次数：2", "最终反馈：未知", "待复盘：True"):
            self.assertIn(expected, html)

    def test_escaped_receipts_and_unknown_usage_do_not_echo_provider_errors(self):
        reply = {"source_commit": {"source_status": "failed", "error_type": "PRIVATE PROVIDER BODY"},
                 "answer_status": "failed_or_unknown", "usage": None}
        html = ui.render_operation_receipts(reply)
        self.assertIn("未知，未填为0", html)
        self.assertNotIn("PRIVATE PROVIDER BODY", html)
        self.assertNotIn("来源已提交", html)

    def test_actual_controlled_file_session_private_input_does_not_enter_next_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            session = ui._new_session(database=Path(directory) / "prototype.sqlite", approved_root=directory,
                                      user_id="fictional-user", run_id="ui-test", background=False)
            try:
                root = self.launch(session)
                private = "我想这周暂停，这条别保存。UI_PRIVATE_BODY"
                self.send(root, private)
                self.assertIn(private, self.rendered(root))
                self.assertNotIn("UI_PRIVATE_BODY", json.dumps(session.snapshot(), ensure_ascii=False))
                self.assertNotIn("UI_PRIVATE_BODY", session.store.connection.execute("SELECT group_concat(text,' ') FROM dm_sources").fetchone()[0] or "")
                self.button(root, "刷新状态").click()
                self.assertNotIn("UI_PRIVATE_BODY", self.rendered(root))
            finally:
                session.close()


class ControlledNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("controlled_notebook_builder", ROOT / "tools/build_controlled_memory_notebook.py")
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_unreleased_entry_has_explicit_placeholder_and_no_execution_outputs(self):
        notebook = self.builder.build()
        self.assertEqual(notebook["metadata"]["controlled_memory_v2"]["executable_revision"], self.builder.UNPINNED)
        sources = "\n".join(c["source"] for c in notebook["cells"])
        self.assertIn("ENABLE_PAID_MODEL = False", sources)
        self.assertIn("COLAB_SECRET_NAME = 'Capstone'", sources)
        self.assertIn("APPROVED_STORAGE_ROOT = ''", sources)
        self.assertIn("DATABASE_FILE = ''", sources)
        self.assertIn("USER_ID = ''", sources)
        self.assertIn("RUN_ID = ''", sources)
        self.assertIn("ControlledDialogueSession.open", sources)
        self.assertNotIn("drive.mount", sources)
        self.assertNotIn("pip install", sources)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_immutable_pin_is_required_and_saved_notebook_matches_builder(self):
        pinned = self.builder.build("a" * 40)
        self.assertEqual(pinned["metadata"]["controlled_memory_v2"]["executable_revision"], "a" * 40)
        with self.assertRaises(ValueError):
            self.builder.build("main")
        current = json.loads((ROOT / "app/Controlled_Memory_V2_Colab.ipynb").read_text())
        revision = current["metadata"]["controlled_memory_v2"]["executable_revision"]
        self.assertEqual(current, self.builder.build(revision))


if __name__ == '__main__':
    unittest.main()
