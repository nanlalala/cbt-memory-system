"""Colab notebook controls for the structured-memory dialogue demonstration.

Importing this module uses the standard library only.  Widgets and the dialogue
service are loaded when ``launch_notebook_demo`` is called inside Colab.  This
module does not load models, make network requests or start a web server.
"""

from __future__ import annotations

import importlib
import sys
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEMO_KIND = "synthetic_research_dialogue_demo"
OFFLINE_MODE = "offline_verbatim_demo"
STATE_LABELS = {
    "candidate": "待确认",
    "active": "已启用",
    "superseded": "旧版本",
    "expired": "已到期",
    "deleted": "已删除",
}


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _paragraph(value: object) -> str:
    return '<div style="white-space:pre-wrap;overflow-wrap:anywhere">' + _text(value) + "</div>"


def validate_snapshot(snapshot: dict[str, Any]) -> None:
    """Accept only the explicitly fictional software-demo view contract."""
    if not isinstance(snapshot, dict) or snapshot.get("kind") != DEMO_KIND:
        raise ValueError("A synthetic dialogue-demo snapshot is required")
    if snapshot.get("clinical_or_benchmark_result") is not False:
        raise ValueError("The notebook demonstration is not an outcome result")
    if not isinstance(snapshot.get("records"), list) or not isinstance(snapshot.get("turns"), list):
        raise ValueError("Snapshot records and turns must be lists")
    if not isinstance(snapshot.get("backend_mode"), str) or not snapshot["backend_mode"]:
        raise ValueError("Snapshot backend mode must be explicit")
    if not isinstance(snapshot.get("live_model_or_network_used"), bool):
        raise ValueError("Snapshot actual execution flag must be boolean")


def render_chat_html(snapshot: dict[str, Any]) -> str:
    validate_snapshot(snapshot)
    blocks = []
    for turn in snapshot["turns"]:
        role = {"user": "你", "assistant": "Agent"}.get(str(turn.get("role")), "记录")
        blocks.append(
            '<div style="border-left:3px solid #cbd5e1;padding:6px 10px;margin:8px 0">'
            f"<strong>{_text(role)}</strong>{_paragraph(turn.get('text', ''))}</div>"
        )
    # Safety bypass deliberately keeps risk text out of the ordinary turn
    # buffer. Its actual response must still be visible in the chat panel.
    reply = snapshot.get("last_reply") or {}
    if reply.get("safety_bypass") is True and reply.get("text"):
        blocks.append(
            '<div style="border-left:3px solid #b45309;padding:6px 10px;margin:8px 0">'
            f"<strong>安全提示</strong>{_paragraph(reply['text'])}</div>"
        )
    return "".join(blocks) if blocks else "<p>当前会话还没有消息。可以先输入一条虚构的目标或经历。</p>"


def render_record_html(record: dict[str, Any], *, conversational: bool = False) -> str:
    """Escape all source/model-controlled text; never render it as markup."""
    source = record.get("source_evidence")
    source = source if isinstance(source, dict) else {}
    status = str(record.get("lifecycle_status", ""))
    source_ids = source.get("turn_ids")
    turn_ids = ", ".join(str(value) for value in source_ids) if isinstance(source_ids, (list, tuple)) else ""
    # This presentation-only field is derived by DialogueSession from the
    # matching operator audit event. Never read an alleged revision or its
    # "verified" flag from model-controlled structured_payload.
    original = record.get("verified_candidate_revision_source")
    original = original if isinstance(original, dict) else {}
    original_html = (
        '<details><summary>修订前的来源原话</summary>'
        + _paragraph(original.get("supporting_text", "")) + "</details>"
        if original else ""
    )
    label = STATE_LABELS.get(status, status)
    trust_html = ""
    if conversational and status == "candidate":
        decision = record.get("memory_use_decision")
        decision = decision if isinstance(decision, dict) else {}
        action = decision.get("action")
        label = (
            "AI 推断 · 未确认" if record.get("origin") == "agent_inference" else
            "AI 整理摘要 · 未核验，仅原话可参考" if decision.get("evidence_label") == "verified_source_only" else
            "历史自报 · 不代表现在仍如此" if action == "use_as_history" else
            "相关时核对" if action == "check_current" else "待核对记录"
        )
        trust_html = "<small>保存不等于确认；时效与来源会影响这条信息能否用于当前对话。</small>"
    return (
        '<div style="padding:8px 0">'
        f"<strong>{_text(label)}</strong>"
        + _paragraph(record.get("content", ""))
        + trust_html
        + '<details><summary>查看原话与来源</summary>'
        + _paragraph(source.get("supporting_text", ""))
        + f"<small>会话：{_text(source.get('session_id', ''))}；原话 ID：{_text(turn_ids)}</small>"
        + "</details>"
        + original_html
        + f"<small>记忆 ID：{_text(record.get('memory_id', ''))}</small></div>"
    )


def render_trace_html(snapshot: dict[str, Any]) -> str:
    validate_snapshot(snapshot)
    reply = snapshot.get("last_reply")
    if not reply:
        return "<p>发送消息后，这里会列出真正进入本次回答上下文的记忆。</p>"
    ids = tuple(str(value) for value in reply.get("memory_ids", ()))
    history_ids = tuple(str(value) for value in reply.get("history_memory_ids", ()))
    dropped = tuple(str(value) for value in reply.get("dropped_memory_ids", ()))
    # Show only sources actually included by the service. Never look up dropped
    # text from the memory inventory or infer answer grounding from inclusion.
    restricted = {str(record.get("memory_id")) for record in snapshot["records"]
                  if record.get("sensitivity") == "restricted"}
    sources = {str(source.get("memory_id")): source for source in reply.get("sources", ())
               if isinstance(source, dict) and source.get("sensitivity") != "restricted"}
    included = []
    for memory_id in ids:
        if snapshot.get("memory_mode") == "conversational" and memory_id in history_ids:
            continue
        if memory_id in restricted:
            continue
        source = sources.get(memory_id, {})
        text = source.get("content") or source.get("supporting_text")
        included.append(
            '<li>' + (_paragraph(text) if text else "内容未提供")
            + '<details><summary>来源编号</summary>' + _paragraph(memory_id) + "</details></li>"
        )
    included_html = "<ul>" + "".join(included) + "</ul>" if included else "<p>无。</p>"
    # Conversational observations are historical quotations, not silently
    # promoted confirmed/current facts. Their trace remains a separate section.
    historical = []
    history_sources = {str(source.get("memory_id")): source for source in reply.get("history_sources", ())
                       if isinstance(source, dict) and source.get("sensitivity") != "restricted"}
    for memory_id in history_ids:
        if memory_id in restricted:
            continue
        source = history_sources.get(memory_id)
        if source is None:
            continue
        historical.append('<li>' + _paragraph(source.get("content") or source.get("supporting_text", "")) + "</li>")
    historical_html = (
        "<p><strong>提供给本次回答的历史自报：</strong></p><ul>" + "".join(historical)
        + "</ul><small>这说明你过去说过这些话，不代表目前仍然适用。</small>"
        if historical else ""
    )
    checks = []
    for check in reply.get("memory_checks", ()):
        if not isinstance(check, dict) or str(check.get("memory_id")) in restricted:
            continue
        if check.get("sensitivity") == "restricted" or not check.get("question"):
            continue
        checks.append('<li>' + _paragraph(check["question"]) + "</li>")
    check_html = (
        "<p><strong>本次为了解当前情况而核对：</strong></p><ul>" + "".join(checks)
        + "</ul><small>这是一次询问；不回答不会被当成确认。</small>"
        if checks else ""
    )
    dropped_html = "、".join(_text(value) for value in dropped) or "无"
    safety = "<p>本次进入演示安全分流，未走普通回答流程。</p>" if reply.get("safety_bypass") else ""
    ordinary_html = (
        f"<p><strong>送入本次回答上下文的长期记忆：</strong></p>{included_html}"
        if included or not historical_html else ""
    )
    return (
        safety
        + ordinary_html
        + historical_html
        + check_html
        + f"<p>因上下文预算未放入的记忆：{dropped_html}</p>"
        + "<small>这只说明信息已提供给模型；回答是否确实依据它、使用是否正确，尚未核验。"
        "未列入不代表已删除。</small>"
    )


def render_extraction_scope_html(snapshot: dict[str, Any]) -> str:
    """Explain the bounded extraction input without claiming full-session recall."""
    scope = snapshot.get("extraction_scope") or {}
    retained = scope.get("retained_user_turn_count")
    if type(retained) is not int:
        retained = sum(turn.get("role") == "user" for turn in snapshot["turns"])
    pending = scope.get("pending_retained_user_turn_count")
    if type(pending) is not int:
        pending = retained
    summary = (
        f"<p><strong>本次整理范围：</strong>窗口内 {pending} 条未整理用户发言。"
        f"当前窗口共保留 {retained} 条用户发言；其余已整理发言只作为理解上下文，"
        "不会再次生成旧候选。"
        "只生成待核对候选，不会自动确认，也不保证覆盖整次谈话。</p>"
    )
    if scope:
        summary += (
            f"<p>本会话累计 {_text(scope.get('total_user_turn_count', retained))} 条用户发言；"
            f"{_text(scope.get('dropped_user_turn_count', 0))} 条已移出窗口，不在此次整理范围。"
            f"尚未整理 {_text(scope.get('pending_user_turn_count', 0))} 条，"
            f"其中 {_text(scope.get('pending_retained_user_turn_count', 0))} 条仍在窗口内。</p>"
        )
        if scope.get("partial_coverage"):
            summary += "<p><strong>当前窗口不能覆盖本次会话开头的全部内容。</strong>已移出的原话不会被此按钮重新读取。</p>"
    return summary


def render_memory_mode_html(snapshot: dict[str, Any]) -> str:
    """Explain actual software behavior without presenting confidence as truth."""
    if snapshot.get("memory_mode") != "conversational":
        return "<p>审核模式：整理候选后，由你选择是否启用。可在独立会话中体验自动记忆模式。</p>"
    enabled = snapshot.get("automatic_memory_enabled") is True
    text = (
        "自动记录已开启：成功发送的、适合保存的用户原话会自动留下历史记录，"
        "不用逐条整理确认。相关但过时或不确定的信息，会在使用时自然询问。"
        if enabled else
        "自动记录已关闭：后续消息不会自动写入长期记录。已有记录仍保留，可在记忆管理中删除。"
    )
    status = snapshot.get("memory_processing")
    status = status if isinstance(status, dict) else {}
    if status.get("status") in {"failed", "error", "auto_write_failed"}:
        text += " 本次自动记录未完成，已成功的聊天回答仍保留。"
    return _paragraph(text) + "<small>目前保存的是你的原话，不会自动把它当作已经确认的目标或结论。历史情况也可能变化。</small>"


def _candidate_revision_policy(record: dict[str, Any]) -> tuple[bool, str]:
    """Use the service's presentation policy; old snapshots fail conservatively."""
    if record.get("lifecycle_status") != "candidate":
        return False, ""
    allowed = record.get("candidate_revision_allowed")
    if type(allowed) is bool:
        return allowed, str(record.get("candidate_revision_reason") or "")
    # Compatibility is limited to old plain-fact/correction snapshots. Complex
    # structured records need the service's explicit policy, not a UI guess.
    if record.get("memory_type") in {"explicit_fact", "user_correction"}:
        return True, ""
    return False, "这条候选的结构字段不能在此直接安全修订。请删除后重新提供明确信息。"


def build_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Build a widget-independent presentation for lightweight regression tests."""
    validate_snapshot(snapshot)
    backend_mode = snapshot["backend_mode"]
    conversational = snapshot.get("memory_mode") == "conversational"
    execution = snapshot.get("provider_execution") or {}
    if execution.get("last_call_status") == "attempted_failed":
        execution_state = "attempted_failed"
        mode_label = "请求已尝试 · 本次失败"
        mode_detail = (
            "本次外部模型请求已尝试，但没有得到有效结果；失败请求仍可能产生费用。"
            "已有成功对话不会被失败内容替换。"
            if execution.get("last_call_external", execution.get("external_request_attempted")) is True else
            "已尝试调用配置的后端，但没有得到有效结果；这不表示发生了模型或网络调用。"
        )
    elif snapshot["live_model_or_network_used"]:
        execution_state = "live_executed"
        mode_label = "Colab 模型请求已执行"
        mode_detail = "此演示已调用外部模型服务；后续发送或整理可能继续产生请求，仅使用虚构资料。"
    elif backend_mode in {OFFLINE_MODE, "injected_offline_test_backend"}:
        execution_state = "offline"
        mode_label = (
            "离线演示 · 原文候选与模板回答"
            if backend_mode == OFFLINE_MODE else "离线测试后端"
        )
        mode_detail = (
            "没有调用语言模型或网络；用于检查流程。"
            if backend_mode == OFFLINE_MODE else
            "使用注入的离线测试结果；没有记录到实际模型或网络调用。"
        )
    else:
        execution_state = "configured_not_called"
        mode_label = "后端已配置 · 尚未执行"
        mode_detail = "尚未记录模型或网络调用；发送消息或整理候选时可能调用已配置的模型服务，仅使用虚构资料。"
    return {
        "mode_label": mode_label,
        "mode_detail": mode_detail,
        "execution_state": execution_state,
        "chat_html": render_chat_html(snapshot),
        "trace_html": render_trace_html(snapshot),
        "extraction_scope_html": render_extraction_scope_html(snapshot),
        "memory_mode": "conversational" if conversational else "review",
        "memory_mode_html": render_memory_mode_html(snapshot),
        "automatic_memory_enabled": conversational and snapshot.get("automatic_memory_enabled") is True,
        "safety_paused": snapshot.get("safety_paused") is True,
        "record_views": [
            {"memory_id": str(record.get("memory_id", "")), "html": render_record_html(record, conversational=conversational),
             "can_confirm": record.get("lifecycle_status") == "candidate",
             "can_revise": _candidate_revision_policy(record)[0],
             "revision_reason": _candidate_revision_policy(record)[1],
             "content": str(record.get("content", "")),
             "can_correct": record.get("lifecycle_status") == "active"}
            for record in snapshot["records"]
        ],
        "clinical_or_benchmark_result": False,
    }


def _require_colab() -> None:
    try:
        importlib.import_module("google.colab")
    except ImportError as exc:
        raise RuntimeError("Open this demonstration in Google Colab; no local server is provided") from exc


def _new_session() -> Any:
    for relative in (
        "modules/agent-integration", "modules/long-term-memory/schema",
        "modules/long-term-memory/storage", "modules/long-term-memory/lifecycle",
        "modules/long-term-memory/extraction", "modules/long-term-memory/retrieval",
    ):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    return importlib.import_module("dialogue_service").DialogueSession(memory_mode="conversational")


def launch_notebook_demo(session: Any = None) -> Any:
    """Display Colab-only controls and return their root widget.

    A caller may pass a Colab-configured ``DialogueSession(provider=...)``.  No
    provider is created or invoked while rendering the UI.  Every model call
    requires a separate send/extract click after the fictional-data checkbox.
    The product default stores source-bound observations automatically. Saving
    does not call confirmation. Review mode remains available for controlled
    comparisons and advanced item management.
    """
    _require_colab()
    widgets = importlib.import_module("ipywidgets")
    display = importlib.import_module("IPython.display").display
    session = _new_session() if session is None else session
    validate_snapshot(session.snapshot())
    initial_view = build_view(session.snapshot())
    conversational = initial_view["memory_mode"] == "conversational"

    title = widgets.HTML("<h3>CBT Memory System · 记忆对话演示</h3>")
    mode = widgets.HTML()
    paused_notice = widgets.HTML()
    notice = widgets.HTML(
        "<p>使用虚构资料体验跨会话记忆、更正和删除。默认数据只在当前 Colab 运行时保存。"
        "这是软件演示，不提供诊断或治疗。</p>"
    )
    acknowledge = widgets.Checkbox(value=False, description="我只输入虚构演示资料", indent=False)
    automatic_memory = widgets.Checkbox(
        value=initial_view["automatic_memory_enabled"], description="自动记录对话中的历史信息", indent=False,
    )
    memory_mode_notice = widgets.HTML()
    chat = widgets.HTML()
    message = widgets.Textarea(
        value="", placeholder="例如：本周的目标是午饭后散步两次。", description="消息",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    send = widgets.Button(description="发送消息", button_style="primary", disabled=True)
    extract = widgets.Button(description="整理当前窗口的记忆候选", disabled=True)
    next_session = widgets.Button(description="开始下一次会话", disabled=True)
    status = widgets.HTML("<p>请先勾选虚构资料声明。</p>")
    memories = widgets.VBox()
    extraction_scope = widgets.HTML()
    decision_panel = widgets.VBox()
    trace = widgets.HTML()
    trace_panel = widgets.Accordion(children=[trace])
    trace_panel.set_title(0, "本次回答收到的记忆参考")
    trace_panel.selected_index = None
    state = {"busy": False, "safety_paused": False, "decision_pending": False, "refreshing_controls": False}
    main_actions = [send, next_session] + ([] if conversational else [extract])
    item_actions: list[Any] = []
    decision_actions: list[Any] = []

    def update_enabled() -> None:
        enabled = acknowledge.value and not state["busy"] and not state["decision_pending"]
        for button in main_actions + item_actions:
            button.disabled = not enabled
        if state["safety_paused"]:
            send.disabled = True
            extract.disabled = True
        message.disabled = not enabled or state["safety_paused"]
        acknowledge.disabled = state["busy"] or state["decision_pending"]
        automatic_memory.disabled = not enabled or state["safety_paused"]
        if conversational:
            extract.disabled = not enabled or state["safety_paused"]
        for button in decision_actions:
            button.disabled = state["busy"] or not acknowledge.value

    def tell(text: str) -> None:
        status.value = _paragraph(text)

    def perform(action: Any, success: str) -> None:
        if not acknowledge.value or state["busy"]:
            return
        state["busy"] = True
        update_enabled()
        try:
            action()
            tell(success)
        except Exception as exc:
            # Provider errors may contain URLs, response bodies or secrets.  Keep
            # their messages and tracebacks out of notebook output and the UI.
            tell(f"操作未完成（{type(exc).__name__}）。请检查输入或模型运行状态；已有记忆仍可查看。")
        finally:
            state["busy"] = False
            refresh()

    def ask_before(action: Any, warning: str, success: str, *, confirm_label: str) -> None:
        if not acknowledge.value or state["busy"] or state["decision_pending"]:
            return
        state["decision_pending"] = True
        proceed = widgets.Button(description=confirm_label, button_style="danger")
        cancel = widgets.Button(description="取消，保留当前内容")
        decision_actions[:] = [proceed, cancel]

        def clear() -> None:
            for button in decision_actions:
                button.disabled = True
            state["decision_pending"] = False
            decision_panel.children = ()
            decision_actions.clear()

        def accept(_: Any) -> None:
            if not state["decision_pending"] or state["busy"]:
                return
            clear()
            perform(action, success)

        def reject(_: Any) -> None:
            if not state["decision_pending"] or state["busy"]:
                return
            clear()
            tell("已取消，当前聊天和记忆保持不变。")
            update_enabled()

        proceed.on_click(accept)
        cancel.on_click(reject)
        decision_panel.children = (widgets.HTML(_paragraph(warning)), widgets.HBox([proceed, cancel]))
        update_enabled()

    def refresh() -> None:
        view = build_view(session.snapshot())
        mode.value = f"<strong>{_text(view['mode_label'])}</strong><p>{_text(view['mode_detail'])}</p>"
        state["safety_paused"] = view["safety_paused"]
        paused_notice.value = (
            "<p><strong>演示已暂停普通回答和记忆整理。</strong>当前需要现实中的人工支持。"
            "“开始下一次会话”只重置演示状态，不表示风险已解除。</p>"
            if view["safety_paused"] else ""
        )
        chat.value = view["chat_html"]
        trace.value = view["trace_html"]
        memory_mode_notice.value = view["memory_mode_html"]
        if automatic_memory.value != view["automatic_memory_enabled"]:
            state["refreshing_controls"] = True
            automatic_memory.value = view["automatic_memory_enabled"]
            state["refreshing_controls"] = False
        extraction_scope.value = view["extraction_scope_html"]
        cards = []
        item_actions.clear()
        for record_view in view["record_views"]:
            memory_id = record_view["memory_id"]
            parts = [widgets.HTML(record_view["html"])]
            buttons = []
            if record_view["can_confirm"]:
                revision_text = None
                if record_view["can_revise"]:
                    revision_text = widgets.Textarea(
                        value=record_view["content"], placeholder="先修订，再决定是否保存",
                        layout=widgets.Layout(width="100%", height="65px"),
                    )
                    revise = widgets.Button(description="保存修订，保持待确认")
                    revise.on_click(lambda _, mid=memory_id, field=revision_text: perform(
                        lambda: session.revise_candidate(mid, field.value),
                        "已按你写的原话修订，仍然待确认。核对无误后再确认；这次修订没有加入普通聊天。",
                    ))
                    parts.append(revision_text)
                    buttons.append(revise)
                    item_actions.append(revise)
                else:
                    parts.append(widgets.HTML(_paragraph(record_view["revision_reason"])))
                confirm = widgets.Button(description="确认这条记忆", button_style="success")

                def confirm_clicked(_: Any, mid=memory_id, field=revision_text,
                                    original=record_view["content"]) -> None:
                    if field is not None and field.value.strip() != original:
                        tell("文字尚未保存。请先保存修订并核对新候选，再单独确认。")
                        return
                    perform(lambda: session.confirm(mid, f"CONFIRM {mid}"),
                            "已确认。后续会话可检索这条记忆。")

                confirm.on_click(confirm_clicked)
                buttons.append(confirm)
                item_actions.append(confirm)
            if record_view["can_correct"]:
                correction_text = widgets.Textarea(
                    placeholder="写下更正后的信息", layout=widgets.Layout(width="100%", height="65px")
                )
                correction = widgets.Button(description="提交更正候选")
                correction.on_click(lambda _, mid=memory_id, field=correction_text: perform(
                    lambda: session.propose_correction(mid, field.value),
                    "更正已保存为待确认候选；单独确认后才会替换旧版本。",
                ))
                parts.append(correction_text)
                buttons.append(correction)
                item_actions.append(correction)
            delete = widgets.Button(description="删除此条记忆", button_style="danger")
            delete.on_click(lambda _, mid=memory_id, content=record_view["content"]: ask_before(
                lambda: session.delete(mid, f"DELETE {mid}"),
                f"选中的记忆：{content[:200]}\n会删除选中的这一条记忆，并重置当前聊天上下文和上一条回答，防止旧信息再次带入。"
                "其他长期记忆不受影响。是否删除？",
                "已删除选中记忆，并重置当前聊天上下文和上一条回答；其他长期记忆不受影响。",
                confirm_label="确认删除并重置当前聊天",
            ))
            buttons.append(delete)
            item_actions.append(delete)
            parts.append(widgets.HBox(buttons))
            cards.append(widgets.VBox(parts, layout=widgets.Layout(border="1px solid #e2e8f0", padding="10px")))
        memories.children = tuple(cards) if cards else (widgets.HTML("<p>还没有长期记忆候选。</p>"),)
        update_enabled()

    def send_clicked(_: Any) -> None:
        text = message.value
        if not text.strip():
            tell("请先输入一条虚构消息。")
            return

        def act() -> None:
            session.send_message(text)
            message.value = ""

        perform(act, "已回复。历史记录会自动处理，需要时再核对；你也可以随时查看、更正或删除。"
                if conversational else "已处理这条消息。可另行整理当前窗口内的记忆候选，范围见按钮下方。")

    def next_session_clicked(_: Any) -> None:
        data = session.snapshot()
        scope = data.get("extraction_scope") or {}
        pending = scope.get("pending_user_turn_count")
        if type(pending) is not int:
            pending = sum(turn.get("role") == "user" for turn in data["turns"])
        success = "已开始下一次会话。已保存的长期记忆候选和已确认记忆仍保留。"
        if pending:
            ask_before(
                session.new_session,
                f"本会话还有 {pending} 条用户发言未整理。继续将清空当前聊天窗口和上一条回答；"
                "尚未整理的内容不会自动保存。取消后可先整理当前窗口。已保存的长期记忆不受影响。",
                success, confirm_label="放弃未整理内容并切换会话",
            )
        else:
            perform(session.new_session, success)

    send.on_click(send_clicked)
    extract.on_click(lambda _: perform(
        session.extract_current_session, "已整理候选。请逐条核对原话，再决定是否确认。"
    ))
    next_session.on_click(next_session_clicked)
    acknowledge.observe(lambda _: update_enabled(), names="value")
    def automatic_changed(change: dict[str, Any]) -> None:
        if state["refreshing_controls"] or not acknowledge.value or state["busy"] or state["decision_pending"]:
            return
        perform(lambda: session.set_automatic_memory_enabled(bool(change["new"])),
                "已更新自动记录设置；已有长期记录未删除。")

    automatic_memory.observe(automatic_changed, names="value")
    if conversational:
        management = widgets.Accordion(children=[widgets.VBox([
            extraction_scope, widgets.HBox([extract]), memories,
        ])])
        management.set_title(0, "查看、修改或删除记忆 · 可选")
        management.selected_index = None
        children = [title, mode, paused_notice, notice, acknowledge, automatic_memory,
                    memory_mode_notice, chat, message, widgets.HBox(main_actions),
                    decision_panel, status, trace_panel, management]
    else:
        children = [title, mode, paused_notice, notice, acknowledge, memory_mode_notice,
                    chat, message, widgets.HBox(main_actions), extraction_scope,
                    decision_panel, status, widgets.HTML("<h4>记忆管理</h4>"), memories, trace_panel]
    root = widgets.VBox(children, layout=widgets.Layout(width="100%", max_width="960px"))
    refresh()
    display(root)
    return root


if __name__ == "__main__":
    raise SystemExit("Use launch_notebook_demo() inside Google Colab; see app/README.md")
