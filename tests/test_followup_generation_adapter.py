"""Small deterministic fixtures; never model or treatment effectiveness."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
import generation_adapter as adapter
from source_replay import SourceReplay


def source(n, text, *, visit=1, role="user", storage="allowed", valid_to=None):
    result = {"source_id": f"A-S{n:02}", "source_order": n, "visit": visit,
              "date": f"2026-09-{visit:02}T10:{n:02}:00Z", "role": role,
              "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "storage_policy": storage}
    if valid_to:
        result["valid_to"] = valid_to
    return result


def event(s):
    return {"variant": "A", "kind": "source", "source": s}


def control(n, action, targets=(), before=None, *, at=None):
    return {"variant": "A", "kind": "control", "control": {
        "event_id": f"event-{n}-{action}", "action": action,
        "target_source_ids": list(targets), "as_of": at or f"2026-09-01T10:{n:02}:00Z",
        "source_order": n, "before_task_id": before}}


def candidate(s, *, content=None, typ="explicit_fact", payload=None, **extra):
    return json.dumps({"candidates": [{"memory_type": typ, "content": content or s["text"],
        "source_turn_ids": [s["source_id"]], "supporting_text": s["text"],
        "structured_payload": payload or {}, "confidence": .8, "importance": .5, **extra}]})


def case(history, *, query="What was my plan?", visit=None, current=None, controls=(), exclusions=()):
    last = history[-1]
    return {"case_id": "A:T01", "variant": "A", "task_id": "T01", "visit": visit or last["visit"],
            "as_of": last["date"], "query": query, "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "history_prefix_source_order": last["source_order"], "start_new_session": False,
            "visible_source_ids": [s["source_id"] for s in history],
            "current_session_source_ids": current if current is not None else [s["source_id"] for s in history if s["visit"] == (visit or last["visit"])],
            "history": history, "controls": list(controls), "exclusions": list(exclusions)}


def test_event_request_has_no_question_future_or_same_turn_deleted_body():
    old = source(1, "secret windmill")
    now = source(2, "delete the previous private description")
    future = source(3, "FUTURE GOLD SHOULD NEVER TRAVEL")
    stream = [event(old), event(now), control(2, "delete", [old["source_id"]]), event(future)]
    messages = adapter.source_event_extraction_request(event(now), stream)
    payload = json.loads(messages[1]["content"])
    assert payload["eligible_source_turn_ids"] == [now["source_id"]]
    assert [s["source_id"] for s in payload["sources"]] == [now["source_id"]]
    assert "secret windmill" not in json.dumps(messages)
    assert "FUTURE GOLD" not in json.dumps(messages)
    assert adapter.source_event_extraction_request(event(source(4, "not stored", storage="session_only")), []) == []
    assert adapter.source_event_extraction_request(event(source(5, "assistant suggestion", role="assistant")), []) == []


def test_model_cannot_override_source_clock_id_or_confirmation():
    s = source(1, "I plan a small practice.")
    replay = SourceReplay(user_id="fictional-A")
    try:
        replay.ingest_source(s, extraction_output=candidate(s, typ="goal", payload={"description": "practice", "status": "planned"},
            observed_at="2099-01-01T00:00:00Z", memory_id="fake", confirmation_state="user_confirmed", valid_to="2100-01-01T00:00:00Z"), extraction_context_ids=[s["source_id"]])
        rows = replay.store.list_for_user(replay.user_id)
        assert len(rows) == 2
        semantic = next(r for r in rows if r.memory_id.startswith("candidate-"))
        assert semantic.observed_at == semantic.created_at == semantic.updated_at == s["date"]
        assert semantic.memory_id != "fake" and semantic.confirmation_state == "unconfirmed"
        assert semantic.lifecycle_status == "candidate" and semantic.valid_to is None
    finally:
        replay.close()


def test_actual_slot_conflicts_and_proposed_correction_do_not_confirm():
    first = source(1, "My plan is ten minutes.")
    later = source(2, "Change my plan to three minutes.")
    correction = source(3, "The newest plan replaces the old one.")
    replay = SourceReplay(user_id="fictional-A")
    try:
        for s in (first, later):
            replay.ingest_source(s, extraction_output=candidate(s, typ="goal", payload={"description": s["text"], "status": "planned", "slot_key": "goal"}), extraction_context_ids=[s["source_id"]])
        assert any(w["action"] == "conflict_requires_confirmation" for w in replay.events[-1]["writes"])
        replay.ingest_source(correction, extraction_output=candidate(correction, typ="user_correction", target_source_id=first["source_id"]), extraction_context_ids=[first["source_id"], correction["source_id"]])
        assert any(w["action"] == "correction_requires_confirmation" for w in replay.events[-1]["writes"])
        assert all(r.lifecycle_status == "candidate" and r.confirmation_state == "unconfirmed" for r in replay.store.list_for_user(replay.user_id))
    finally:
        replay.close()


def invalid_semantics_retains_quote_and_exposes_failure(mutation):
    s = source(1, "Only the current original statement.")
    raw = json.loads(candidate(s))
    if mutation == "wrong_quote": raw["candidates"][0]["supporting_text"] = "fabricated"
    elif mutation == "past_citation": raw["candidates"][0]["source_turn_ids"] = ["OTHER"]
    elif mutation == "foreign_link":
        raw["candidates"][0]["memory_type"] = "user_correction"
        raw["candidates"][0]["target_source_id"] = "foreign"
    else: raw["candidates"] *= 13
    replay = SourceReplay(user_id="fictional-A")
    try:
        replay.ingest_source(s, extraction_output=json.dumps(raw), extraction_context_ids=[s["source_id"]])
        assert replay.semantic_statuses[s["source_id"]] == "extraction_invalid"
        assert len(replay.store.list_for_user(replay.user_id)) == 1
    finally:
        replay.close()


def test_delete_purges_store_raw_session_and_dependent_cache():
    old = source(1, "private windmill")
    child = source(2, "a later allowable observation")
    replay = SourceReplay(user_id="fictional-A")
    try:
        replay.ingest_source(old, extraction_output=candidate(old), extraction_context_ids=[old["source_id"]])
        replay.ingest_source(child, extraction_output=candidate(child, content="index referring to private windmill"), extraction_context_ids=[old["source_id"], child["source_id"]])
        before = replay.state_hash()
        replay.apply_control(control(2, "delete", [old["source_id"]])["control"])
        assert replay.state_hash() != before
        assert old["source_id"] not in replay.raw_sources and old["source_id"] not in replay.session_sources
        assert not replay.extraction_cache and not replay.extraction_cache_dependencies
        assert len(replay.store.list_for_user(replay.user_id)) == 1  # only child exact quote
        assert all("private windmill" not in json.dumps(r.to_dict()) for r in replay.store.list_for_user(replay.user_id))
        with unittest.TestCase().assertRaisesRegex(ValueError, "resurrection"):
            replay.ingest_source(old)
    finally:
        replay.close()


def test_semantic_index_changes_retrieval_but_summary_not_rendered():
    old = source(1, "I will begin with a single sentence before meetings.")
    now = source(2, "Today we continue.", visit=2)
    task = case([old, now], query="homework assignment")
    outputs = {old["source_id"]: candidate(old, content="homework assignment SEMANTIC_ONLY_SECRET"), now["source_id"]: '{"candidates": []}'}
    result = adapter.prepare_case(task, replay_events=[event(old), event(now)], extraction_outputs=outputs)
    raw = result["prepared_by_arm"]["raw_history_rag"]
    managed = result["prepared_by_arm"]["managed_memory_rag"]
    assert old["source_id"] not in raw["trace"]["included_source_ids"]
    assert old["source_id"] in managed["trace"]["included_source_ids"]
    assert old["text"] in managed["messages"][0]["content"]
    assert "SEMANTIC_ONLY_SECRET" not in json.dumps(managed["messages"])
    assert result["trace"]["all_records_unconfirmed"]


def test_all_actual_draft_prefixes_and_privacy_controls_only_with_mocks():
    import prepare_generation_materials as compiler
    ledger = json.loads((ROOT / 'evaluation/dialogue-benefit/task_materials_draft.json').read_text())
    materials, events, _ = compiler.compile_materials(ledger)
    for task in materials['tasks']:
        outputs = {e['source']['source_id']: '{"candidates": []}' for e in events if e['kind'] == 'source' and e['source']['role'] == 'user' and e['source']['storage_policy'] == 'allowed'}
        result = adapter.prepare_case(task, replay_events=events, extraction_outputs=outputs)
        for prepared in result['prepared_by_arm'].values():
            assert len(prepared['messages'][0]['content']) < 12000
            assert prepared['trace']['context_characters'] <= 6000
            assert set(prepared['trace']['included_source_ids']) <= set(task['visible_source_ids'])
            if task['visit'] == 4:
                assert task['variant'] + '-S13' not in json.dumps(prepared['messages'])
                assert task['variant'] + '-S15' not in json.dumps(prepared['messages'])
        if task['start_new_session']:
            assert result['prepared_by_arm']['short_term']['trace']['included_source_ids'] == []
        assert not result['trace']['generated_answers_reingested']


def test_budget_drops_and_partial_short_spans_are_visible():
    old = source(1, "A long earlier source. " * 80)
    now = source(2, "Current long source. " * 80, visit=2)
    task = case([old, now], query="source")
    result = adapter.prepare_case(task, replay_events=[event(old), event(now)], extraction_outputs={old['source_id']: '{"candidates": []}', now['source_id']: '{"candidates": []}'}, settings={'max_context_chars': 600, 'minimum_relevance': 0})
    for prepared in result['prepared_by_arm'].values():
        assert prepared['trace']['context_characters'] <= 600
        assert now['source_id'] in prepared['trace']['short_source_inclusion_unknown_due_to_truncation']
        assert all(not p.get('complete_source_rendered') for p in prepared['trace']['rendered_source_spans'] if p['channel'] == 'short_term')
    assert result['prepared_by_arm']['managed_memory_rag']['trace']['budget_dropped_memory_ids']


def test_explicit_source_validity_filters_quote_without_confirming_it():
    s = source(1, "Temporary dated statement", valid_to="2026-09-02T00:00:00Z")
    replay = SourceReplay(user_id='fictional-A')
    try:
        replay.ingest_source(s)
        assert s['source_id'] in replay.eligible_sources(as_of=s['date'])
        assert not replay.eligible_sources(as_of="2026-09-03T00:00:00Z")
        assert replay.store.list_for_user(replay.user_id)[0].lifecycle_status == 'candidate'
    finally:
        replay.close()


class TestFollowupGenerationAdapter(unittest.TestCase):
    """Run the pure functions with the repository's dependency-free runner."""


for _name, _test in list(globals().items()):
    if _name.startswith('test_') and callable(_test):
        setattr(TestFollowupGenerationAdapter, _name, lambda self, f=_test: f())
for _mutation in ('wrong_quote', 'past_citation', 'foreign_link', 'too_many'):
    setattr(TestFollowupGenerationAdapter, 'test_invalid_semantics_' + _mutation,
            lambda self, mutation=_mutation: invalid_semantics_retains_quote_and_exposes_failure(mutation))
del _name, _test, _mutation
