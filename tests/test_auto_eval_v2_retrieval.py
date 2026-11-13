"""Invented source-only regression fixtures, never a CBT effect estimate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import unittest

PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/auto_eval_v2/retrieval.py"
SPEC = importlib.util.spec_from_file_location("auto_eval_v2_retrieval", PATH)
retrieval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retrieval)


def raises(exception, *, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(exception, match) if match else case.assertRaises(exception)


def source(sid, order, content, role="user", **changes):
    return {"source_id": sid, "source_order": order, "role": role, "content": content,
            "retained": True, "use_allowed": True, "external_allowed": True, **changes}


def run(sources, raw, indexes=None, **kwargs):
    return retrieval.select_evidence(sources, raw, indexes or {}, "What small step fits now?",
        cutoff_order=kwargs.pop("cutoff_order", 100), recent_count=kwargs.pop("recent_count", 0),
        **kwargs)


def test_literal_index_recovers_older_barrier_without_sending_index_prose():
    rows = [source("a", 0, "I am worried about the next walk."),
            source("b", 1, "We discussed a diary."),
            source("c", 2, "Please pause the walking task; my ankle still hurts.")]
    result = run(rows, ["a", "b", "c"], {"barrier": [{"source_id": "c",
        "quote": "my ankle still hurts", "summary": "DO NOT SEND THIS INFERRED PROFILE"}]}, top_k=1)
    assert result["selected_source_ids"] == ["c"]
    assert json.loads(result["raw_baseline"]["messages"][1]["content"])["older_evidence"][0]["source_id"] == "a"
    assert result["older_evidence"][0]["content"] == rows[2]["content"]
    assert "DO NOT SEND" not in result["serialized_request"]
    assert result["trace"]["promoted_index_facts"] == 0


def test_current_six_whole_messages_are_direct_and_not_index_displaceable():
    rows = [source(f"s{i}", i, "original " + str(i), "assistant" if i % 2 else "user") for i in range(10)]
    result = run(rows, [f"s{i}" for i in range(10)], {"semantic": [
        {"source_id": "s0", "quote": "original 0"}]}, recent_count=6, top_k=1)
    assert [r["source_id"] for r in result["recent_messages"]] == [f"s{i}" for i in range(4, 10)]
    assert [r["role"] for r in result["recent_messages"]] == [r["role"] for r in rows[4:]]
    assert [r["content"] for r in result["recent_messages"]] == [r["content"] for r in rows[4:]]
    assert set(result["selected_source_ids"]).isdisjoint(result["trace"]["recent_source_ids"])


def test_future_deleted_and_denied_sources_never_reach_wire():
    rows = [source("ok", 0, "allowed older quote"), source("deleted", 1, "SECRET DELETE", retained=False),
            source("denied", 2, "SECRET DENIED", external_allowed=False), source("future", 8, "SECRET FUTURE")]
    result = run(rows, ["deleted", "denied", "future", "ok"], {"update": [
        {"source_id": r["source_id"], "quote": r["content"]} for r in rows]}, cutoff_order=7, recent_count=2)
    assert "SECRET" not in result["serialized_request"]
    assert set(result["trace"]["excluded_sources"]) == {"deleted", "denied", "future"}
    # The unavailable recent messages are removed, not backfilled with older text.
    assert result["recent_messages"] == []


def test_invalid_quote_never_admits_index_only_source():
    rows = [source("a", 0, "actual text"), source("b", 1, "another text")]
    result = run(rows, ["a"], {"task": [{"source_id": "b", "quote": "user agreed"}]})
    assert result["selected_source_ids"] == ["a"]
    assert result["trace"]["index_rejections"][0]["reason"] == "literal_quote_binding_failed"


def test_equal_source_set_restores_exact_raw_request_order():
    rows = [source("z", 0, "first original"), source("a", 1, "second original")]
    result = run(rows, ["z", "a"], {"semantic": [{"source_id": "a", "quote": "second original"}]}, top_k=2)
    assert result["selected_source_ids"] == ["z", "a"]
    assert result["messages"] == result["raw_baseline"]["messages"]
    assert result["request_sha256"] == result["raw_baseline"]["request_sha256"]
    assert result["trace"]["same_source_set_preserves_raw_order"] is True


def test_source_level_deduplication_and_job_uuid_do_not_change_rank():
    rows = [source("a", 0, "first"), source("b", 1, "second")]
    first = run(rows, ["a", "b"], {"semantic": [{"source_id": "b", "quote": "second", "job_id": "uuid-one"}]}, top_k=1)
    repeated = run(rows, ["a", "a", "b"], {"semantic": [
        {"source_id": "b", "quote": "second", "job_id": "uuid-two"},
        {"source_id": "b", "quote": "second", "job_id": "uuid-three"}]}, top_k=1)
    assert first["messages"] == repeated["messages"]
    assert first["trace"]["score_parts"] == repeated["trace"]["score_parts"]


def test_role_has_no_ranking_bonus_and_is_preserved():
    rows = [source("a", 0, "same topic", "assistant"), source("b", 1, "same topic", "user")]
    result = run(rows, ["a", "b"], top_k=1)
    assert result["older_evidence"][0]["role"] == "assistant"
    assert result["trace"]["source_role_preference"] is False


def test_identical_text_from_distinct_original_events_is_not_merged():
    rows = [source("a", 0, "same utterance"), source("b", 1, "same   utterance"), source("c", 2, "useful other utterance")]
    result = run(rows, ["a", "b", "c"], top_k=2)
    assert result["selected_source_ids"] == ["a", "b"]
    assert len(result["older_evidence"]) == 2


def test_neighbor_requires_literal_original_adjacency_and_budget():
    rows = [source("a", 0, "I tried the task."), source("b", 1, "What got in the way?", "assistant"),
            source("c", 2, "My pain got worse."), source("far", 4, "unrelated far message")]
    result = run(rows, [], {"barrier": [{"source_id": "a", "quote": "tried", "neighbor_ids": ["b", "far"]}]}, top_k=3)
    assert result["selected_source_ids"] == ["a", "b"]
    assert result["trace"]["neighbor_additions"] == [{"source_id": "b", "neighbor_of": "a"}]


def test_complete_escaped_request_budget_is_counted_not_text_lengths():
    rows = [source("a", 0, 'a "quoted" \\ example\n' * 10), source("b", 1, "small")]
    baseline = run(rows, [], top_k=1)
    limit = baseline["request_chars"] + 170
    result = run(rows, ["a", "b"], top_k=1, max_request_chars=limit,
                 request_options={"model": "independent-answer", "max_tokens": 1024})
    assert result["request_chars"] <= limit
    assert len(result["serialized_request"]) == result["request_chars"]
    assert result["selected_source_ids"] == ["b"]
    assert result["trace"]["packing_rejections"][0]["source_id"] == "a"


def test_recent_over_budget_is_error_not_truncation():
    rows = [source("a", 0, "x" * 3000)]
    with raises(ValueError, match="whole_recent"):
        run(rows, [], recent_count=6, max_request_chars=1000)


def test_ordinal_mode_does_not_invent_calendar_metadata():
    result = run([source("a", 0, "I said last Monday in the original text.")], ["a"])
    assert "date" not in result["older_evidence"][0]
    assert "last Monday" in result["older_evidence"][0]["content"]
    with raises(ValueError, match="calendar_dates"):
        run([source("a", 0, "old", date="2026-11-12")], ["a"])


def test_strict_source_schema():
    for changes in [{"retained": 1}, {"source_order": True}, {"source_id": ""}]:
        with raises(ValueError):
            run([source("a", 0, "text", **changes)], ["a"])


def test_unknown_or_nonfinite_channel_weights_fail():
    rows = [source("a", 0, "text")]
    for weights in ({"raw": float("nan")}, {"raw": True}, {"invented": 1}):
        with raises(ValueError):
            run(rows, ["a"], channel_weights=weights)


def test_unknown_provider_fields_cannot_carry_assessor_gold():
    with raises(ValueError, match="nonprovider"):
        run([source("a", 0, "text")], ["a"], request_options={"gold": "hidden answer"})


def test_cutoff_is_inclusive_and_whitespace_pointer_is_rejected():
    rows = [source("at", 4, "visible at cutoff"), source("after", 5, "future")]
    result = run(rows, ["at", "after"], {"semantic": [{"source_id": "at", "quote": " "}]}, cutoff_order=4)
    assert result["selected_source_ids"] == ["at"]
    assert result["trace"]["excluded_sources"] == {"after": ["future_after_cutoff"]}
    assert result["trace"]["index_rejections"][0]["reason"] == "literal_quote_binding_failed"


def test_unicode_budget_counts_escaped_complete_request():
    rows = [source("a", 0, "我想暂停这项练习。")]
    result = run(rows, ["a"])
    assert result["request_chars"] == len(result["serialized_request"])
    assert result["serialized_request"].isascii()
    assert json.loads(result["serialized_request"])["messages"] == result["messages"]
    assert result["older_evidence"][0]["content"] == rows[0]["content"]


def load_tests(loader, standard_tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    unittest.main()
