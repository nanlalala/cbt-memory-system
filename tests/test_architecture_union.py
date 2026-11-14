"""Union managed arm: raw-RAG selection is always present; items only add."""
import json
from pathlib import Path
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
from architecture_retest import runner
from architecture_union import VERSION, union_requests as u
from architecture_union.layers import UnionConfig, fact_key_candidates, reliability, rerank_items
import cactus_materials


class FakeDense:
    def rank(self, query, documents):
        return list(range(len(documents)))


def fixture():
    compiled = cactus_materials.compile_rows(cactus_materials.invented_fixture_rows(), family_limit=1, fixture_only=True)
    case = compiled["generation"][-1]
    return case, runner.source_views(case)


def user_source(sources):
    return next(s for s in sources[:-6] if s["role"] == "user")


def item_on(source, name="m", value="A deliberately processed fixture representation.", origin="explicit_user_statement"):
    return dict(item_id=name, section="case_state", value=value, origin=origin,
                evidence=[dict(source_id=source["source_id"], start=0, end=len(source["content"]), text=source["content"])])


class UnionManagedArm(unittest.TestCase):
    def test_every_raw_selected_message_is_in_the_managed_request_and_items_are_added(self):
        case, sources = fixture()
        plans = u.prepare_case_requests_union(case, sources, [item_on(user_source(sources))], FakeDense(), threading.Lock())
        self.assertEqual(set(plans), set(runner.ARMS))
        raw_ids = plans["raw_history_rag"]["selection_audit"]["raw_selected_source_ids"]
        self.assertTrue(raw_ids)
        managed = json.dumps(plans["managed_memory_rag"]["payload"])
        by_id = {s["source_id"]: s for s in sources}
        for sid in raw_ids:
            self.assertIn(json.dumps(by_id[sid]["content"])[1:-1], managed)
        audit = plans["managed_memory_rag"]["selection_audit"]
        self.assertTrue(audit["raw_selection_is_subset_of_managed"])
        self.assertEqual(audit["condition"], u.UNION_CONDITION)
        self.assertEqual(audit["union_version"], VERSION)
        self.assertEqual(audit["union_item_ids"], ["m"])
        self.assertIn("deliberately processed fixture representation", managed)

    def test_other_arms_are_identical_to_the_frozen_runner(self):
        case, sources = fixture()
        frozen = runner.prepare_case_requests(case, sources, [item_on(user_source(sources))], FakeDense(), threading.Lock())
        plans = u.prepare_case_requests_union(case, sources, [item_on(user_source(sources))], FakeDense(), threading.Lock())
        for arm in ("no_rag", "raw_history_rag"):
            self.assertEqual(plans[arm]["payload"], frozen[arm]["payload"])
        self.assertNotEqual(plans["managed_memory_rag"]["payload"], frozen["managed_memory_rag"]["payload"])

    def test_oversized_item_is_skipped_but_raw_selection_is_never_dropped(self):
        case, sources = fixture()
        huge = item_on(user_source(sources), "huge", "x" * (runner.SETTINGS["max_answer_chars"] + 10))
        plans = u.prepare_case_requests_union(case, sources, [huge], FakeDense(), threading.Lock())
        audit = plans["managed_memory_rag"]["selection_audit"]
        self.assertEqual(audit["union_item_ids"], [])
        self.assertEqual([r["item_id"] for r in audit["union_item_rejections"]], ["huge"])
        self.assertTrue(audit["raw_selection_is_subset_of_managed"])

    def test_item_the_builder_silently_drops_is_not_counted_as_added(self):
        case, sources = fixture()
        wrong = item_on(next(s for s in sources[:-6] if s["role"] == "assistant"), "wrong_role")
        plans = u.prepare_case_requests_union(case, sources, [wrong], FakeDense(), threading.Lock())
        audit = plans["managed_memory_rag"]["selection_audit"]
        self.assertEqual(audit["union_item_ids"], [])
        self.assertEqual(audit["union_item_rejections"][0]["item_id"], "wrong_role")

    def test_no_items_still_gives_raw_selection_in_managed_arm(self):
        case, sources = fixture()
        plans = u.prepare_case_requests_union(case, sources, [], FakeDense(), threading.Lock())
        self.assertTrue(plans["managed_memory_rag"]["selection_audit"]["raw_selection_is_subset_of_managed"])


def long_fixture(turns=24, recent_only=False):
    """A fictional case with enough older messages that the raw arm cannot take all."""
    case, _ = fixture()
    count = 6 if recent_only else turns
    history = [dict(message_id="turn-%03d" % n, role="assistant" if n % 2 == 0 else "user",
                    content="FICTIONAL %s line %d about topic%d." % ("question" if n % 2 == 0 else "report", n, n),
                    date=None, source_order=n) for n in range(count)]
    case = {**case, "history": history, "history_cutoff_order_exclusive": count,
            "input": "FICTIONAL current report about walking practice."}
    return case, runner.source_views(case)


def memory_block(plans):
    message = next(m for m in plans["managed_memory_rag"]["payload"]["messages"]
                   if m["content"].startswith("SOURCED MEMORY DATA ONLY."))
    return json.loads(message["content"].split("\n", 1)[1])


def item_ids(plans):
    return plans["managed_memory_rag"]["selection_audit"]["union_item_ids"]


class AdditiveLayers(unittest.TestCase):
    def run_union(self, case, sources, items, **config):
        return u.prepare_case_requests_union(case, sources, items, FakeDense(), threading.Lock(),
                                             UnionConfig(**config) if config else None)

    def test_default_config_is_the_tested_union_and_records_itself(self):
        case, sources = fixture()
        audit = self.run_union(case, sources, [item_on(user_source(sources))])["managed_memory_rag"]["selection_audit"]
        self.assertEqual(audit["union_config"], UnionConfig().as_dict())
        self.assertEqual(audit["union_fact_key_source_ids"], [])
        self.assertIsNone(audit["union_rerank"])

    def test_frozen_arms_unchanged_with_every_layer_on(self):
        case, sources = long_fixture()
        items = [item_on(user_source(sources))]
        frozen = runner.prepare_case_requests(case, sources, items, FakeDense(), threading.Lock())
        plans = self.run_union(case, sources, items, renderer="v5", fact_key_sources=3, pin_profile=True, rerank=True)
        for arm in ("no_rag", "raw_history_rag"):
            self.assertEqual(plans[arm]["payload"], frozen[arm]["payload"])
        self.assertTrue(plans["managed_memory_rag"]["selection_audit"]["raw_selection_is_subset_of_managed"])

    def test_fact_key_adds_at_most_n_unselected_older_messages_once_each(self):
        case, sources = long_fixture()
        older = [s for s in sources[:-runner.SETTINGS["recent_messages"]]]
        frozen = runner.prepare_case_requests(case, sources, [], FakeDense(), threading.Lock())
        missed = set(frozen["raw_history_rag"]["selection_audit"]["raw_omitted_source_ids"])
        target = [s for s in older if s["role"] == "user" and s["source_id"] in missed][0]
        items = [item_on(target, "walk", "Plans walking practice three times a week.")]
        plans = self.run_union(case, sources, items, fact_key_sources=3)
        audit = plans["managed_memory_rag"]["selection_audit"]
        raw_ids, fact_ids = audit["union_raw_source_ids"], audit["union_fact_key_source_ids"]
        self.assertTrue(0 < len(fact_ids) <= 3)
        self.assertFalse(set(fact_ids) & set(raw_ids))
        self.assertTrue(set(fact_ids) <= {s["source_id"] for s in older})
        block = memory_block(plans)
        by_id = {s["source_id"]: s for s in sources}
        quoted = [c["claim_id"] for c in block["claims"] if c["claim_id"].startswith("quote:")]
        self.assertEqual(sorted(quoted), sorted("quote:" + sid for sid in [*raw_ids, *fact_ids]))
        texts = [c["value"] for c in block["claims"] if c["claim_id"].startswith("quote:")]
        for sid in [*raw_ids, *fact_ids]:
            self.assertEqual(texts.count(by_id[sid]["content"]), 1)
        # The message the raw arm missed is found through the item that cites it.
        self.assertNotIn(target["source_id"], raw_ids)
        self.assertEqual(fact_ids[0], target["source_id"])

    def test_fact_key_never_returns_a_source_outside_the_eligible_pool(self):
        pool = [{"source_id": "a", "content": "walk"}, {"source_id": "b", "content": "walk walk"}]
        items = [dict(item_id="x", value="walk", evidence=[dict(source_id="deleted", start=0, end=1, text="w")])]
        got = fact_key_candidates("walk", pool, items, ["a"], lambda q, docs: list(range(len(docs))), 5)
        self.assertEqual(got, ["b"])

    def test_profile_items_are_tried_first_and_hypotheses_never_pinned(self):
        case, sources = long_fixture()
        users = [s for s in sources[:-runner.SETTINGS["recent_messages"]] if s["role"] == "user"]
        items = [dict(item_on(users[0], "bg", "Background about walking practice."), section="relevant_background"),
                 dict(item_on(users[1], "hyp", "Avoidance may keep worry going."), section="working_formulation",
                      origin="agent_inference"),
                 dict(item_on(users[2], "fix", "Corrected: five minutes, not ten."), origin="user_correction"),
                 item_on(users[3], "state", "Currently dreading the meeting.")]
        plans = self.run_union(case, sources, items, pin_profile=True)
        audit = plans["managed_memory_rag"]["selection_audit"]
        self.assertEqual(audit["union_profile_item_ids"], ["fix", "state"])
        self.assertEqual(item_ids(plans)[:2], ["fix", "state"])
        self.assertNotIn("hyp", audit["union_profile_item_ids"])

    def test_v5_renderer_pins_corrections_and_keeps_raw_superset(self):
        case, sources = long_fixture()
        user = [s for s in sources[:-runner.SETTINGS["recent_messages"]] if s["role"] == "user"][0]
        items = [dict(item_on(user, "fix", "Corrected: five minutes."), origin="user_correction")]
        row = self.run_union(case, sources, items, renderer="v5")["managed_memory_rag"]
        self.assertEqual(row["trace"]["assembly_version"], "cbt-context-v5-draft")
        self.assertEqual(row["trace"]["pinned_item_ids"], ["fix"])
        self.assertTrue(row["selection_audit"]["raw_selection_is_subset_of_managed"])

    def test_rerank_with_zero_weights_keeps_frozen_order(self):
        case, sources = long_fixture()
        users = [s for s in sources[:-runner.SETTINGS["recent_messages"]] if s["role"] == "user"]
        items = [item_on(users[i], "i%d" % i, "Report %d." % i) for i in range(4)]
        orders = {s["source_id"]: s["source_order"] for s in sources}
        roles = {s["source_id"]: s["role"] for s in sources}
        zero = UnionConfig(rerank=True, recency_weight=0, importance_weight=0, reliability_weight=0)
        ordered, rows = rerank_items(items, orders, roles, len(sources), zero)
        self.assertEqual([i["item_id"] for i in ordered], ["i0", "i1", "i2", "i3"])
        self.assertEqual(len(rows), 4)

    def test_rerank_lets_a_later_state_overtake_only_a_close_neighbour(self):
        # 20 items: one relevance place is worth 0.05, the recency gap here about 0.06.
        case, sources = long_fixture()
        users = [s for s in sources[:-runner.SETTINGS["recent_messages"]] if s["role"] == "user"]
        early, late = item_on(users[0], "early", "State then."), item_on(users[-1], "late", "State now.")
        filler = [dict(item_on(users[1], "f%d" % i, "Other %d." % i), section="relevant_background") for i in range(18)]
        orders = {s["source_id"]: s["source_order"] for s in sources}
        roles = {s["source_id"]: s["role"] for s in sources}
        ordered, _ = rerank_items([early, late, *filler], orders, roles, len(sources), UnionConfig(rerank=True))
        self.assertEqual([i["item_id"] for i in ordered][:2], ["late", "early"])
        ordered, _ = rerank_items([*filler, early, late], orders, roles, len(sources), UnionConfig(rerank=True))
        self.assertEqual(ordered[0]["item_id"], "f0")

    def test_reliability_uses_evidence_grade_and_repetition_never_lifts_a_hypothesis(self):
        roles = {"u1": "user", "u2": "user", "a1": "assistant"}
        ev = lambda *ids: [dict(source_id=i, start=0, end=1, text="x") for i in ids]
        stated = dict(item_id="s", origin="explicit_user_statement", evidence=ev("u1"))
        repeated = dict(item_id="r", origin="explicit_user_statement", evidence=ev("u1", "u2"))
        hypothesis = dict(item_id="h", origin="agent_inference", evidence=ev("u1", "u2"))
        no_user = dict(item_id="n", origin="explicit_user_statement", evidence=ev("a1"))
        self.assertGreater(reliability(repeated, roles)[0], reliability(stated, roles)[0])
        self.assertEqual(reliability(hypothesis, roles)[1], "low")
        self.assertEqual(reliability(no_user, roles)[1], "low")

    def test_sparse_history_falls_back_to_recent_messages_only(self):
        case, sources = long_fixture(recent_only=True)
        plans = self.run_union(case, sources, [], renderer="v5", fact_key_sources=3, pin_profile=True, rerank=True)
        audit = plans["managed_memory_rag"]["selection_audit"]
        self.assertEqual(audit["union_raw_source_ids"], [])
        self.assertEqual(audit["union_fact_key_source_ids"], [])
        self.assertEqual(audit["union_item_ids"], [])

    def test_bad_config_is_refused(self):
        for bad in ({"renderer": "v6"}, {"fact_key_sources": -1}, {"recency_weight": 2}):
            with self.assertRaises(ValueError):
                UnionConfig(**bad)


if __name__ == "__main__":
    unittest.main()
