from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for relative in ("modules/long-term-memory/schema", "modules/long-term-memory/retrieval"):
    sys.path.insert(0, str(ROOT / relative))

from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from memory_use_policy import MemoryUsePolicy, assess_memory_use  # noqa: E402


NOW = "2026-11-05T10:00:00Z"


def memory(text="I keep a notebook beside the sofa.", **changes):
    values = dict(
        schema_version="1.0", memory_id="m1", user_id="alice",
        memory_type="explicit_fact", content=text, origin="explicit_user_statement",
        confirmation_state="unconfirmed", lifecycle_status="candidate",
        sensitivity="standard", confidence=0.99, importance=0.5,
        created_at="2026-11-01T10:00:00Z", observed_at="2026-11-01T10:00:00Z",
        updated_at="2026-11-01T10:00:00Z",
        source_evidence=SourceEvidence("s1", ["u1"], text),
    )
    values.update(changes)
    return MemoryRecord(**values)


class ConversationalUsePolicyTests(unittest.TestCase):
    def decide(self, record, **kwargs):
        return assess_memory_use(record, user_id="alice", as_of=NOW, **kwargs)

    def test_confidence_float_and_model_flags_cannot_authenticate_a_candidate(self):
        record = memory()
        record.structured_payload = {"source_verified": True, "confidence": 1.0}
        before = deepcopy(record.to_dict())
        decision = self.decide(record)
        self.assertEqual(decision.action, "exclude")
        self.assertEqual(decision.reference_text, "")
        self.assertEqual(record.to_dict(), before)

    def test_verified_quote_can_be_used_only_as_a_dated_past_statement(self):
        record = memory()
        decision = self.decide(record, trusted_quote_ids={"m1"})
        self.assertEqual(decision.action, "use_as_history")
        self.assertEqual(decision.reference_text, record.content)
        self.assertEqual(decision.currentness_label, "past_statement_only")
        self.assertEqual(record.lifecycle_status, "candidate")
        self.assertEqual(record.confirmation_state, "unconfirmed")

    def test_verified_source_does_not_make_a_hallucinated_summary_usable(self):
        record = memory("The user lives in Paris.")
        record.source_evidence.supporting_text = "I like walking after lunch."
        decision = self.decide(record, trusted_quote_ids={"m1"})
        self.assertEqual(decision.action, "exclude")
        decision = self.decide(record, verified_source_ids={"m1"}, current_use=True)
        self.assertEqual(decision.action, "check_current")
        self.assertEqual(decision.reference_text, "I like walking after lunch.")
        self.assertNotIn("Paris", decision.reference_text)
        self.assertIn("summary_not_verified", decision.reason_codes)

    def test_historical_source_only_recall_uses_verified_quote_without_current_check(self):
        record = memory("The user lives in Paris.")
        record.source_evidence.supporting_text = "I like walking after lunch."
        historical = self.decide(record, verified_source_ids={"m1"}, current_use=False)
        self.assertEqual(historical.action, "use_as_history")
        self.assertEqual(historical.reference_text, "I like walking after lunch.")
        self.assertEqual(historical.currentness_label, "past_statement_only")
        self.assertEqual(historical.evidence_label, "verified_source_only")
        self.assertNotIn("Paris", historical.reference_text)
        self.assertEqual(self.decide(record, current_use=False).action, "exclude")

    def test_current_cbt_follow_up_checks_the_state_but_historical_query_does_not(self):
        for text in (
            "这周做思维记录让我觉得很有压力。",
            "本周我的目标是午饭后散步两次。",
            "My homework goal is one short walk after lunch.",
        ):
            with self.subTest(text=text):
                record = memory(text)
                current = self.decide(record, trusted_quote_ids={"m1"}, current_use=True)
                historical = self.decide(record, trusted_quote_ids={"m1"}, current_use=False)
                self.assertEqual(current.action, "check_current")
                self.assertIn("dynamic_state_requires_current_check", current.reason_codes)
                self.assertEqual(historical.action, "use_as_history")

    def test_recent_confirmed_goal_is_not_assumed_to_be_current_or_completed(self):
        record = memory(
            "Try one brief walk", memory_type="goal", lifecycle_status="active",
            confirmation_state="user_confirmed", observed_at=NOW,
            structured_payload={"description": "Try one brief walk", "status": "planned"},
        )
        decision = self.decide(record, current_use=True)
        self.assertEqual(decision.action, "check_current")
        self.assertEqual(record.structured_payload["status"], "planned")

    def test_preference_change_and_staleness_trigger_checks_only_for_current_use(self):
        changed = memory("我不再喜欢游泳，觉得准备起来很麻烦。")
        current = self.decide(changed, trusted_quote_ids={"m1"}, current_use=True)
        self.assertEqual(current.action, "check_current")
        self.assertEqual(self.decide(changed, trusted_quote_ids={"m1"}).action, "use_as_history")
        english = self.decide(memory("I don't like swimming anymore."), trusted_quote_ids={"m1"}, current_use=True)
        self.assertEqual(english.action, "check_current")
        stale = memory("I prefer a quiet activity.", observed_at="2026-09-01T10:00:00Z")
        check = self.decide(stale, trusted_quote_ids={"m1"}, current_use=True)
        self.assertEqual(check.action, "check_current")
        self.assertIn("freshness_limit_reached", check.reason_codes)

    def test_unresolved_conflict_is_not_resolved_by_higher_model_confidence(self):
        record = memory("My usual workday ends at six.", confidence=1.0)
        decision = self.decide(record, trusted_quote_ids={"m1"}, conflicting_ids={"m1"}, current_use=True)
        self.assertEqual(decision.action, "check_current")
        self.assertIn("unresolved_conflict", decision.reason_codes)

    def test_time_scope_lifecycle_and_restricted_boundaries_come_before_confidence(self):
        variants = (
            {"user_id": "bob"}, {"lifecycle_status": "deleted"},
            {"lifecycle_status": "expired"}, {"lifecycle_status": "superseded"},
            {"sensitivity": "restricted"}, {"valid_to": NOW},
            {"valid_from": "2026-11-06T10:00:00Z"},
            {"observed_at": "2026-11-06T10:00:00Z"},
        )
        for changes in variants:
            with self.subTest(changes=changes):
                record = memory(**changes)
                result = self.decide(record, trusted_quote_ids={"m1"}, verified_source_ids={"m1"})
                self.assertEqual(result.action, "exclude")
                self.assertEqual(result.reference_text, "")
                self.assertEqual(result.observed_at, "")

    def test_age_cutoff_is_configurable_and_not_a_probability(self):
        record = memory("I prefer a quiet activity.")
        default = self.decide(record, trusted_quote_ids={"m1"}, current_use=True)
        configured = self.decide(
            record, trusted_quote_ids={"m1"}, current_use=True,
            policy=MemoryUsePolicy(preference_recheck_days=2),
        )
        self.assertEqual(default.action, "use_as_history")
        self.assertEqual(configured.action, "check_current")
        self.assertNotIn("confidence", configured.to_dict())
        self.assertTrue(configured.confidence_label)

    def test_invalid_policy_and_naive_clock_are_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MemoryUsePolicy(preference_recheck_days=value)
        with self.assertRaises(ValueError):
            assess_memory_use(memory(), user_id="alice", as_of="2026-11-05T10:00:00")


if __name__ == "__main__":
    unittest.main()
