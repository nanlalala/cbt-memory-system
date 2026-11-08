from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))

from evidence_bundles import (
    EvidenceBundle, EvidenceClaim, EvidenceSpan, OutboundBudgetError,
    build_outbound_messages, select_knowledge_evidence,
)


BASE = [{"role": "system", "content": "Use only supported fields."}, {"role": "user", "content": "这周作业怎样了？"}]


def claim(cid, field, value, text, **kwargs):
    span = EvidenceSpan("s1", cid + ":span", text)
    return EvidenceClaim(cid, field, value, spans=(span,), support_status="rule_verified", **kwargs)


class EvidenceBundleTests(unittest.TestCase):
    def pack(self, claims=(), **kwargs):
        return build_outbound_messages(BASE, [EvidenceBundle("task", tuple(claims))],
                                       max_outbound_chars=kwargs.pop("max_outbound_chars", 4000), **kwargs)

    def test_complete_escaped_request_and_fixed_metadata_are_counted(self):
        package = self.pack([claim("pause", "engagement", "paused", "这周暂停")],
                            request_metadata={"model": "fixed-model", "temperature": 0})
        self.assertEqual(package.char_count, len(package.serialized_request))
        self.assertEqual(json.loads(package.serialized_request), package.request_payload)
        self.assertIn("\\u8fd9", package.serialized_request)
        self.assertEqual(package.request_payload["model"], "fixed-model")
        self.assertEqual(package.request_payload["max_tokens"], 4096)
        self.assertGreater(package.char_count, sum(len(m["content"]) for m in package.messages))

    def test_oversized_mandatory_request_is_rejected_without_truncating_input(self):
        with self.assertRaises(OutboundBudgetError):
            self.pack(max_outbound_chars=20)
        self.assertEqual(BASE[-1]["content"], "这周作业怎样了？")

    def test_output_budget_admission_does_not_silently_increase_limit(self):
        with self.assertRaises(OutboundBudgetError):
            self.pack(max_response_tokens=100, remaining_output_tokens=99)
        with self.assertRaises(ValueError):
            self.pack(max_response_tokens=1, request_metadata={"max_tokens": True})

    def test_short_pause_survives_when_quantity_history_does_not_fit(self):
        pause = claim("pause", "engagement", "paused", "这周暂停")
        quantity = claim("count", "reported_count", 2, "过去练习很长的历程" * 150)
        package = self.pack([quantity, pause], max_outbound_chars=1300)
        self.assertEqual(package.included_claim_ids, ("pause",))
        self.assertEqual(package.unknown_claim_ids, ("count",))
        self.assertIn("paused", package.serialized_request)
        self.assertNotIn('"reported_count"', package.serialized_request)
        self.assertIn("Absent fields remain unknown", package.messages[1]["content"])

    def test_shortest_sufficient_alternative_preserves_exact_quote(self):
        short = EvidenceSpan("s1", "short", "暂停")
        long = EvidenceSpan("s1", "long", "很长的历程" * 100 + "暂停")
        alternative = EvidenceClaim("p", "engagement", "paused", evidence_options=((long,), (short,)),
                                    support_status="rule_verified")
        package = self.pack([alternative], max_outbound_chars=1200)
        self.assertEqual(package.included_claim_ids, ("p",))
        self.assertIn('"short"', package.messages[1]["content"])
        self.assertNotIn('"long"', package.messages[1]["content"])

    def test_unverified_semantics_and_model_inference_do_not_enter_current_view(self):
        span = EvidenceSpan("s1", "neg", "我没答应每天做")
        unverified = EvidenceClaim("wrong", "agreement", "agreed", spans=(span,))
        inferred = EvidenceClaim("hypothesis", "pattern", "avoidance", spans=(span,),
                                 support_status="rule_verified", origin="agent_inference")
        package = self.pack([unverified, inferred])
        self.assertEqual(package.included_claim_ids, ())
        self.assertEqual(set(package.unknown_claim_ids), {"wrong", "hypothesis"})
        self.assertEqual(package.messages, tuple(BASE))

    def test_source_only_quote_does_not_certify_a_new_statement(self):
        span = EvidenceSpan("s1", "neg", "我没答应每天做")
        bad = EvidenceClaim("bad", "history_quote", "同意每天做", spans=(span,), support_status="source_quote")
        good = EvidenceClaim("good", "history_quote", span.text, spans=(span,), support_status="source_quote")
        package = self.pack([bad, good])
        self.assertEqual(package.included_claim_ids, ("good",))

    def test_quote_substring_cannot_strip_negation(self):
        span = EvidenceSpan("s1", "neg", "我没答应每天做")
        chopped = EvidenceClaim("chopped", "history_quote", "答应每天做", spans=(span,), support_status="source_quote")
        package = self.pack([chopped])
        self.assertEqual(package.included_claim_ids, ())

    def test_same_source_multiple_spans_count_once_but_are_retained(self):
        a = claim("walk", "reported_count", 2, "散步两次")
        b = claim("record", "engagement", "paused", "记录暂停")
        package = self.pack([a, b])
        self.assertEqual(package.source_ids, ("s1",))
        self.assertEqual(set(package.included_claim_ids), {"walk", "record"})
        self.assertIn("walk:span", package.messages[1]["content"])
        self.assertIn("record:span", package.messages[1]["content"])

    def test_missing_usage_is_unknown_and_partial_usage_is_not_zero_filled(self):
        self.assertIsNone(self.pack().usage)
        self.assertEqual(self.pack(usage={"input_tokens": 5}).usage, {"input_tokens": 5})

    def test_bad_quote_offsets_duplicate_claims_and_metadata_override_fail(self):
        with self.assertRaises(ValueError):
            EvidenceSpan("s", "span", "暂停", start=2, end=3)
        c = claim("same", "engagement", "paused", "暂停")
        with self.assertRaises(ValueError):
            self.pack([c, c])
        with self.assertRaises(ValueError):
            self.pack(request_metadata={"messages": []})


class KnowledgeExitTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {"source_id": "reviewed", "chunk_id": "ch1", "text": "Review the agreed task.",
                      "page_start": 3, "page_end": 4, "relevance": .8, "permission_checked": True}
        self.manifest = {"ch1": dict(self.chunk)}

    def select(self, candidates, **kwargs):
        return select_knowledge_evidence(candidates, task_requires_knowledge=kwargs.pop("task_requires_knowledge", True),
                                         enabled_source_ids=kwargs.pop("enabled_source_ids", {"reviewed"}),
                                         corpus_chunks=kwargs.pop("corpus_chunks", self.manifest), **kwargs)

    def test_disabled_sources_or_unrequested_route_are_explicit_empty_exits(self):
        self.assertEqual(self.select([self.chunk], enabled_source_ids=set()).status, "unavailable")
        self.assertEqual(self.select([self.chunk], task_requires_knowledge=False).status, "not_requested")

    def test_no_hit_and_missing_registry_proof_do_not_claim_textbook_support(self):
        self.assertEqual(self.select([{**self.chunk, "relevance": .01}]).status, "no_hit")
        self.assertEqual(self.select([{**self.chunk, "permission_checked": False}]).status, "no_hit")
        self.assertEqual(self.select([self.chunk], corpus_chunks=None).reason, "verified_chunk_manifest_required")

    def test_chunk_and_page_must_reconcile_with_the_validated_corpus(self):
        self.assertEqual(self.select([{**self.chunk, "page_start": 2}]).status, "no_hit")
        self.assertEqual(self.select([{**self.chunk, "text": "fabricated"}]).status, "no_hit")
        good = self.select([self.chunk])
        self.assertEqual(good.status, "ready")
        self.assertEqual(good.evidence[0]["chunk_id"], "ch1")


if __name__ == "__main__":
    unittest.main()
