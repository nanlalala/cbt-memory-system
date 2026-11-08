"""No-call checks for archive identity, privacy and blinded reviewer projection."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "evaluation/cbt-response-review/prepare_archived_review.py"
SPEC = importlib.util.spec_from_file_location("cbt_archived_packets", SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def fixture():
    packets, allocation, planned = [], [], []
    for case_id in sorted(review.EXPECTED_CASES):
        withheld = case_id.split(":")[1] in {"T09", "T10", "T12"}
        for arm in review.ARMS:
            packet_id = None if withheld else f"OLD-{len(packets):03d}"
            planned.append({
                "case_id": case_id, "arm": arm, "packet_id": packet_id,
                "packet_export_status": "privacy_withheld" if withheld else "exported",
                "privacy_reference_withheld": withheld, "completion_status": "success",
            })
            if withheld:
                continue
            source_text = f"Fictional dated user source for {case_id}."
            envelope = {
                "answer": f"The visible answer for {case_id}.",
                "used_source_ids": ["ORIGINAL-ID"],
                "clarification": {"source_ids": ["ORIGINAL-ID"], "question": "A separate question?"},
            }
            raw = json.dumps(envelope, ensure_ascii=False, sort_keys=True)
            packet = {
                "packet_id": packet_id, "query": f"Current question for {case_id}?",
                "original_response": raw, "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "source_history": [{"source_id": "ORIGINAL-ID", "text": source_text,
                                    "text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                                    "role": "user", "date": "2026-11-01", "storage_policy": "allowed"}],
                "completion_criterion": "GOLD_SECRET_CRITERION",
                "scoring_source_references": ["GOLD_SECRET_REFERENCE"],
                "old_model_score": "OLD_SCORE_NOT_REVIEW_EVIDENCE",
                "arm": arm,
            }
            packet["packet_sha256"] = review.stable(packet)
            packets.append(packet)
            allocation.append({"case_id": case_id, "arm": arm, "packet_id": packet_id})
    controller = {"executable_commit": review.EXECUTABLE, "protocol_sha256": review.OLD_PROTOCOL,
                  "planned_review_rows": planned}
    summary = {"executable_commit": review.EXECUTABLE, "protocol_sha256": review.OLD_PROTOCOL,
               "cost": {"recorded_provider_attempts": 202, "reported_tokens": {"total_tokens": 217053},
                        "currency_amount": None}}
    return packets, allocation, controller, summary


def reseal(packet):
    packet["packet_sha256"] = review.stable({k: v for k, v in packet.items() if k != "packet_sha256"})


class ProjectionTests(unittest.TestCase):
    def test_complete_denominator_and_canonical_response_sharing(self):
        result = review.build_review(*fixture())
        self.assertEqual(result["counts"]["planned_rows"], 108)
        self.assertEqual(result["counts"]["reviewable_rows"], 81)
        self.assertEqual(result["counts"]["privacy_withheld_rows"], 27)
        self.assertEqual(len(result["controller_rows"]), 108)
        # All three fixture arms really have the same complete visible reply.
        self.assertEqual(len(result["canonical_packets"]), 27)
        self.assertEqual(result["counts"]["reused_canonical_rows"], 54)
        for packet in result["canonical_packets"]:
            self.assertEqual(set(packet), review.PACKET_FIELDS)
            self.assertEqual(packet["packet_sha256"], review.stable(
                {key: packet[key] for key in ("history", "input", "response")}))

    def test_review_projection_removes_allocation_oracle_and_old_scores(self):
        result = review.build_review(*fixture())
        visible = json.dumps(result["packets"], ensure_ascii=False)
        for forbidden in ("GOLD_SECRET", "OLD_SCORE", "completion_criterion", "scoring_source_references",
                          "short_term", "raw_history_rag", "managed_memory_rag", "ORIGINAL-ID"):
            self.assertNotIn(forbidden, visible)
        for packet in result["packets"]:
            self.assertEqual(set(packet), review.PACKET_FIELDS)
            self.assertEqual(set(packet["history"][0]), {"message_id", "role", "content", "date"})
            self.assertEqual(packet["history"][0]["message_id"], "u1")

    def test_answer_is_preserved_without_inventing_a_displayed_clarification(self):
        source = fixture()
        original = deepcopy(source)
        result = review.build_review(*source)
        self.assertEqual(source, original)
        self.assertTrue(all("separate question" not in p["response"] for p in result["packets"]))
        pending = [r for r in result["controller_rows"] if r["review_status"] == "pending"]
        self.assertTrue(all(r["source_clarification"]["question"] == "A separate question?" for r in pending))
        self.assertTrue(all(r["clarification_question_exactly_contained_in_answer"] is False for r in pending))
        self.assertTrue(all(r["response_view"] == "answer_field_only" for r in pending))

    def test_reviewer_preserves_role_date_and_exact_source_text(self):
        packets, allocation, controller, summary = fixture()
        result = review.build_review(packets, allocation, controller, summary)
        expected = {(p["source_history"][0]["text"], p["source_history"][0]["role"],
                     p["source_history"][0]["date"]) for p in packets}
        self.assertEqual({(p["history"][0]["content"], p["history"][0]["role"],
                           p["history"][0]["date"]) for p in result["packets"]}, expected)

    def test_withheld_rows_cannot_be_resurrected(self):
        packets, allocation, controller, summary = fixture()
        withheld = next(r for r in controller["planned_review_rows"] if r["privacy_reference_withheld"])
        withheld["packet_id"] = packets[0]["packet_id"]
        with self.assertRaisesRegex(ValueError, "privacy-withheld"):
            review.build_review(packets, allocation, controller, summary)

    def test_dropping_a_planned_failure_or_packet_is_rejected(self):
        for target in ("planned", "packet"):
            with self.subTest(target=target):
                packets, allocation, controller, summary = fixture()
                if target == "planned":
                    controller["planned_review_rows"].pop()
                else:
                    packets.pop()
                with self.assertRaises(ValueError):
                    review.build_review(packets, allocation, controller, summary)

    def test_equal_answer_with_different_history_cannot_share_review(self):
        packets, allocation, controller, summary = fixture()
        packets[0]["source_history"][0]["text"] = "Different history from one arm."
        packets[0]["source_history"][0]["text_sha256"] = hashlib.sha256(
            packets[0]["source_history"][0]["text"].encode()).hexdigest()
        reseal(packets[0])
        with self.assertRaisesRegex(ValueError, "share review history"):
            review.build_review(packets, allocation, controller, summary)

    def test_tampered_response_and_source_hashes_are_rejected(self):
        for field in ("response", "source"):
            with self.subTest(field=field):
                packets, _, _, _ = fixture()
                packet = packets[0]
                if field == "response":
                    packet["original_response"] += " "
                else:
                    packet["source_history"][0]["text"] += " changed"
                reseal(packet)
                with self.assertRaises(ValueError):
                    review.visible_payload(packet)

    def test_restricted_source_history_fails_closed(self):
        packets, _, _, _ = fixture()
        packets[0]["source_history"][0]["storage_policy"] = "dont_save"
        reseal(packets[0])
        with self.assertRaisesRegex(ValueError, "explicitly permitted"):
            review.visible_payload(packets[0])

    def test_historical_identity_and_cost_cannot_be_relabelled(self):
        for target in ("executable", "cost"):
            with self.subTest(target=target):
                packets, allocation, controller, summary = fixture()
                if target == "executable":
                    summary["executable_commit"] = "CURRENT_PRODUCT_NOT_HISTORICAL"
                else:
                    summary["cost"]["recorded_provider_attempts"] = 0
                with self.assertRaises(ValueError):
                    review.build_review(packets, allocation, controller, summary)


class ArchiveTests(unittest.TestCase):
    def test_duplicate_json_keys_and_nonfinite_numbers_are_rejected(self):
        for body in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.subTest(body=body), self.assertRaises(ValueError):
                review.read_json(body)

    def test_checksum_change_rejects_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sources = [root / name for name in ("reviewer.zip", "controller.zip", "report.json")]
            for path in sources:
                path.write_bytes(b"not the frozen source")
            out = root / "new-review"
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                review.prepare(*sources, out)
            self.assertFalse(out.exists())

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with self.assertRaisesRegex(ValueError, "fresh output"):
                review.prepare(out / "missing1", out / "missing2", out / "missing3", out)

    def test_traversal_archive_members_are_not_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../untrusted.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unsafe archive"):
                review._read_archive(path)


if __name__ == "__main__":
    unittest.main()
