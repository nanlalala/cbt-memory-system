"""Actual archive export integrity; no models, human scores or participant data."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/dialogue-benefit"))
import prepare_legacy_cbt_review as review


class LegacyCBTReviewTests(unittest.TestCase):
    def test_complete_archive_is_masked_and_unscored_with_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "review"
            receipt = review.build_review(target)
            self.assertEqual(receipt["archived_responses"], 32)
            self.assertEqual(receipt["human_ratings"], 0)
            with review.SOURCE_REPLIES.open(encoding="utf-8") as handle:
                source = {(row["case_id"], row["arm"]): row["answer"] for row in csv.DictReader(handle)}
            keys = {row["packet_id"]: row for row in json.loads((target / "restricted/allocation.json").read_text())}
            orders = []
            for label in ("reviewer_1", "reviewer_2"):
                folder = target / label
                packets = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
                orders.append([packet["packet_id"] for packet in packets])
                self.assertEqual(len(packets), 32)
                for packet in packets:
                    key = keys[packet["packet_id"]]
                    self.assertEqual(packet["response"], source[(key["case_id"], key["arm"])])
                    self.assertEqual(packet["response_sha256"], hashlib.sha256(packet["response"].encode()).hexdigest())
                    self.assertNotIn("arm", packet)
                    self.assertNotIn("latency_ms", packet)
                    self.assertNotIn("model_assisted", packet)
                with (folder / "ratings.csv").open(encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        self.assertTrue(all(row[dim] == "" for dim in review.DIMENSIONS))
                self.assertFalse((folder / "allocation.json").exists())
            self.assertEqual(set(orders[0]), set(orders[1]))
            self.assertNotEqual(orders[0], orders[1])
            checksums = json.loads((target / "SHA256SUMS.json").read_text())
            for path, checksum in checksums.items():
                self.assertEqual(hashlib.sha256((target / path).read_bytes()).hexdigest(), checksum)
            with self.assertRaises(FileExistsError):
                review.build_review(target)

    def test_duplicate_unknown_or_missing_archive_rows_rejected_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with review.SOURCE_REPLIES.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                columns, rows = list(reader.fieldnames), list(reader)
            for index, bad in enumerate((rows[:-1], rows + [rows[0]], [dict(rows[0], arm="unknown")] + rows[1:],
                                         [dict(rows[0], answer="")] + rows[1:])):
                source = path / f"bad{index}.csv"
                with source.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(bad)
                with self.assertRaises(ValueError):
                    review.build_review(path / f"out{index}", source_replies=source)
                self.assertFalse((path / f"out{index}").exists())

    def test_source_case_loader_does_not_execute_module(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.py"
            path.write_text("raise RuntimeError('do not import me')\nCASES=[{'id':'one','category':'task','dialogue':'fictional'}]\n")
            self.assertEqual(review.load_cases(path)[0]["id"], "one")


if __name__ == "__main__":
    unittest.main()
