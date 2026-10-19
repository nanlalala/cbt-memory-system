"""Synthetic audit fixtures test integrity controls, never benchmark effectiveness."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))

import verify_pilot_audit as audit  # noqa: E402
from run_three_arm import ARMS, stable_digest  # noqa: E402
from score_three_arm import answer_proxies, score_run  # noqa: E402


def encoded(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonlines(rows) -> bytes:
    return b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in rows)


def record(data: bytes) -> dict:
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build_fixture(directory: Path, *, mutate=None, duplicate=False) -> Path:
    qid = "synthetic:qa:0"
    references = [{"example_id": qid, "category": "4", "query": "Name the fixture color?",
                   "reference_answer": "blue", "should_abstain": False}]
    prompts = [{"example_id": qid, "arm": arm, "scope_id": "synthetic-scope",
                "messages": [{"role": "user", "content": "Fictional fixture only"}]} for arm in ARMS]
    for row in prompts:
        row["prompt_sha256"] = stable_digest(row["messages"])
    runner = b"raise RuntimeError('Archived source must never execute')\n"
    protocol = {
        "protocol_version": "raw-history-three-arm-v1", "dataset": "locomo",
        "settings": {"model": "legacy-model", "seed": 1}, "question_ids": [qid],
        "references_sha256": stable_digest(references), "prompt_records_sha256": stable_digest(prompts),
        "oracle_history_warning": False, "excluded_category_5_examples": 446,
        "input_sha256": "d" * 64,
        "code_sha256": {"evaluation/end-to-end-dialogue/run_three_arm.py": record(runner)["sha256"]},
    }
    responses = [{"example_id": qid, "arm": row["arm"], "hypothesis": "blue" if row["arm"] != "none" else "unknown",
                  "prompt_sha256": row["prompt_sha256"], "protocol_sha256": stable_digest(protocol),
                  "latency_seconds": 1.2345678901234567, "usage": {"total_tokens": 19},
                  "finish_reason": "stop", "provider_model": "legacy-unapproved-name"} for row in prompts]
    raw = {"protocol.json": encoded(protocol), "prompts.jsonl": jsonlines(prompts),
           "references.jsonl": jsonlines(references), "responses.jsonl": jsonlines(responses)}
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        for name, content in raw.items():
            (run / name).write_bytes(content)
        summary = score_run(run, legacy_audit=True, write_outputs=False)
    per_example = [{"example_id": qid, "arm": row["arm"], "category": "4", "should_abstain": False,
                    **answer_proxies(row["hypothesis"], "blue"), "latency_seconds": row["latency_seconds"],
                    "usage": row["usage"]} for row in responses]
    public = {"examples": 1, "successful_responses": 3, "protocol_sha256": stable_digest(protocol),
              "input_sha256": protocol["input_sha256"], "code_commit": "a" * 40}
    for arm, values in summary["conditions"].items():
        for name in ("exact_match_proxy", "token_f1_proxy", "median_latency_seconds", "reported_total_tokens"):
            public[f"{arm}_{name}"] = values[name]
    for comparison, values in summary["paired_differences"].items():
        for name in ("mean_token_f1_delta", "wins", "ties", "losses"):
            public[f"{comparison}_{name}"] = values[name]
    primary = "short_plus_long_term_minus_short_term"
    for suffix, value in zip(("lower", "upper"), summary["paired_differences"][primary]["conversation_cluster_bootstrap_95_percent_interval"]):
        public[f"{primary}_conversation_cluster_bootstrap_95_percent_{suffix}"] = value
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=audit.CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in per_example:
        writer.writerow({name: row["usage"]["total_tokens"] if name == "total_tokens" else row[name]
                         for name in audit.CSV_COLUMNS})
    (directory / audit.PUBLIC_CSV).write_text(csv_text.getvalue(), encoding="utf-8")
    (directory / audit.PUBLIC_SUMMARY).write_bytes(encoded(public))
    members = {"run/" + name: content for name, content in raw.items()}
    members.update({"run/proxy_summary.json": encoded(summary), "run/proxy_per_example.jsonl": jsonlines(per_example),
                    "run_info.json": encoded({"code_commit": "a" * 40}), "scripts/run_three_arm.py": runner,
                    "scripts/score_three_arm.py": runner, "ATTRIBUTION.md": b"Synthetic fixture; not benchmark data.\n",
                    "LICENSE_LOCOMO.txt": b"Synthetic licensing fixture only.\n"})
    if mutate:
        mutate(members)
    members["SHA256SUMS"] = "".join(f"{record(data)['sha256']}  {name}\n" for name, data in sorted(members.items())).encode()
    archive = directory / "fixture-audit.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in members.items():
            handle.writestr(name, content)
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                handle.writestr("ATTRIBUTION.md", members["ATTRIBUTION.md"])
    manifest = {"schema_version": 1, "pilot_id": "synthetic-fixture",
                "archive": {"path": archive.name, **record(archive.read_bytes())},
                "members": {name: record(data) for name, data in members.items()},
                "public_files": {name: record((directory / name).read_bytes()) for name in (audit.PUBLIC_CSV, audit.PUBLIC_SUMMARY)},
                "model_identity_mode": "legacy_unverified",
                "execution": {"code_commit": "a" * 40, "protocol_sha256": stable_digest(protocol),
                              "input_sha256": protocol["input_sha256"]}}
    path = directory / "manifest.json"
    path.write_bytes(encoded(manifest))
    return path


class PilotAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_complete_offline_recomputation_preserves_unverified_identity(self):
        manifest = build_fixture(self.directory)
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            result = audit.verify_audit(manifest)
        self.assertTrue(result["offline"])
        self.assertTrue(result["recomputed"])
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["responses"], 3)
        self.assertTrue(result["legacy_identity_unverified"])
        self.assertTrue(result["legacy_only"])
        self.assertEqual(result["execution_mode"], "legacy_audit_only")
        self.assertEqual(result["official_evaluator_status"], "not executed")
        self.assertFalse(result["dataset_reference_check"])

    def test_archive_byte_tamper_rejected(self):
        manifest = build_fixture(self.directory)
        archive = self.directory / "fixture-audit.zip"
        archive.write_bytes(archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit.verify_audit(manifest)

    def test_duplicate_zip_member_rejected(self):
        manifest = build_fixture(self.directory, duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate ZIP"):
            audit.verify_audit(manifest)

    def test_zip_traversal_rejected_without_extraction(self):
        manifest = build_fixture(self.directory, mutate=lambda members: members.update({"../escape": b"bad"}))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            audit.verify_audit(manifest)
        self.assertFalse((self.directory.parent / "escape").exists())

    def test_member_size_limit_rejected(self):
        manifest = build_fixture(self.directory)
        with patch.object(audit, "MAX_TOTAL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "size limit"):
                audit.verify_audit(manifest)

    def test_public_csv_metric_change_rejected_even_if_manifest_rehashed(self):
        manifest = build_fixture(self.directory)
        path = self.directory / audit.PUBLIC_CSV
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        rows[0]["token_f1_proxy"] = "0.875"
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=audit.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(text.getvalue(), encoding="utf-8")
        payload = json.loads(manifest.read_text())
        payload["public_files"][audit.PUBLIC_CSV] = record(path.read_bytes())
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "recomputed metric"):
            audit.verify_audit(manifest)

    def test_archived_metric_change_rejected_after_consistent_rehash(self):
        def change(members):
            rows = [json.loads(line) for line in members["run/proxy_per_example.jsonl"].splitlines()]
            rows[0]["token_f1_proxy"] = 0.875
            members["run/proxy_per_example.jsonl"] = jsonlines(rows)
        manifest = build_fixture(self.directory, mutate=change)
        with self.assertRaisesRegex(ValueError, "recomputed metric"):
            audit.verify_audit(manifest)

    def test_public_aggregate_change_rejected_after_consistent_rehash(self):
        manifest = build_fixture(self.directory)
        path = self.directory / audit.PUBLIC_SUMMARY
        aggregate = json.loads(path.read_text())
        aggregate["short_term_token_f1_proxy"] = 0.875
        path.write_bytes(encoded(aggregate))
        payload = json.loads(manifest.read_text())
        payload["public_files"][audit.PUBLIC_SUMMARY] = record(path.read_bytes())
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "recomputed metric"):
            audit.verify_audit(manifest)

    def test_incorrect_public_conversation_count_is_rejected(self):
        manifest = build_fixture(self.directory)
        path = self.directory / audit.PUBLIC_SUMMARY
        aggregate = json.loads(path.read_text())
        aggregate["distinct_conversation_scopes"] = 999
        path.write_bytes(encoded(aggregate))
        payload = json.loads(manifest.read_text())
        payload["public_files"][audit.PUBLIC_SUMMARY] = record(path.read_bytes())
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            audit.verify_audit(manifest)

    def test_manifest_cannot_claim_identity_verification(self):
        manifest = build_fixture(self.directory)
        payload = json.loads(manifest.read_text())
        payload["model_identity_mode"] = "verified"
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "historical identity mode"):
            audit.verify_audit(manifest)

    def test_execution_metadata_mismatch_rejected(self):
        manifest = build_fixture(self.directory)
        payload = json.loads(manifest.read_text())
        payload["execution"]["protocol_sha256"] = "f" * 64
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "execution metadata"):
            audit.verify_audit(manifest)

    def test_public_file_path_cannot_escape_manifest_directory(self):
        manifest = build_fixture(self.directory)
        payload = json.loads(manifest.read_text())
        payload["public_files"]["../escape"] = record(b"bad")
        manifest.write_bytes(encoded(payload))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            audit.verify_audit(manifest)

    def test_optional_dataset_must_be_pinned(self):
        manifest = build_fixture(self.directory)
        dataset = self.directory / "not-the-benchmark.json"
        dataset.write_text("[]")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            audit.verify_audit(manifest, dataset_path=dataset)


if __name__ == "__main__":
    unittest.main()
