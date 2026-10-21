#!/usr/bin/env python3
"""Recompute the 21 October Colab retrieval metrics from released evidence IDs.

Standard-library audit only: no model, API, GPU or full benchmark download.
This checks the published rankings and arithmetic, not conversational benefit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import zipfile


HERE = Path(__file__).resolve().parent
ARCHIVE_SHA256 = "de7f13bfc932518b1aa2faa430f733d7bc2b54d280364587a61c9f292f1cc59e"
REPORT_SHA256 = "0f351aa1e51ff51ed02e8f945355bf7457b1da7ef0d4f296cac523e15ab46c0e"
RANKINGS_SHA256 = "bdffaa46c5364cc9fbcd63909a75cddefafb9a485f7f35a2dda6f4d55996090e"


def verify() -> dict:
    archive_path = HERE / "colab_20261021_public_audit.zip"
    report_path = HERE / "colab_20261021_verification.json"
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive differs from the fixed Colab export")
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != REPORT_SHA256:
        raise ValueError("Public report differs from the fixed Colab export")
    with zipfile.ZipFile(archive_path) as archive:
        expected = {"colab_20261021_verification.json", "colab_20261021_rankings.jsonl",
                    "SHA256SUMS", "ATTRIBUTION.txt"}
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
            raise ValueError("Unexpected or duplicate archive members")
        report_bytes = archive.read("colab_20261021_verification.json")
        rankings_bytes = archive.read("colab_20261021_rankings.jsonl")
        if (report_bytes != report_path.read_bytes()
                or hashlib.sha256(rankings_bytes).hexdigest() != RANKINGS_SHA256):
            raise ValueError("Archived report or rankings identity mismatch")
        checksums = dict(line.split("  ", 1)[::-1] for line in
                         archive.read("SHA256SUMS").decode().splitlines())
        if checksums != {"colab_20261021_verification.json": REPORT_SHA256,
                         "colab_20261021_rankings.jsonl": RANKINGS_SHA256}:
            raise ValueError("Archive checksum manifest mismatch")
    report = json.loads(report_bytes)
    rows = [json.loads(line) for line in rankings_bytes.splitlines()]
    if len(rows) != 5958:
        raise ValueError("Expected 1,986 questions across three retrieval backends")
    by_backend = {name: [] for name in ("lexical", "dense", "hybrid")}
    annotations = {}
    for row in rows:
        if set(row) != {"backend", "example_id", "category", "gold_evidence_ids",
                        "predicted_ids", "should_abstain"} or row["backend"] not in by_backend:
            raise ValueError("Unexpected ranking fields or backend")
        if not all(isinstance(item, str) for item in row["gold_evidence_ids"] + row["predicted_ids"]):
            raise ValueError("Evidence IDs must be strings")
        signature = (row["category"], row["gold_evidence_ids"], row["should_abstain"])
        if row["example_id"] in annotations and annotations[row["example_id"]] != signature:
            raise ValueError("Reference annotations changed between retrieval backends")
        annotations[row["example_id"]] = signature
        by_backend[row["backend"]].append(row)
    results = {}
    for run in report["retrieval_runs"]:
        name = run["backend"]
        selected = by_backend[name]
        if len(selected) != 1986 or len({row["example_id"] for row in selected}) != 1986:
            raise ValueError("Missing or duplicate question/backend row")
        eligible = [row for row in selected if not row["should_abstain"] and row["gold_evidence_ids"]]
        calculated = {"examples": len(selected), "n": len(eligible)}
        for k in (1, 5, 10):
            recalls, ranks = [], []
            for row in eligible:
                gold, predicted = set(row["gold_evidence_ids"]), row["predicted_ids"][:k]
                recalls.append(len(gold.intersection(predicted)) / len(gold))
                first = next((i for i, value in enumerate(predicted, 1) if value in gold), None)
                ranks.append(1 / first if first else 0.0)
            calculated[f"evidence_recall_at_{k}"] = sum(recalls) / len(eligible)
            calculated[f"mrr_at_{k}"] = sum(ranks) / len(eligible)
        if set(calculated) != set(run["metrics"]) or any(
                not math.isclose(value, run["metrics"][key], rel_tol=0, abs_tol=1e-12)
                for key, value in calculated.items()):
            raise ValueError("Recomputed rankings do not match the published metrics")
        results[name] = calculated
    return {"verified": True, "ranking_rows": len(rows), "metrics": results,
            "model_or_api_calls": 0, "effectiveness_claim": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
