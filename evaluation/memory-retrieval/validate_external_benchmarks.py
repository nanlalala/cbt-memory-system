"""Verify downloaded benchmark identity and parse the official data formats."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_adapters import load_locomo, load_longmemeval


ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "benchmark_manifest.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_identity(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"unexpected file size for {path}")
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longmemeval", type=Path)
    parser.add_argument("--locomo", type=Path)
    args = parser.parse_args()
    if not args.longmemeval and not args.locomo:
        raise SystemExit("Provide --longmemeval and/or --locomo")

    if args.longmemeval:
        spec = MANIFEST["benchmarks"]["longmemeval_oracle_cleaned"]
        require_identity(args.longmemeval, spec)
        examples = load_longmemeval(args.longmemeval)
        abstention = sum(item.should_abstain for item in examples)
        if len(examples) != spec["expected_examples"]:
            raise ValueError("unexpected LongMemEval example count")
        if abstention != spec["expected_abstention_examples"]:
            raise ValueError("unexpected LongMemEval abstention count")
        print(f"LongMemEval verified: {len(examples)} examples, {abstention} abstention.")

    if args.locomo:
        spec = MANIFEST["benchmarks"]["locomo10"]
        require_identity(args.locomo, spec)
        examples = load_locomo(args.locomo)
        conversations = {item.example_id.split(":qa:")[0] for item in examples}
        adversarial = sum(item.category == "5" for item in examples)
        if len(examples) != spec["expected_qa_examples"]:
            raise ValueError("unexpected LoCoMo QA count")
        if len(conversations) != spec["expected_conversations"]:
            raise ValueError("unexpected LoCoMo conversation count")
        if adversarial != spec["expected_adversarial_examples"]:
            raise ValueError("unexpected LoCoMo adversarial count")
        print(
            f"LoCoMo verified: {len(conversations)} conversations, "
            f"{len(examples)} QA examples, {adversarial} adversarial."
        )


if __name__ == "__main__":
    main()
