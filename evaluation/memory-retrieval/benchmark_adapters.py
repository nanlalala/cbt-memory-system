"""Adapters for the official LongMemEval and LoCoMo dataset formats."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    example_id: str
    category: str
    query: str
    answer: str
    evidence_ids: tuple[str, ...]
    sessions: tuple[dict, ...]
    should_abstain: bool = False


def load_longmemeval(path: str | Path) -> list[RetrievalExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[RetrievalExample] = []
    for item in data:
        required = {
            "question_id",
            "question_type",
            "question",
            "answer",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
            "answer_session_ids",
        }
        missing = required - item.keys()
        if missing:
            raise ValueError(f"LongMemEval item is missing: {sorted(missing)}")
        session_ids = item["haystack_session_ids"]
        dates = item["haystack_dates"]
        sessions = item["haystack_sessions"]
        if not (len(session_ids) == len(dates) == len(sessions)):
            raise ValueError("LongMemEval session ids, dates and contents must align")
        packed = tuple(
            {"session_id": sid, "date": date, "turns": turns}
            for sid, date, turns in zip(session_ids, dates, sessions)
        )
        question_id = str(item["question_id"])
        examples.append(
            RetrievalExample(
                example_id=question_id,
                category=str(item["question_type"]),
                query=str(item["question"]),
                answer=str(item["answer"]),
                evidence_ids=tuple(str(value) for value in item["answer_session_ids"]),
                sessions=packed,
                should_abstain=question_id.endswith("_abs"),
            )
        )
    return examples


def load_locomo(path: str | Path) -> list[RetrievalExample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples: list[RetrievalExample] = []
    for sample in data:
        conversation = sample["conversation"]
        session_names = sorted(
            (
                key
                for key in conversation
                if key.startswith("session_")
                and not key.endswith("_date_time")
                and key[len("session_") :].isdigit()
            ),
            key=lambda value: int(value.split("_")[1]),
        )
        sessions = tuple(
            {
                "session_id": f"{sample['sample_id']}:{name}",
                "date": conversation.get(f"{name}_date_time"),
                "turns": conversation[name],
            }
            for name in session_names
        )
        for index, qa in enumerate(sample.get("qa", [])):
            answer = qa.get("answer", qa.get("adversarial_answer"))
            if answer is None:
                raise ValueError("LoCoMo QA item has neither answer nor adversarial_answer")
            examples.append(
                RetrievalExample(
                    example_id=f"{sample['sample_id']}:qa:{index}",
                    category=str(qa.get("category", "unknown")),
                    query=str(qa["question"]),
                    answer=str(answer),
                    evidence_ids=tuple(str(value) for value in qa.get("evidence", [])),
                    sessions=sessions,
                    should_abstain=False,
                )
            )
    return examples


def evidence_retrieval_metrics(
    examples: Iterable[RetrievalExample], predictions: dict[str, list[str]], *, k: int = 5
) -> dict[str, float]:
    """Compute evidence Recall@K and MRR using benchmark-provided evidence IDs."""

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for example in examples:
        if example.should_abstain or not example.evidence_ids:
            continue
        predicted = predictions.get(example.example_id, [])[:k]
        gold = set(example.evidence_ids)
        recalls.append(len(gold.intersection(predicted)) / len(gold))
        first = next((rank for rank, value in enumerate(predicted, 1) if value in gold), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
    if not recalls:
        return {f"evidence_recall_at_{k}": 0.0, f"mrr_at_{k}": 0.0, "n": 0}
    return {
        f"evidence_recall_at_{k}": sum(recalls) / len(recalls),
        f"mrr_at_{k}": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "n": len(recalls),
    }


def write_longmemeval_hypotheses(
    answers: dict[str, str], output_path: str | Path
) -> None:
    """Write the exact JSONL format consumed by LongMemEval's official evaluator."""

    output = Path(output_path)
    with output.open("w", encoding="utf-8") as handle:
        for question_id, hypothesis in answers.items():
            handle.write(
                json.dumps(
                    {"question_id": question_id, "hypothesis": hypothesis}, ensure_ascii=False
                )
                + "\n"
            )
