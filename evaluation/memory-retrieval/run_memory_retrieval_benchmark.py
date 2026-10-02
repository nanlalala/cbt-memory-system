#!/usr/bin/env python3
"""Run evidence retrieval on the official LongMemEval or LoCoMo release.

This evaluator measures retrieval against dataset-provided evidence IDs.  It does
not use locally authored questions or model-generated gold answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "modules/long-term-memory/schema"))
sys.path.insert(0, str(ROOT / "modules/long-term-memory/retrieval"))

from benchmark_adapters import (  # noqa: E402
    RetrievalExample,
    evidence_retrieval_metrics,
    load_locomo,
    load_longmemeval,
)
from memory_retriever import MultilingualE5Encoder  # noqa: E402


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    document_id: str
    text: str
    date: str | None = None


def documents_for_example(example: RetrievalExample, dataset: str) -> list[BenchmarkDocument]:
    documents: list[BenchmarkDocument] = []
    for session in example.sessions:
        turns = session["turns"]
        if dataset == "longmemeval":
            text = "\n".join(
                f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns
            )
            documents.append(BenchmarkDocument(str(session["session_id"]), text, session.get("date")))
        else:
            for turn in turns:
                documents.append(
                    BenchmarkDocument(
                        str(turn["dia_id"]),
                        f"{turn.get('speaker', 'unknown')}: {turn.get('text', '')}",
                        session.get("date"),
                    )
                )
    return documents


def _dense_scores(encoder: MultilingualE5Encoder, query: str, document_vectors: Any) -> list[float]:
    query_vector = encoder.encode([query], is_query=True)[0]
    # E5 returns normalized embeddings; dot product is cosine similarity.
    return [sum(float(a) * float(b) for a, b in zip(query_vector, vector)) for vector in document_vectors]


def run_benchmark(
    examples: Sequence[RetrievalExample],
    *,
    dataset: str,
    backend: str = "lexical",
    top_k: int = 10,
    model_name: str = "intfloat/multilingual-e5-small",
    device: str | None = None,
) -> tuple[dict[str, float], list[dict]]:
    encoder = MultilingualE5Encoder(model_name, device=device) if backend in {"dense", "hybrid"} else None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise RuntimeError("The benchmark runner requires scikit-learn") from exc
    predictions: dict[str, list[str]] = {}
    details: list[dict] = []
    corpus_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, example in enumerate(examples, 1):
        documents = documents_for_example(example, dataset)
        texts = [item.text for item in documents]
        corpus_key = tuple(item.document_id for item in documents)
        cached = corpus_cache.get(corpus_key)
        if cached is None:
            vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
            document_matrix = vectorizer.fit_transform(texts)
            cached = {"vectorizer": vectorizer, "document_matrix": document_matrix}
            if encoder:
                cached["dense_vectors"] = encoder.encode(texts, is_query=False)
            corpus_cache[corpus_key] = cached
        query_vector = cached["vectorizer"].transform([example.query])
        lexical = (cached["document_matrix"] @ query_vector.T).toarray().ravel().tolist()
        dense = (
            _dense_scores(encoder, example.query, cached["dense_vectors"])
            if encoder
            else [0.0] * len(texts)
        )
        if backend == "lexical":
            scores = lexical
        elif backend == "dense":
            scores = dense
        else:
            # Simple score fusion; component values remain available in the result file.
            scores = [0.45 * l_score + 0.55 * d_score for l_score, d_score in zip(lexical, dense)]
        order = sorted(range(len(documents)), key=lambda value: (-scores[value], documents[value].document_id))
        ranked = [documents[value].document_id for value in order[:top_k]]
        predictions[example.example_id] = ranked
        details.append(
            {
                "example_id": example.example_id,
                "category": example.category,
                "query": example.query,
                "gold_evidence_ids": list(example.evidence_ids),
                "predicted_ids": ranked,
                "should_abstain": example.should_abstain,
            }
        )
        if index % 100 == 0:
            print(f"processed {index}/{len(examples)}", file=sys.stderr)

    metrics: dict[str, float] = {"examples": len(examples)}
    for k in (1, 5, 10):
        metrics.update(evidence_retrieval_metrics(examples, predictions, k=k))
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("longmemeval", "locomo"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backend", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = load_longmemeval(args.input) if args.dataset == "longmemeval" else load_locomo(args.input)
    metrics, details = run_benchmark(
        examples,
        dataset=args.dataset,
        backend=args.backend,
        top_k=args.top_k,
        model_name=args.model,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "backend": args.backend,
        "model": args.model if args.backend != "lexical" else None,
        "metrics": metrics,
        "predictions": details,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
