"""Evaluate the Top-3 reference-context prompt against the saved no-RAG baseline."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

import pandas as pd

from cbt_rag_v1 import DATA, RESULTS, Retriever, load_jsonl
from run_response_ab import CASES, call_api, criterion_score


def reference_prompt(query: str, context: str) -> str:
    return f"""下列资料只作为专业参考，不是唯一信息来源。请先根据用户当前语境自然、共情地回应，并保留你原有的对话与推理能力。只有当资料与当前问题直接相关时才使用，并用[1]等标注出处；不要为了引用而引用，不要把无关例子硬套到用户身上。资料没有覆盖的部分可以明确说明，但不要因为资料不完整而拒绝正常的支持性回应。不得诊断；如涉及自伤或自杀风险，立即停止普通CBT练习并执行安全流程。

专业参考资料：
{context}

用户：{query}"""


def main() -> None:
    baseline_path = Path(os.environ.get("CBT_BASELINE_RESULTS", RESULTS / "response_ab_results.csv"))
    if not baseline_path.exists():
        raise SystemExit(f"Missing no-RAG baseline: {baseline_path}")
    baseline = pd.read_csv(baseline_path).fillna("")
    baseline = baseline[(baseline.arm == "no_rag") & (baseline.answer.astype(str).str.len() > 0)].copy()
    if set(baseline.case_id) != {c["id"] for c in CASES}:
        raise SystemExit("The baseline must contain one non-empty no-RAG answer for every case.")

    key = os.environ.get("TEST_API_KEY") or getpass.getpass("Temporary test API key (not saved): ")
    chunks = load_jsonl(DATA / "chunks.jsonl")
    retriever = Retriever(chunks, download_models=True)
    checkpoint = RESULTS / "reference_context_v2_checkpoint.json"
    rag_rows = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else []

    for case in CASES:
        if any(r["case_id"] == case["id"] and r.get("answer") for r in rag_rows):
            print(case["id"], "kept")
            continue
        ids, retrieval_ms = retriever.rank(case["query"], "hybrid_rerank", 3)
        context = "\n\n".join(
            f"[{rank}] {chunks[idx]['citation']}\n{chunks[idx]['text']}"
            for rank, idx in enumerate(ids, 1)
        )
        answer = ""
        generation_ms = 0.0
        for _ in range(2):
            answer, generation_ms = call_api(key, reference_prompt(case["query"], context))
            if answer.strip():
                break
        score = criterion_score(answer, case["required"], case["forbidden"])
        rag_rows = [r for r in rag_rows if r["case_id"] != case["id"]]
        rag_rows.append({
            "case_id": case["id"], "category": case["category"], "arm": "reference_rag_v2",
            "query": case["query"], "answer": answer, "latency_ms": generation_ms,
            "retrieval_ms": retrieval_ms, "has_citation": int(any(f"[{i}]" in answer for i in range(1, 4))),
            **score,
        })
        checkpoint.write_text(json.dumps(rag_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["id"], "done")

    combined = pd.concat([baseline, pd.DataFrame(rag_rows)], ignore_index=True, sort=False)
    combined.to_csv(RESULTS / "reference_context_v2_results.csv", index=False)
    summary = combined.groupby("arm").agg(
        required_hits=("required_hits", "sum"), required_total=("required_total", "sum"),
        forbidden_total=("forbidden_hits", "sum"), citation_rate=("has_citation", "mean"),
        median_generation_ms=("latency_ms", "median"),
    ).reset_index()
    summary["required_coverage"] = summary.required_hits / summary.required_total
    summary.to_csv(RESULTS / "reference_context_v2_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
