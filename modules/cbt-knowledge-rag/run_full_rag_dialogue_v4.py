"""Fresh no-RAG vs full seven-source hybrid+reranker dialogue evaluation.

This experiment deliberately does not reuse answers or scores from the public-reference
recovery experiment. Both arms are generated afresh with identical decoding settings.
"""
from __future__ import annotations

import getpass
import json
import os
import random
import re
from pathlib import Path

import pandas as pd

from cbt_rag_v1 import DATA, RESULTS, Retriever, load_jsonl
from run_expanded_dialogue_v3 import (
    CASES,
    RETRIEVAL_QUERIES,
    RUBRIC,
    SYSTEM,
    api_call,
    behavior_coverage,
    judge_prompt,
    parse_json,
    plain_prompt,
    rag_prompt,
)


ARMS = ("no_rag", "full_hybrid_rerank_rag")
CHECKPOINT = RESULTS / "full_rag_dialogue_v4_checkpoint.json"
SCORE_CHECKPOINT = RESULTS / "full_rag_dialogue_v4_score_checkpoint.json"
DETAIL_CSV = RESULTS / "full_rag_dialogue_v4_results.csv"
SCORE_CSV = RESULTS / "full_rag_dialogue_v4_model_assisted_scores.csv"
SUMMARY_CSV = RESULTS / "full_rag_dialogue_v4_summary.csv"
RETRIEVAL_AUDIT_CSV = RESULTS / "full_rag_dialogue_v4_retrieval_audit.csv"


def validate_full_corpus(chunks: list[dict]) -> None:
    expected = {
        "who_mhgap_2023",
        "va_brief_cbt",
        "va_mybriefcbt",
        "va_cbt_depression",
        "cci_lipi",
        "beck_cbt_3e_sample",
        "tolin_doing_cbt_2e_sample",
    }
    present = {c["source_id"] for c in chunks}
    if present != expected:
        raise RuntimeError(
            "This script requires the complete seven-source corpus. "
            f"Missing={sorted(expected-present)}, unexpected={sorted(present-expected)}"
        )


def main() -> None:
    if not os.environ.get("CBT_TEST_BASE_URL") or not os.environ.get("CBT_TEST_MODEL"):
        raise SystemExit("Set CBT_TEST_BASE_URL and CBT_TEST_MODEL")

    # run_expanded_dialogue_v3 reads these variables at import time; validate their values.
    from run_expanded_dialogue_v3 import BASE_URL, MODEL

    if not BASE_URL or not MODEL:
        raise SystemExit("Endpoint/model environment variables were not visible at import time")

    key = os.environ.get("TEST_API_KEY") or getpass.getpass("Temporary test API key (not saved): ")
    chunks = load_jsonl(DATA / "chunks.jsonl")
    validate_full_corpus(chunks)
    retriever = Retriever(chunks, download_models=True)

    retrieval: dict[str, tuple[list[int], float]] = {}
    audit_rows: list[dict] = []
    for case in CASES:
        ids, latency = retriever.rank(RETRIEVAL_QUERIES[case["id"]], "hybrid_rerank", 3)
        retrieval[case["id"]] = (ids, latency)
        for rank, idx in enumerate(ids, 1):
            chunk = chunks[idx]
            audit_rows.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "source_id": chunk["source_id"],
                    "citation": chunk["citation"],
                    "topics": ",".join(chunk["cbt_topics"]),
                    "excerpt": chunk["text"][:500].replace("\n", " "),
                    "retrieval_ms": latency,
                }
            )
    pd.DataFrame(audit_rows).to_csv(RETRIEVAL_AUDIT_CSV, index=False)

    rows = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else []
    for case in CASES:
        ids, retrieval_ms = retrieval[case["id"]]
        context = "\n\n".join(
            f"[{n}] {chunks[idx]['citation']}\n{chunks[idx]['text']}"
            for n, idx in enumerate(ids, 1)
        )
        prompts = {
            "no_rag": plain_prompt(case["dialogue"]),
            "full_hybrid_rerank_rag": rag_prompt(case["dialogue"], context),
        }
        for arm in ARMS:
            if any(r["case_id"] == case["id"] and r["arm"] == arm and r.get("answer") for r in rows):
                continue
            answer, latency = api_call(key, SYSTEM, prompts[arm], 1400)
            rows = [r for r in rows if not (r["case_id"] == case["id"] and r["arm"] == arm)]
            rows.append(
                {
                    "case_id": case["id"],
                    "category": case["category"],
                    "arm": arm,
                    "answer": answer,
                    "latency_ms": latency,
                    "retrieval_ms": retrieval_ms if arm.endswith("rag") else 0,
                    "has_citation": int(bool(re.search(r"\[[1-3]\]", answer))),
                    "expected_proxy_hits": behavior_coverage(answer, case["expected"]),
                    "expected_total": len(case["expected"]),
                    "finish_truncated": int(bool(answer) and not re.search(r"[。！？.!?）)]$", answer.strip())),
                }
            )
            CHECKPOINT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(case["id"], arm, "done", flush=True)

    answers = {(r["case_id"], r["arm"]): r["answer"] for r in rows}
    scores = json.loads(SCORE_CHECKPOINT.read_text()) if SCORE_CHECKPOINT.exists() else []
    rng = random.Random(20260828)
    for case in CASES:
        if any(s["case_id"] == case["id"] for s in scores):
            continue
        no = answers[(case["id"], "no_rag")]
        rag = answers[(case["id"], "full_hybrid_rerank_rag")]
        rag_is_a = bool(rng.getrandbits(1))
        a, b = (rag, no) if rag_is_a else (no, rag)
        judged = None
        last_raw = ""
        for _ in range(3):
            last_raw, _ = api_call(
                key,
                "你只执行严格、可复核的匿名量表评分，并输出合法JSON。",
                judge_prompt(case, a, b),
                3000,
            )
            try:
                candidate = parse_json(last_raw)
                if all(
                    label in candidate and all(dim in candidate[label] for dim in RUBRIC)
                    for label in ("A", "B")
                ):
                    judged = candidate
                    break
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        if judged is None:
            (RESULTS / f"full_rag_v4_judge_parse_failure_{case['id']}.txt").write_text(
                last_raw, encoding="utf-8"
            )
            raise RuntimeError(f"Judge returned invalid JSON for {case['id']} after 3 attempts")

        mapping = {
            "A": "full_hybrid_rerank_rag" if rag_is_a else "no_rag",
            "B": "no_rag" if rag_is_a else "full_hybrid_rerank_rag",
        }
        item = {
            "case_id": case["id"],
            "category": case["category"],
            "rag_was_answer_a": int(rag_is_a),
            "winner": mapping.get(judged.get("winner"), "tie"),
            "reason": judged.get("reason", ""),
        }
        for label, arm in mapping.items():
            vals = judged[label]
            for dim in RUBRIC:
                item[f"{arm}_{dim}"] = int(vals[dim])
            item[f"{arm}_total"] = sum(int(vals[d]) for d in RUBRIC)
        scores.append(item)
        SCORE_CHECKPOINT.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        print(case["id"], "scored", flush=True)

    detail = pd.DataFrame(rows)
    score_df = pd.DataFrame(scores)
    detail.to_csv(DETAIL_CSV, index=False)
    score_df.to_csv(SCORE_CSV, index=False)

    summary: list[dict] = []
    for arm in ARMS:
        arm_rows = detail[detail.arm == arm]
        summary.append(
            {
                "arm": arm,
                "n": len(CASES),
                "mean_model_assisted_score_12": score_df[f"{arm}_total"].mean(),
                "mean_expected_proxy_coverage": arm_rows.expected_proxy_hits.sum()
                / arm_rows.expected_total.sum(),
                "citation_rate": arm_rows.has_citation.mean(),
                "truncation_rate": arm_rows.finish_truncated.mean(),
                "median_latency_ms": arm_rows.latency_ms.median(),
                "pairwise_wins": int((score_df.winner == arm).sum()),
                "ties": int((score_df.winner == "tie").sum()),
            }
        )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
