from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pypdf import PdfReader
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Small dependency-free BM25 fallback for reproducible CPU tests."""
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus = corpus
            self.k1, self.b = k1, b
            self.doc_len = np.asarray([len(d) for d in corpus], dtype=float)
            self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 1.0
            self.tf = [Counter(d) for d in corpus]
            df = Counter()
            for d in corpus:
                df.update(set(d))
            n = max(1, len(corpus))
            self.idf = {t: math.log(1 + (n - f + .5) / (f + .5)) for t, f in df.items()}

        def get_scores(self, query: list[str]) -> np.ndarray:
            scores = np.zeros(len(self.corpus), dtype=float)
            for term in query:
                idf = self.idf.get(term, 0.0)
                for i, freq in enumerate(self.tf):
                    f = freq.get(term, 0)
                    if f:
                        denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                        scores[i] += idf * f * (self.k1 + 1) / denom
            return scores
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
for d in (DATA, RESULTS, CACHE):
    d.mkdir(parents=True, exist_ok=True)

DENSE_MODEL = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

TAXONOMY = {
    "automatic_thoughts": ["automatic thought", "automatic thoughts", "hot thought", "cognition", "thought record"],
    "cognitive_restructuring": ["cognitive restructuring", "evaluate thought", "evidence for", "evidence against", "alternative thought"],
    "behavioural_activation": ["behavioral activation", "behavioural activation", "activity scheduling", "mastery", "pleasure"],
    "behavioural_experiment": ["behavioral experiment", "behavioural experiment", "test a prediction", "experiment"],
    "exposure": ["exposure", "fear hierarchy", "avoidance", "safety behavior", "safety behaviour"],
    "problem_solving": ["problem solving", "problem-solving", "brainstorm", "solution"],
    "goal_setting": ["goal", "smart goal", "action plan", "homework", "assignment", "between-session"],
    "case_formulation": ["case formulation", "conceptualization", "conceptualisation", "maintenance cycle", "formulation"],
    "core_beliefs": ["core belief", "intermediate belief", "conditional assumption", "schema"],
    "depression": ["depression", "depressive", "low mood", "anhedonia"],
    "anxiety": ["anxiety", "panic", "worry", "fear"],
    "suicide_safety": ["suicide", "suicidal", "self-harm", "self harm", "immediate risk", "crisis", "death",
                       "自杀", "自伤", "结束生命", "醒不过来", "死亡", "不想活"],
    "scope_referral": ["refer", "referral", "specialist", "emergency", "scope", "assessment", "diagnosis",
                       "转介", "紧急", "诊断", "专业人员"],
    "measurement": ["measure", "rating", "questionnaire", "monitor", "outcome", "progress"],
    "therapeutic_relationship": ["therapeutic relationship", "collaborative", "empathy", "rapport", "guided discovery"],
}


def norm_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\xa0", " ")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repeated_edge_lines(pages: list[str]) -> set[str]:
    counts: Counter[str] = Counter()
    for p in pages:
        lines = [re.sub(r"\s+", " ", x.strip()) for x in p.splitlines() if x.strip()]
        for line in lines[:3] + lines[-3:]:
            if 3 <= len(line) <= 100:
                counts[line] += 1
    threshold = max(4, int(len(pages) * 0.18))
    return {x for x, n in counts.items() if n >= threshold}


def clean_page(text: str, repeated: set[str]) -> str:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw.strip())
        if not line or line in repeated or re.fullmatch(r"(?:page\s*)?\d+", line, re.I):
            continue
        lines.append(line)
    return norm_text("\n".join(lines))


def is_heading(line: str) -> bool:
    s = line.strip()
    if not 4 <= len(s) <= 120 or s.endswith((".", ";", ",")):
        return False
    words = s.split()
    numbered = bool(re.match(r"^(?:chapter|session|module|part|step|\d+(?:\.\d+)*)\b", s, re.I))
    titleish = len(words) <= 12 and (s.isupper() or sum(w[:1].isupper() for w in words) >= max(2, len(words) * .65))
    return numbered or titleish


def tag_text(text: str) -> list[str]:
    low = text.lower()
    return sorted([tag for tag, terms in TAXONOMY.items() if any(t in low for t in terms)])


def usable_english_chunk(text: str) -> bool:
    """Reject common broken-font/OCR output without deleting ordinary tables."""
    if re.search(r"/g\d+|\b\d{3}:N|N\*:", text):
        return False
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 20:
        return True
    vowel_words = sum(bool(re.search(r"[aeiouy]", w, re.I)) for w in words)
    if vowel_words / len(words) < 0.28:
        return False
    controls = sum(ord(c) < 32 and c not in "\n\t" for c in text)
    return controls <= 2


def make_chunks(source: dict[str, Any], target_chars: int = 1400, overlap_chars: int = 180) -> list[dict[str, Any]]:
    pdf_path = (ROOT / source["file"]).resolve()
    reader = PdfReader(str(pdf_path))
    raw_pages = [(p.extract_text() or "") for p in reader.pages]
    repeated = repeated_edge_lines(raw_pages)
    chunks: list[dict[str, Any]] = []
    section = "Front matter"
    buffer = ""
    start_page = 1

    def flush(end_page: int, force: bool = False) -> None:
        nonlocal buffer, start_page
        while len(buffer) >= target_chars or (force and len(buffer) >= 220):
            cut = min(len(buffer), target_chars)
            if cut < len(buffer):
                candidates = [buffer.rfind(x, 0, cut) for x in ("\n\n", ". ", "; ")]
                best = max(candidates)
                if best > int(target_chars * .65):
                    cut = best + 1
            body = buffer[:cut].strip()
            if usable_english_chunk(section + "\n" + body):
                digest = hashlib.sha1(f"{source['id']}:{start_page}:{len(chunks)}:{body[:100]}".encode()).hexdigest()[:12]
                tags = tag_text(section + "\n" + body)
                risk = "high" if "suicide_safety" in tags else ("medium" if "scope_referral" in tags else "low")
                chunks.append({
                    "chunk_id": f"{source['id']}:{digest}", "source_id": source["id"], "title": source["title"],
                    "section": section, "page_start": start_page, "page_end": end_page,
                    "role": source["role"], "access": source["access"], "coverage": source["coverage"],
                    "cbt_topics": tags, "risk_level": risk, "text": body,
                    "citation": f"{source['title']}, pp. {start_page}-{end_page}",
                })
            if cut >= len(buffer):
                buffer = ""
            else:
                buffer = buffer[max(0, cut - overlap_chars):].strip()
                start_page = end_page

    for page_no, raw in enumerate(raw_pages, 1):
        page = clean_page(raw, repeated)
        if not page:
            continue
        for line in page.splitlines():
            if is_heading(line):
                flush(page_no, force=True)
                section = line[:120]
                start_page = page_no
            buffer += line + "\n"
            flush(page_no)
        buffer += "\n"
    flush(len(raw_pages), force=True)
    return chunks


def build_corpus() -> list[dict[str, Any]]:
    config_path = Path(os.environ.get("CBT_KB_CONFIG", ROOT / "knowledge_sources.yaml"))
    config = yaml.safe_load(config_path.read_text())
    all_chunks: list[dict[str, Any]] = []
    for source in config["sources"]:
        chunks = make_chunks(source)
        print(f"{source['id']}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    out = DATA / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for x in all_chunks:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    pd.DataFrame([{k: v for k, v in x.items() if k != "text"} for x in all_chunks]).to_csv(DATA / "chunk_catalog.csv", index=False)
    return all_chunks


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.lower())


class Retriever:
    def __init__(self, chunks: list[dict[str, Any]], download_models: bool = True):
        self.chunks = chunks
        self.texts = [f"{c['title']} {c['section']} {' '.join(c['cbt_topics'])} {c['text']}" for c in chunks]
        self.tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_features=80000).fit(self.texts)
        self.tfidf_docs = self.tfidf.transform(self.texts)
        self.bm25 = BM25Okapi([tokens(t) for t in self.texts])
        self.dense = self.reranker = self.embeddings = None
        self._query_embeddings: dict[str, np.ndarray] = {}
        self._bm_scores: dict[str, np.ndarray] = {}
        self._rank_cache: dict[tuple[str, str], tuple[list[int], float]] = {}
        if download_models:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            self.dense = SentenceTransformer(DENSE_MODEL, device="cpu")
            emb_path = CACHE / "e5_embeddings.npy"
            if emb_path.exists():
                arr = np.load(emb_path)
                if len(arr) == len(chunks):
                    self.embeddings = arr
            if self.embeddings is None:
                passages = ["passage: " + t for t in self.texts]
                self.embeddings = self.dense.encode(passages, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
                np.save(emb_path, self.embeddings)
            try:
                self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")
            except Exception as exc:
                print(f"Reranker unavailable; using metadata rerank: {exc}")

    def rank(self, query: str, mode: str, k: int = 10) -> tuple[list[int], float]:
        cached = self._rank_cache.get((query, mode))
        if cached is not None and len(cached[0]) >= k:
            return cached[0][:k], cached[1]
        t0 = time.perf_counter()
        if query not in self._bm_scores:
            self._bm_scores[query] = np.asarray(self.bm25.get_scores(tokens(query)))
        if self.dense is not None and query not in self._query_embeddings:
            self._query_embeddings[query] = self.dense.encode(["query: " + query], normalize_embeddings=True)[0]
        if mode == "tfidf":
            scores = cosine_similarity(self.tfidf.transform([query]), self.tfidf_docs).ravel()
            order = np.argsort(-scores)
        elif mode == "bm25":
            scores = self._bm_scores[query]
            order = np.argsort(-scores)
        elif mode == "lexical_hybrid":
            tf = cosine_similarity(self.tfidf.transform([query]), self.tfidf_docs).ravel()
            bm = self._bm_scores[query]
            tf_rank = np.argsort(-tf)
            bm_rank = np.argsort(-bm)
            scores = np.zeros(len(self.chunks), dtype=float)
            for ranking in (tf_rank, bm_rank):
                for r, idx in enumerate(ranking[:100]):
                    scores[idx] += 1.0 / (60 + r + 1)
            qtags = set(tag_text(query))
            scores += np.asarray([.02 * len(qtags.intersection(c["cbt_topics"])) for c in self.chunks])
            if "suicide_safety" in qtags:
                scores += np.asarray([100.0 if "suicide_safety" in c["cbt_topics"] else 0.0 for c in self.chunks])
            order = np.argsort(-scores)
        elif mode == "dense":
            q = self._query_embeddings[query]
            order = np.argsort(-(self.embeddings @ q))
        else:
            # Reciprocal Rank Fusion across lexical and semantic rankers.
            bm = np.argsort(-self._bm_scores[query])
            q = self._query_embeddings[query]
            de = np.argsort(-(self.embeddings @ q))
            scores = np.zeros(len(self.chunks), dtype=float)
            for ranking in (bm, de):
                for r, idx in enumerate(ranking[:100]):
                    scores[idx] += 1.0 / (60 + r + 1)
            order = np.argsort(-scores)
            if mode == "hybrid_rerank":
                pool = order[:30]
                if self.reranker is not None:
                    pairs = [(query, self.texts[i]) for i in pool]
                    rr = np.asarray(self.reranker.predict(pairs, batch_size=16, show_progress_bar=False))
                else:
                    qtags = set(tag_text(query))
                    rr = np.asarray([scores[i] + .02 * len(qtags.intersection(self.chunks[i]["cbt_topics"])) for i in pool])
                # Deterministic safety routing sits above the generative/RAG layer.
                # A clinical similarity score must not push explicit risk guidance out of context.
                qtags = set(tag_text(query))
                if "suicide_safety" in qtags:
                    rr = rr + np.asarray([
                        100.0 if "suicide_safety" in self.chunks[i]["cbt_topics"] else 0.0 for i in pool
                    ])
                order = pool[np.argsort(-rr)]
        result = [int(i) for i in order[:max(k, 10)]]
        latency = (time.perf_counter() - t0) * 1000
        self._rank_cache[(query, mode)] = (result, latency)
        return result[:k], latency


def relevant(chunk: dict[str, Any], item: dict[str, Any]) -> bool:
    sources = set(item.get("expected_sources", []))
    topics = set(item.get("expected_topics", []))
    keywords = [x.lower() for x in item.get("gold_keywords", [])]
    source_ok = not sources or chunk["source_id"] in sources
    topic_ok = not topics or bool(topics.intersection(chunk["cbt_topics"]))
    keyword_ok = not keywords or any(x in chunk["text"].lower() for x in keywords)
    return source_ok and (topic_ok or keyword_ok)


def evaluate(retriever: Retriever, questions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    modes = ["tfidf", "bm25", "dense", "hybrid", "hybrid_rerank"]
    for mode in modes:
        for q in questions:
            ids, latency = retriever.rank(q["query"], mode, 10)
            rels = [relevant(retriever.chunks[i], q) for i in ids]
            first = next((i + 1 for i, x in enumerate(rels) if x), None)
            rows.append({
                "method": mode, "question_id": q["question_id"], "category": q["category"],
                "recall_at_5": int(any(rels[:5])), "recall_at_10": int(any(rels)),
                "mrr_at_10": 0 if first is None else 1 / first,
                "context_precision_at_5": sum(rels[:5]) / 5, "latency_ms": latency,
                "top_chunk_ids": json.dumps([retriever.chunks[i]["chunk_id"] for i in ids[:5]]),
                "top_citations": json.dumps([retriever.chunks[i]["citation"] for i in ids[:5]], ensure_ascii=False),
                "safety_critical": q.get("safety_critical", False),
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "retrieval_per_question.csv", index=False)
    summary = df.groupby("method", sort=False).agg(
        recall_at_5=("recall_at_5", "mean"), recall_at_10=("recall_at_10", "mean"),
        mrr_at_10=("mrr_at_10", "mean"), context_precision_at_5=("context_precision_at_5", "mean"),
        median_latency_ms=("latency_ms", "median"),
    ).reset_index()
    safety = df[df.safety_critical].groupby("method")["recall_at_5"].mean().rename("safety_recall_at_5")
    summary = summary.merge(safety, on="method", how="left")
    summary.to_csv(RESULTS / "retrieval_summary.csv", index=False)
    (RESULTS / "retrieval_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    return summary


def export_examples(retriever: Retriever, questions: list[dict[str, Any]]) -> None:
    rows = []
    for q in questions:
        ids, _ = retriever.rank(q["query"], "hybrid_rerank", 5)
        for rank, i in enumerate(ids, 1):
            c = retriever.chunks[i]
            rows.append({"question_id": q["question_id"], "query": q["query"], "rank": rank,
                         "relevant_by_gold_rule": relevant(c, q), "citation": c["citation"],
                         "topics": ",".join(c["cbt_topics"]), "excerpt": c["text"][:600]})
    pd.DataFrame(rows).to_csv(RESULTS / "hybrid_rerank_examples.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "evaluate", "all"])
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()
    if args.command in ("build", "all"):
        chunks = build_corpus()
    else:
        chunks = load_jsonl(DATA / "chunks.jsonl")
    if args.command in ("evaluate", "all"):
        questions = load_jsonl(ROOT / "rag_eval_set.jsonl")
        retriever = Retriever(chunks, download_models=not args.no_models)
        print(evaluate(retriever, questions).to_string(index=False))
        export_examples(retriever, questions)


if __name__ == "__main__":
    main()
