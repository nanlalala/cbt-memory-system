from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
nb=nbf.v4.new_notebook()
nb["metadata"]["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
nb["metadata"]["language_info"]={"name":"python","version":"3.12"}
cells=[]
cells.append(nbf.v4.new_markdown_cell("""# CBT Knowledge Base RAG v1 — Week 1 Evaluation

**Yinan Jin · Supervisor: Dr Judice LY Koh**

This notebook evaluates the supporting CBT knowledge layer for *A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking*. RAG is frozen before the later memory comparison so that memory—not changing domain knowledge—is the experimental variable."""))
cells.append(nbf.v4.new_markdown_cell("""## Frozen architecture

`PDF QA → section/page chunks + metadata → BM25 + multilingual E5 → reciprocal-rank fusion → multilingual cross-encoder → bilingual safety route → cited context`

Commercial books are represented by official Beck/Tolin sample chapters only. Full clinical/safety coverage comes from official WHO, VA and CCI resources."""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, pandas as pd, matplotlib.pyplot as plt
ROOT=Path.cwd()
if not (ROOT/'results').exists():
    ROOT=ROOT/'cbt_knowledge_rag_v1'
summary=pd.read_csv(ROOT/'results/retrieval_summary.csv')
per_q=pd.read_csv(ROOT/'results/retrieval_per_question.csv')
summary"""))
cells.append(nbf.v4.new_markdown_cell("""### Final measured result

| Method | Recall@5 | Recall@10 | MRR@10 | Context P@5 | Median latency | Safety R@5 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | .52 | .60 | .372 | .324 | 58.1 ms | .875 |
| BM25 | .52 | .62 | .323 | .328 | cached | .750 |
| E5 dense | .64 | .72 | .411 | .320 | cached | 1.000 |
| Hybrid | .64 | .72 | .480 | .348 | cached | .875 |
| **Hybrid + reranker** | **.80** | **.84** | **.583** | **.440** | **1.96 s** | **1.000** |

The final configuration retrieves at least one weak-label relevant chunk in the top five for 40/50 questions and for 8/8 safety questions."""))
cells.append(nbf.v4.new_code_cell("""plot=summary.set_index('method')[['recall_at_5','recall_at_10','mrr_at_10']]
ax=plot.plot(kind='bar',figsize=(10,5),ylim=(0,1),rot=20,title='Retrieval quality on 50 CBT questions')
ax.set_ylabel('Score'); ax.grid(axis='y',alpha=.25); plt.tight_layout(); plt.show()"""))
cells.append(nbf.v4.new_code_cell("""final=per_q[per_q.method=='hybrid_rerank']
by_category=final.groupby('category')[['recall_at_5','recall_at_10','mrr_at_10','context_precision_at_5']].mean().round(3)
by_category"""))
cells.append(nbf.v4.new_markdown_cell("""## Why cleaning changed the result

Some VA depression-manual pages used broken embedded fonts and extracted as `/g40/g84...`. Before filtering, E5 Recall@5 was .44 and hybrid+rerank Recall@5 was .76. After rejecting those blocks and rebuilding embeddings, the values became .64 and .80. The notebook therefore reports only the cleaned index as the final retrieval result."""))
cells.append(nbf.v4.new_code_cell("""chunks=[json.loads(x) for x in (ROOT/'data/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
pd.Series([c['source_id'] for c in chunks],name='chunks').value_counts().to_frame()"""))
cells.append(nbf.v4.new_markdown_cell("""## Exploratory response A/B (pre-clean corpus)

The API pilot is diagnostic, not a final efficacy claim. Automatic required-term coverage was 50% in both arms; RAG citation rate was 62.5% versus 0%. Manual scoring on seven complete pairs favored no-RAG (7.93/8) over pre-clean RAG (7.21/8), mainly because noise and an overly restrictive context prompt sometimes reduced empathy or actionability. One RAG response was empty due to gateway token behavior.

The cleaned book excerpts were not resent to the unverified external gateway. A final response study must use an approved institutional endpoint or a local generation model."""))
cells.append(nbf.v4.new_code_cell("""ab=pd.read_csv(ROOT/'results/response_ab_results.csv')
manual=pd.read_csv(ROOT/'results/response_manual_scoring_pilot.csv')
display(ab.groupby('arm').agg(required_hits=('required_hits','sum'),required_total=('required_total','sum'),citation_rate=('has_citation','mean'),median_latency_ms=('latency_ms','median')))
display(manual)"""))
cells.append(nbf.v4.new_markdown_cell("""## Re-run locally

```bash
../rag_v1_env/bin/python cbt_rag_v1.py build
HF_HOME=cache/huggingface ../rag_v1_env/bin/python cbt_rag_v1.py evaluate
```

For the thesis evaluation, replace weak source/topic/keyword labels with reviewer-confirmed exact chunk IDs. Keep this RAG fixed across all memory conditions."""))
nb["cells"]=cells
nbf.write(nb,ROOT/'CBT_Knowledge_RAG_v1_Evaluation.ipynb')
print(ROOT/'CBT_Knowledge_RAG_v1_Evaluation.ipynb')
