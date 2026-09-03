# Week 1 — CBT Knowledge Base RAG v1

**Project:** A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking  
**Student:** Yinan Jin  
**Supervisor:** Dr Judice LY Koh

## Scope decision

CBT Knowledge Base RAG is a supporting capability, not the thesis's main experimental variable. This week freezes a technically credible RAG v1. In the later memory experiment, the same RAG must be used for all three conditions: no cross-session memory, short-term memory only, and short-term plus manageable long-term memory.

## Completed system

The corpus contains 2,250 cleaned, cited chunks from seven sources. WHO, VA and CCI documents supply full public clinical/safety coverage. Judith Beck and David Tolin are represented only by official publisher sample chapters, so conclusions must not imply full-book coverage.

The retrieval pipeline is:

1. PDF cleaning and broken-font/OCR rejection;
2. section/page-aware chunks with overlap and metadata;
3. BM25 lexical retrieval plus multilingual E5 dense retrieval;
4. reciprocal-rank fusion;
5. multilingual cross-encoder reranking;
6. deterministic safety routing for direct and indirect self-harm language;
7. citations retaining title and PDF page range.

## Gold-set design

The pilot set contains 50 questions: 8 CBT concepts, 8 formulation, 8 goals/homework, 12 techniques, 6 scope questions and 8 safety-critical questions. Relevance is currently defined by an auditable combination of expected source, CBT topic and keywords. This is useful for engineering iteration but remains a weak-label evaluation; a domain expert should confirm exact relevant chunk IDs before the dissertation's final evaluation.

## Final retrieval results

| Method | Recall@5 | Recall@10 | MRR@10 | Context precision@5 | Median retrieval latency | Safety Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.52 | 0.60 | 0.372 | 0.324 | 58.1 ms | 0.875 |
| BM25 | 0.52 | 0.62 | 0.323 | 0.328 | 0.02 ms* | 0.750 |
| E5 dense | 0.64 | 0.72 | 0.411 | 0.320 | 0.10 ms* | 1.000 |
| Hybrid | 0.64 | 0.72 | 0.480 | 0.348 | 0.20 ms* | 0.875 |
| Hybrid + reranker | **0.80** | **0.84** | **0.583** | **0.440** | **1,963.9 ms** | **1.000** |

\*Dense query encoding was cached across methods inside one evaluation run, so these sub-millisecond values measure ranking after cached query features, not cold end-to-end latency. The hybrid-plus-reranker latency is the most meaningful end-to-end comparison for the final configuration.

Category Recall@5 for the final retriever was: CBT concepts 0.875, formulation 0.625, goals/homework 0.875, techniques 0.750, scope 0.667 and safety 1.000. Formulation and scope are the clearest areas for further gold-set and corpus improvement.

## Important cleaning finding

Before broken-font filtering, the final retriever reached Recall@5 0.76 and dense retrieval only 0.44. A small number of VA depression-manual pages were extracted as strings such as `/g40/g84...`; these chunks distorted embeddings and caused the response model to call the supplied context “garbled”. After rejection and re-indexing, dense Recall@5 rose to 0.64 and hybrid-plus-reranker Recall@5 rose to 0.80. This demonstrates why document QA is part of RAG engineering, not a cosmetic preprocessing step.

## Exploratory response A/B

An initial DeepSeek pilot compared the same system prompt and generation settings with and without retrieved context. The automatic proxy score showed 50% required-term coverage for both arms; RAG produced citations in 62.5% of answers versus 0% without RAG. One RAG response was empty because the gateway exhausted its token budget in hidden reasoning, and the pilot was run before the final broken-font cleanup. On seven complete pairs, model-assisted dialogue scoring averaged 7.93/8 for no-RAG and 7.21/8 for pre-clean RAG.

This is not evidence that RAG is ineffective. It shows that high retrieval recall alone does not guarantee a better answer: noisy chunks, over-restrictive “only use context” prompting, and incomplete coverage can make a strong base model less helpful. The final cleaned retrieval system could not be re-sent to the unverified external gateway because doing so would disclose local book excerpts. A final answer-level study should be run only through an approved endpoint or a local generation model.

### Exploratory response pilot: Reference-context prompt v2

After explicit authorization to use the configured test gateway, the cleaned retriever was retested with a revised policy: at most three chunks are supplied as optional professional references, while the model retains its original conversational and empathic capability. The same eight scenarios and saved no-RAG answers were used.

| Evaluation aspect | Result | Interpretation |
|---|---:|---|
| Model-assisted dialogue score | 7.88/8 vs 7.94/8 | Nearly identical observed scores in this eight-scenario pilot |
| Required-term coverage | +3.1 percentage points | Slight improvement |
| Citation coverage | 0% → 62.5% | Clear improvement in professional grounding |
| Median generation latency | −0.08 s | No added latency |

Two automatic “forbidden term” matches were false positives because the terms occurred in negated statements (“does not mean you are unsuitable for CBT” and “not forcing yourself to pull yourself together”). Post-hoc inspection found no corresponding harmful recommendation.

The revised prompt therefore removed the clear quality degradation seen with the pre-clean, context-only RAG (7.21/8). In this small pilot, it preserved approximately the same observed dialogue quality while adding professional grounding and citations; it did not demonstrate a material overall dialogue-quality advantage over the strong base model.

This component-level pilot is exploratory and does not provide a final conclusion on the value of RAG. After Long-term Memory RAG and Agent integration are complete, the no-RAG and RAG configurations will be compared again in the end-to-end evaluation. The final comparison will examine dialogue quality, cross-session continuity, CBT assignment tracking, professional boundaries, safety, latency and token cost.

## Safety conclusion

Safety cannot be delegated to vector retrieval or the final LLM. The first version missed one “continue thought challenging despite self-harm” query. Adding bilingual deterministic risk detection and forcing safety-tagged chunks into the reranked context increased safety Recall@5 to 8/8. The production prototype still needs a separate crisis classifier/route, local-resource configuration and human-reviewed response templates.

## Handoff to the memory project

The RAG output contract should be frozen as:

```json
{
  "query": "current user need or CBT task",
  "retrieved_chunks": [
    {
      "chunk_id": "source:hash",
      "text": "...",
      "source_id": "...",
      "section": "...",
      "page_start": 1,
      "page_end": 1,
      "cbt_topics": ["goal_setting"],
      "risk_level": "low",
      "citation": "Title, pp. 1-1"
    }
  ],
  "retrieval_method": "hybrid_rerank",
  "safety_route_triggered": false
}
```

The response agent can then receive: current session summary + current assignment state + selected long-term memories + frozen CBT RAG context + safety rules. Memory evaluation must not change the RAG corpus or retriever between conditions.

## Next action

Before freezing v1.0 for the six-week project: have the supervisor or a CBT-trained reviewer validate exact gold chunks for the 50 questions; improve formulation/scope coverage; approve a local or institutional model endpoint for answer-level evaluation; then integrate this fixed retrieval contract into the memory manager.
