# CBT Knowledge Base RAG v1

Week-1 supporting module for **A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking** (Yinan Jin, supervisor: Dr Judice LY Koh).

This module builds a frozen CBT knowledge layer. It is deliberately separate from the project's research variable: the later memory experiment must use this same RAG configuration in all memory conditions.

## What is implemented

- Seven local sources: WHO mhGAP, three VA provider/therapist manuals, CCI clinician guide, and official Beck/Tolin sample chapters.
- Page-aware and section-aware PDF extraction, repeated header/footer removal, broken-font/OCR filtering, overlapping chunks, source/page citations and CBT/safety metadata.
- TF-IDF baseline, BM25, multilingual E5 dense retrieval, reciprocal-rank hybrid fusion and multilingual cross-encoder reranking.
- A deterministic safety route above similarity retrieval.
- A 50-query bilingual/domain evaluation set covering CBT concepts, formulation, goals/homework, techniques, scope and safety.
- Retrieval metrics and an exploratory no-RAG/RAG response pilot.

## Reproduce

Use a CPU-only PyTorch build to avoid unnecessary CUDA packages:

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
.venv/bin/pip install -r modules/cbt-knowledge-rag/requirements.txt
cd modules/cbt-knowledge-rag
../../.venv/bin/python cbt_rag_v1.py all
```

The first run downloads `intfloat/multilingual-e5-small` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` from Hugging Face and caches document embeddings locally.

For an optional response test, set an OpenAI-compatible endpoint and model locally. The key is requested with hidden input and is never saved:

```bash
export CBT_TEST_BASE_URL='https://your-authorized-gateway.example/v1'
export CBT_TEST_MODEL='your-model-id'
../../.venv/bin/python run_response_ab.py
```

Do not upload copyrighted full books. The two commercial sources in this prototype are official sample chapters only.

## Main result

The cleaned `hybrid_rerank` system achieved Recall@5 **0.80**, Recall@10 **0.84**, MRR@10 **0.583**, context precision@5 **0.44**, and safety Recall@5 **1.00** on the 50-query pilot. See `CBT_Knowledge_RAG_v1_Evaluation.ipynb` and `CBT_RAG_v1_Report.md` for limitations and interpretation.


## Expanded dialogue test

`run_expanded_dialogue_v3.py` adds 16 multi-turn scenarios and blinded model-assisted A/B scoring. A recovery configuration using only legally downloadable WHO material and official Beck/Tolin sample chapters is defined in `knowledge_sources_public_test.yaml`. Its result was 11.875/12 for no RAG and 11.313/12 for public-reference RAG; see `CBT_RAG_v3_Expanded_Evaluation.md`. This is a model-assisted development score, not a human clinical rating.

