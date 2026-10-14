# CBT Knowledge Base RAG v1

Week-1 supporting module for **A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking** (Yinan Jin, supervisor: Dr Judice LY Koh).

This module builds a frozen CBT knowledge layer. It is deliberately separate from the project's research variable: the later memory experiment must use the same knowledge configuration in all memory conditions.

## Current source policy

`source_registry.json` is now the authority for source identity, version, file SHA-256, license review and `retrieval_enabled`. All seven historical sources are currently **disabled**: none has both an approved project-use decision and a pinned local PDF. This is a licensing/provenance gate, not a finding that the clinical content is invalid.

WHO's official record identifies a CC BY-NC-SA 3.0 IGO license, but the exact local PDF and project-use conditions remain pending. The CCI [copyright notice](https://www.cci.health.wa.gov.au/Utilities/Copyright-and-Disclaimer) restricts reproduction outside personal downloading and legally permitted uses. VA hosting and Beck/Tolin publisher samples are not treated as automatic retrieval permission. The two publisher samples remain disabled pending permission review.

Three approvals are independent: `retrieval` for local indexing/search, `external_model_transmission` for sending excerpts to a model API, and `corpus_redistribution` for sharing derived corpora. Enabling retrieval does not enable either of the other uses. The registry documents a reviewer decision; code cannot establish legal permission or authenticate a reviewer by itself.

Run `python modules/cbt-knowledge-rag/source_registry.py` to inspect policy without downloading PDFs or models. The source selection YAMLs now contain IDs only, so titles, editions and permission state have one canonical location.

To approve a source: verify the exact document and reuse terms; record the license evidence URL, project reviewer, dated approval reference and approved uses; pin the local document's SHA-256; then set `retrieval_enabled: true`. `build_corpus()` skips disabled sources and refuses to build an all-disabled corpus. `Retriever()` rejects disabled, unregistered, legacy or stale chunks before any model is loaded. Changing the registry requires a corpus rebuild. No legacy results have been revised by this policy change.

## Evidence cards and frozen runs

Corpus builds write `data/corpus_manifest.json` with registry/corpus fingerprints, chunker configuration, enabled source IDs and chunk count. Each chunk binds the registry, source version, source bytes and exact chunk text. This provides reproducible identity, not a signed authenticity guarantee.

`Retriever.evidence_cards(indices)` returns source URL/version, license review, independent use approvals, page/section/citation, bounded excerpt and hashes. `for_external_model=True` requires explicit external-transmission approval. The four response pilot runners apply the same gate before sending retrieved excerpts to an API. Cards describe retrieved evidence, not medical endorsement or proof that a generated answer is supported.

```python
from source_registry import SourceRegistry
from cbt_rag_v1 import Retriever

registry = SourceRegistry.from_path()
retriever = Retriever(approved_chunks, download_models=False, registry=registry)
indices, latency_ms = retriever.rank("homework review", "tfidf", 3)
cards = retriever.evidence_cards(indices, max_excerpt_chars=600)
frozen_knowledge_id = retriever.corpus_fingerprint
```

Keep the same registry/corpus fingerprint in all three memory conditions. Until approvals are complete, the end-to-end experiment must explicitly declare a fixed disabled/empty knowledge layer; do not label it as a clinical seven-source RAG comparison.

## What is implemented

- Seven historical source candidates: WHO mhGAP, three VA provider/therapist manuals, CCI clinician guide, and official Beck/Tolin sample chapters; currently gated as described above.
- Page-aware and section-aware PDF extraction, repeated header/footer removal, broken-font/OCR filtering, overlapping chunks, source/page citations and CBT/safety metadata.
- TF-IDF baseline, BM25, multilingual E5 dense retrieval, reciprocal-rank hybrid fusion and multilingual cross-encoder reranking.
- Keyword/tag-based safety prioritization within the pilot retriever. This is not a deployed crisis detector or an application safety route.
- A 50-query bilingual/domain evaluation set covering CBT concepts, formulation, goals/homework, techniques, scope and safety.
- Retrieval metrics and an exploratory no-RAG/RAG response pilot.

## Reproduce after source approval

Use a CPU-only PyTorch build to avoid unnecessary CUDA packages:

```bash
python -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu
.venv/bin/pip install -r modules/cbt-knowledge-rag/requirements.txt
cd modules/cbt-knowledge-rag
../../.venv/bin/python cbt_rag_v1.py all
```

The first run downloads `intfloat/multilingual-e5-small` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` from Hugging Face and caches document embeddings locally. The cache is bound to the exact ordered chunk texts, cache format and model name; any mismatch rebuilds the embeddings instead of silently reusing stale vectors.

For an optional response test, set an OpenAI-compatible endpoint and model locally. The key is requested with hidden input and is never saved:

```bash
export CBT_TEST_BASE_URL='https://your-authorized-gateway.example/v1'
export CBT_TEST_MODEL='your-model-id'
../../.venv/bin/python run_response_ab.py
```

The build command currently stops at the all-disabled policy gate. Do not bypass it to reproduce historical pilots. Do not upload copyrighted full books or samples, and do not infer permission from public downloading.

## Historical pilot result

The pre-registry cleaned `hybrid_rerank` system achieved Recall@5 **0.80**, Recall@10 **0.84**, MRR@10 **0.583**, context precision@5 **0.44**, and safety Recall@5 **1.00** on the 50-query pilot. These archived values do not describe the currently disabled knowledge layer. See `CBT_Knowledge_RAG_v1_Evaluation.ipynb` and `CBT_RAG_v1_Report.md` for limitations and interpretation.


## Historical dialogue tests

`run_expanded_dialogue_v3.py` was a recovery experiment using downloadable WHO material and official Beck/Tolin sample chapters. It scored 11.875/12 for no RAG and 11.313/12 for the limited public-reference configuration, helping identify coverage and irrelevant-context failures. Its use of downloadable material preceded the source registry and does not establish reuse authorization.

`run_full_rag_dialogue_v4.py` was the full-corpus comparison. Across the same 16 scenarios, hybrid-and-reranked RAG scored 11.6875/12 versus 11.375/12 for no RAG, with six RAG wins, two no-RAG wins and eight ties. The difference is small and does not establish broad clinical or conversational superiority. These scores remain archived model-assisted development results, not human clinical ratings or results under the new source policy.
