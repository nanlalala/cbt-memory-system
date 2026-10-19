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

The ID-only YAML is parsed as an immutable `KnowledgeConfig`. A source must be both approved in the registry and selected in that configuration. Schema-2 corpus manifests bind the registry, exact selected/enabled source sets, normalized configuration, chunker, model pins and ordered chunk digests. Build, load, ranking, evaluation and export all check the manifest; an approved but unselected source is rejected. A retriever's corpus and indexed text cannot change after initialization.

Schema-2 chunk digests cover ID, text, section, page range, citation, topics, risk level and source metadata. Evidence-card creation checks them again. Missing or changed metadata and old unprotected chunks require a rebuild; they are never automatically re-signed. These hashes detect changes relative to the frozen build, not prove that a page attribution or license decision is true.

`Retriever.evidence_cards(indices)` returns source URL/version, license review, independent use approvals, page/section/citation, bounded excerpt and hashes. `for_external_model=True` requires explicit external-transmission approval. The four response pilot runners apply the same gate before sending retrieved excerpts to an API. Cards describe retrieved evidence, not medical endorsement or proof that a generated answer is supported.

```python
from source_registry import SourceRegistry
from cbt_rag_v1 import Retriever, load_corpus, load_knowledge_config

registry = SourceRegistry.from_path()
config = load_knowledge_config(registry=registry)
chunks = load_corpus(knowledge_config=config, registry=registry)
retriever = Retriever(chunks, download_models=False, registry=registry,
                      knowledge_config=config)
indices, latency_ms = retriever.rank("homework review", "tfidf", 3)
cards = retriever.evidence_cards(indices, max_excerpt_chars=600)
frozen_knowledge_id = retriever.corpus_fingerprint
```

This example requires an approved schema-2 build and its `data/corpus_manifest.json`; current all-disabled defaults correctly fail closed. For a custom corpus, supply its explicit configuration, registry and manifest. Keep the same config/registry/corpus fingerprints in every memory condition. Until approvals are complete, the end-to-end experiment must declare an empty knowledge layer.

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

Model files are pinned in `../model_snapshots.json`:

| Role | Repository | Immutable revision |
|---|---|---|
| Dense | `intfloat/multilingual-e5-small` | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `1427fd652930e4ba29e8149678df786c240d8825` |

The shared loader downloads only locked files at those commits, then verifies actual file sizes and SHA-256 hashes before local safetensors loading. It disables remote code and refuses unlisted loading/tokenizer files or silent model substitution. `evaluation_run_manifest.json` records actual model-file identities and installed model-library versions. Embedding caches also bind those identities, preprocessing and the ordered corpus; cache bytes are hash-checked.

For predownloaded, read-only snapshots, pass `--dense-snapshot PATH --rerank-snapshot PATH --model-local-files-only` to the knowledge CLI, using clean directories containing the locked file sets. Extra unused exports from a full Hub checkout may need to be left out of these directories. The loader's `local_path` mode never downloads or writes to that directory. Light regression tests use fake snapshots; historical model scores have not been re-run with these newly locked weights.

After an approved corpus build, `python cbt_rag_v1.py evaluate --no-models` evaluates TF-IDF, BM25 and lexical hybrid and exports a supported lexical example file. It does not download E5/reranker weights. Direct requests for dense/hybrid/reranking without those capabilities raise a clear `ValueError`.

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
