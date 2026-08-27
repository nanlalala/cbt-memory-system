# CBT Memory System

Research prototype for **A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking**.

## Research focus

This project studies how a long-term CBT reflective journaling agent should form, store, retrieve, update, correct and forget memories across repeated sessions. It is not intended to diagnose users or replace therapists.

The principal comparison is:

1. no cross-session memory;
2. short-term memory only;
3. short-term memory plus manageable long-term memory.

The CBT knowledge-base RAG is a fixed supporting module and must remain unchanged across these three conditions.

## Repository structure

```text
cbt-memory-system/
├── docs/                         # Proposal, architecture and research decisions
├── modules/
│   ├── cbt-knowledge-rag/        # Completed Week-1 RAG module
│   ├── memory-schema/            # Structured CBT and memory records
│   ├── memory-manager/           # Formation/update/conflict/forgetting lifecycle
│   └── agent-integration/        # Context assembly and response-agent interface
├── evaluation/                   # Multi-session scenarios and comparison metrics
└── app/                          # Journal, assignments, memory controls and summaries
```

## Current status

Week 1 is complete: the CBT knowledge module implements PDF cleaning, cited chunks, BM25, multilingual E5 dense retrieval, hybrid fusion, cross-encoder reranking, bilingual safety routing and a 50-question evaluation.

Final cleaned retrieval result: Recall@5 **0.80**, Recall@10 **0.84**, MRR@10 **0.583**, and safety Recall@5 **1.00**.

See [`modules/cbt-knowledge-rag/CBT_RAG_v1_Report.md`](modules/cbt-knowledge-rag/CBT_RAG_v1_Report.md).

