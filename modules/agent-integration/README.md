# Agent integration

This module orchestrates two independent retrieval channels and assembles the final response context.

## Inputs

- current conversation window and session summary;
- current goal and active assignment state;
- qualified long-term memories;
- qualified CBT Knowledge RAG evidence;
- deterministic safety-policy output.

## Responsibilities

1. classify the current task and safety state;
2. decide whether CBT knowledge retrieval is needed;
3. decide whether long-term memory retrieval is needed;
4. build separate task-focused queries for each retriever;
5. enforce relevance, lifecycle and sensitivity gates;
6. deduplicate and budget the final context;
7. preserve provenance so the system can explain why information was used;
8. generate a response without treating retrieved context as automatically true.

## Context contract

The two retrieval channels remain distinguishable in the prompt:

- **professional evidence**: source, page/section, topic and safety label;
- **personal memory**: memory ID, provenance, time, confidence, confirmation and lifecycle status.

The context builder must allow either channel to return zero items. Deterministic crisis routing runs before retrieval and is not replaced by RAG.
