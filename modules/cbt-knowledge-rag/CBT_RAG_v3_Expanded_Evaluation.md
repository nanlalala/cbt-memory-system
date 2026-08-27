# Expanded Dialogue Evaluation — Public Reference RAG v3

## What was tested

Sixteen new multi-turn scenarios compared the same `deepseek-v4-flash-0731` response agent under two conditions:

- `no_rag`: dialogue plus the fixed CBT/safety system instruction;
- `public_reference_rag`: the same input plus Top-3 retrieved passages.

The corpus used in this recovery run contained 763 cleaned chunks: the full WHO mhGAP Guideline (3rd edition), Judith Beck's official chapter 11 sample, and David Tolin's official chapter 1 sample. Commercial full books were not used.

Scenarios covered guided discovery, emotional reasoning, thought records, homework barriers and review, problem solving, graded exposure, relapse prevention, behavioural experiments, diagnosis and medication boundaries, self-harm, urgent psychosis-like symptoms, grief, medical referral, and user correction.

## Evaluation method

The qualitative result is a **model-assisted score**, not a human or clinical expert score. A fixed judge prompt received anonymous A/B answers in deterministic random order. Six dimensions were scored 0–2: CBT accuracy, collaboration/empathy, guided discovery, actionability, context fit, and safety/scope. Case-specific expected and prohibited behaviours were supplied. Citations did not earn points by themselves.

## Results

| Metric | No RAG | Public-reference RAG |
|---|---:|---:|
| Cases | 16 | 16 |
| Mean model-assisted score (/12) | **11.875** | **11.313** |
| Median model-assisted score (/12) | 12 | 12 |
| Pairwise judge preference | **6** | **1** |
| Ties | 9 | 9 |
| Citation rate | 0% | 18.75% |
| Median generation latency | 16.50 s | 15.56 s |

Mean paired score difference (RAG minus no RAG) was -0.5625. A fixed-seed case bootstrap gave a 95% percentile interval of -1.25 to -0.125. This interval describes this small model-judged test set only; it is not clinical evidence. Total-score differences occurred in five cases and all favoured no RAG; a two-sided exact sign test was p=0.0625. The rubric was near its ceiling, so small differences should be interpreted cautiously.

### Dimension means

| Dimension (/2) | No RAG | RAG |
|---|---:|---:|
| CBT accuracy | 1.938 | 1.875 |
| Collaboration/empathy | 2.000 | 1.938 |
| Guided discovery | 1.938 | 1.875 |
| Actionability | **2.000** | **1.750** |
| Context fit | 2.000 | 2.000 |
| Safety/scope | **2.000** | **1.875** |

## Why RAG scored lower

1. **Coverage mismatch.** Beck chapter 11 mainly concerns session structure, while Tolin chapter 1 is an overview. They do not adequately cover all requested techniques such as exposure, relapse prevention, grief, or medical triage.
2. **Irrelevant retrieval.** Lexical retrieval sometimes returned WHO evidence tables, bibliography fragments, or unrelated recommendations for ordinary CBT questions. Supplying weak context does not help a strong base model.
3. **One truncated RAG response.** B02 stopped mid-sentence, reducing actionability. This was retained as an end-to-end generation failure rather than silently regenerated.
4. **One unsafe/poor behavioural experiment.** In B15 the RAG answer recommended intentionally leaving a minor error in an email. The case prohibited sending a deliberately erroneous email; the judge scored RAG 7/12 versus 12/12.
5. **Citations were sparse.** Only three RAG answers cited a supplied passage, suggesting most retrieved context was not sufficiently useful.

## Interpretation

This test does not show that a CBT knowledge base is intrinsically harmful. It shows that **adding retrieved text without coverage and relevance control can lower dialogue quality**. The strong system prompt already produced high-quality baseline answers, while the small sample corpus often added little and occasionally introduced a poor intervention.

Before freezing the RAG module, add relevance gating: inject a passage only when it matches the current CBT task and passes a minimum score; otherwise return no knowledge context. Expand the legal corpus with technique-specific clinician manuals, then repeat the same frozen 16-case comparison. RAG should remain a fixed supporting module when the later short-/long-term-memory experiment begins.

## Limitations

- The judge was the same model family used for response generation and may favour its own style.
- There was no CBT-trained human reviewer.
- Sixteen cases are a development set, not a clinical validation sample.
- The automatic keyword proxy covered only a narrow subset of behaviours and is retained as an engineering diagnostic, not a quality score.
- The public sample chapters cannot represent the full Beck or Tolin books.
