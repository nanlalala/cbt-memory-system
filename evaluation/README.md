# Evaluation

Evaluation is split into four layers so that retrieval quality is not confused with dialogue quality.

## 1. Knowledge retrieval

Evaluate CBT Knowledge RAG with Recall@K, MRR, context precision, citation support, cross-lingual retrieval and safety-topic recall.

## 2. Memory retrieval

Evaluate long-term memory retrieval with Recall@K, latest-state accuracy, correction compliance, contradiction rate, superseded-memory leakage, provenance accuracy, empty-result accuracy and sensitivity-policy compliance.

## 3. End-to-end dialogue

Compare:

1. no cross-session memory;
2. short-term memory only;
3. short-term plus manageable long-term memory RAG.

Freeze the response model, CBT Knowledge RAG, safety rules and scenarios across conditions. Measure continuity, personalisation, homework tracking, contextual correctness, safety, latency and token cost.

## 4. Safety

Test crisis recognition, role boundaries, unsupported diagnosis, referral behaviour and leakage of sensitive, deleted or superseded memories.

Model-assisted scores are development signals rather than human clinical ratings.
