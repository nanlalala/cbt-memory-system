# V4 changed-input development repair — 5 November 2026

This is a **completed development diagnostic**, not an independent confirmatory
study or a contemporaneous full rerun. Genuine hosted Colab generated 206 changed-
input replies; 2,194 unchanged-prompt replies were reused from the earlier frozen
run. Per-row provenance is retained for all 800 cases × 3 arms.

| Arm | Original v3 correct | Current correct | Gains / losses vs v3 | New / reused replies |
| --- | ---: | ---: | ---: | ---: |
| Short-term | 10/800 | 10/800 | 0 / 0 | 0 / 800 |
| Raw-history RAG | 419/800 | 419/800 | 2 / 2 | 110 / 690 |
| Managed-memory RAG | 480/800 | 483/800 | 3 / 0 | 96 / 704 |

Managed–raw is now +8.0 percentage points (72 wins / 720 ties / 8 losses).
Managed's incremental change is +3/800, or +0.375 percentage points, not the whole
eight-point gap. Single-hop managed correctness is 385/400; multi-hop remains
98/400 (24.5%). All length variants form one dependent fact-consolidation family;
`not_estimable_one_family` is retained rather than an IID-question confidence
interval. PersonaMem is unchanged. No CBT/user/clinical benefit is established.

New cost: 206 successful attempts, zero failures or unresolved attempts,
462,998 prompt + 449 completion = **463,447 reported tokens**. Currency cost is
unknown. Reused replies are not charged again in this batch; their historical
cost remains in the original run. API time is not wall time.

- [Public summary](public_summary.json)
- [2,400 text-free per-question score/provenance rows](public_per_question_scores.jsonl)
- [Original Colab report manifest](report_manifest.json)
- [Independent derived-metadata audit](independent_derived_audit.json)
- [Chinese report and English boundaries](../../../../docs/objective-v4-repair-results-20261105.md)

The independent audit verifies the complete grid, numeric aggregates, prompt-
change/reuse identities and manifest hashes. It does not reopen source/question/
reply/gold text outside Colab, independently authenticate the provider or audit
billing. Official and strict secondary scores were recomputed inside Colab;
metadata reconciliation here is not independent semantic annotation. Answer-
error causal stages remain unresolved without independent evidence labels.

Base execution commit: `efd75116f985749a96a6070cb80c4b38d5c94dba`.
Repair-helper SHA-256: `5e1f862f615a871a09e498da0199d43b43b09a3b745c827a6c8152aeeae89bdc`.
Protocol identity: `bd14178402c2b713e5a4e9eeac250477c4ee66951ff3a06493295ae62a3552ae`.
The protocol and original run remain separate; this result does not overwrite them.
