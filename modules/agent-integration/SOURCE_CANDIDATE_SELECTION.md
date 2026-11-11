# Candidate supplements count original sources

`controlled-memory-v2.2-source-aware-candidates` changes the live candidate-index
selection stage. It does not change the frozen quotation experiments, their
contexts, answers or scores. A new pinned protocol and run identity are required
to evaluate this version; previous paid-call budgets do not carry over.

Previously the candidate retriever retained its first three records before
restoring original quotations and removing duplicates. Three summaries of one
source could consume every record slot. The source might already be present in
the original-history channel, leaving no supplemental evidence despite relevant
independent sources farther down the candidate ranking.

The current path ranks the full eligible candidate pool with the same relevance
rule, then selects at most three distinct new original user sources. Duplicate
candidate identities, sources already in the original-history channel and
unresolvable exact locators consume no source slot. Both candidate permissions
and all dependencies of a multi-source candidate must be currently eligible.
The supplemental quota is a source quota, not a requirement to add three facts.

The answer still receives complete original quotations. Candidate summaries
remain unconfirmed indexing descriptions and never become confirmed user facts
or current-state claims. Original-history ranks are preserved at packing time;
all candidate supplements have lower priority than the ranked original-history
channel. The existing complete-request budget can drop any whole supplement.

The trace records the selector version, original candidate ranks, selected
source IDs, exclusion reasons and each supplement's complete candidate-source
dependencies. Only dependencies of supplements actually included in the final
request are added to durable send/publication admission. Withdrawing an indexing
dependency after sending discards the old reply even if that dependency's text
was not quoted. The response model, recent messages, knowledge channel, output
budget and clarification policy remain owned by the common comparison protocol.

The default live product has up to six original-history sources and three
candidate supplements. That is not, by itself, a fair raw-versus-managed study:
a raw arm limited to six while a managed arm can use nine has a capacity
difference. The new professional-reference evaluation must give raw retrieval
the same maximum nine-source opportunity and room to fill the same complete
request budget, using a competent common lexical/dense baseline. Do not adopt
the older fixed-top-six replay as that new comparator or attribute source-count
capacity differences to memory management. All source-count limits are explicit
in the new product protocol.

Source-ID deduplication assumes an original-history hit contains the full source,
as the current live durable-source path does. If a future retriever returns only
partial spans, its adapter must distinguish partial from complete coverage
before passing `already_present_source_ids` to the selector.

## Verification

Twelve invented software fixtures cover duplicate-source starvation, exact-source
restoration, negative qualifiers, unknown/withdrawn permissions, role and subject
boundaries, multi-source dependencies, actual live context wiring and raw ranking
under a tight complete-request budget. In a regression fixture where the former
three candidate records all refer to an already retrieved source, the new path
can reach the next three distinct sources. This verifies selection behavior,
not clinical relevance or response-quality benefit.

The existing 31 controlled-dialogue and 22 controlled-adapter regression tests
also passed. No dataset, model, provider, reference answer or patient record was
used in these tests. The next genuine hosted-Colab evaluation must separately
measure necessary historical facts used correctly, source/state errors and CBT
response quality against the same capable original-history comparator.
