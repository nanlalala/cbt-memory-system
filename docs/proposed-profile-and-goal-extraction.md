# Proposed: background, core goal and wording cues (draft, not wired in)

Status: design proposal. Nothing here changes the frozen extraction prompt in
`modules/agent-integration/cbt_projection_contract.py`, whose hash is pinned by
published evaluation records. Changing it would require a new contract version
and a new evaluation run; it would not be comparable with existing results.

## What already exists
- `relevant_background`: relevant goal, preference or strength.
- `working_formulation`: tentative hypotheses (`origin=agent_inference`) with exact quotes.
- Schema 2.0: V1 current profile/preferences, V3 current goals/plans, V2 recent emotion/problem state.

## Gaps this draft addresses
1. **Wording-based emotion cues**: `cbt_wording_cues.py` produces tentative
   `working_formulation` items (e.g. "possibly discouraged") from user wording
   only, with the exact quote as evidence. It never rates or diagnoses, and it
   binds under the frozen `bind_candidates`. It is not called by the pipeline yet.
2. **Core goal vs current goal**: not distinguished today. Proposal: a `goal_level`
   field in the `relevant_background` value (`core` | `current`), set only when the
   user states the goal. Needs a prompt change in a new contract version.
3. **Background (role, context)**: covered only if the user states a relevant
   strength or preference. Proposal: allow `role_or_context` in `relevant_background`
   values, user-stated only, never inferred.

## To adopt (needs real model calls)
- New contract version with the two prompt additions, plus a small fictional
  extraction check against the DeepSeek and Qwen verifier.
- Wire `wording_cues` into extraction as extra candidates, then pass them through
  the same verifier as other hypotheses.
- Re-run any evaluation you report with the new contract; do not mix with old results.
