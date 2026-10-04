# Conditional JEV adapter: offline integration readout

2026-10-04. Implementation, the complete authored fixture run and independent verification passed. This stage has **zero model inference** and no semantic-accuracy or Answer F1 result. The [aggregate record](results/conditional_jev_contract_20261004.json) preserves the run and verification hashes.

The [adapter](../../SLAC/retrieval/decision/conditional.py) now distinguishes standalone support from set-conditioned added information and a separate conflict judgment. Whole evidence units, source order, endpoint/model identities and schema/prompt/question/renderer versions participate in its immutable bindings. Changing the current pack invalidates conditional cached results; standalone results remain isolated in their own scope.

The decoder follows the documented Choice response shape while validating requested dimensions independently. A malformed or missing dimension becomes unknown without discarding a valid sibling. Model/request identity mismatches or unexpected response identities invalidate the whole envelope. Optional probability and confidence fields do not decide labels or supply a calibrated confidence threshold.

The [authored fixture](fixtures/conditional_jev_contract_v1.json) contains eight pairs, sixteen cases. The query and candidate stay fixed inside a pair while the current pack changes. Nine cases also have authored source relations and predetermined endpoint-permutation controls. Controls keep visible text/order, relation count and serialized relation-metadata byte length equal; they are not claimed to have equal model token counts or costs.

Expected semantic labels live only in the supervision portion of the fixture. The [runner](run_conditional_contract.py) uses explicit input projection, mutates the hidden supervision to verify isolation, and supplies fixed simulated responses independent of those labels. It therefore measures input/output and cache behavior, not whether JEV understands the cases. Relation endpoints and character spans are mechanically valid; this does not certify a relation's semantic truth.

## Implementation checks

The core's 41 synthetic tests and the runner's 16 tests passed. Independent core verification also passed for nine state changes, eight model/version changes, ten malformed states, five invalid counters, ten malformed whole-response cases and five per-dimension response faults. The [module README example](../../SLAC/retrieval/decision/README.md) ran successfully with an explicitly fake transport.

The frozen run produced 50 simulated request records with 42 unique request keys and eight cache hits. All 50 exact evidence-budget boundaries accepted the complete units and rejected a budget one unit smaller. Six malformed-response probes, a stale-cache probe and cache-hit transport suppression passed. Independent verification checked all 16 authored cases, their input projections and source ordering, the 50 records and nine relation controls. Changing hidden supervision and fixture identifiers in all 16 cases left the model payloads unchanged. Full inspection is confined to this artificial fixture; no real dataset or credentials were read.

The immutable local summary SHA-256 is `3cb9eb47eee8adbbc50d196b4b18e534b5c03d9b68b93f5c9f82bcb8500143fd`. These counts are mechanical checks, not correct model predictions. The fake standalone label is always `unknown`; added information is always `yes`; conflict is always `no`.

Evidence budgets cover the exact rendered candidate or current-pack/candidate union, using an injected counter. The fixture uses whitespace counts and the independent core checker uses a different synthetic counter. Both test accounting boundaries; neither is a production tokenizer or an API-cost measurement. A separate payload-byte cap checks the complete serialized input. Current native SLAC unit conversion and a live provider/runtime/cost admission layer are not yet implemented here.

## Research decision

The [novelty gate](SET_CONDITIONED_JEV_GATE_20261004.md) remains unchanged: ordinary set-conditioned evidence selection has direct precedents. The adapter makes a future comparison between plain conditioning and explicit source relations possible; it supplies no evidence that the latter helps. The earlier single positive-support decrease remains a limited observation, not a selected validation case for this module.

The next uncertainty is whether a real decision model can discriminate the fixed paired states. [Read-only local feasibility work](LOCAL_DECISION_FEASIBILITY_20261004.md) identified a possible small open-weight baseline, with platform and dependency compatibility still unverified. A local model would be a distinct baseline, not a substitute for JEV evidence. Any semantic execution must freeze its runtime/model and resource limits first; the old 900-question paid evaluation remains paused.

The [protocol](CONDITIONAL_JEV_CONTRACT_PROTOCOL_20261004.md) remains the fixed scope. Public authored fixtures, code and aggregate records do not constitute a real-data evaluation, a new selection algorithm or a quality improvement.
