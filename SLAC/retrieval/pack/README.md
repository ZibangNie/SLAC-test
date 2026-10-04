# Budget-aware selection

`budget_selection.py` is an optional, pure Python selector. It does not change
the current `evidence_packer.pack_evidence` default, invoke models, or load
reference answers. The caller supplies candidate utilities and an exact cost
function for each complete evidence pack.

Run this synthetic example from the repository root:

```python
from SLAC.retrieval.pack.budget_selection import BudgetCandidate, select_budgeted

candidates = [
    BudgetCandidate("a", 0.9, "native-text-a", priority=0),
    BudgetCandidate("b", 0.8, "native-text-b", priority=1),
    BudgetCandidate("c", 0.8, "native-text-c", priority=2),
]
# Synthetic full-pack counts, including the empty pack. In a real integration,
# replace this lookup with tokenization of the complete, consistently rendered
# pack, including headers, separators, and the configured special tokens.
pack_tokens = {
    (): 0, (0,): 8, (1,): 5, (2,): 5,
    (0, 1): 13, (0, 2): 13, (1, 2): 10, (0, 1, 2): 18,
}
result = select_budgeted(candidates, pack_tokens.__getitem__, budget_tokens=10)
assert result.greedy.selected_indexes == (0,)
assert result.optimal.selected_indexes == (1, 2)
assert result.optimal.utility == 1.6
assert result.optimal.cost_tokens == 10
assert result.optimal.optimality_proven
```

The callback receives a tuple of original candidate indexes in increasing
order, including `()`. It must be deterministic and side-effect free, return a
nonnegative integer, and make the empty pack feasible. Count the complete
rendered pack: adding standalone unit counts is not generally equivalent.
Costs can be nonadditive or nonmonotone; the exact search therefore keeps a
candidate even when its singleton cost exceeds the budget. Each distinct
subset is counted at most once within a selection call.

Lower `priority` values are scanned first; equal priorities preserve input
order. Candidate IDs must be unique. Candidates with equal `duplicate_key`
strings are mutually exclusive. For evidence use, the caller should supply
the exact native-text duplicate grouping it intends to enforce. Utilities
must be finite and nonnegative. Counts, priorities, and limits must be
integers; booleans and floating token counts are rejected.

`select_greedy` provides just the priority scan with the same inputs and caps.
`select_budgeted` also enumerates every duplicate-compatible subset up to
`max_units`, maximizing `math.fsum` of utilities. It retains the greedy pack
when its utility is optimal. Other equal-utility optima prefer inclusion of
the earliest priority candidate, then the next. It does not break ties by
token count or use a numeric tolerance.

Defaults allow 16 candidates and 3 selected units, at most 697 subsets
including the empty subset. An independent `max_subsets=10000` guard rejects
larger searches before any cost calls; there is no silent approximate
fallback. `max_units=0` and empty candidate pools are supported. The result
reports exhaustive-search counts and whether the greedy pack also attained
the optimum.

This is standard constrained subset optimization. Optimality applies to the
supplied pool, utilities, duplicate rule, cardinality limit, and full-pack
costs. It does not establish calibrated probabilities, improved evidence or
answer quality, or a novel research method.

Synthetic verification:

```text
python -m pytest tests/research/test_budget_selection.py -q
```
