# Confirmation analysis amendment: numerical degeneracy guard (V2)

2026-09-27. This amendment precedes any new-cohort predictions or quality outcomes. It supersedes only the numerical-degeneracy handling in the [V1 analysis commitment](CONFIRMATION_ANALYSIS_PROTOCOL_20260927.md). The already published V1 source, tests, protocol and machine protocol remain unchanged as a historical record. V2 is a new implementation, not a rewrite of prior results. No new QA, model or API call is used to make this correction.

## Synthetic finding and fixed correction

Independent review of V1 found that binary floating-point subtraction can turn a nominally constant family gain into a tiny nonzero sample SD. For three singleton synthetic families, baseline scores `[0.1, 0.2, 0.3]` and score-method scores `[0.2, 0.3, 0.4]` yield family gains very close to 0.1, but with SD about `3.25e-17`. V1's exact-equality guard therefore constructs a collapsed-looking positive interval and marks a positive effect. This conflicts with the protocol's conservative intention that constant gains without estimated variation must not automatically count as confirmation.

V2 computes and preserves the unrounded family mean differences, their actual sample SD (`ddof=1`) and their range. The fixed rule is:

```text
family_delta_range = max(family_deltas) - min(family_deltas)
numerically_degenerate = (family_delta_range <= 1e-12)
```

For exact zero SD, keep `unavailable_zero_sample_sd`. For nonzero SD with range at most `1e-12`, use `unavailable_numerically_degenerate_family_differences`. In either case the interval and half-width are null, both directional-support flags are false, and the observed mean, actual SD, range, threshold and degeneracy flag remain in the output. Do not replace the observed SD with zero or suppress a comparison. All five means and all 16 descriptive bootstrap intervals are still reported.

The tolerance is fixed from a synthetic numerical counterexample, not fitted to new results. It is a conservative computational guard, not a practical-effect margin, equivalence test or power statement. It can also withhold inference for real variation below this resolution; that limitation is intentional and visible. Variation above the threshold remains eligible for the same t-interval calculation.

**The threshold is not applied to confidence-interval endpoints.** For a nondegenerate comparison, positive support still requires the unrounded lower endpoint to be strictly greater than zero; an endpoint exactly equal to zero does not pass. The former endpoint tests are retained.

## Scope held fixed

The five methods, common candidate/packing contracts, family-balanced official Answer F1 estimand, three comparisons, two-sided Bonferroni alpha 0.05, t quantile, all descriptive dual-weight intervals, family resampling seed/draws, complete-cohort requirement and conditional 193-family planning flag remain as in V1. The t quantile algorithm is unchanged. No method, family, question, comparator or output-length setting is added or selected by this correction. No paid run or dataset is admitted.

The new files are [analyze_qasper_confirmation_v2.py](analyze_qasper_confirmation_v2.py) and [its synthetic tests](../../tests/research/test_qasper_confirmation_statistics_v2.py). Their SPEC and result schemas explicitly identify V2 and record the range threshold/action. The synthetic suite inherits all 83 V1 cases and adds the decimal constant-gain counterexample, threshold boundaries, preserved actual SD, above-threshold variation and schema assertions. A new public machine protocol must bind V2, this amendment and the unchanged metadata plan before any later cohort execution; V1's hashes must not be silently replaced.

This amendment addresses numerical consistency with the declared conservative inference rule. It does not validate family independence, remove exposure uncertainty, guarantee t-interval coverage or supply independent confirmation evidence.
