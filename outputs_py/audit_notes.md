# Experiment Audit Notes

This run preserves raw simulation statistics and avoids post-simulation calibration that changes mechanism ordering, success rates, participation, offloaded counts, or utility.

## Notes
- **raw_results** (passed): Simulation outputs are aggregated directly; calibration and forced ordering post-processing have been removed.
- **display_scaling** (documented): Utility plot fields use display_value = raw_value / 100; raw utility fields are preserved as *_raw.
- **dim_strategy** (documented): Primary DIM runs use strategy=F; Figure 3-8 still reports F/R/RF strategy comparisons.
- **dim_secondary_bids** (documented): DIM-derived rounds may allow secondary positive real-task bids when dim_allow_secondary_positive_bids=true; these bids are generated before allocation and are part of the raw bid book.
- **selection_metrics** (documented): Figure 3-6 plot fields use raw bid intensity for A/B/goal/compete and paired decoy-influenced node counts for tau_decoy; raw selection counts are preserved as tau_*_raw.
- **participation_metrics** (documented): Figure 4-2 uses raw participation intensity; unique participant counts are preserved as *_unique_participants_raw.
- **paired_task_curves** (documented): Figure 4-5 uses common-random-number task prefixes and reports standard errors; no monotonic smoothing is applied.
- **representative_heatmap** (documented): Task-level bid/price heatmaps use the first repeat at comparison_node_count, not a ranking-optimized representative sample.
- **truthfulness_DIM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_PRM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_TRAIM** (failed): truthful bid is not utility-maximizing in this fixed-bid scan

## Truthfulness Scan
- DIM: target=node_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.5;0.8;1.0;1.2;1.5;2.0, passed=True
- PRM: target=node_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.8;1.0;1.2;1.5;2.0, passed=True
- TRAIM: target=bs_3, truthful_utility=0.0, max_utility=85.31358550580585, best_bid_multipliers=2.0, passed=False

## Truthfulness Method Notes
- DIM and PRM scans reuse the realized bid book and recompute winners/payments with all non-target bids fixed.
- PRM dynamic preference updates are not re-simulated for each report; this is an ex-post implementation audit.
- TRAIM scans multiply one base station's reported cost for allocation/payment while holding physical coverage and true costs fixed.
