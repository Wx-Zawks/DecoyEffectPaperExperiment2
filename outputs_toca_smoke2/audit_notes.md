# Experiment Audit Notes

This run preserves raw simulation statistics and avoids post-simulation calibration that changes mechanism ordering, success rates, participation, offloaded counts, or utility.

## Notes
- **raw_results** (passed): Simulation outputs are aggregated directly; calibration and forced ordering post-processing have been removed.
- **display_scaling** (documented): Utility plot fields use display_value = raw_value / 100; raw utility fields are preserved as *_raw.
- **dim_strategy** (documented): Primary DIM runs use strategy=F; Figure 3-8 still reports F/R/RF strategy comparisons.
- **dim_secondary_bids** (documented): DIM-derived rounds may allow secondary positive real-task bids when dim_allow_secondary_positive_bids=true; these bids are generated before allocation and are part of the raw bid book.
- **zero_cost_bids** (documented): When allow_zero_cost_bids=true, truthful non-exaggerated DIM bids in [-150, 0] and PRM bids in [-1000, 0] are clipped to the zero reserve and kept as valid zero-price auction bids; monitored exaggerated bids remain cancelled.
- **selection_metrics** (documented): Figure 3-6 plots raw positive-bid before/after participation intensity for A/B/goal/compete; zero-reserve bids remain in the bid book and completion statistics but are not counted as positive-bid intensity. tau_decoy is preserved in CSV only and is not plotted.
- **participation_metrics** (documented): Figure 4-2 uses raw resource-provider participation intensity for DIM/PRM/TRAIM; TOCA participant CSV fields are preserved but not plotted there because TOCA participants are accepted SMD task requests.
- **toca_simplification** (documented): TOCA is implemented as a comparable simplified online combinatorial-auction MEC baseline: it preserves online arrivals, candidate offloading schemes, position coverage, deadlines, resource constraints, dynamic resource prices, and accept/reject decisions.
- **toca_theory_scope** (documented): The TOCA baseline simplifies the full paper mechanism by omitting the primal-dual theoretical price update proof machinery and complex VM-type enumeration; no post-simulation calibration is applied to raw TOCA data.
- **toca_figures** (documented): TOCA is included in task success-rate, offloaded-task, mechanism-summary, task-level heatmap, utility, and truthfulness figures. It is skipped in the resource-node participation figure because its participant semantics differ.
- **paired_task_curves** (documented): Figure 4-5 uses common-random-number task prefixes and reports standard errors; no monotonic smoothing is applied.
- **representative_heatmap** (documented): Task-level bid/price heatmaps use the first repeat at comparison_node_count, not a ranking-optimized representative sample.
- **truthfulness_DIM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_PRM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_TOCA** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_TRAIM** (passed): truthful bid maximizes utility in this fixed-bid scan

## Truthfulness Scan
- DIM: target=node_2, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1;1.3;1.5, passed=True
- PRM: target=node_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1;1.3;1.5, passed=True
- TOCA: target=task_3, truthful_utility=1540.738813448024, max_utility=1540.738813448024, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1;1.3;1.5, passed=True
- TRAIM: target=bs_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1;1.3;1.5, passed=True

## Truthfulness Method Notes
- DIM and PRM scans reuse the realized bid book and recompute winners/payments with all non-target bids fixed.
- PRM dynamic preference updates are not re-simulated for each report; this is an ex-post implementation audit.
- TRAIM scans multiply one base station's reported cost for allocation/payment while holding physical coverage and true costs fixed.
- TOCA scans multiply one target SMD task's reported bid while holding all other tasks, positions, deadlines, and base-station capacities fixed.
