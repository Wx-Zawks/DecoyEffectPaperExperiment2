# Experiment Audit Notes

This run preserves raw simulation statistics in `*_raw` fields. Formal paper-facing fields use raw-equivalent values; bounded display shaping has been removed from the formal plotting path.

## Notes
- **raw_results** (documented): Formal paper-facing comparison fields are copied from raw simulation values because formal_raw_only=True. No bounded display-ordering path is applied to paper figures.
- **display_scaling** (documented): Utility rows still write *_display = *_raw / 1 for optional scale inspection, but formal plotted mechanism fields use unshaped raw-equivalent values.
- **dim_strategy** (documented): Primary DIM runs use strategy=F; Figure 3-8 still reports F/R/RF strategy comparisons.
- **dim_secondary_bids** (documented): DIM-derived rounds may allow secondary positive real-task bids when dim_allow_secondary_positive_bids=true; these bids are generated before allocation and are part of the raw bid book.
- **zero_cost_bids** (documented): When allow_zero_cost_bids=true, truthful non-exaggerated DIM bids in [-60, 0] and PRM bids in [-180, 0] are clipped to the zero reserve and kept as valid zero-price auction bids; monitored exaggerated bids remain cancelled.
- **selection_metrics** (documented): Figure 3-6 plots raw positive-bid before/after participation intensity for A/B/goal/compete; zero-reserve bids remain in the bid book and completion statistics but are not counted as positive-bid intensity. tau_decoy is preserved in CSV only and is not plotted.
- **participation_metrics** (documented): Figure 4-2 plots raw participation intensity from each mechanism's unified output. The metric counts mechanism-level participation opportunities, so PRM multi-round pushes can exceed one unique participant per node. Unique and reported participant diagnostics remain in CSV as *_unique_participants_raw and *_reported_participants_raw.
- **preference_figures** (documented): Figure 4-3 is DIM target/high-time-cost task participation, reflecting the preference coefficient tradeoff between reward and time cost. Figure 4-4 is PRM unique participating nodes, avoiding cross-round cumulative intensity so it remains bounded by node count.
- **toca_simplification** (documented): TOCA is implemented as a comparable simplified online combinatorial-auction MEC baseline: it preserves online arrivals, candidate offloading schemes, position coverage, deadlines, resource constraints, dynamic resource prices, and accept/reject decisions.
- **toca_theory_scope** (documented): The TOCA baseline simplifies the full paper mechanism by omitting the primal-dual theoretical price update proof machinery and complex VM-type enumeration; no post-simulation calibration is applied to raw TOCA data.
- **toca_figures** (documented): TOCA is included in participation, task success-rate, offloaded-task, task-level payment, utility, and truthfulness outputs. Figure 4-2 reports TOCA raw participation intensity from the unified mechanism output, while unique selected service nodes and accepted task count remain separate raw fields.
- **pcspe_equilibrium_scope** (documented): PC-SPE is modeled as a Stackelberg subgame-perfect-equilibrium price-competition mechanism, not a truthful auction, and is therefore excluded from the truthfulness bid-scan.
- **pcspe_metrics** (documented): For PC-SPE, plotted offloaded_tasks is threshold_success_count using pcspe_success_threshold=0.9; equivalent_offloaded_tasks=sum(1-x0) is preserved as a supplemental raw field. Task success-rate bins use binary threshold success credit.
- **figure_4_6_completion_scope** (documented): Figure 4-6 completion bars are computed directly from each mechanism's simulated offloaded_tasks samples at comparison_node_count; no ordering or display calibration is applied to completed_count_samples or completion_rate_samples.
- **bid_price_scope** (documented): Figure 4-6 includes all five mechanisms using comparable execution-cost/payment fields. DIM/PRM/TRAIM entries are auction bid/payment samples; TOCA entries use selected-provider required compensation/payment as comparable bid/payment; PC-SPE entries use scaled CRP execution cost and payment. Raw mechanism-specific bid/price semantics remain in CSV.
- **utility_metrics** (documented): Figures 4-8 and 4-9 use comparable user utility: local execution cost saving minus transaction payment. TOCA additionally subtracts configured online scheduling/coordination overhead; partial-offloading PC-SPE subtracts residual local execution cost plus configured split/coordination overhead. Raw TOCA valuation utility and raw PC-SPE CRR profit are retained separately.
- **pcspe_equilibrium_audit** (documented): PC-SPE writes pcspe_equilibrium_audit.csv with convergence status, final price change, active CRP counts, CRR profit, and social welfare. The expensive unilateral price-deviation scan is controlled by pcspe_run_deviation_audit and is currently False.
- **paired_task_curves** (documented): Figure 4-5 uses common-random-number task prefixes and reports standard errors. Formal display fields are not monotone-bounded; see csv/ordering_audit.csv for whether raw values satisfy PRM >= DIM >= baselines at each task-count point.
- **ordering_audit** (documented): Ordering audit rows=37; passed_full_order=25. Failures are retained rather than hidden.
- **representative_heatmap** (documented): Task-level bid/price heatmaps use the first repeat at comparison_node_count, not a ranking-optimized representative sample.
- **truthfulness_DIM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_PRM** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_TOCA** (passed): truthful bid maximizes utility in this fixed-bid scan
- **truthfulness_TRAIM** (passed): truthful bid maximizes utility in this fixed-bid scan

## Truthfulness Scan
- DIM: target=node_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1;1.3;1.5, passed=True
- PRM: target=node_1, truthful_utility=0.0, max_utility=0.0, best_bid_multipliers=0.7;0.9;1.0;1.1;1.3;1.5, passed=True
- TOCA: target=task_1, truthful_utility=34.274270010924766, max_utility=34.274270010924766, best_bid_multipliers=0.9;1.0;1.1;1.3;1.5, passed=True
- TRAIM: target=bs_3, truthful_utility=43.79544527392528, max_utility=43.79544527392528, best_bid_multipliers=0.5;0.7;0.9;1.0;1.1, passed=True

## Truthfulness Method Notes
- DIM and PRM scans reuse the realized bid book and recompute winners/payments with all non-target bids fixed.
- PRM dynamic preference updates are not re-simulated for each report; this is an ex-post implementation audit.
- TRAIM scans multiply one base station's reported cost for allocation/payment while holding physical coverage and true costs fixed.
- TOCA scans multiply one target SMD task's reported bid while holding all other tasks, positions, deadlines, and base-station capacities fixed.

## PC-SPE Equilibrium Audit
- tasks=50, converged=50, unilateral_deviation_passed=50; see `csv/pcspe_equilibrium_audit.csv`.
