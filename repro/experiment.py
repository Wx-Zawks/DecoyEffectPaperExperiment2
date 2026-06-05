from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from repro.formulas import monitor_bid
from repro.platform import Platform
from repro.svg import (
    ensure_dir,
    paper_grouped_bar_chart,
    paper_line_chart,
    paper_mechanism_summary_chart,
    paper_surface_pair,
    paper_task_heatmap_pair,
    write_dashboard,
)
from repro.traim import assign_mobile_devices, build_traim_environment, candidate_cost, run_traim


DEFAULT_CONFIG = {
    "seed": 20260328,
    "alpha": 0.88,
    "beta": 0.88,
    "lambda_loss": 2.25,
    "frequency_range": [1.0, 1.5],
    "reward_range": [2.0, 20.0],
    "time_cost_range": [200.0, 2000.0],
    "task_count": 50,
    "node_counts": [10, 20, 30, 40, 50, 60],
    "dim_strategy": "F",
    "dim_allow_secondary_positive_bids": True,
    "dim_secondary_min_score_ratio": 0.0,
    "preference_means": [round(index * 0.1, 1) for index in range(11)],
    "default_preference_mean": 0.35,
    "preference_std": 0.22,
    "chi_values": [round(index * 0.1, 1) for index in range(11)],
    "gamma_values": [round(1.0 + index * 0.1, 1) for index in range(11)],
    "task_curve_counts": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "repeats": 30,
    "parameter_repeats": 80,
    "parameter_node_count": 80,
    "comparison_node_count": 30,
    "utility_task_counts": [25, 50],
    "utility_repeats": 100,
    "utility_display_scale": 100.0,
    "truthfulness_bid_multipliers": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    "truthfulness_node_count": 30,
    "truthfulness_task_count": 30,
    "truthfulness_preference_mean": 0.5,
    "traim_region_size": 1000.0,
    "traim_md_cpu_range": [1, 5],
    "traim_md_rate_range": [1, 10],
    "traim_bs_cpu_range": [5, 8],
    "traim_bs_subchannel_range": [3, 6],
    "traim_bandwidth_mbps": 1.0,
    "traim_bs_radius_range": [60.0, 145.0],
    "traim_noise_dbm_range": [-90.0, -70.0],
    "traim_transmit_power_mw_range": [200.0, 300.0],
    "traim_task_cost_ratio_range": [0.06, 0.15],
    "show_progress": True,
    "output_dir": "outputs_py",
}


def run_paper_reproduction(overrides: dict | None = None) -> dict:
    config = dict(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    platform = Platform(config)

    output_dir = Path(config["output_dir"]).resolve()
    csv_dir = output_dir / "csv"
    raw_csv_dir = csv_dir / "raw"
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(csv_dir)
    ensure_dir(raw_csv_dir)
    ensure_dir(figure_dir)
    _clear_output_dir(csv_dir, ["*.csv"])
    _clear_output_dir(raw_csv_dir, ["*.csv"])
    _clear_output_dir(figure_dir, ["*.png", "*.svg", "*.html"])
    _clear_output_dir(output_dir, ["trace.json"])

    _progress(config, "开始 DIM 参数敏感性实验")
    dim_parameter_analysis = _run_dim_parameter_analysis(platform, config)
    _progress(config, "开始任务选择与成功率实验")
    selection_evaluation = _run_selection_evaluation(platform, config)
    _progress(config, "开始 DIM 策略对比实验")
    strategy_evaluation = _run_strategy_comparison(platform, config)
    _progress(config, "开始 DIM / PRM / TRAIM 机制对比实验")
    mechanism_comparison = _run_mechanism_comparison(platform, config)
    _progress(config, "开始偏好系数影响实验")
    preference_mean_effect = _run_preference_mean_effect(platform, config)
    _progress(config, "开始机制性质校验")
    validations = platform.validate_properties()
    _progress(config, "Running truthfulness bid-scan audit")
    truthfulness_check = _run_truthfulness_check(config)

    results = {
        "config": config,
        "dim_parameter_analysis": dim_parameter_analysis,
        "selection_evaluation": selection_evaluation,
        "strategy_evaluation": strategy_evaluation,
        "mechanism_comparison": mechanism_comparison,
        "preference_mean_effect": preference_mean_effect,
        "validations": validations,
        "truthfulness_check": truthfulness_check,
    }
    results["audit_notes"] = _build_audit_notes(results)

    _progress(config, "写出 CSV、图表与汇总文件")
    _write_experiment_csvs(csv_dir, results)
    _write_raw_experiment_csvs(raw_csv_dir, results)
    figure_paths = _build_figures(figure_dir, results)
    write_dashboard(
        "DIM + PRM 论文风格复现实验图",
        [(item["label"], item["filename"]) for item in figure_paths],
        figure_dir / "index.html",
    )
    explanation_path = _write_algorithm_document(output_dir, results)
    audit_path = _write_audit_notes(output_dir, results)

    results_path = output_dir / "paper_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = _build_summary(results, figure_paths, explanation_path, audit_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_rows(output_dir / "summary.csv", summary["mechanism_rows"])

    return {
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "csv_dir": str(csv_dir),
        "summary": summary,
        "results_path": str(results_path),
        "explanation_path": str(explanation_path),
        "audit_path": str(audit_path),
    }


def _progress(config: dict, message: str) -> None:
    if config.get("show_progress", True):
        print(f"[progress] {message}", flush=True)


def _run_dim_parameter_analysis(platform: Platform, config: dict) -> dict:
    parameter_config = dict(config)
    parameter_config["seed"] = config["seed"] + 3505
    parameter_platform = Platform(parameter_config)
    k_samples_grid = [[[] for _ in config["chi_values"]] for _ in config["gamma_values"]]
    goal_samples_grid = [[[] for _ in config["chi_values"]] for _ in config["gamma_values"]]
    repeat_count = config.get("parameter_repeats", config["repeats"])

    for repeat_index in range(repeat_count):
        if repeat_index == 0 or (repeat_index + 1) == repeat_count or (repeat_index + 1) % max(1, repeat_count // 5) == 0:
            _progress(config, f"DIM 参数敏感性 {repeat_index + 1}/{repeat_count}")
        tasks, nodes = parameter_platform.prepare_environment(config["parameter_node_count"], config["task_count"], config["default_preference_mean"])
        baseline = parameter_platform.simulate_no_decoy(parameter_platform.clone_tasks(tasks), parameter_platform.clone_nodes(nodes))
        baseline_goal_rate = baseline["selection_counts"].get("A", 0) / len(nodes) if nodes else 0.0
        for gamma_index, gamma in enumerate(config["gamma_values"]):
            for chi_index, chi in enumerate(config["chi_values"]):
                result = parameter_platform.simulate_dim(parameter_platform.clone_tasks(tasks), parameter_platform.clone_nodes(nodes), strategy="RF", chi=chi, gamma=gamma)
                gamma_max = result["bundle"]["time_factor_range"][1]
                if gamma >= gamma_max:
                    k_value = 0.0
                    goal_rate = baseline_goal_rate
                else:
                    k_value = result["bundle"]["k_value"]
                    goal_rate = result["goal_selection_rate"]
                k_samples_grid[gamma_index][chi_index].append(k_value)
                goal_samples_grid[gamma_index][chi_index].append(goal_rate)

    raw_k_matrix: list[list[float]] = []
    matrix_goal_rate: list[list[float]] = []
    for gamma_index, _ in enumerate(config["gamma_values"]):
        raw_k_matrix.append([_average(values) for values in k_samples_grid[gamma_index]])
        matrix_goal_rate.append([_average(values) for values in goal_samples_grid[gamma_index]])

    empirical_goal_rate_matrix = matrix_goal_rate
    matrix_k = raw_k_matrix
    k_display_scale = 1.0
    goal_rate_display_range = (
        min((value for row in empirical_goal_rate_matrix for value in row), default=0.0),
        max((value for row in empirical_goal_rate_matrix for value in row), default=0.0),
    )
    rows: list[dict] = []
    for gamma_index, gamma in enumerate(config["gamma_values"]):
        for chi_index, chi in enumerate(config["chi_values"]):
            rows.append(
                {
                    "chi": chi,
                    "gamma": gamma,
                    "raw_k": raw_k_matrix[gamma_index][chi_index],
                    "k": matrix_k[gamma_index][chi_index],
                    "empirical_goal_selection_rate": empirical_goal_rate_matrix[gamma_index][chi_index],
                    "goal_selection_rate": matrix_goal_rate[gamma_index][chi_index],
                }
            )
    return {
        "x_values": config["chi_values"],
        "y_values": config["gamma_values"],
        "k_matrix": matrix_k,
        "raw_k_matrix": raw_k_matrix,
        "k_display_scale": k_display_scale,
        "empirical_goal_rate_matrix": empirical_goal_rate_matrix,
        "goal_rate_display_range": goal_rate_display_range,
        "goal_rate_matrix": matrix_goal_rate,
        "rows": rows,
    }


def _run_selection_evaluation(platform: Platform, config: dict) -> dict:
    selection_rows: list[dict] = []
    time_cost_records: list[float] = []
    success_records: dict[str, list[tuple[float, float]]] = {"TRAIM": [], "DIM": [], "PRM": []}
    node_counts = config["node_counts"]
    dim_strategy = config.get("dim_strategy", "R")
    max_node_count = max(node_counts)
    no_decoy_counts = {node_count: {"A": [], "B": []} for node_count in node_counts}
    no_decoy_bid_intensity = {node_count: {"A": [], "B": []} for node_count in node_counts}
    dim_counts = {node_count: {"goal": [], "compete": [], "decoy": []} for node_count in node_counts}
    dim_bid_intensity = {node_count: {"goal": [], "compete": []} for node_count in node_counts}
    decoy_influence_counts = {node_count: [] for node_count in node_counts}

    for repeat_index in range(config["repeats"]):
        if repeat_index == 0 or (repeat_index + 1) == config["repeats"] or (repeat_index + 1) % max(1, config["repeats"] // 5) == 0:
            _progress(config, f"任务选择实验 {repeat_index + 1}/{config['repeats']}")
        tasks, nodes = platform.prepare_environment(max_node_count, config["task_count"], config["default_preference_mean"])
        for node_count in node_counts:
            node_subset = nodes[:node_count]
            no_decoy = platform.simulate_no_decoy(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))
            dim = platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset), strategy=dim_strategy)
            prm = None
            traim = None
            if node_count == config["comparison_node_count"]:
                prm = platform.simulate_prm(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))
                traim = platform.simulate_traim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))

            no_decoy_counts[node_count]["A"].append(no_decoy["selection_counts"].get("A", 0))
            no_decoy_counts[node_count]["B"].append(no_decoy["selection_counts"].get("B", 0))
            dim_counts[node_count]["goal"].append(dim["selection_counts"].get("A", 0))
            dim_counts[node_count]["compete"].append(dim["selection_counts"].get("B", 0))
            no_decoy_intensity = _bid_intensity_by_kind(no_decoy)
            dim_intensity = _bid_intensity_by_kind(dim)
            no_decoy_bid_intensity[node_count]["A"].append(no_decoy_intensity.get("A", 0))
            no_decoy_bid_intensity[node_count]["B"].append(no_decoy_intensity.get("B", 0))
            dim_bid_intensity[node_count]["goal"].append(dim_intensity.get("A", 0))
            dim_bid_intensity[node_count]["compete"].append(dim_intensity.get("B", 0))
            decoy_influence_counts[node_count].append(_decoy_influenced_node_count(no_decoy, dim))
            dim_counts[node_count]["decoy"].append(dim["rounds"][0]["display_selection_counts"].get("DECOY", 0))

            if node_count == config["comparison_node_count"]:
                time_cost_records.extend(task.time_cost for task in tasks)
                if traim is not None:
                    _append_success_records(success_records["TRAIM"], tasks, traim)
                _append_success_records(success_records["DIM"], tasks, dim)
                if prm is not None:
                    _append_success_records(success_records["PRM"], tasks, prm)

    for node_count in node_counts:
        selection_rows.append(
            {
                "node_count": node_count,
                "tau_A_raw": _average(no_decoy_counts[node_count]["A"]),
                "tau_B_raw": _average(no_decoy_counts[node_count]["B"]),
                "tau_goal_raw": _average(dim_counts[node_count]["goal"]),
                "tau_compete_raw": _average(dim_counts[node_count]["compete"]),
                "tau_decoy_raw": _average(dim_counts[node_count]["decoy"]),
                "tau_A_bid_intensity_raw": _average(no_decoy_bid_intensity[node_count]["A"]),
                "tau_B_bid_intensity_raw": _average(no_decoy_bid_intensity[node_count]["B"]),
                "tau_goal_bid_intensity_raw": _average(dim_bid_intensity[node_count]["goal"]),
                "tau_compete_bid_intensity_raw": _average(dim_bid_intensity[node_count]["compete"]),
                "tau_decoy_influenced_nodes_raw": _average(decoy_influence_counts[node_count]),
                "tau_A": _average(no_decoy_bid_intensity[node_count]["A"]),
                "tau_B": _average(no_decoy_bid_intensity[node_count]["B"]),
                "tau_goal": _average(dim_bid_intensity[node_count]["goal"]),
                "tau_compete": _average(dim_bid_intensity[node_count]["compete"]),
                "tau_decoy": _average(decoy_influence_counts[node_count]),
            }
        )

    return {
        "selection_rows": selection_rows,
        "time_cost_success_rates": _time_cost_success_rate_rows(time_cost_records, success_records, bin_count=5),
    }


def _run_strategy_comparison(platform: Platform, config: dict) -> dict:
    rows: list[dict] = []
    for node_count in config["node_counts"]:
        _progress(config, f"DIM 策略对比：节点数 {node_count}")
        strategy_values: dict[str, list[float]] = {"F": [], "R": [], "RF": []}
        for _ in range(config["repeats"]):
            tasks, nodes = platform.prepare_environment(node_count, config["task_count"], config["default_preference_mean"])
            for strategy in ("F", "R", "RF"):
                result = platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(nodes), strategy=strategy)
                strategy_values[strategy].append(result["selection_counts"].get("A", 0))
        rows.append(
            {
                "node_count": node_count,
                "F": _average(strategy_values["F"]),
                "R": _average(strategy_values["R"]),
                "RF": _average(strategy_values["RF"]),
            }
        )
    return {"rows": rows}


def _run_mechanism_comparison(platform: Platform, config: dict) -> dict:
    participant_rows: list[dict] = []
    offloaded_curve_rows: list[dict] = []
    utility_rows: dict[int, list[dict]] = {task_count: [] for task_count in config["utility_task_counts"]}
    representative_bids: dict[str, list[float]] = {}
    representative_prices: dict[str, list[float]] = {}
    representative_task_order: list[int] = []
    representative_selected = False
    mechanisms = ["DIM", "PRM", "TRAIM"]
    task_statistics = {
        "mechanisms": mechanisms,
        "total_tasks": config["task_count"],
        "completion_rate_samples": {mechanism: [] for mechanism in mechanisms},
        "completed_count_samples": {mechanism: [] for mechanism in mechanisms},
        "winning_bid_samples": {mechanism: [] for mechanism in mechanisms},
        "transaction_price_samples": {mechanism: [] for mechanism in mechanisms},
        "common_task_ratio_rows": [],
    }

    for node_count in config["node_counts"]:
        _progress(config, f"机制参与数对比：节点数 {node_count}")
        participant_samples: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
        unique_participant_samples: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
        for repeat_index in range(config["repeats"]):
            comparison = platform.compare_mechanisms(node_count, config["task_count"], config["default_preference_mean"])
            for mechanism in mechanisms:
                participant_samples[mechanism].append(_mechanism_participation_intensity(comparison[mechanism]))
                unique_participant_samples[mechanism].append(_mechanism_active_node_count(comparison[mechanism]))

            if node_count == config["comparison_node_count"]:
                if not representative_selected:
                    representative_selected = True
                    representative_bids = _task_level_metric_pair(comparison, "winning_bid")
                    representative_prices = _task_level_metric_pair(comparison, "transaction_price")
                    representative_task_order = _task_order_by_time_cost(comparison)
                _append_task_statistics(task_statistics, comparison, repeat_index)

        row = {"node_count": node_count}
        for mechanism in mechanisms:
            raw_value = _average(participant_samples[mechanism])
            unique_value = _average(unique_participant_samples[mechanism])
            row[f"{mechanism}_unique_participants_raw"] = unique_value
            row[f"{mechanism}_participation_intensity_raw"] = raw_value
            row[f"{mechanism}_raw"] = raw_value
            row[mechanism] = raw_value
        participant_rows.append(row)

    _run_utility_curves(platform, config, utility_rows)

    offloaded_samples_by_task_count = {
        task_count: {mechanism: [] for mechanism in mechanisms}
        for task_count in config["task_curve_counts"]
    }
    max_curve_task_count = max(config["task_curve_counts"])
    for repeat_index in range(config["repeats"]):
        if repeat_index == 0 or (repeat_index + 1) == config["repeats"] or (repeat_index + 1) % max(1, config["repeats"] // 5) == 0:
            _progress(config, f"offloaded task curve {repeat_index + 1}/{config['repeats']}")
        comparisons_by_task_count = None
        for _ in range(256):
            tasks, nodes = platform.prepare_environment(config["comparison_node_count"], max_curve_task_count, config["default_preference_mean"])
            try:
                comparisons_by_task_count = {
                    task_count: _compare_mechanisms_on_environment(platform, tasks[:task_count], nodes, config["default_preference_mean"])
                    for task_count in config["task_curve_counts"]
                }
                break
            except ValueError:
                continue
        if comparisons_by_task_count is None:
            raise RuntimeError("Failed to sample paired task-prefix environments with valid DIM A/B classes.")
        for task_count, comparison in comparisons_by_task_count.items():
            for mechanism in mechanisms:
                offloaded_samples_by_task_count[task_count][mechanism].append(comparison[mechanism]["offloaded_tasks"])
    for task_count in config["task_curve_counts"]:
        _progress(config, f"机制卸载数曲线：任务数 {task_count}")
        offloaded_samples = offloaded_samples_by_task_count[task_count]
        for _ in range(0):
            comparison = platform.compare_mechanisms(config["comparison_node_count"], task_count, config["default_preference_mean"])
            for mechanism in mechanisms:
                offloaded_samples[mechanism].append(comparison[mechanism]["offloaded_tasks"])
        row = {"task_count": task_count}
        for mechanism in mechanisms:
            raw_value = _average(offloaded_samples[mechanism])
            row[f"{mechanism}_raw"] = raw_value
            row[f"{mechanism}_std_error"] = _standard_error(offloaded_samples[mechanism])
            row[mechanism] = raw_value
        offloaded_curve_rows.append(row)

    return {
        "participant_rows": participant_rows,
        "offloaded_curve_rows": offloaded_curve_rows,
        "task_bid_rows": representative_bids,
        "task_price_rows": representative_prices,
        "task_order_by_time_cost": representative_task_order,
        "task_statistics": task_statistics,
        "utility_rows": utility_rows,
    }


def _run_utility_curves(platform: Platform, config: dict, utility_rows: dict[int, list[dict]]) -> None:
    scale = config.get("utility_display_scale", 1.0)
    repeat_count = config.get("utility_repeats", config["repeats"])
    utility_platform = Platform(dict(config))
    node_counts = config["node_counts"]
    max_node_count = max(node_counts)
    mechanisms = ["DIM", "PRM", "TRAIM"]
    for utility_task_count in config["utility_task_counts"]:
        _progress(config, f"用户效用曲线：任务数 {utility_task_count}")
        samples = {
            node_count: {mechanism: [] for mechanism in mechanisms}
            for node_count in node_counts
        }
        for _ in range(repeat_count):
            tasks, nodes = utility_platform.prepare_environment(max_node_count, utility_task_count, config["default_preference_mean"])
            for node_count in node_counts:
                node_subset = nodes[:node_count]
                dim = utility_platform.simulate_dim(utility_platform.clone_tasks(tasks), utility_platform.clone_nodes(node_subset), strategy=config.get("dim_strategy", "R"))
                prm = utility_platform.simulate_prm(utility_platform.clone_tasks(tasks), utility_platform.clone_nodes(node_subset))
                traim = utility_platform.simulate_traim(utility_platform.clone_tasks(tasks), utility_platform.clone_nodes(node_subset))
                samples[node_count]["DIM"].append(dim["user_total_utility"])
                samples[node_count]["PRM"].append(prm["user_total_utility"])
                samples[node_count]["TRAIM"].append(traim["user_total_utility"])
        for node_count in node_counts:
            row = {"node_count": node_count}
            for mechanism in mechanisms:
                raw_value = _average(samples[node_count][mechanism])
                row[f"{mechanism}_raw"] = raw_value
                row[f"{mechanism}_display"] = raw_value / scale
                row[mechanism] = row[f"{mechanism}_display"]
            utility_rows[utility_task_count].append(row)


def _compare_mechanisms_on_environment(platform: Platform, tasks: list, nodes: list, preference_mean: float) -> dict:
    return {
        "task_count": len(tasks),
        "node_count": len(nodes),
        "preference_mean": preference_mean,
        "tasks": [
            {
                "task_id": task.task_id,
                "task_index": int(task.task_id.split("_")[-1]),
                "reward": task.reward,
                "time_cost": task.time_cost,
                "clock_frequency": task.clock_frequency,
            }
            for task in tasks
        ],
        "DIM": platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(nodes), strategy=platform.config.get("dim_strategy", "R")),
        "PRM": platform.simulate_prm(platform.clone_tasks(tasks), platform.clone_nodes(nodes)),
        "TRAIM": platform.simulate_traim(platform.clone_tasks(tasks), platform.clone_nodes(nodes)),
    }


def _run_preference_mean_effect(platform: Platform, config: dict) -> dict:
    dim_rows: list[dict] = []
    prm_rows: list[dict] = []
    node_counts = config["node_counts"]
    max_node_count = max(node_counts)
    for preference_mean in config["preference_means"]:
        _progress(config, f"偏好系数影响：d={preference_mean}")
        dim_goal_samples = {node_count: [] for node_count in node_counts}
        dim_total_samples = {node_count: [] for node_count in node_counts}
        dim_threshold_samples = {node_count: [] for node_count in node_counts}
        prm_samples = {node_count: [] for node_count in node_counts}
        for _ in range(config["repeats"]):
            tasks, nodes = platform.prepare_environment(max_node_count, config["task_count"], preference_mean)
            for node_count in node_counts:
                node_subset = nodes[:node_count]
                dim = platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset), strategy=config.get("dim_strategy", "R"))
                prm = platform.simulate_prm(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))
                dim_goal_samples[node_count].append(dim["goal_participants"])
                dim_total_samples[node_count].append(dim["participants"])
                dim_threshold_samples[node_count].append(dim["goal_choice_threshold"])
                prm_samples[node_count].append(prm["participants"])
        dim_goal_series = [_average(dim_goal_samples[node_count]) for node_count in node_counts]
        dim_total_series = [_average(dim_total_samples[node_count]) for node_count in node_counts]
        dim_threshold_series = [
            _average([value for value in dim_threshold_samples[node_count] if value is not None])
            for node_count in node_counts
        ]
        prm_series = [_average(prm_samples[node_count]) for node_count in node_counts]
        dim_rows.append(
            {
                "preference_mean": preference_mean,
                "values": dim_goal_series,
                "total_values": dim_total_series,
                "threshold_values": dim_threshold_series,
            }
        )
        prm_rows.append({"preference_mean": preference_mean, "values": prm_series})
    return {"dim_rows": dim_rows, "prm_rows": prm_rows}


def _run_truthfulness_check(config: dict) -> dict:
    audit_config = dict(config)
    audit_config["seed"] = int(config["seed"]) + 9101
    audit_platform = Platform(audit_config)
    node_count = int(config.get("truthfulness_node_count", config["comparison_node_count"]))
    task_count = int(config.get("truthfulness_task_count", config["task_count"]))
    preference_mean = float(config.get("truthfulness_preference_mean", config["default_preference_mean"]))
    multipliers = [float(value) for value in config.get("truthfulness_bid_multipliers", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])]

    tasks, nodes = audit_platform.prepare_environment(node_count, task_count, preference_mean)
    dim = audit_platform.simulate_dim(audit_platform.clone_tasks(tasks), audit_platform.clone_nodes(nodes), strategy=audit_config.get("dim_strategy", "R"))
    prm = audit_platform.simulate_prm(audit_platform.clone_tasks(tasks), audit_platform.clone_nodes(nodes))
    mobile_devices, base_stations = build_traim_environment(audit_platform.clone_tasks(tasks), audit_platform.clone_nodes(nodes), audit_platform.rng, audit_config)

    rows: list[dict] = []
    rows.extend(_auction_bid_scan_rows("DIM", dim, multipliers))
    rows.extend(_auction_bid_scan_rows("PRM", prm, multipliers))
    rows.extend(_traim_bid_scan_rows(mobile_devices, base_stations, multipliers))
    audit_rows = _truthfulness_audit_rows(rows)
    return {
        "rows": rows,
        "audit_rows": audit_rows,
        "notes": [
            "DIM and PRM scans reuse the realized bid book and recompute winners/payments with all non-target bids fixed.",
            "PRM dynamic preference updates are not re-simulated for each report; this is an ex-post implementation audit.",
            "TRAIM scans multiply one base station's reported cost for allocation/payment while holding physical coverage and true costs fixed.",
        ],
    }


def _auction_bid_scan_rows(mechanism: str, result: dict, multipliers: list[float]) -> list[dict]:
    bid_records = [
        dict(bid)
        for round_result in result.get("rounds", [])
        for bid in round_result.get("bids", [])
        if not bid.get("is_decoy")
    ]
    target_node_id = _first_target_bidder(result, bid_records)
    rows: list[dict] = []
    for multiplier in multipliers:
        payment, utility, won_task_count = _auction_scan_outcome(bid_records, target_node_id, multiplier)
        rows.append(
            {
                "mechanism": mechanism,
                "target_id": target_node_id,
                "bid_multiplier": multiplier,
                "won": won_task_count > 0,
                "won_task_count": won_task_count,
                "payment": payment if won_task_count > 0 else None,
                "utility": utility,
                "truthful_bid_reference": "1.0c",
            }
        )
    return rows


def _first_target_bidder(result: dict, bid_records: list[dict]) -> str:
    for node_id in result.get("participant_ids", []):
        if any(bid.get("node_id") == node_id for bid in bid_records):
            return node_id
    return bid_records[0]["node_id"] if bid_records else ""


def _auction_scan_outcome(bid_records: list[dict], target_node_id: str, multiplier: float) -> tuple[float, float, int]:
    bids_by_task: dict[str, list[dict]] = {}
    for bid in bid_records:
        adjusted = dict(bid)
        if bid.get("node_id") == target_node_id:
            truthful_value = float(bid.get("truthful_bid", bid.get("accepted_bid", 0.0)))
            reported_bid = truthful_value * multiplier
            accepted_bid, cancelled = monitor_bid(reported_bid, truthful_value)
            if cancelled or accepted_bid <= 0.0:
                continue
            adjusted["accepted_bid"] = accepted_bid
            adjusted["reported_bid"] = reported_bid
        elif float(adjusted.get("accepted_bid", 0.0)) <= 0.0:
            continue
        bids_by_task.setdefault(adjusted["task_id"], []).append(adjusted)

    payment = 0.0
    utility = 0.0
    won_task_count = 0
    for bids in bids_by_task.values():
        ordered_bids = sorted(bids, key=lambda item: (float(item["accepted_bid"]), item["node_id"]))
        winner = ordered_bids[0]
        second_price = float(ordered_bids[1]["accepted_bid"]) if len(ordered_bids) > 1 else float(winner["accepted_bid"])
        if winner.get("node_id") != target_node_id:
            continue
        won_task_count += 1
        payment += second_price
        utility += second_price - float(winner.get("truthful_bid", winner.get("execution_cost", 0.0)))
    return payment, utility, won_task_count


def _traim_bid_scan_rows(mobile_devices: list, base_stations: list, multipliers: list[float]) -> list[dict]:
    baseline = run_traim(mobile_devices, base_stations)
    target_bs_id = baseline.bidder_ids[0] if baseline.bidder_ids else (base_stations[0].bs_id if base_stations else "")
    rows: list[dict] = []
    for multiplier in multipliers:
        winners, candidates, payments = _run_traim_with_report_multiplier(mobile_devices, base_stations, target_bs_id, multiplier)
        md_by_id = {device.md_id: device for device in mobile_devices}
        target_candidate = candidates.get(target_bs_id)
        true_cost = candidate_cost(_bs_by_id(base_stations)[target_bs_id], target_candidate, md_by_id) if target_candidate else 0.0
        payment = payments.get(target_bs_id)
        utility = (payment - true_cost) if payment is not None else 0.0
        rows.append(
            {
                "mechanism": "TRAIM",
                "target_id": target_bs_id,
                "bid_multiplier": multiplier,
                "won": target_bs_id in winners,
                "won_task_count": len(target_candidate.md_ids) if target_candidate and target_bs_id in winners else 0,
                "payment": payment,
                "utility": utility,
                "truthful_bid_reference": "1.0c",
            }
        )
    return rows


def _run_traim_with_report_multiplier(
    mobile_devices: list,
    base_stations: list,
    target_bs_id: str,
    multiplier: float,
) -> tuple[list[str], dict, dict[str, float]]:
    if not mobile_devices or not base_stations:
        return [], {}, {}
    f_max = max(base_station.cpu_cores for base_station in base_stations)
    theta_max = max(base_station.subchannels for base_station in base_stations)
    remaining_bs_ids = {base_station.bs_id for base_station in base_stations}
    unserved_md_ids = {device.md_id for device in mobile_devices}
    bs_by_id = _bs_by_id(base_stations)
    md_by_id = {device.md_id: device for device in mobile_devices}
    winners: list[str] = []
    winning_candidates: dict = {}
    while unserved_md_ids and remaining_bs_ids:
        round_candidates = {
            bs_id: assign_mobile_devices(mobile_devices, bs_by_id[bs_id], set(unserved_md_ids), f_max, theta_max)
            for bs_id in sorted(remaining_bs_ids)
        }
        round_candidates = {
            bs_id: candidate
            for bs_id, candidate in round_candidates.items()
            if candidate.cpr_denominator > 0.0
        }
        if not round_candidates:
            break
        selected_bs_id = min(
            round_candidates,
            key=lambda bs_id: (
                _reported_traim_cost(bs_by_id[bs_id], round_candidates[bs_id], md_by_id, target_bs_id, multiplier)
                / round_candidates[bs_id].cpr_denominator,
                _reported_traim_cost(bs_by_id[bs_id], round_candidates[bs_id], md_by_id, target_bs_id, multiplier),
                bs_id,
            ),
        )
        selected_candidate = round_candidates[selected_bs_id]
        winners.append(selected_bs_id)
        winning_candidates[selected_bs_id] = selected_candidate
        remaining_bs_ids.remove(selected_bs_id)
        unserved_md_ids.difference_update(selected_candidate.md_ids)
    payments = _reported_traim_payments(mobile_devices, base_stations, winners, winning_candidates, target_bs_id, multiplier)
    return winners, winning_candidates, payments


def _reported_traim_payments(
    mobile_devices: list,
    base_stations: list,
    winners: list[str],
    winning_candidates: dict,
    target_bs_id: str,
    multiplier: float,
) -> dict[str, float]:
    bs_by_id = _bs_by_id(base_stations)
    md_by_id = {device.md_id: device for device in mobile_devices}
    f_max = max(base_station.cpu_cores for base_station in base_stations)
    theta_max = max(base_station.subchannels for base_station in base_stations)
    all_md_ids = {device.md_id for device in mobile_devices}
    payments: dict[str, float] = {}
    for winner_id in winners:
        winner_candidate = winning_candidates[winner_id]
        target_ids = set(winner_candidate.md_ids)
        remaining_bs_ids = {base_station.bs_id for base_station in base_stations if base_station.bs_id != winner_id}
        locally_unserved = set(all_md_ids)
        locally_served: set[str] = set()
        critical_payment = 0.0
        while target_ids - locally_served and remaining_bs_ids:
            candidates = {
                bs_id: assign_mobile_devices(mobile_devices, bs_by_id[bs_id], set(locally_unserved), f_max, theta_max)
                for bs_id in sorted(remaining_bs_ids)
            }
            candidates = {bs_id: candidate for bs_id, candidate in candidates.items() if candidate.cpr_denominator > 0.0}
            if not candidates:
                break
            substitute_id = min(
                candidates,
                key=lambda bs_id: (
                    _reported_traim_cost(bs_by_id[bs_id], candidates[bs_id], md_by_id, target_bs_id, multiplier)
                    / candidates[bs_id].cpr_denominator,
                    _reported_traim_cost(bs_by_id[bs_id], candidates[bs_id], md_by_id, target_bs_id, multiplier),
                    bs_id,
                ),
            )
            substitute = candidates[substitute_id]
            substitute_cpr = (
                _reported_traim_cost(bs_by_id[substitute_id], substitute, md_by_id, target_bs_id, multiplier)
                / substitute.cpr_denominator
            )
            remaining_bs_ids.remove(substitute_id)
            locally_unserved.difference_update(substitute.md_ids)
            locally_served.update(substitute.md_ids)
            if target_ids.issubset(locally_served):
                critical_payment = substitute_cpr * winner_candidate.cpr_denominator
                break
        winner_reported_cost = _reported_traim_cost(bs_by_id[winner_id], winner_candidate, md_by_id, target_bs_id, multiplier)
        payments[winner_id] = max(critical_payment, winner_reported_cost)
    return payments


def _reported_traim_cost(base_station, candidate, md_by_id: dict, target_bs_id: str, multiplier: float) -> float:
    true_cost = candidate_cost(base_station, candidate, md_by_id)
    return true_cost * multiplier if base_station.bs_id == target_bs_id else true_cost


def _bs_by_id(base_stations: list) -> dict:
    return {base_station.bs_id: base_station for base_station in base_stations}


def _truthfulness_audit_rows(rows: list[dict]) -> list[dict]:
    audit_rows: list[dict] = []
    for mechanism in sorted({row["mechanism"] for row in rows}):
        mechanism_rows = [row for row in rows if row["mechanism"] == mechanism]
        truthful_rows = [row for row in mechanism_rows if math.isclose(float(row["bid_multiplier"]), 1.0, abs_tol=1e-9)]
        truthful_utility = float(truthful_rows[0]["utility"]) if truthful_rows else None
        max_utility = max((float(row["utility"]) for row in mechanism_rows), default=0.0)
        passed = truthful_utility is not None and truthful_utility >= max_utility - 1e-9
        best_multipliers = [
            row["bid_multiplier"]
            for row in mechanism_rows
            if math.isclose(float(row["utility"]), max_utility, rel_tol=1e-9, abs_tol=1e-9)
        ]
        audit_rows.append(
            {
                "mechanism": mechanism,
                "target_id": mechanism_rows[0]["target_id"] if mechanism_rows else "",
                "truthful_utility": truthful_utility,
                "max_utility": max_utility,
                "best_bid_multipliers": ";".join(str(value) for value in best_multipliers),
                "passed_truthfulness_check": passed,
                "audit_note": "truthful bid maximizes utility in this fixed-bid scan" if passed else "truthful bid is not utility-maximizing in this fixed-bid scan",
            }
        )
    return audit_rows


def _build_figures(figure_dir: Path, results: dict) -> list[dict]:
    figures: list[dict] = []

    dim_parameters = results["dim_parameter_analysis"]
    paper_surface_pair(
        x_values=dim_parameters["x_values"],
        y_values=dim_parameters["y_values"],
        z_top=dim_parameters["k_matrix"],
        z_bottom=dim_parameters["goal_rate_matrix"],
        zlabels=("诱饵效应强度系数 K", "目标任务被选率"),
        subtitles=("（a）诱饵因子对强度系数 K 的影响", "（b）诱饵因子对目标任务被选率的影响"),
        caption="图 3-5 诱饵因子对 DIM 机制的影响",
        output_path=figure_dir / "figure_3_5.png",
    )
    figures.append({"label": "图 3-5 诱饵因子对 DIM 机制的影响", "filename": "figure_3_5.png"})

    selection_rows = results["selection_evaluation"]["selection_rows"]
    x_nodes = [row["node_count"] for row in selection_rows]
    paper_line_chart(
        x_values=x_nodes,
        series=[
            ("τ_compete", [row["tau_compete"] for row in selection_rows]),
            ("τ_goal", [row["tau_goal"] for row in selection_rows]),
            ("τ_decoy", [row["tau_decoy"] for row in selection_rows]),
            ("τ_A", [row["tau_A"] for row in selection_rows]),
            ("τ_B", [row["tau_B"] for row in selection_rows]),
        ],
        x_label="环境中雾节点数",
        y_label="参与强度 / 受诱饵影响节点数",
        caption="图 3-6 各任务参与强度与诱饵影响节点数比较",
        output_path=figure_dir / "figure_3_6.png",
    )
    figures.append({"label": "图 3-6 各任务参与节点数量的比较", "filename": "figure_3_6.png"})

    success_rows = results["selection_evaluation"]["time_cost_success_rates"]
    bin_labels = [row["bin_label"] for row in success_rows]
    paper_grouped_bar_chart(
        x_labels=bin_labels,
        series=[
            ("TRAIM", [row["TRAIM"] for row in success_rows]),
            ("DIM", [row["DIM"] for row in success_rows]),
            ("PRM", [row["PRM"] for row in success_rows]),
        ],
        x_label="任务时间成本区间",
        y_label="任务卸载成功率",
        caption="图 3-7 不同时间成本区间下任务卸载成功率比较",
        output_path=figure_dir / "figure_3_7.png",
        figsize=(6.8, 4.9),
        y_lim=(0.0, 1.0),
    )
    figures.append({"label": "图 3-7 不同时间成本区间下任务卸载成功率比较", "filename": "figure_3_7.png"})

    strategy_rows = results["strategy_evaluation"]["rows"]
    paper_line_chart(
        x_values=[row["node_count"] for row in strategy_rows],
        series=[
            ("R策略", [row["R"] for row in strategy_rows]),
            ("F策略", [row["F"] for row in strategy_rows]),
            ("RF策略", [row["RF"] for row in strategy_rows]),
        ],
        x_label="雾节点数",
        y_label="参与节点个数",
        caption="图 3-8 诱饵效应不同策略的比较",
        output_path=figure_dir / "figure_3_8.png",
    )
    figures.append({"label": "图 3-8 诱饵效应不同策略的比较", "filename": "figure_3_8.png"})

    mechanism = results["mechanism_comparison"]
    participant_rows = mechanism["participant_rows"]
    paper_line_chart(
        x_values=[row["node_count"] for row in participant_rows],
        series=[
            ("TRAIM", [row["TRAIM"] for row in participant_rows]),
            ("PRM机制", [row["PRM"] for row in participant_rows]),
            ("DIM机制", [row["DIM"] for row in participant_rows]),
        ],
        x_label="雾节点数",
        y_label="节点参与强度",
        caption="图 4-2 DIM、PRM 与 TRAIM 机制下节点参与强度",
        output_path=figure_dir / "figure_4_2.png",
    )
    figures.append({"label": "图 4-2 DIM、PRM 与 TRAIM 节点参与强度", "filename": "figure_4_2.png"})

    preference = results["preference_mean_effect"]
    paper_line_chart(
        x_values=results["config"]["node_counts"],
        series=[(f"d={row['preference_mean']:.1f}", row["values"]) for row in preference["dim_rows"]],
        x_label="雾节点数",
        y_label="目标任务参与节点数",
        caption="图 4-3 DIM 机制下偏好系数对目标任务参与节点数的影响",
        output_path=figure_dir / "figure_4_3.png",
        figsize=(6.3, 5.2),
    )
    figures.append({"label": "图 4-3 DIM 机制下偏好系数影响", "filename": "figure_4_3.png"})

    paper_line_chart(
        x_values=results["config"]["node_counts"],
        series=[(f"d={row['preference_mean']:.1f}", row["values"]) for row in preference["prm_rows"]],
        x_label="雾节点数",
        y_label="雾节点参与总数",
        caption="图 4-4 PRM 机制下偏好系数对雾节点参与总数的影响",
        output_path=figure_dir / "figure_4_4.png",
        figsize=(6.3, 5.2),
    )
    figures.append({"label": "图 4-4 PRM 机制下偏好系数影响", "filename": "figure_4_4.png"})

    offloaded_curve = mechanism["offloaded_curve_rows"]
    paper_line_chart(
        x_values=[row["task_count"] for row in offloaded_curve],
        series=[
            ("TRAIM", [row["TRAIM"] for row in offloaded_curve]),
            ("PRM机制", [row["PRM"] for row in offloaded_curve]),
            ("DIM机制", [row["DIM"] for row in offloaded_curve]),
        ],
        x_label="任务数",
        y_label="卸载成功任务数",
        caption="图 4-5 DIM、PRM 与 TRAIM 机制下卸载成功任务数比较",
        output_path=figure_dir / "figure_4_5.png",
    )
    figures.append({"label": "图 4-5 DIM、PRM 与 TRAIM 卸载成功任务数比较", "filename": "figure_4_5.png"})

    task_statistics = mechanism["task_statistics"]
    paper_mechanism_summary_chart(
        mechanisms=task_statistics["mechanisms"],
        completion_rates=task_statistics["completion_rate_samples"],
        completed_counts=task_statistics["completed_count_samples"],
        total_tasks=task_statistics["total_tasks"],
        bid_samples=task_statistics["winning_bid_samples"],
        price_samples=task_statistics["transaction_price_samples"],
        caption="图 4-6 DIM、PRM 与 TRAIM 机制下任务完成率、出价与成交报酬分布对比",
        output_path=figure_dir / "figure_4_6.png",
    )
    figures.append({"label": "图 4-6 DIM、PRM 与 TRAIM 机制级统计对比", "filename": "figure_4_6.png"})

    task_order = mechanism["task_order_by_time_cost"] or list(range(1, len(mechanism["task_bid_rows"]["DIM"]) + 1))
    paper_task_heatmap_pair(
        task_indexes=task_order,
        bid_values=mechanism["task_bid_rows"],
        price_values=mechanism["task_price_rows"],
        mechanisms=task_statistics["mechanisms"],
        caption="图 4-7 DIM、PRM 与 TRAIM 机制下任务级成交情况热力图（× 表示未成交）",
        output_path=figure_dir / "figure_4_7.png",
    )
    figures.append({"label": "图 4-7 DIM、PRM 与 TRAIM 任务级成交热力图", "filename": "figure_4_7.png"})

    for utility_task_count, utility_rows in mechanism["utility_rows"].items():
        figure_name = f"figure_4_{8 if utility_task_count == 25 else 9}.png"
        utility_scale = results["config"].get("utility_display_scale", 1.0)
        paper_line_chart(
            x_values=[row["node_count"] for row in utility_rows],
            series=[
                ("TRAIM", [row["TRAIM"] for row in utility_rows]),
                ("PRM机制", [row["PRM"] for row in utility_rows]),
                ("DIM机制", [row["DIM"] for row in utility_rows]),
            ],
            x_label="雾节点数",
            y_label=f"移动设备总效用 / {utility_scale:g}",
            caption=f"图 4-{8 if utility_task_count == 25 else 9} 任务数为 {utility_task_count} 时 DIM、PRM 与 TRAIM 机制下用户终端设备总效用比较",
            output_path=figure_dir / figure_name,
        )
        figures.append({"label": f"图 4-{8 if utility_task_count == 25 else 9} 任务数为 {utility_task_count} 的用户总效用", "filename": figure_name})

    truthfulness_rows = results.get("truthfulness_check", {}).get("rows", [])
    if truthfulness_rows:
        multipliers = sorted({float(row["bid_multiplier"]) for row in truthfulness_rows})
        mechanisms = ["DIM", "PRM", "TRAIM"]
        paper_line_chart(
            x_values=multipliers,
            series=[
                (
                    mechanism_name,
                    [
                        _truthfulness_utility_at(truthfulness_rows, mechanism_name, multiplier)
                        for multiplier in multipliers
                    ],
                )
                for mechanism_name in mechanisms
            ],
            x_label="reported bid multiplier",
            y_label="target utility",
            caption="Truthfulness audit: fixed other bids, scanned target report multipliers",
            output_path=figure_dir / "truthfulness_check.png",
            figsize=(6.6, 4.8),
        )
        figures.append({"label": "Truthfulness bid-scan audit", "filename": "truthfulness_check.png"})

    return figures


def _truthfulness_utility_at(rows: list[dict], mechanism: str, multiplier: float) -> float:
    for row in rows:
        if row["mechanism"] == mechanism and math.isclose(float(row["bid_multiplier"]), multiplier, abs_tol=1e-9):
            return float(row["utility"])
    return math.nan


def _build_summary(results: dict, figure_paths: list[dict], explanation_path: Path, audit_path: Path) -> dict:
    participant_rows = results["mechanism_comparison"]["participant_rows"]
    utility_rows = results["mechanism_comparison"]["utility_rows"]
    utility_task_count = 50 if 50 in utility_rows else sorted(utility_rows)[-1]
    utility_rows_selected = utility_rows[utility_task_count]
    price_curve = results["mechanism_comparison"]["task_price_rows"]
    return {
        "validations": results["validations"],
        "mechanism_rows": [
            {
                "metric": "avg_participation_intensity",
                "DIM": _average([row["DIM"] for row in participant_rows]),
                "PRM": _average([row["PRM"] for row in participant_rows]),
                "TRAIM": _average([row["TRAIM"] for row in participant_rows]),
            },
            {
                "metric": "avg_transaction_price_task_level",
                "DIM": _average(price_curve["DIM"]),
                "PRM": _average(price_curve["PRM"]),
                "TRAIM": _average(price_curve["TRAIM"]),
            },
            {
                "metric": f"avg_user_utility_task{utility_task_count}",
                "DIM": _average([row["DIM_raw"] for row in utility_rows_selected]),
                "PRM": _average([row["PRM_raw"] for row in utility_rows_selected]),
                "TRAIM": _average([row["TRAIM_raw"] for row in utility_rows_selected]),
            },
        ],
        "figure_files": [item["filename"] for item in figure_paths],
        "explanation_file": explanation_path.name,
        "audit_file": audit_path.name,
    }


def _write_experiment_csvs(csv_dir: Path, results: dict) -> None:
    _write_rows(csv_dir / "dim_parameter_analysis.csv", results["dim_parameter_analysis"]["rows"])
    _write_rows(csv_dir / "dim_selection_counts.csv", results["selection_evaluation"]["selection_rows"])
    _write_rows(csv_dir / "time_cost_success_rates.csv", results["selection_evaluation"]["time_cost_success_rates"])
    _write_rows(csv_dir / "dim_strategy_compare.csv", results["strategy_evaluation"]["rows"])
    _write_rows(csv_dir / "mechanism_participants.csv", results["mechanism_comparison"]["participant_rows"])
    _write_rows(csv_dir / "mechanism_offloaded_curve.csv", results["mechanism_comparison"]["offloaded_curve_rows"])
    _write_rows(csv_dir / "mechanism_task_statistics.csv", _flatten_task_statistics(results["mechanism_comparison"]["task_statistics"]))
    _write_rows(csv_dir / "common_task_ratios.csv", results["mechanism_comparison"]["task_statistics"]["common_task_ratio_rows"])
    _write_rows(csv_dir / "task_level_bids.csv", _to_task_rows(results["mechanism_comparison"]["task_bid_rows"]))
    _write_rows(csv_dir / "task_level_prices.csv", _to_task_rows(results["mechanism_comparison"]["task_price_rows"]))
    for utility_task_count, rows in results["mechanism_comparison"]["utility_rows"].items():
        _write_rows(csv_dir / f"utility_task_{utility_task_count}.csv", rows)
    _write_rows(csv_dir / "dim_preference_mean_effect.csv", _flatten_dim_preference_rows(results["preference_mean_effect"]["dim_rows"], results["config"]["node_counts"]))
    _write_rows(csv_dir / "prm_preference_mean_effect.csv", _flatten_preference_rows(results["preference_mean_effect"]["prm_rows"], results["config"]["node_counts"]))
    _write_rows(csv_dir / "validations.csv", [results["validations"]])
    _write_rows(csv_dir / "truthfulness_check.csv", results["truthfulness_check"]["rows"])
    _write_rows(csv_dir / "truthfulness_audit.csv", results["truthfulness_check"]["audit_rows"])


def _write_raw_experiment_csvs(csv_dir: Path, results: dict) -> None:
    _write_rows(csv_dir / "dim_parameter_analysis_raw.csv", results["dim_parameter_analysis"]["rows"])
    _write_rows(csv_dir / "dim_selection_counts_raw.csv", results["selection_evaluation"]["selection_rows"])
    _write_rows(csv_dir / "time_cost_success_rates_raw.csv", results["selection_evaluation"]["time_cost_success_rates"])
    _write_rows(csv_dir / "mechanism_participants_raw.csv", results["mechanism_comparison"]["participant_rows"])
    _write_rows(csv_dir / "mechanism_offloaded_curve_raw.csv", results["mechanism_comparison"]["offloaded_curve_rows"])
    _write_rows(csv_dir / "mechanism_task_statistics_raw.csv", _flatten_task_statistics(results["mechanism_comparison"]["task_statistics"]))
    _write_rows(csv_dir / "task_level_bids_raw.csv", _to_task_rows(results["mechanism_comparison"]["task_bid_rows"]))
    _write_rows(csv_dir / "task_level_prices_raw.csv", _to_task_rows(results["mechanism_comparison"]["task_price_rows"]))
    for utility_task_count, rows in results["mechanism_comparison"]["utility_rows"].items():
        _write_rows(csv_dir / f"utility_task_{utility_task_count}_raw.csv", rows)
    _write_rows(csv_dir / "truthfulness_check_raw.csv", results["truthfulness_check"]["rows"])


def _build_audit_notes(results: dict) -> list[dict]:
    notes = [
        {
            "topic": "raw_results",
            "status": "passed",
            "note": "Simulation outputs are aggregated directly; calibration and forced ordering post-processing have been removed.",
        },
        {
            "topic": "display_scaling",
            "status": "documented",
            "note": f"Utility plot fields use display_value = raw_value / {results['config'].get('utility_display_scale', 1.0):g}; raw utility fields are preserved as *_raw.",
        },
        {
            "topic": "dim_strategy",
            "status": "documented",
            "note": f"Primary DIM runs use strategy={results['config'].get('dim_strategy', 'R')}; Figure 3-8 still reports F/R/RF strategy comparisons.",
        },
        {
            "topic": "dim_secondary_bids",
            "status": "documented",
            "note": "DIM-derived rounds may allow secondary positive real-task bids when dim_allow_secondary_positive_bids=true; these bids are generated before allocation and are part of the raw bid book.",
        },
        {
            "topic": "selection_metrics",
            "status": "documented",
            "note": "Figure 3-6 plot fields use raw bid intensity for A/B/goal/compete and paired decoy-influenced node counts for tau_decoy; raw selection counts are preserved as tau_*_raw.",
        },
        {
            "topic": "participation_metrics",
            "status": "documented",
            "note": "Figure 4-2 uses raw participation intensity; unique participant counts are preserved as *_unique_participants_raw.",
        },
        {
            "topic": "paired_task_curves",
            "status": "documented",
            "note": "Figure 4-5 uses common-random-number task prefixes and reports standard errors; no monotonic smoothing is applied.",
        },
        {
            "topic": "representative_heatmap",
            "status": "documented",
            "note": "Task-level bid/price heatmaps use the first repeat at comparison_node_count, not a ranking-optimized representative sample.",
        },
    ]
    for audit_row in results.get("truthfulness_check", {}).get("audit_rows", []):
        notes.append(
            {
                "topic": f"truthfulness_{audit_row['mechanism']}",
                "status": "passed" if audit_row["passed_truthfulness_check"] else "failed",
                "note": audit_row["audit_note"],
            }
        )
    return notes


def _write_audit_notes(output_dir: Path, results: dict) -> Path:
    path = output_dir / "audit_notes.md"
    lines = [
        "# Experiment Audit Notes",
        "",
        "This run preserves raw simulation statistics and avoids post-simulation calibration that changes mechanism ordering, success rates, participation, offloaded counts, or utility.",
        "",
        "## Notes",
    ]
    for note in results.get("audit_notes", []):
        lines.append(f"- **{note['topic']}** ({note['status']}): {note['note']}")
    lines.extend(["", "## Truthfulness Scan"])
    for audit_row in results.get("truthfulness_check", {}).get("audit_rows", []):
        lines.append(
            "- {mechanism}: target={target_id}, truthful_utility={truthful_utility}, "
            "max_utility={max_utility}, best_bid_multipliers={best_bid_multipliers}, "
            "passed={passed_truthfulness_check}".format(**audit_row)
        )
    lines.extend(["", "## Truthfulness Method Notes"])
    for note in results.get("truthfulness_check", {}).get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_rows(output_dir / "audit_notes.csv", results.get("audit_notes", []))
    return path


def _write_algorithm_document(output_dir: Path, results: dict) -> Path:
    path = output_dir / "DIM_PRM_算法实现说明.md"
    utility_rows = results["mechanism_comparison"]["utility_rows"]
    utility_task_count = 50 if 50 in utility_rows else sorted(utility_rows)[-1]
    content = f"""# DIM、PRM 与 TRAIM 机制实现说明

## 1. 这次输出做了什么调整

1. 按要求去除了所有面向旧基线 PMMRA 的公开实验对比与图表展示。
2. 保留了 DIM 与 PRM 两个论文核心机制，并把 DIM 章节中的“加入诱饵前/后”作为内部对照。
3. 所有图统一改为 `matplotlib` 生成的 `PNG` 文件，尽量贴近论文中的图形风格。
4. 输出目录中新增了本说明文档，详细解释两个机制及实验落地方式。

## 2. DIM 机制如何实现

### 2.1 目标

DIM 的目标是在不增加平台直接报酬支出的前提下，通过设计诱饵任务改变参考点，使雾节点更倾向于选择高报酬高时间成本的目标任务。

### 2.2 代码对应

- 任务分类与类任务画像：`repro/dim.py`
- 公式 `(2)(3)(4)(5)(6)`：`repro/formulas.py`
- DIM 主发布流程：`Platform.simulate_dim()` in `repro/platform.py`

### 2.3 实现步骤

1. 对原始任务报酬执行线性归一化，使其和时间成本在同一量纲上。
2. 计算全部任务的平均报酬与平均时间成本，构造参考点。
3. 按论文算法 1 将任务划分为 `A / B / C` 三类：
   - `A`：高报酬高时间成本
   - `B`：低报酬低时间成本
   - `C`：其余任务
4. 将 `A` 类平均任务作为 `goal`，`B` 类平均任务作为 `compete`。
5. 根据论文定理 3-1、3-2、3-3 设定诱饵参数，默认优先采用 `R` 策略，并在实验中额外计算 `F / R / RF` 三种策略。
6. 通过公式 `(6)` 计算诱饵效应强度系数 `K`。
7. 将诱饵任务与原任务统一发布给雾节点，统计目标任务被选率和高时间成本任务的卸载变化。

### 2.4 我在实验里怎么落地 DIM

1. 图 3-5 使用 `(χ, γ)` 网格扫描，生成两个三维曲面：
   - `K` 随诱饵因子的变化
   - 目标任务被选率随诱饵因子的变化
2. 图 3-6 比较加入诱饵前后的任务选择节点数，保留 `τ_A / τ_B` 与 `τ_goal / τ_compete / τ_decoy` 五条曲线。
3. 图 3-7 用散点图展示加入诱饵前后成功卸载任务的时间成本分布。
4. 图 3-8 对比 `F / R / RF` 三种诱饵策略下参与目标任务的节点数量。

## 3. PRM 机制如何实现

### 3.1 目标

PRM 在 DIM 的基础上进一步引入偏好逆转，通过动态更新雾节点偏好系数、分组推送剩余任务、降低任务出价和成交价，提升总参与量和用户终端总效用。

### 3.2 代码对应

- 偏好分组与诱饵推送：`repro/prm.py`
- 公式 `(15)(16)(17)(20)(21)(24)(26)`：`repro/formulas.py`
- PRM 多轮发布、估计、分组与任务推送：`Platform.simulate_prm()` in `repro/platform.py`

### 3.3 实现步骤

1. 首轮先执行一次 DIM 发布。
2. 雾节点依据偏好值选择任务类型，并按公式 `(21)` 形成报价。
3. 平台对每个任务采用 Vickrey 次低价拍卖：
   - 最低出价节点赢得任务
   - 次低价作为最终成交价
4. 平台根据已观察到的出价折扣反推出偏好满足差值，并按公式 `(24)` 估计偏好。
5. 按论文定理 4-1，将节点划分为：
   - `NPRN`
   - `PRN`
   - `PRIN`
6. 对落选任务重新分类，并针对不同组使用论文定理 4-2 中的诱饵参数进行推送。
7. 重复执行，直到无剩余任务或剩余任务不再适合继续细分推送。

### 3.4 我在实验里怎么落地 PRM

1. 图 4-2 比较 DIM 与 PRM 的总参与节点数。
2. 图 4-3 / 图 4-4 分别测试不同偏好系数均值 `d` 对 DIM 与 PRM 的影响。
3. 图 4-5 对比不同任务规模下两机制的成功卸载任务数。
4. 图 4-6 / 图 4-7 输出任务级别的节点出价与成交报酬柱状图。
5. 图 4-8 / 图 4-9 比较任务数为 `25` 与 `50` 时的用户终端总效用。

## 4. 为了让结果更贴近论文，我具体做了哪些工程处理

1. 保持论文给定核心参数不变：
   - `J=50`
   - `I=10~60`
   - `f_i, f_j ∈ [1, 1.5]`
   - `v_j ∈ [2, 20]`
   - `t_j ∈ [200, 2000]`
   - `α=β=0.88`
   - `λ=2.25`
2. 偏好系数使用 `0~1` 截断正态分布，并支持 `d=0.0~1.0` 的偏好均值实验。
3. 将图形全部改为白底、细虚线网格、论文风格图例框、PNG 输出。
4. PRM 的剩余任务采用多轮推送，并在任务级别汇总最终出价和成交价。
5. 对真实性、个体理性和计算复杂度做了程序化校验。

## 5. 当前输出中可以直接查看的结果

- 图像目录：`figures/`
- 明细数据：`csv/`
- 完整实验数据：`paper_results.json`
- 汇总摘要：`summary.json`

## 6. 当前一轮完整结果摘要

- DIM 平均节点参与强度：{_average([row["DIM"] for row in results["mechanism_comparison"]["participant_rows"]]):.4f}
- PRM 平均节点参与强度：{_average([row["PRM"] for row in results["mechanism_comparison"]["participant_rows"]]):.4f}
- TRAIM 平均节点参与强度：{_average([row["TRAIM"] for row in results["mechanism_comparison"]["participant_rows"]]):.4f}
- DIM 任务级平均成交报酬：{_average(results["mechanism_comparison"]["task_price_rows"]["DIM"]):.4f}
- PRM 任务级平均成交报酬：{_average(results["mechanism_comparison"]["task_price_rows"]["PRM"]):.4f}
- TRAIM 任务级平均临界支付：{_average(results["mechanism_comparison"]["task_price_rows"]["TRAIM"]):.4f}
- DIM 在任务数为 {utility_task_count} 时平均用户总效用：{_average([row["DIM_raw"] for row in utility_rows[utility_task_count]]):.4f}
- PRM 在任务数为 {utility_task_count} 时平均用户总效用：{_average([row["PRM_raw"] for row in utility_rows[utility_task_count]]):.4f}
- TRAIM 在任务数为 {utility_task_count} 时平均用户总效用：{_average([row["TRAIM_raw"] for row in utility_rows[utility_task_count]]):.4f}
"""
    compliance_note = (
        "> Compliance note: this document is generated from raw simulation outputs. "
        "The current experiment pipeline does not calibrate, floor, cap, smooth, or reorder raw results after simulation. "
        "Plot-scale fields are explicitly marked, for example `*_display = *_raw / utility_display_scale`. "
        "See `audit_notes.md` for the truthfulness scan and any failed audit result.\n\n"
    )
    path.write_text(compliance_note + content, encoding="utf-8")
    return path


def _scatter_points_from_result(result: dict) -> tuple[list[float], list[float]]:
    assignments = []
    task_time_costs: dict[str, float] = {}
    for round_result in result["rounds"]:
        for bid in round_result["bids"]:
            if not bid["is_decoy"]:
                task_time_costs.setdefault(bid["task_id"], bid["task_time_cost"])
        assignments.extend(
            assignment for assignment in round_result["assignments"] if not assignment["is_decoy"]
        )
    assignments = sorted(assignments, key=lambda item: _task_index(item["task_id"]))
    x_values = list(range(1, len(assignments) + 1))
    y_values = [task_time_costs.get(assignment["task_id"], 0.0) for assignment in assignments]
    return x_values, y_values


def _append_success_records(records: list[tuple[float, float]], tasks: list, result: dict) -> None:
    assigned_task_ids = {
        assignment["task_id"]
        for round_result in result["rounds"]
        for assignment in round_result["assignments"]
        if not assignment["is_decoy"]
    }
    for task in tasks:
        records.append((task.time_cost, 1.0 if task.task_id in assigned_task_ids else 0.0))


def _bid_intensity_by_kind(result: dict) -> dict[str, int]:
    counts: dict[str, set[tuple[str, str]]] = {}
    for round_result in result.get("rounds", []):
        for bid in round_result.get("bids", []):
            if bid.get("is_decoy"):
                continue
            kind = bid.get("selected_kind") or bid.get("task_kind")
            counts.setdefault(kind, set()).add((bid.get("node_id"), bid.get("task_id")))
    return {kind: len(values) for kind, values in counts.items()}


def _first_selected_kind_by_node(result: dict) -> dict[str, str]:
    selected: dict[str, str] = {}
    for round_result in result.get("rounds", []):
        for bid in round_result.get("bids", []):
            if bid.get("is_decoy"):
                continue
            selected.setdefault(bid.get("node_id"), bid.get("selected_kind") or bid.get("task_kind"))
    return selected


def _decoy_influenced_node_count(no_decoy_result: dict, dim_result: dict) -> int:
    no_decoy_choices = _first_selected_kind_by_node(no_decoy_result)
    dim_choices = _first_selected_kind_by_node(dim_result)
    return sum(
        1
        for node_id, dim_kind in dim_choices.items()
        if dim_kind == "A" and no_decoy_choices.get(node_id) != "A"
    )


def _time_cost_success_rate_rows(
    time_costs: list[float],
    success_records: dict[str, list[tuple[float, float]]],
    bin_count: int,
) -> list[dict]:
    if not time_costs:
        return []
    edges = _quantile_edges(time_costs, bin_count)
    rows: list[dict] = []
    for index in range(len(edges) - 1):
        lower = edges[index]
        upper = edges[index + 1]
        row = {
            "bin_index": index + 1,
            "bin_label": f"{lower:.0f}-{upper:.0f}",
            "lower_time_cost": lower,
            "upper_time_cost": upper,
        }
        for mechanism, records in success_records.items():
            in_bin = [
                success_credit
                for time_cost, success_credit in records
                if _is_in_bin(time_cost, lower, upper, is_last=index == len(edges) - 2)
            ]
            raw_value = sum(in_bin) / len(in_bin) if in_bin else None
            row[f"{mechanism}_raw"] = raw_value
            row[mechanism] = raw_value
        rows.append(row)
    return rows


def _quantile_edges(values: list[float], bin_count: int) -> list[float]:
    ordered = sorted(values)
    edges = []
    for index in range(bin_count + 1):
        position = round(index * (len(ordered) - 1) / bin_count)
        edges.append(ordered[position])
    for index in range(1, len(edges)):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1e-6
    return edges


def _is_in_bin(value: float, lower: float, upper: float, is_last: bool) -> bool:
    if is_last:
        return lower <= value <= upper
    return lower <= value < upper


def _append_task_statistics(task_statistics: dict, comparison: dict, repeat_index: int) -> None:
    bid_rows = _task_level_metric_pair(comparison, "winning_bid")
    price_rows = _task_level_metric_pair(comparison, "transaction_price")
    for mechanism in task_statistics["mechanisms"]:
        completed_count = comparison[mechanism]["offloaded_tasks"]
        task_statistics["completed_count_samples"][mechanism].append(completed_count)
        task_statistics["completion_rate_samples"][mechanism].append(completed_count / comparison["task_count"])
        task_statistics["winning_bid_samples"][mechanism].extend(_present_values(bid_rows[mechanism]))
        task_statistics["transaction_price_samples"][mechanism].extend(_present_values(price_rows[mechanism]))
    for metric_name, rows in (("winning_bid", bid_rows), ("transaction_price", price_rows)):
        dim_values = rows["DIM"]
        prm_values = rows["PRM"]
        common_indexes = [
            index
            for index, (dim_value, prm_value) in enumerate(zip(dim_values, prm_values), start=1)
            if dim_value is not None and prm_value is not None
        ]
        if not common_indexes:
            continue
        dim_common = [dim_values[index - 1] for index in common_indexes]
        prm_common = [prm_values[index - 1] for index in common_indexes]
        dim_mean = _average(dim_common)
        prm_mean = _average(prm_common)
        task_statistics["common_task_ratio_rows"].append(
            {
                "repeat_index": repeat_index,
                "metric": metric_name,
                "common_task_count": len(common_indexes),
                "DIM_mean": dim_mean,
                "PRM_mean": prm_mean,
                "PRM_over_DIM": prm_mean / dim_mean if dim_mean > 0.0 else None,
            }
        )


def _mechanism_active_node_count(result: dict) -> float:
    return float(result.get("participants", len(result.get("participant_ids", []))))


def _mechanism_participation_intensity(result: dict) -> float:
    if result.get("mechanism") == "PRM":
        return float(sum(round_result.get("participants_real", 0) for round_result in result.get("rounds", [])))
    return _mechanism_active_node_count(result)


def _present_values(values: list[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _task_order_by_time_cost(comparison: dict) -> list[int]:
    return [
        task["task_index"]
        for task in sorted(comparison.get("tasks", []), key=lambda item: (item["time_cost"], item["task_index"]))
    ]


def _task_level_metric_pair(comparison: dict, metric_name: str) -> dict[str, list[float | None]]:
    output: dict[str, list[float | None]] = {}
    for mechanism_name in ("DIM", "PRM", "TRAIM"):
        if mechanism_name not in comparison:
            continue
        result = comparison[mechanism_name]
        max_task_index = comparison["task_count"]
        values: list[float | None] = [None for _ in range(max_task_index)]
        if metric_name == "time_cost":
            for round_result in result["rounds"]:
                for bid in round_result["bids"]:
                    if bid["is_decoy"]:
                        continue
                    task_index = _task_index(bid["task_id"])
                    values[task_index - 1] = bid["task_time_cost"]
            output[mechanism_name] = values
            continue
        for round_result in result["rounds"]:
            for assignment in round_result["assignments"]:
                if assignment["is_decoy"]:
                    continue
                task_index = _task_index(assignment["task_id"])
                values[task_index - 1] = assignment[metric_name]
        output[mechanism_name] = values
    return output


def _task_index(task_id: str) -> int:
    return int(task_id.split("_")[-1])


def _to_task_rows(task_rows: dict[str, list[float | None]]) -> list[dict]:
    task_count = len(next(iter(task_rows.values()), []))
    rows: list[dict] = []
    for index in range(task_count):
        row = {"task_index": index + 1}
        for mechanism, values in task_rows.items():
            row[mechanism] = values[index]
        rows.append(row)
    return rows


def _flatten_task_statistics(task_statistics: dict) -> list[dict]:
    rows: list[dict] = []
    for mechanism in task_statistics["mechanisms"]:
        completion_rates = task_statistics["completion_rate_samples"][mechanism]
        completed_counts = task_statistics["completed_count_samples"][mechanism]
        bid_samples = task_statistics["winning_bid_samples"][mechanism]
        price_samples = task_statistics["transaction_price_samples"][mechanism]
        rows.append(
            {
                "mechanism": mechanism,
                "total_tasks": task_statistics["total_tasks"],
                "mean_completed_tasks": _average(completed_counts),
                "mean_completion_rate": _average(completion_rates),
                "winning_bid_sample_count": len(bid_samples),
                "winning_bid_mean": _average(bid_samples),
                "winning_bid_median": _median(bid_samples),
                "transaction_price_sample_count": len(price_samples),
                "transaction_price_mean": _average(price_samples),
                "transaction_price_median": _median(price_samples),
            }
        )
    return rows


def _flatten_preference_rows(rows: list[dict], node_counts: list[int]) -> list[dict]:
    flattened: list[dict] = []
    for row in rows:
        for node_count, value in zip(node_counts, row["values"]):
            flattened.append({"preference_mean": row["preference_mean"], "node_count": node_count, "participants": value})
    return flattened


def _flatten_dim_preference_rows(rows: list[dict], node_counts: list[int]) -> list[dict]:
    flattened: list[dict] = []
    for row in rows:
        for index, node_count in enumerate(node_counts):
            flattened.append(
                {
                    "preference_mean": row["preference_mean"],
                    "node_count": node_count,
                    "goal_participants": row["values"][index],
                    "total_participants": row["total_values"][index],
                    "goal_choice_threshold": row["threshold_values"][index],
                }
            )
    return flattened


def _average(values: list[float | None]) -> float:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return 0.0
    return sum(clean_values) / len(clean_values)


def _standard_error(values: list[float | None]) -> float | None:
    clean_values = [float(value) for value in values if value is not None]
    if len(clean_values) < 2:
        return None
    mean_value = sum(clean_values) / len(clean_values)
    variance = sum((value - mean_value) ** 2 for value in clean_values) / (len(clean_values) - 1)
    return math.sqrt(variance / len(clean_values))


def _median(values: list[float | None]) -> float:
    clean_values = sorted(value for value in values if value is not None)
    if not clean_values:
        return 0.0
    middle = len(clean_values) // 2
    if len(clean_values) % 2 == 1:
        return clean_values[middle]
    return (clean_values[middle - 1] + clean_values[middle]) / 2.0


def _clear_output_dir(path: Path, patterns: list[str]) -> None:
    for pattern in patterns:
        for item in path.glob(pattern):
            if item.is_file():
                item.unlink()


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
