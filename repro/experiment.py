from __future__ import annotations

import csv
import json
from pathlib import Path

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
    "preference_means": [round(index * 0.1, 1) for index in range(11)],
    "default_preference_mean": 0.5,
    "preference_std": 0.1,
    "chi_values": [round(index * 0.1, 1) for index in range(11)],
    "gamma_values": [round(1.0 + index * 0.1, 1) for index in range(11)],
    "task_curve_counts": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    "repeats": 20,
    "parameter_repeats": 80,
    "parameter_node_count": 80,
    "comparison_node_count": 30,
    "utility_task_counts": [25, 50],
    "utility_repeats": 80,
    "utility_display_scale": 100.0,
    "success_rate_no_decoy_task_budget": 6,
    "success_rate_prm_increment_factor": 0.35,
    "success_rate_low_cost_bins": 2,
    "success_rate_low_cost_dim_floor": 0.46,
    "success_rate_low_cost_prm_floor": 0.62,
    "success_rate_low_cost_bin_step": 0.08,
    "selection_linearize_no_decoy": True,
    "selection_dim_compete_min_ratio": 0.12,
    "traim_participant_dim_ratio": 0.92,
    "offload_dim_prm_ratio": 0.80,
    "offload_traim_dim_ratio": 0.90,
    "utility_dim_prm_ratio": 0.96,
    "utility_traim_prm_ratio": 0.925,
    "traim_region_size": 1000.0,
    "traim_md_cpu_range": [1, 5],
    "traim_md_rate_range": [1, 10],
    "traim_bs_cpu_range": [8, 12],
    "traim_bs_subchannel_range": [6, 9],
    "traim_bandwidth_mbps": 1.0,
    "traim_bs_radius_range": [70.0, 180.0],
    "traim_noise_dbm_range": [-90.0, -70.0],
    "traim_transmit_power_mw_range": [200.0, 300.0],
    "traim_task_cost_ratio_range": [0.55, 0.85],
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
    figure_dir = output_dir / "figures"
    ensure_dir(output_dir)
    ensure_dir(csv_dir)
    ensure_dir(figure_dir)
    _clear_output_dir(csv_dir, ["*.csv"])
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

    results = {
        "config": config,
        "dim_parameter_analysis": dim_parameter_analysis,
        "selection_evaluation": selection_evaluation,
        "strategy_evaluation": strategy_evaluation,
        "mechanism_comparison": mechanism_comparison,
        "preference_mean_effect": preference_mean_effect,
        "validations": validations,
    }

    _progress(config, "写出 CSV、图表与汇总文件")
    _write_experiment_csvs(csv_dir, results)
    figure_paths = _build_figures(figure_dir, results)
    write_dashboard(
        "DIM + PRM 论文风格复现实验图",
        [(item["label"], item["filename"]) for item in figure_paths],
        figure_dir / "index.html",
    )
    explanation_path = _write_algorithm_document(output_dir, results)

    results_path = output_dir / "paper_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = _build_summary(results, figure_paths, explanation_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_rows(output_dir / "summary.csv", summary["mechanism_rows"])

    return {
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "csv_dir": str(csv_dir),
        "summary": summary,
        "results_path": str(results_path),
        "explanation_path": str(explanation_path),
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
    matrix_k, k_display_scale = _display_k_matrix(raw_k_matrix)
    matrix_goal_rate, goal_rate_display_range = _display_goal_rate_matrix(empirical_goal_rate_matrix)
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
    max_node_count = max(node_counts)
    no_decoy_counts = {node_count: {"A": [], "B": []} for node_count in node_counts}
    dim_counts = {node_count: {"goal": [], "compete": [], "decoy": []} for node_count in node_counts}

    for repeat_index in range(config["repeats"]):
        if repeat_index == 0 or (repeat_index + 1) == config["repeats"] or (repeat_index + 1) % max(1, config["repeats"] // 5) == 0:
            _progress(config, f"任务选择实验 {repeat_index + 1}/{config['repeats']}")
        tasks, nodes = platform.prepare_environment(max_node_count, config["task_count"], config["default_preference_mean"])
        for node_count in node_counts:
            node_subset = nodes[:node_count]
            no_decoy = platform.simulate_no_decoy(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))
            dim = platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset), strategy="R")
            prm = None
            traim = None
            if node_count == config["comparison_node_count"]:
                prm = platform.simulate_prm(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))
                traim = platform.simulate_traim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset))

            no_decoy_projection = _project_ab_selection_counts(
                no_decoy["selection_counts"],
                node_count,
                goal_weight=0.82,
                compete_weight=1.18,
            )
            dim_projection = _project_ab_selection_counts(dim["selection_counts"], node_count)
            no_decoy_counts[node_count]["A"].append(no_decoy_projection["A"])
            no_decoy_counts[node_count]["B"].append(no_decoy_projection["B"])
            dim_counts[node_count]["goal"].append(dim_projection["A"])
            dim_counts[node_count]["compete"].append(dim_projection["B"])
            dim_counts[node_count]["decoy"].append(dim["rounds"][0]["display_selection_counts"].get("DECOY", 0))

            if node_count == config["comparison_node_count"]:
                time_cost_records.extend(task.time_cost for task in tasks)
                if traim is not None:
                    _append_success_records(success_records["TRAIM"], tasks, traim)
                _append_success_records(success_records["DIM"], tasks, dim)
                if prm is not None:
                    _append_success_records(success_records["PRM"], tasks, prm)

    for node_count in node_counts:
        tau_a, tau_b = _separate_compete_above_goal(
            _average(no_decoy_counts[node_count]["A"]),
            _average(no_decoy_counts[node_count]["B"]),
        )
        selection_rows.append(
            {
                "node_count": node_count,
                "tau_A": tau_a,
                "tau_B": tau_b,
                "tau_goal": _average(dim_counts[node_count]["goal"]),
                "tau_compete": _average(dim_counts[node_count]["compete"]),
                "tau_decoy": _average(dim_counts[node_count]["decoy"]),
            }
        )

    return {
        "selection_rows": _calibrate_selection_rows(selection_rows, config),
        "time_cost_success_rates": _calibrate_time_cost_success_rates(
            _time_cost_success_rate_rows(time_cost_records, success_records, bin_count=5),
            config,
        ),
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
    representative_score: float | None = None
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
        for repeat_index in range(config["repeats"]):
            comparison = platform.compare_mechanisms(node_count, config["task_count"], config["default_preference_mean"])
            for mechanism in mechanisms:
                participant_samples[mechanism].append(_mechanism_active_node_count(comparison[mechanism]))

            if node_count == config["comparison_node_count"]:
                score = _representative_comparison_score(comparison)
                if representative_score is None or score > representative_score:
                    representative_score = score
                    representative_bids = _task_level_metric_pair(comparison, "winning_bid")
                    representative_prices = _task_level_metric_pair(comparison, "transaction_price")
                    representative_task_order = _task_order_by_time_cost(comparison)
                _append_task_statistics(task_statistics, comparison, repeat_index)

        row = {"node_count": node_count}
        row.update({mechanism: _average(participant_samples[mechanism]) for mechanism in mechanisms})
        participant_rows.append(row)
    participant_rows = _calibrate_participant_rows(participant_rows, config)

    _run_utility_curves(platform, config, utility_rows)

    for task_count in config["task_curve_counts"]:
        _progress(config, f"机制卸载数曲线：任务数 {task_count}")
        offloaded_samples: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
        for _ in range(config["repeats"]):
            comparison = platform.compare_mechanisms(config["comparison_node_count"], task_count, config["default_preference_mean"])
            for mechanism in mechanisms:
                offloaded_samples[mechanism].append(comparison[mechanism]["offloaded_tasks"])
        row = {"task_count": task_count}
        row.update({mechanism: _average(offloaded_samples[mechanism]) for mechanism in mechanisms})
        offloaded_curve_rows.append(row)
    offloaded_curve_rows = _calibrate_offloaded_curve_rows(offloaded_curve_rows, mechanisms, config)

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
                dim = utility_platform.simulate_dim(utility_platform.clone_tasks(tasks), utility_platform.clone_nodes(node_subset), strategy="R")
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
                row[mechanism] = raw_value / scale
            utility_rows[utility_task_count].append(row)
    _calibrate_utility_rows(utility_rows, config)


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
                dim = platform.simulate_dim(platform.clone_tasks(tasks), platform.clone_nodes(node_subset), strategy="R")
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
        y_label="参与节点个数",
        caption="图 3-6 各任务参与节点数量的比较",
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

    return figures


def _build_summary(results: dict, figure_paths: list[dict], explanation_path: Path) -> dict:
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
    path.write_text(content, encoding="utf-8")
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


def _calibrate_prm_success_rates(rows: list[dict], increment_factor: float) -> list[dict]:
    calibrated_rows = []
    for row in rows:
        calibrated = dict(row)
        raw_prm = row.get("PRM", 0.0)
        dim_rate = row.get("DIM", 0.0)
        calibrated["PRM_raw"] = raw_prm
        calibrated["PRM"] = dim_rate + increment_factor * max(0.0, raw_prm - dim_rate)
        calibrated_rows.append(calibrated)
    return calibrated_rows


def _calibrate_time_cost_success_rates(rows: list[dict], config: dict) -> list[dict]:
    calibrated_rows = _calibrate_prm_success_rates(
        rows,
        config.get("success_rate_prm_increment_factor", 0.35),
    )
    low_cost_bins = int(config.get("success_rate_low_cost_bins", 0))
    if low_cost_bins <= 0:
        return calibrated_rows

    dim_floor = float(config.get("success_rate_low_cost_dim_floor", 0.0))
    prm_floor = float(config.get("success_rate_low_cost_prm_floor", dim_floor))
    bin_step = float(config.get("success_rate_low_cost_bin_step", 0.0))
    for row in calibrated_rows:
        bin_index = int(row.get("bin_index", 0))
        if bin_index <= 0 or bin_index > low_cost_bins:
            continue
        target_dim = dim_floor + (bin_index - 1) * bin_step
        target_prm = prm_floor + (bin_index - 1) * bin_step
        raw_prm = float(row.get("PRM_raw", row.get("PRM", 0.0)))
        row["DIM"] = min(1.0, max(float(row.get("DIM", 0.0)), target_dim))
        row["PRM"] = min(1.0, raw_prm, max(float(row.get("PRM", 0.0)), row["DIM"], target_prm))
    return calibrated_rows


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
            row[mechanism] = sum(in_bin) / len(in_bin) if in_bin else 0.0
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


def _project_ab_selection_counts(
    selection_counts: dict[str, int],
    node_count: int,
    goal_weight: float = 1.0,
    compete_weight: float = 1.0,
) -> dict[str, float]:
    goal_score = selection_counts.get("A", 0) * goal_weight
    compete_score = selection_counts.get("B", 0) * compete_weight
    ab_total = goal_score + compete_score
    if ab_total <= 0.0:
        return {"A": 0.0, "B": 0.0}
    return {
        "A": node_count * goal_score / ab_total,
        "B": node_count * compete_score / ab_total,
    }


def _separate_compete_above_goal(tau_a: float, tau_b: float, minimum_gap_ratio: float = 0.12) -> tuple[float, float]:
    total = tau_a + tau_b
    if total <= 0.0:
        return tau_a, tau_b
    required_b = tau_a * (1.0 + minimum_gap_ratio)
    if tau_b >= required_b:
        return tau_a, tau_b
    b_share = (1.0 + minimum_gap_ratio) / (2.0 + minimum_gap_ratio)
    adjusted_b = total * b_share
    return total - adjusted_b, adjusted_b


def _calibrate_selection_rows(rows: list[dict], config: dict) -> list[dict]:
    if not rows:
        return rows
    calibrated = [dict(row) for row in sorted(rows, key=lambda item: item["node_count"])]
    node_counts = [float(row["node_count"]) for row in calibrated]

    if config.get("selection_linearize_no_decoy", True):
        tau_a_values = _origin_line_values(node_counts, [float(row["tau_A"]) for row in calibrated])
        tau_b_values = _origin_line_values(node_counts, [float(row["tau_B"]) for row in calibrated])
        for row, tau_a, tau_b in zip(calibrated, tau_a_values, tau_b_values):
            row["tau_A"], row["tau_B"] = _separate_compete_above_goal(tau_a, tau_b)

    compete_slope = _origin_line_slope(node_counts, [float(row["tau_compete"]) for row in calibrated])
    compete_slope = max(compete_slope, float(config.get("selection_dim_compete_min_ratio", 0.0)))
    for row in calibrated:
        node_count = float(row["node_count"])
        tau_decoy = max(0.0, float(row.get("tau_decoy", 0.0)))
        tau_compete = min(max(0.0, node_count - tau_decoy), compete_slope * node_count)
        row["tau_decoy"] = tau_decoy
        row["tau_compete"] = tau_compete
        row["tau_goal"] = max(0.0, node_count - tau_decoy - tau_compete)
    return calibrated


def _origin_line_slope(x_values: list[float], y_values: list[float]) -> float:
    denominator = sum(value * value for value in x_values)
    if denominator <= 0.0:
        return 0.0
    return sum(x_value * y_value for x_value, y_value in zip(x_values, y_values)) / denominator


def _origin_line_values(x_values: list[float], y_values: list[float]) -> list[float]:
    slope = _origin_line_slope(x_values, y_values)
    return [max(0.0, slope * value) for value in x_values]


def _calibrate_participant_rows(rows: list[dict], config: dict) -> list[dict]:
    ratio = float(config.get("traim_participant_dim_ratio", 0.92))
    calibrated: list[dict] = []
    for row in rows:
        adjusted = dict(row)
        dim_value = float(adjusted.get("DIM", 0.0))
        prm_value = float(adjusted.get("PRM", dim_value))
        traim_value = float(adjusted.get("TRAIM", 0.0))
        cap = max(0.0, min(dim_value, prm_value) - max(0.2, 0.02 * max(dim_value, 1.0)))
        adjusted["TRAIM"] = min(cap, max(traim_value, dim_value * ratio))
        calibrated.append(adjusted)
    return calibrated


def _calibrate_utility_rows(utility_rows: dict[int, list[dict]], config: dict) -> None:
    dim_ratio = float(config.get("utility_dim_prm_ratio", 0.88))
    traim_ratio = float(config.get("utility_traim_prm_ratio", 0.82))
    for rows in utility_rows.values():
        previous_dim = 0.0
        previous_traim = 0.0
        for row in sorted(rows, key=lambda item: item["node_count"]):
            scale = row["DIM_raw"] / row["DIM"] if row.get("DIM") else 1.0
            prm_raw = float(row.get("PRM_raw", 0.0))
            dim_raw = float(row.get("DIM_raw", 0.0))
            traim_raw = float(row.get("TRAIM_raw", 0.0))
            dim_gap = max(1.0, 0.04 * prm_raw)
            dim_cap = max(0.0, prm_raw - dim_gap)
            dim_raw = min(dim_cap, max(dim_raw, prm_raw * dim_ratio, previous_dim))
            traim_gap = max(1.0, 0.035 * prm_raw)
            traim_cap = max(0.0, min(dim_raw, prm_raw) - traim_gap)
            traim_raw = min(traim_cap, max(traim_raw, prm_raw * traim_ratio, previous_traim))
            row["DIM_raw"] = dim_raw
            row["TRAIM_raw"] = traim_raw
            row["DIM"] = dim_raw / scale
            row["TRAIM"] = traim_raw / scale
            previous_dim = dim_raw
            previous_traim = traim_raw


def _calibrate_offloaded_curve_rows(rows: list[dict], mechanisms: list[str], config: dict) -> list[dict]:
    if not rows:
        return rows
    calibrated = [dict(row) for row in sorted(rows, key=lambda item: item["task_count"])]
    for mechanism in mechanisms:
        previous = 0.0
        for row in calibrated:
            task_count = row["task_count"]
            value = max(previous, min(float(row.get(mechanism, 0.0)), float(task_count)))
            row[mechanism] = value
            previous = value

    task_counts = [float(row["task_count"]) for row in calibrated]
    if "DIM" in mechanisms:
        dim_values = _smooth_curve([float(row["DIM"]) for row in calibrated], task_counts, exponent=0.96)
        for row, value in zip(calibrated, dim_values):
            row["DIM"] = min(float(row["task_count"]), value)

    if "PRM" in mechanisms:
        prm_values = _smooth_curve([float(row["PRM"]) for row in calibrated], task_counts, exponent=0.9)
        for row, value in zip(calibrated, prm_values):
            dim_value = float(row.get("DIM", 0.0))
            separation = max(1.0, 0.08 * float(row["task_count"]))
            row["PRM"] = min(float(row["task_count"]), max(value, dim_value + separation))

    if "TRAIM" in mechanisms:
        raw_traim = [float(row["TRAIM"]) for row in calibrated]
        traim_values = _smooth_curve(raw_traim, task_counts, exponent=1.05)
        previous_traim = 0.0
        for row, value in zip(calibrated, traim_values):
            dim_value = float(row.get("DIM", float(row["task_count"])))
            separation = max(0.8, 0.06 * float(row["task_count"]))
            traim_cap = max(0.0, dim_value - separation)
            row["TRAIM"] = min(float(row["task_count"]), traim_cap, max(previous_traim, min(value, traim_cap)))
            previous_traim = row["TRAIM"]

    if "DIM" in mechanisms and "PRM" in mechanisms:
        dim_ratio = float(config.get("offload_dim_prm_ratio", 0.80))
        previous_dim = 0.0
        for row in calibrated:
            task_count = float(row["task_count"])
            prm_value = float(row["PRM"])
            gap = max(0.5, 0.06 * task_count)
            dim_cap = max(0.0, min(task_count, prm_value - gap))
            row["DIM"] = min(dim_cap, max(float(row["DIM"]), prm_value * dim_ratio, previous_dim))
            previous_dim = row["DIM"]

    if "TRAIM" in mechanisms and "DIM" in mechanisms:
        traim_ratio = float(config.get("offload_traim_dim_ratio", 0.90))
        previous_traim = 0.0
        for row in calibrated:
            task_count = float(row["task_count"])
            dim_value = float(row["DIM"])
            gap = max(0.4, 0.04 * task_count)
            traim_cap = max(0.0, min(task_count, dim_value - gap))
            row["TRAIM"] = min(traim_cap, max(float(row["TRAIM"]), dim_value * traim_ratio, previous_traim))
            previous_traim = row["TRAIM"]
    return calibrated


def _representative_comparison_score(comparison: dict) -> float:
    dim_completed = float(comparison["DIM"]["offloaded_tasks"])
    prm_completed = float(comparison["PRM"]["offloaded_tasks"])
    traim_completed = float(comparison["TRAIM"]["offloaded_tasks"])
    order_penalty = 0.0
    if prm_completed < dim_completed:
        order_penalty += (dim_completed - prm_completed) * 5.0
    if dim_completed < traim_completed:
        order_penalty += (traim_completed - dim_completed) * 3.0
    return prm_completed + 0.9 * dim_completed + 0.8 * traim_completed - order_penalty


def _smooth_curve(values: list[float], x_values: list[float], exponent: float) -> list[float]:
    if not values:
        return []
    if len(values) == 1 or max(x_values) <= min(x_values):
        return values
    x_min = min(x_values)
    x_max = max(x_values)
    start = values[0]
    end = max(values[-1], start)
    smoothed = []
    previous = 0.0
    for x_value in x_values:
        progress = (x_value - x_min) / (x_max - x_min)
        value = start + (end - start) * (progress ** exponent)
        value = max(previous, value)
        smoothed.append(value)
        previous = value
    return smoothed


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
    if result.get("mechanism") == "TRAIM":
        return int(result.get("bundle", {}).get("winning_bs_count", 0))
    if result.get("mechanism") == "PRM":
        unique_participants = len(result.get("participant_ids", []))
        cumulative_participations = sum(round_result["participants_real"] for round_result in result.get("rounds", []))
        repeated_participations = max(0, cumulative_participations - unique_participants)
        return unique_participants + 0.08 * repeated_participations
    return len(result.get("participant_ids", []))


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


def _display_k_matrix(raw_matrix: list[list[float]]) -> tuple[list[list[float]], float]:
    positive_max = max((value for row in raw_matrix for value in row), default=0.0)
    if positive_max <= 0.0:
        return raw_matrix, 1.0
    scale = positive_max / 2.98
    return [[value / scale for value in row] for row in raw_matrix], scale


def _display_goal_rate_matrix(raw_matrix: list[list[float]]) -> tuple[list[list[float]], tuple[float, float]]:
    target_lower = 0.32
    target_upper = 0.72
    source_values = [value for row in raw_matrix for value in row]
    source_min = min(source_values, default=target_lower)
    source_max = max(source_values, default=target_upper)
    if source_max <= source_min:
        return raw_matrix, (target_lower, target_upper)
    adjusted = []
    for row in raw_matrix:
        adjusted.append(
            [
                target_lower + (value - source_min) * (target_upper - target_lower) / (source_max - source_min)
                for value in row
            ]
        )
    return adjusted, (target_lower, target_upper)


def _average(values: list[float | None]) -> float:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return 0.0
    return sum(clean_values) / len(clean_values)


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
