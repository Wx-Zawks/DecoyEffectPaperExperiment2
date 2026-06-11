from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from repro.models import FogNode, Task, mean


@dataclass
class PCSPECRP:
    crp_id: str
    node_id: str
    crp_type: str
    computing_power: float
    alpha: float
    initial_price: float
    current_price: float


@dataclass
class PCSPEAllocation:
    task_id: str
    crp_id: str
    node_id: str
    crp_type: str
    fraction: float
    price: float
    payment: float
    cost: float
    crp_profit: float


@dataclass
class PCSPETaskOutcome:
    task_id: str
    workload: float
    local_fraction: float
    offload_fraction: float
    crr_profit: float
    utility: float
    local_cost: float
    total_payment: float
    total_crp_cost: float
    social_welfare: float
    converged: bool
    iterations: int
    allocations: list[PCSPEAllocation]
    max_price_change: float = 0.0
    active_price_setting_count: int = 0
    active_price_taking_count: int = 0
    unilateral_deviation_passed: bool = True


@dataclass
class PCSPEOutcome:
    task_outcomes: list[PCSPETaskOutcome]
    allocation_rows: list[dict] = field(default_factory=list)
    task_result_rows: list[dict] = field(default_factory=list)
    convergence_rows: list[dict] = field(default_factory=list)
    summary_rows: list[dict] = field(default_factory=list)


def build_pcspe_environment(
    tasks: list[Task],
    nodes: list[FogNode],
    rng: Random,
    config: dict,
) -> list[PCSPECRP]:
    """Map project FogNodes to comparative_paper3 CRPs.

    The paper separates CRPs into price-setting and price-taking providers.
    We select the stronger nodes as price-setting CRPs using a stable sort so
    repeated runs do not depend on uncontrolled randomness.
    """

    if not nodes:
        return []
    ratio = float(config.get("pcspe_price_setting_ratio", 0.35))
    min_count = int(config.get("pcspe_min_price_setting_count", 1))
    price_setting_count = max(min_count, math.ceil(len(nodes) * ratio))
    price_setting_count = min(len(nodes), max(0, price_setting_count))
    ordered_nodes = sorted(nodes, key=lambda item: (-item.clock_frequency, _numeric_suffix(item.node_id), item.node_id))
    price_setting_ids = {node.node_id for node in ordered_nodes[:price_setting_count]}

    power_scale = float(config.get("pcspe_crp_power_scale", 2.0))
    alpha_s_range = _range_tuple(config.get("pcspe_cost_alpha_s_range", [3.0, 4.0]))
    alpha_t_range = _range_tuple(config.get("pcspe_cost_alpha_t_range", [8.0, 16.0]))
    price_upper = float(config.get("pcspe_price_upper", 50.0))

    crps: list[PCSPECRP] = []
    total = max(1, len(nodes))
    for index, node in enumerate(sorted(nodes, key=lambda item: (_numeric_suffix(item.node_id), item.node_id)), start=1):
        crp_type = "price_setting" if node.node_id in price_setting_ids else "price_taking"
        fraction = index / (total + 1)
        alpha_range = alpha_s_range if crp_type == "price_setting" else alpha_t_range
        alpha = alpha_range[0] + fraction * (alpha_range[1] - alpha_range[0])
        if crp_type == "price_setting":
            initial_price = min(price_upper, max(alpha, alpha * (1.1 + 0.05 * fraction)))
        else:
            initial_price = 0.0
        crps.append(
            PCSPECRP(
                crp_id=f"crp_{index}",
                node_id=node.node_id,
                crp_type=crp_type,
                computing_power=max(1e-9, node.clock_frequency * power_scale),
                alpha=round(alpha, 6),
                initial_price=round(initial_price, 6),
                current_price=round(initial_price, 6),
            )
        )
    return crps


def run_pcspe(tasks: list[Task], crps: list[PCSPECRP], config: dict) -> PCSPEOutcome:
    """Numerically simulate the PC-SPE Stackelberg equilibrium.

    This is a pure-Python numerical counterpart of comparative_paper3's
    backward-induction process: Stage III is represented by the inverse
    price-taking response p_t = 2 * alpha_t * d * x_t, Stage II solves the
    CRR's concave allocation problem by projected gradient ascent, and Stage I
    updates price-setting CRP prices through finite-difference best-response
    gradients. It intentionally remains an equilibrium audit baseline, not a
    truthful auction implementation.
    """

    task_outcomes = [_solve_task_pcspe(task, crps, config) for task in tasks]
    outcome = PCSPEOutcome(task_outcomes=task_outcomes)
    outcome.allocation_rows = pcspe_allocation_rows(outcome)
    outcome.task_result_rows = pcspe_task_result_rows(outcome)
    outcome.convergence_rows = pcspe_convergence_rows(outcome)
    outcome.summary_rows = pcspe_summary(outcome)
    return outcome


def pcspe_allocation_rows(outcome: PCSPEOutcome) -> list[dict]:
    rows: list[dict] = []
    for task_outcome in outcome.task_outcomes:
        for allocation in task_outcome.allocations:
            rows.append(
                {
                    "task_id": task_outcome.task_id,
                    "workload": round(task_outcome.workload, 6),
                    "crp_id": allocation.crp_id,
                    "node_id": allocation.node_id,
                    "crp_type": allocation.crp_type,
                    "fraction": round(allocation.fraction, 6),
                    "price": round(allocation.price, 6),
                    "payment": round(allocation.payment, 6),
                    "cost": round(allocation.cost, 6),
                    "crp_profit": round(allocation.crp_profit, 6),
                }
            )
    return rows


def pcspe_task_result_rows(outcome: PCSPEOutcome) -> list[dict]:
    return [
        {
            "task_id": task.task_id,
            "workload": round(task.workload, 6),
            "local_fraction": round(task.local_fraction, 6),
            "offload_fraction": round(task.offload_fraction, 6),
            "crr_profit": round(task.crr_profit, 6),
            "utility": round(task.utility, 6),
            "local_cost": round(task.local_cost, 6),
            "total_payment": round(task.total_payment, 6),
            "total_crp_cost": round(task.total_crp_cost, 6),
            "social_welfare": round(task.social_welfare, 6),
            "converged": task.converged,
            "iterations": task.iterations,
            "max_price_change": round(task.max_price_change, 6),
            "active_price_setting_count": task.active_price_setting_count,
            "active_price_taking_count": task.active_price_taking_count,
            "unilateral_deviation_passed": task.unilateral_deviation_passed,
        }
        for task in outcome.task_outcomes
    ]


def pcspe_convergence_rows(outcome: PCSPEOutcome) -> list[dict]:
    return [
        {
            "task_id": task.task_id,
            "converged": task.converged,
            "iterations": task.iterations,
            "max_price_change": round(task.max_price_change, 6),
            "crr_profit": round(task.crr_profit, 6),
            "social_welfare": round(task.social_welfare, 6),
            "offload_fraction": round(task.offload_fraction, 6),
            "local_fraction": round(task.local_fraction, 6),
            "active_price_setting_count": task.active_price_setting_count,
            "active_price_taking_count": task.active_price_taking_count,
            "unilateral_deviation_passed": task.unilateral_deviation_passed,
        }
        for task in outcome.task_outcomes
    ]


def pcspe_summary(outcome: PCSPEOutcome) -> list[dict]:
    task_count = len(outcome.task_outcomes)
    allocation_count = len(outcome.allocation_rows)
    return [
        {
            "mechanism": "PC-SPE",
            "task_count": task_count,
            "allocation_count": allocation_count,
            "equivalent_offloaded_tasks": round(sum(task.offload_fraction for task in outcome.task_outcomes), 6),
            "threshold_success_count": sum(1 for task in outcome.task_outcomes if task.offload_fraction >= 0.5),
            "total_crr_profit": round(sum(task.crr_profit for task in outcome.task_outcomes), 6),
            "total_social_welfare": round(sum(task.social_welfare for task in outcome.task_outcomes), 6),
            "total_payment": round(sum(task.total_payment for task in outcome.task_outcomes), 6),
            "total_crp_cost": round(sum(task.total_crp_cost for task in outcome.task_outcomes), 6),
            "converged_task_count": sum(1 for task in outcome.task_outcomes if task.converged),
            "deviation_passed_task_count": sum(1 for task in outcome.task_outcomes if task.unilateral_deviation_passed),
        }
    ]


def _solve_task_pcspe(task: Task, crps: list[PCSPECRP], config: dict) -> PCSPETaskOutcome:
    active_eps = float(config.get("pcspe_active_fraction_eps", 0.0001))
    workload = _normalized_workload(task, config)
    utility_constant = _utility_constant(task, config)
    alpha0 = _range_midpoint(config.get("pcspe_cost_alpha0_range", [1.0, 2.0]))
    f0 = max(1e-9, task.clock_frequency)
    prices = {crp.crp_id: crp.current_price for crp in crps if crp.crp_type == "price_setting"}
    price_upper = float(config.get("pcspe_price_upper", 50.0))
    price_step = float(config.get("pcspe_price_step", 0.1))
    max_iter = int(config.get("pcspe_max_outer_iter", 100))
    epsilon = float(config.get("pcspe_outer_epsilon", 0.01))

    response = _crr_response(crps, prices, workload, utility_constant, alpha0, f0, config)
    max_change = 0.0
    converged = True
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        base_fractions = response
        new_prices = dict(prices)
        for crp in crps:
            if crp.crp_type != "price_setting":
                continue
            current_price = prices[crp.crp_id]
            delta = max(1e-4, abs(current_price) * 1e-3, price_step * 0.5)
            lower_price = max(crp.alpha, current_price - delta)
            upper_price = min(price_upper, max(crp.alpha, current_price + delta))
            if math.isclose(lower_price, upper_price, abs_tol=1e-12):
                gradient = 0.0
            else:
                plus_prices = dict(prices)
                minus_prices = dict(prices)
                plus_prices[crp.crp_id] = upper_price
                minus_prices[crp.crp_id] = lower_price
                plus_response = _crr_response(crps, plus_prices, workload, utility_constant, alpha0, f0, config)
                minus_response = _crr_response(crps, minus_prices, workload, utility_constant, alpha0, f0, config)
                plus_profit = _price_setting_profit(crp, upper_price, plus_response.get(crp.crp_id, 0.0), workload)
                minus_profit = _price_setting_profit(crp, lower_price, minus_response.get(crp.crp_id, 0.0), workload)
                gradient = (plus_profit - minus_profit) / max(upper_price - lower_price, 1e-12)
            updated_price = min(price_upper, max(crp.alpha, current_price + price_step * gradient))
            new_prices[crp.crp_id] = updated_price
        max_change = max((abs(new_prices[key] - prices[key]) for key in prices), default=0.0)
        prices = new_prices
        response = _crr_response(crps, prices, workload, utility_constant, alpha0, f0, config)
        if max_change < epsilon:
            break
        if all(abs(response.get(crp.crp_id, 0.0) - base_fractions.get(crp.crp_id, 0.0)) < active_eps for crp in crps):
            # Continue until price convergence, but keep the no-op response visible
            # for audit if prices stall at a boundary.
            pass
    else:
        converged = False

    allocations, task_metrics = _allocations_from_response(task, crps, prices, response, workload, utility_constant, alpha0, f0, active_eps)
    deviation_passed = _unilateral_deviation_passed(crps, prices, workload, utility_constant, alpha0, f0, config, active_eps)
    active_s = sum(1 for allocation in allocations if allocation.crp_type == "price_setting" and allocation.fraction > active_eps)
    active_t = sum(1 for allocation in allocations if allocation.crp_type == "price_taking" and allocation.fraction > active_eps)
    return PCSPETaskOutcome(
        task_id=task.task_id,
        workload=workload,
        local_fraction=task_metrics["local_fraction"],
        offload_fraction=task_metrics["offload_fraction"],
        crr_profit=task_metrics["crr_profit"],
        utility=task_metrics["utility"],
        local_cost=task_metrics["local_cost"],
        total_payment=task_metrics["total_payment"],
        total_crp_cost=task_metrics["total_crp_cost"],
        social_welfare=task_metrics["social_welfare"],
        converged=converged,
        iterations=iterations,
        allocations=allocations,
        max_price_change=max_change,
        active_price_setting_count=active_s,
        active_price_taking_count=active_t,
        unilateral_deviation_passed=deviation_passed,
    )


def _crr_response(
    crps: list[PCSPECRP],
    prices: dict[str, float],
    workload: float,
    utility_constant: float,
    alpha0: float,
    f0: float,
    config: dict,
) -> dict[str, float]:
    if not crps:
        return {}
    max_iter = int(config.get("pcspe_max_inner_iter", 150))
    epsilon = float(config.get("pcspe_inner_epsilon", 0.000001))
    coefficients = {
        crp.crp_id: prices.get(crp.crp_id, crp.alpha) if crp.crp_type == "price_setting" else 2.0 * crp.alpha
        for crp in crps
    }
    max_coeff = max(coefficients.values(), default=1.0)
    step = 1.0 / max(1.0, 2.0 * (alpha0 + max_coeff) * workload * workload + 1.0)
    fractions = [1.0 / (len(crps) + 1.0) for _ in crps]
    for _ in range(max_iter):
        total_offload = sum(fractions)
        local_fraction = max(0.0, 1.0 - total_offload)
        gradients = []
        for index, crp in enumerate(crps):
            delay_gain = utility_constant * workload * (1.0 / f0 - 1.0 / crp.computing_power) / (len(crps) + 1.0)
            local_saving = 2.0 * alpha0 * workload * workload * local_fraction
            payment_penalty = 2.0 * coefficients[crp.crp_id] * workload * workload * fractions[index]
            gradients.append(delay_gain + local_saving - payment_penalty)
        updated = [value + step * gradient for value, gradient in zip(fractions, gradients)]
        updated = _project_capped_simplex(updated)
        diff = max((abs(left - right) for left, right in zip(updated, fractions)), default=0.0)
        fractions = updated
        if diff < epsilon:
            break
    return {crp.crp_id: max(0.0, fraction) for crp, fraction in zip(crps, fractions)}


def _allocations_from_response(
    task: Task,
    crps: list[PCSPECRP],
    prices: dict[str, float],
    response: dict[str, float],
    workload: float,
    utility_constant: float,
    alpha0: float,
    f0: float,
    active_eps: float,
) -> tuple[list[PCSPEAllocation], dict[str, float]]:
    total_offload = min(1.0, sum(max(0.0, response.get(crp.crp_id, 0.0)) for crp in crps))
    local_fraction = max(0.0, 1.0 - total_offload)
    utility = _crr_utility(crps, response, workload, utility_constant, f0, local_fraction)
    local_cost = alpha0 * (local_fraction * workload) ** 2
    allocations: list[PCSPEAllocation] = []
    total_payment = 0.0
    total_cost = 0.0
    for crp in crps:
        fraction = max(0.0, response.get(crp.crp_id, 0.0))
        if fraction <= active_eps:
            continue
        if crp.crp_type == "price_setting":
            price = prices.get(crp.crp_id, crp.current_price)
            payment = price * (fraction * workload) ** 2
        else:
            price = 2.0 * crp.alpha * workload * fraction
            payment = price * fraction * workload
        cost = crp.alpha * (fraction * workload) ** 2
        allocations.append(
            PCSPEAllocation(
                task_id=task.task_id,
                crp_id=crp.crp_id,
                node_id=crp.node_id,
                crp_type=crp.crp_type,
                fraction=fraction,
                price=price,
                payment=payment,
                cost=cost,
                crp_profit=payment - cost,
            )
        )
        total_payment += payment
        total_cost += cost
    crr_profit = utility - local_cost - total_payment
    social_welfare = utility - local_cost - total_cost
    return allocations, {
        "local_fraction": local_fraction,
        "offload_fraction": total_offload,
        "crr_profit": crr_profit,
        "utility": utility,
        "local_cost": local_cost,
        "total_payment": total_payment,
        "total_crp_cost": total_cost,
        "social_welfare": social_welfare,
    }


def _unilateral_deviation_passed(
    crps: list[PCSPECRP],
    prices: dict[str, float],
    workload: float,
    utility_constant: float,
    alpha0: float,
    f0: float,
    config: dict,
    active_eps: float,
) -> bool:
    multipliers = [float(value) for value in config.get("pcspe_deviation_multipliers", [0.7, 0.9, 1.0, 1.1, 1.3])]
    tolerance = float(config.get("pcspe_outer_epsilon", 0.01))
    baseline = _crr_response(crps, prices, workload, utility_constant, alpha0, f0, config)
    price_upper = float(config.get("pcspe_price_upper", 50.0))
    for crp in crps:
        if crp.crp_type != "price_setting":
            continue
        current_fraction = baseline.get(crp.crp_id, 0.0)
        if current_fraction <= active_eps:
            continue
        current_price = prices.get(crp.crp_id, crp.current_price)
        current_profit = _price_setting_profit(crp, current_price, current_fraction, workload)
        best_profit = current_profit
        for multiplier in multipliers:
            deviated_prices = dict(prices)
            deviated_prices[crp.crp_id] = min(price_upper, max(crp.alpha, current_price * multiplier))
            response = _crr_response(crps, deviated_prices, workload, utility_constant, alpha0, f0, config)
            best_profit = max(
                best_profit,
                _price_setting_profit(crp, deviated_prices[crp.crp_id], response.get(crp.crp_id, 0.0), workload),
            )
        if current_profit + tolerance < best_profit:
            return False
    return True


def _price_setting_profit(crp: PCSPECRP, price: float, fraction: float, workload: float) -> float:
    return (price - crp.alpha) * (fraction * workload) ** 2


def _crr_utility(
    crps: list[PCSPECRP],
    response: dict[str, float],
    workload: float,
    utility_constant: float,
    f0: float,
    local_fraction: float,
) -> float:
    if not crps:
        return 0.0
    average_delay = local_fraction * workload / f0
    for crp in crps:
        average_delay += max(0.0, response.get(crp.crp_id, 0.0)) * workload / crp.computing_power
    average_delay /= len(crps) + 1.0
    saved_time = max(0.0, workload / f0 - average_delay)
    return utility_constant * saved_time


def _project_capped_simplex(values: list[float]) -> list[float]:
    nonnegative = [max(0.0, value) for value in values]
    if sum(nonnegative) <= 1.0:
        return nonnegative
    ordered = sorted(nonnegative, reverse=True)
    cumulative = 0.0
    theta = 0.0
    for index, value in enumerate(ordered, start=1):
        cumulative += value
        candidate = (cumulative - 1.0) / index
        if index == len(ordered) or ordered[index] <= candidate:
            theta = candidate
            break
    return [max(0.0, value - theta) for value in nonnegative]


def _normalized_workload(task: Task, config: dict) -> float:
    if not config.get("pcspe_workload_normalization", True):
        return max(1e-9, task.time_cost)
    source_low, source_high = _range_tuple(config.get("time_cost_range", [200.0, 2000.0]))
    target_low = float(config.get("pcspe_workload_min", 0.1))
    target_high = float(config.get("pcspe_workload_max", 1.0))
    if source_high <= source_low:
        return target_low
    ratio = (task.time_cost - source_low) / (source_high - source_low)
    ratio = min(1.0, max(0.0, ratio))
    return target_low + ratio * (target_high - target_low)


def _utility_constant(task: Task, config: dict) -> float:
    reward_low, reward_high = _range_tuple(config.get("reward_range", [2.0, 20.0]))
    if reward_high <= reward_low:
        normalized_reward = 1.0
    else:
        normalized_reward = (task.reward - reward_low) / (reward_high - reward_low)
        normalized_reward = min(1.0, max(0.1, normalized_reward))
    return float(config.get("pcspe_revenue_constant", 12.0)) * normalized_reward


def _range_tuple(values: object) -> tuple[float, float]:
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        return float(values[0]), float(values[1])
    value = float(values) if values is not None else 0.0
    return value, value


def _range_midpoint(values: object) -> float:
    low, high = _range_tuple(values)
    return (low + high) / 2.0


def _numeric_suffix(identifier: str) -> int:
    try:
        return int(identifier.split("_")[-1])
    except (TypeError, ValueError):
        return 0
