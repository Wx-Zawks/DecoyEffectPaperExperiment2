from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from repro.dim import classify_tasks
from repro.formulas import local_execution_cost
from repro.models import FogNode, Task, mean


@dataclass
class TOCATask:
    task_id: str
    arrival_time: int
    deadline: int
    value: float
    bid: float
    cpu_cycles: float
    cpu_demand: int
    subchannel_demand: int
    input_size: float
    output_size: float
    is_goal_task: bool
    task_class: str
    position: tuple[float, float]
    reward: float
    time_cost: float
    clock_frequency: float


@dataclass
class TOCABaseStation:
    bs_id: str
    node_id: str
    cpu_capacity: int
    subchannel_capacity: int
    coverage_radius: float
    position: tuple[float, float]
    cpu_unit_cost: float
    subchannel_unit_cost: float
    used_cpu: list[int] = field(default_factory=list)
    used_subchannels: list[int] = field(default_factory=list)

    def clone_empty(self) -> "TOCABaseStation":
        return TOCABaseStation(
            bs_id=self.bs_id,
            node_id=self.node_id,
            cpu_capacity=self.cpu_capacity,
            subchannel_capacity=self.subchannel_capacity,
            coverage_radius=self.coverage_radius,
            position=self.position,
            cpu_unit_cost=self.cpu_unit_cost,
            subchannel_unit_cost=self.subchannel_unit_cost,
            used_cpu=[0 for _ in self.used_cpu],
            used_subchannels=[0 for _ in self.used_subchannels],
        )


@dataclass
class TOCAScheme:
    task_id: str
    bs_id: str
    time_slots: list[int]
    cpu_demand: int
    subchannel_demand: int
    operational_cost: float
    dynamic_price: float
    total_payment: float
    net_utility: float


@dataclass
class TOCATaskResult:
    task_id: str
    arrival_time: int
    deadline: int
    task_class: str
    bid: float
    accepted: bool
    selected_bs: str | None
    selected_time_slots: list[int]
    payment: float
    operational_cost: float
    dynamic_price: float
    smd_utility: float
    social_welfare_contribution: float
    cpu_demand: int
    subchannel_demand: int
    candidate_count: int
    rejection_reason: str


@dataclass
class TOCAOutcome:
    tasks: list[TOCATask]
    base_stations: list[TOCABaseStation]
    task_results: list[TOCATaskResult]
    accepted_schemes: list[TOCAScheme]

    @property
    def accepted_results(self) -> list[TOCATaskResult]:
        return [result for result in self.task_results if result.accepted]

    @property
    def rejected_results(self) -> list[TOCATaskResult]:
        return [result for result in self.task_results if not result.accepted]


def build_toca_environment(
    tasks: list[Task],
    nodes: list[FogNode],
    rng: Random,
    config: dict,
) -> tuple[list[TOCATask], list[TOCABaseStation]]:
    time_slots = int(config.get("toca_time_slots", 20))
    deadline_window = int(config.get("toca_deadline_window", 5))
    region_size = float(config.get("traim_region_size", 1000.0))
    cpu_capacity_range = tuple(config.get("toca_cpu_capacity_range", config.get("traim_bs_cpu_range", [5, 9])))
    subchannel_capacity_range = tuple(
        config.get("toca_subchannel_capacity_range", config.get("traim_bs_subchannel_range", [4, 8]))
    )
    bs_radius_range = tuple(config.get("toca_bs_radius_range", config.get("traim_bs_radius_range", [140.0, 300.0])))
    cpu_unit_cost = float(config.get("toca_cpu_unit_cost", 1.0))
    subchannel_unit_cost = float(config.get("toca_subchannel_unit_cost", 0.5))

    task_classes = _task_class_map(tasks)
    toca_tasks: list[TOCATask] = []
    for index, task in enumerate(tasks):
        arrival = _arrival_time(index, len(tasks), time_slots, deadline_window)
        deadline = min(time_slots - 1, arrival + deadline_window)
        cpu_demand = _scale_to_int(task.time_cost, _min_time(tasks), _max_time(tasks), 1, 5)
        subchannel_demand = _scale_to_int(task.reward, _min_reward(tasks), _max_reward(tasks), 1, 3)
        local_value = local_execution_cost(task)
        bid = max(0.0, (local_value + task.reward) * float(config.get("toca_value_scale", 1.0)))
        task_class = task_classes.get(task.task_id, "RAW")
        toca_tasks.append(
            TOCATask(
                task_id=task.task_id,
                arrival_time=arrival,
                deadline=deadline,
                value=bid,
                bid=bid,
                cpu_cycles=task.time_cost,
                cpu_demand=cpu_demand,
                subchannel_demand=subchannel_demand,
                input_size=round(max(1.0, task.time_cost / 100.0), 6),
                output_size=round(max(0.5, task.reward / 2.0), 6),
                is_goal_task=task_class == "A",
                task_class=task_class,
                position=(rng.uniform(0.0, region_size), rng.uniform(0.0, region_size)),
                reward=task.reward,
                time_cost=task.time_cost,
                clock_frequency=task.clock_frequency,
            )
        )

    base_stations: list[TOCABaseStation] = []
    for index, node in enumerate(nodes):
        base_stations.append(
            TOCABaseStation(
                bs_id=f"bs_{index + 1}",
                node_id=node.node_id,
                cpu_capacity=rng.randint(int(cpu_capacity_range[0]), int(cpu_capacity_range[1])),
                subchannel_capacity=rng.randint(int(subchannel_capacity_range[0]), int(subchannel_capacity_range[1])),
                coverage_radius=rng.uniform(float(bs_radius_range[0]), float(bs_radius_range[1])),
                position=(rng.uniform(0.0, region_size), rng.uniform(0.0, region_size)),
                cpu_unit_cost=cpu_unit_cost,
                subchannel_unit_cost=subchannel_unit_cost,
                used_cpu=[0 for _ in range(time_slots)],
                used_subchannels=[0 for _ in range(time_slots)],
            )
        )
    return toca_tasks, base_stations


def run_toca(
    tasks: list[TOCATask],
    base_stations: list[TOCABaseStation],
    config: dict,
    bid_overrides: dict[str, float] | None = None,
) -> TOCAOutcome:
    bid_overrides = bid_overrides or {}
    task_results: list[TOCATaskResult] = []
    accepted_schemes: list[TOCAScheme] = []
    task_order = sorted(tasks, key=lambda task: (task.arrival_time, _numeric_suffix(task.task_id)))

    for task in task_order:
        effective_task = _with_reported_bid(task, bid_overrides.get(task.task_id))
        candidates = generate_candidate_schemes(effective_task, base_stations, config)
        best_scheme = max(candidates, key=lambda scheme: (scheme.net_utility, -scheme.total_payment, scheme.bs_id), default=None)
        if best_scheme is not None and best_scheme.net_utility > 0.0:
            selected_bs = _bs_by_id(base_stations)[best_scheme.bs_id]
            for slot in best_scheme.time_slots:
                selected_bs.used_cpu[slot] += best_scheme.cpu_demand
                selected_bs.used_subchannels[slot] += best_scheme.subchannel_demand
            accepted_schemes.append(best_scheme)
            result = _accepted_result(effective_task, best_scheme, len(candidates))
        else:
            result = _rejected_result(effective_task, len(candidates), "non_positive_utility" if candidates else "no_feasible_scheme")
        task_results.append(result)

    return TOCAOutcome(tasks=tasks, base_stations=base_stations, task_results=task_results, accepted_schemes=accepted_schemes)


def generate_candidate_schemes(task: TOCATask, base_stations: list[TOCABaseStation], config: dict) -> list[TOCAScheme]:
    schemes: list[TOCAScheme] = []
    for base_station in base_stations:
        if _distance(task.position, base_station.position) > base_station.coverage_radius:
            continue
        slot_prices = []
        for slot in range(task.arrival_time, task.deadline + 1):
            if slot >= len(base_station.used_cpu):
                continue
            if base_station.used_cpu[slot] + task.cpu_demand > base_station.cpu_capacity:
                continue
            if base_station.used_subchannels[slot] + task.subchannel_demand > base_station.subchannel_capacity:
                continue
            slot_prices.append((slot, _dynamic_resource_price(task, base_station, slot, config)))
        required_slots = _required_slot_count(task)
        if len(slot_prices) < required_slots:
            continue
        selected_slots = sorted(slot_prices, key=lambda item: (item[1], item[0]))[:required_slots]
        time_slots = sorted(slot for slot, _ in selected_slots)
        dynamic_price = sum(price for _, price in selected_slots)
        operational_cost = _operational_cost(task, base_station, required_slots, config)
        total_payment = operational_cost + dynamic_price
        schemes.append(
            TOCAScheme(
                task_id=task.task_id,
                bs_id=base_station.bs_id,
                time_slots=time_slots,
                cpu_demand=task.cpu_demand,
                subchannel_demand=task.subchannel_demand,
                operational_cost=round(operational_cost, 6),
                dynamic_price=round(dynamic_price, 6),
                total_payment=round(total_payment, 6),
                net_utility=round(task.bid - total_payment, 6),
            )
        )
    return schemes


def toca_summary(outcome: TOCAOutcome) -> dict:
    accepted = outcome.accepted_results
    rejected = outcome.rejected_results
    total_goal_count = sum(1 for task in outcome.tasks if task.is_goal_task)
    accepted_goal_count = sum(1 for task in outcome.tasks if task.is_goal_task and _is_accepted(task.task_id, accepted))
    total_payment = sum(result.payment for result in accepted)
    total_operational_cost = sum(result.operational_cost for result in accepted)
    total_utility = sum(result.smd_utility for result in accepted)
    social_welfare = sum(result.social_welfare_contribution for result in accepted)
    return {
        "accepted_task_count": len(accepted),
        "rejected_task_count": len(rejected),
        "accepted_goal_task_count": accepted_goal_count,
        "total_goal_task_count": total_goal_count,
        "total_payment": round(total_payment, 6),
        "total_operational_cost": round(total_operational_cost, 6),
        "user_total_utility": round(total_utility, 6),
        "social_welfare": round(social_welfare, 6),
        "resource_utilization": round(resource_utilization(outcome.base_stations), 6),
        "individual_rationality": min((result.smd_utility for result in accepted), default=0.0) >= -1e-9,
    }


def task_result_rows(outcome: TOCAOutcome) -> list[dict]:
    return [
        {
            "task_id": result.task_id,
            "arrival_time": result.arrival_time,
            "deadline": result.deadline,
            "task_class": result.task_class,
            "bid": round(result.bid, 6),
            "accepted": result.accepted,
            "selected_bs": result.selected_bs,
            "selected_time_slots": ";".join(str(slot) for slot in result.selected_time_slots),
            "payment": round(result.payment, 6),
            "operational_cost": round(result.operational_cost, 6),
            "dynamic_price": round(result.dynamic_price, 6),
            "smd_utility": round(result.smd_utility, 6),
            "social_welfare_contribution": round(result.social_welfare_contribution, 6),
            "cpu_demand": result.cpu_demand,
            "subchannel_demand": result.subchannel_demand,
            "candidate_count": result.candidate_count,
            "rejection_reason": result.rejection_reason,
        }
        for result in sorted(outcome.task_results, key=lambda item: _numeric_suffix(item.task_id))
    ]


def round_rows(outcome: TOCAOutcome) -> list[dict]:
    return [
        {
            "round_index": index,
            "task_id": result.task_id,
            "arrival_time": result.arrival_time,
            "deadline": result.deadline,
            "candidate_count": result.candidate_count,
            "decision": "ACCEPTED" if result.accepted else "REJECTED",
            "selected_bs": result.selected_bs,
            "selected_time_slots": ";".join(str(slot) for slot in result.selected_time_slots),
            "net_utility": round(result.smd_utility, 6),
            "rejection_reason": result.rejection_reason,
        }
        for index, result in enumerate(sorted(outcome.task_results, key=lambda item: (item.arrival_time, _numeric_suffix(item.task_id))))
    ]


def resource_usage_rows(outcome: TOCAOutcome) -> list[dict]:
    rows: list[dict] = []
    for base_station in outcome.base_stations:
        for slot, (used_cpu, used_channels) in enumerate(zip(base_station.used_cpu, base_station.used_subchannels)):
            rows.append(
                {
                    "bs_id": base_station.bs_id,
                    "node_id": base_station.node_id,
                    "time_slot": slot,
                    "used_cpu": used_cpu,
                    "cpu_capacity": base_station.cpu_capacity,
                    "cpu_utilization": round(used_cpu / base_station.cpu_capacity if base_station.cpu_capacity else 0.0, 6),
                    "used_subchannels": used_channels,
                    "subchannel_capacity": base_station.subchannel_capacity,
                    "subchannel_utilization": round(
                        used_channels / base_station.subchannel_capacity if base_station.subchannel_capacity else 0.0,
                        6,
                    ),
                }
            )
    return rows


def clone_base_stations_empty(base_stations: list[TOCABaseStation]) -> list[TOCABaseStation]:
    return [base_station.clone_empty() for base_station in base_stations]


def resource_utilization(base_stations: list[TOCABaseStation]) -> float:
    cpu_ratios = []
    channel_ratios = []
    for base_station in base_stations:
        cpu_ratios.extend(used / base_station.cpu_capacity for used in base_station.used_cpu if base_station.cpu_capacity)
        channel_ratios.extend(
            used / base_station.subchannel_capacity for used in base_station.used_subchannels if base_station.subchannel_capacity
        )
    return mean(cpu_ratios + channel_ratios)


def _task_class_map(tasks: list[Task]) -> dict[str, str]:
    categories, _, _ = classify_tasks([task.clone() for task in tasks])
    mapping: dict[str, str] = {}
    for task_class, class_tasks in categories.items():
        for task in class_tasks:
            mapping[task.task_id] = task_class
    return mapping


def _arrival_time(index: int, task_count: int, time_slots: int, deadline_window: int) -> int:
    if time_slots <= 1 or task_count <= 1:
        return 0
    latest_arrival = max(0, time_slots - max(1, deadline_window) - 1)
    return min(latest_arrival, int(round(index * latest_arrival / max(1, task_count - 1))))


def _required_slot_count(task: TOCATask) -> int:
    return max(1, math.ceil(task.cpu_demand / 2.0))


def _operational_cost(task: TOCATask, base_station: TOCABaseStation, slot_count: int, config: dict) -> float:
    base_cost = (
        task.cpu_demand * base_station.cpu_unit_cost
        + task.subchannel_demand * base_station.subchannel_unit_cost
    ) * slot_count
    return base_cost * float(config.get("toca_cost_scale", 1.0))


def _dynamic_resource_price(task: TOCATask, base_station: TOCABaseStation, slot: int, config: dict) -> float:
    growth = float(config.get("toca_price_growth_factor", 3.0))
    cpu_usage = base_station.used_cpu[slot] / max(base_station.cpu_capacity, 1)
    channel_usage = base_station.used_subchannels[slot] / max(base_station.subchannel_capacity, 1)
    cpu_price = base_station.cpu_unit_cost * (1.0 + growth * cpu_usage) * task.cpu_demand
    channel_price = base_station.subchannel_unit_cost * (1.0 + growth * channel_usage) * task.subchannel_demand
    return (cpu_price + channel_price) * float(config.get("toca_cost_scale", 1.0))


def _accepted_result(task: TOCATask, scheme: TOCAScheme, candidate_count: int) -> TOCATaskResult:
    return TOCATaskResult(
        task_id=task.task_id,
        arrival_time=task.arrival_time,
        deadline=task.deadline,
        task_class=task.task_class,
        bid=task.bid,
        accepted=True,
        selected_bs=scheme.bs_id,
        selected_time_slots=scheme.time_slots,
        payment=scheme.total_payment,
        operational_cost=scheme.operational_cost,
        dynamic_price=scheme.dynamic_price,
        smd_utility=task.bid - scheme.total_payment,
        social_welfare_contribution=task.value - scheme.operational_cost,
        cpu_demand=task.cpu_demand,
        subchannel_demand=task.subchannel_demand,
        candidate_count=candidate_count,
        rejection_reason="",
    )


def _rejected_result(task: TOCATask, candidate_count: int, reason: str) -> TOCATaskResult:
    return TOCATaskResult(
        task_id=task.task_id,
        arrival_time=task.arrival_time,
        deadline=task.deadline,
        task_class=task.task_class,
        bid=task.bid,
        accepted=False,
        selected_bs=None,
        selected_time_slots=[],
        payment=0.0,
        operational_cost=0.0,
        dynamic_price=0.0,
        smd_utility=0.0,
        social_welfare_contribution=0.0,
        cpu_demand=task.cpu_demand,
        subchannel_demand=task.subchannel_demand,
        candidate_count=candidate_count,
        rejection_reason=reason,
    )


def _with_reported_bid(task: TOCATask, reported_bid: float | None) -> TOCATask:
    if reported_bid is None:
        return task
    return TOCATask(
        task_id=task.task_id,
        arrival_time=task.arrival_time,
        deadline=task.deadline,
        value=task.value,
        bid=max(0.0, reported_bid),
        cpu_cycles=task.cpu_cycles,
        cpu_demand=task.cpu_demand,
        subchannel_demand=task.subchannel_demand,
        input_size=task.input_size,
        output_size=task.output_size,
        is_goal_task=task.is_goal_task,
        task_class=task.task_class,
        position=task.position,
        reward=task.reward,
        time_cost=task.time_cost,
        clock_frequency=task.clock_frequency,
    )


def _is_accepted(task_id: str, accepted: list[TOCATaskResult]) -> bool:
    return any(result.task_id == task_id for result in accepted)


def _bs_by_id(base_stations: list[TOCABaseStation]) -> dict[str, TOCABaseStation]:
    return {base_station.bs_id: base_station for base_station in base_stations}


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _scale_to_int(value: float, source_low: float, source_high: float, target_low: int, target_high: int) -> int:
    if source_high <= source_low:
        return target_low
    ratio = (value - source_low) / (source_high - source_low)
    scaled = target_low + ratio * (target_high - target_low)
    return min(target_high, max(target_low, int(round(scaled))))


def _min_time(tasks: list[Task]) -> float:
    return min((task.time_cost for task in tasks), default=0.0)


def _max_time(tasks: list[Task]) -> float:
    return max((task.time_cost for task in tasks), default=1.0)


def _min_reward(tasks: list[Task]) -> float:
    return min((task.reward for task in tasks), default=0.0)


def _max_reward(tasks: list[Task]) -> float:
    return max((task.reward for task in tasks), default=1.0)


def _numeric_suffix(identifier: str) -> int:
    return int(identifier.split("_")[-1])
