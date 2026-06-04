from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random

from repro.models import FogNode, Task


@dataclass
class MobileDevice:
    md_id: str
    task_id: str
    x: float
    y: float
    cpu_cores: int
    data_rate: int
    transmit_power_mw: float
    noise_mw: float
    local_value: float
    task_reward: float
    task_time_cost: float


@dataclass
class BaseStation:
    bs_id: str
    node_id: str
    x: float
    y: float
    cpu_cores: int
    subchannels: int
    bandwidth_mbps: float
    coverage_radius: float
    bid_cost: float
    true_cost: float
    task_cost_ratio: float


@dataclass
class TRAIMCandidate:
    bs_id: str
    md_ids: list[str]
    phi_by_md: dict[str, int]
    resource_value: float
    coverage_ids: set[str]

    @property
    def cpr_denominator(self) -> float:
        return self.resource_value


@dataclass
class TRAIMOutcome:
    winners: list[str]
    bidder_ids: list[str]
    candidates: dict[str, TRAIMCandidate]
    payments: dict[str, float]
    unserved_md_ids: list[str]


def build_traim_environment(
    tasks: list[Task],
    nodes: list[FogNode],
    rng: Random,
    config: dict,
) -> tuple[list[MobileDevice], list[BaseStation]]:
    region_size = float(config.get("traim_region_size", 1000.0))
    md_cpu_range = tuple(config.get("traim_md_cpu_range", [1, 5]))
    md_rate_range = tuple(config.get("traim_md_rate_range", [1, 10]))
    bs_cpu_range = tuple(config.get("traim_bs_cpu_range", [8, 12]))
    bs_channel_range = tuple(config.get("traim_bs_subchannel_range", [6, 9]))
    radius_range = tuple(config.get("traim_bs_radius_range", [70.0, 180.0]))
    noise_dbm_range = tuple(config.get("traim_noise_dbm_range", [-90.0, -70.0]))
    power_range = tuple(config.get("traim_transmit_power_mw_range", [200.0, 300.0]))
    bandwidth = float(config.get("traim_bandwidth_mbps", 1.0))
    task_cost_ratio_range = tuple(config.get("traim_task_cost_ratio_range", [0.55, 0.85]))
    time_range = tuple(config.get("time_cost_range", [200.0, 2000.0]))
    reward_range = tuple(config.get("reward_range", [2.0, 20.0]))

    mobile_devices: list[MobileDevice] = []
    for index, task in enumerate(tasks):
        md_cpu = _scale_to_int(task.time_cost, float(time_range[0]), float(time_range[1]), int(md_cpu_range[0]), int(md_cpu_range[1]))
        md_rate = _scale_to_int(task.reward, float(reward_range[0]), float(reward_range[1]), int(md_rate_range[0]), int(md_rate_range[1]))
        noise_dbm = rng.uniform(float(noise_dbm_range[0]), float(noise_dbm_range[1]))
        mobile_devices.append(
            MobileDevice(
                md_id=f"md_{index + 1}",
                task_id=task.task_id,
                x=rng.uniform(0.0, region_size),
                y=rng.uniform(0.0, region_size),
                cpu_cores=md_cpu,
                data_rate=md_rate,
                transmit_power_mw=rng.uniform(float(power_range[0]), float(power_range[1])),
                noise_mw=10 ** (noise_dbm / 10.0),
                local_value=task.time_cost / max(task.clock_frequency, 1e-9),
                task_reward=task.reward,
                task_time_cost=task.time_cost,
            )
        )

    base_stations: list[BaseStation] = []
    for index, node in enumerate(nodes):
        cpu_cores = rng.randint(int(bs_cpu_range[0]), int(bs_cpu_range[1]))
        subchannels = rng.randint(int(bs_channel_range[0]), int(bs_channel_range[1]))
        cpu_unit_cost = rng.uniform(0.05, 1.0)
        channel_unit_cost = rng.uniform(0.05, 1.0)
        true_cost = cpu_unit_cost * cpu_cores + channel_unit_cost * subchannels
        bid_cost = true_cost
        base_stations.append(
            BaseStation(
                bs_id=f"bs_{index + 1}",
                node_id=node.node_id,
                x=rng.uniform(0.0, region_size),
                y=rng.uniform(0.0, region_size),
                cpu_cores=cpu_cores,
                subchannels=subchannels,
                bandwidth_mbps=bandwidth,
                coverage_radius=rng.uniform(float(radius_range[0]), float(radius_range[1])),
                bid_cost=bid_cost,
                true_cost=true_cost,
                task_cost_ratio=rng.uniform(float(task_cost_ratio_range[0]), float(task_cost_ratio_range[1])),
            )
        )
    return mobile_devices, base_stations


def run_traim(mobile_devices: list[MobileDevice], base_stations: list[BaseStation]) -> TRAIMOutcome:
    if not mobile_devices or not base_stations:
        return TRAIMOutcome([], [], {}, {}, [device.md_id for device in mobile_devices])

    f_max = max(base_station.cpu_cores for base_station in base_stations)
    theta_max = max(base_station.subchannels for base_station in base_stations)
    remaining_bs_ids = {base_station.bs_id for base_station in base_stations}
    unserved_md_ids = {device.md_id for device in mobile_devices}
    winners: list[str] = []
    winning_candidates: dict[str, TRAIMCandidate] = {}
    bs_by_id = {base_station.bs_id: base_station for base_station in base_stations}
    md_by_id = {device.md_id: device for device in mobile_devices}
    candidate_cache: dict[tuple[str, frozenset[str]], TRAIMCandidate] = {}
    all_md_ids = frozenset(device.md_id for device in mobile_devices)
    bidder_ids = [
        base_station.bs_id
        for base_station in base_stations
        if _cached_assignment(mobile_devices, bs_by_id[base_station.bs_id], all_md_ids, f_max, theta_max, candidate_cache).cpr_denominator > 0.0
    ]

    while unserved_md_ids and remaining_bs_ids:
        round_candidates: dict[str, TRAIMCandidate] = {}
        for bs_id in list(remaining_bs_ids):
            candidate = _cached_assignment(
                mobile_devices,
                bs_by_id[bs_id],
                frozenset(unserved_md_ids),
                f_max,
                theta_max,
                candidate_cache,
            )
            if candidate.cpr_denominator <= 0.0:
                remaining_bs_ids.remove(bs_id)
                continue
            round_candidates[bs_id] = candidate
        if not round_candidates:
            break
        selected_bs_id = min(
            round_candidates,
            key=lambda item: (
                candidate_cost(bs_by_id[item], round_candidates[item], md_by_id) / round_candidates[item].cpr_denominator,
                candidate_cost(bs_by_id[item], round_candidates[item], md_by_id),
                item,
            ),
        )
        selected_candidate = round_candidates[selected_bs_id]
        winners.append(selected_bs_id)
        winning_candidates[selected_bs_id] = selected_candidate
        remaining_bs_ids.remove(selected_bs_id)
        unserved_md_ids.difference_update(selected_candidate.md_ids)

    payments = critical_value_payments(mobile_devices, base_stations, winners, winning_candidates, f_max, theta_max, candidate_cache)
    return TRAIMOutcome(winners, bidder_ids, winning_candidates, payments, sorted(unserved_md_ids, key=_numeric_suffix))


def assign_mobile_devices(
    mobile_devices: list[MobileDevice],
    base_station: BaseStation,
    available_md_ids: set[str],
    f_max: int,
    theta_max: int,
) -> TRAIMCandidate:
    candidate_devices = [
        device
        for device in sorted(mobile_devices, key=lambda item: _numeric_suffix(item.md_id))
        if device.md_id in available_md_ids and _distance(device, base_station) <= base_station.coverage_radius
    ]
    coverage_ids = {device.md_id for device in candidate_devices}
    feasible: list[tuple[MobileDevice, int, float]] = []
    for device in candidate_devices:
        required_subchannels = subchannels_required(device, base_station)
        if device.cpu_cores <= base_station.cpu_cores and required_subchannels <= base_station.subchannels:
            value = device.cpu_cores / f_max + required_subchannels / theta_max
            feasible.append((device, required_subchannels, value))
    if not feasible:
        return TRAIMCandidate(base_station.bs_id, [], {}, 0.0, coverage_ids)

    cpu_capacity = base_station.cpu_cores
    channel_capacity = base_station.subchannels
    dp = [
        [[0.0 for _ in range(channel_capacity + 1)] for _ in range(cpu_capacity + 1)]
        for _ in range(len(feasible) + 1)
    ]
    take = [
        [[False for _ in range(channel_capacity + 1)] for _ in range(cpu_capacity + 1)]
        for _ in range(len(feasible) + 1)
    ]

    for item_index, (device, required_subchannels, value) in enumerate(feasible, start=1):
        for cpu in range(cpu_capacity + 1):
            for channels in range(channel_capacity + 1):
                best = dp[item_index - 1][cpu][channels]
                if cpu >= device.cpu_cores and channels >= required_subchannels:
                    with_device = dp[item_index - 1][cpu - device.cpu_cores][channels - required_subchannels] + value
                    if with_device > best + 1e-12:
                        best = with_device
                        take[item_index][cpu][channels] = True
                dp[item_index][cpu][channels] = best

    selected_ids: list[str] = []
    phi_by_md: dict[str, int] = {}
    cpu = cpu_capacity
    channels = channel_capacity
    for item_index in range(len(feasible), 0, -1):
        if take[item_index][cpu][channels]:
            device, required_subchannels, _ = feasible[item_index - 1]
            selected_ids.append(device.md_id)
            phi_by_md[device.md_id] = required_subchannels
            cpu -= device.cpu_cores
            channels -= required_subchannels

    selected_ids.sort(key=_numeric_suffix)
    return TRAIMCandidate(base_station.bs_id, selected_ids, phi_by_md, dp[len(feasible)][cpu_capacity][channel_capacity], coverage_ids)


def critical_value_payments(
    mobile_devices: list[MobileDevice],
    base_stations: list[BaseStation],
    winners: list[str],
    winning_candidates: dict[str, TRAIMCandidate],
    f_max: int,
    theta_max: int,
    candidate_cache: dict[tuple[str, frozenset[str]], TRAIMCandidate] | None = None,
) -> dict[str, float]:
    bs_by_id = {base_station.bs_id: base_station for base_station in base_stations}
    md_by_id = {device.md_id: device for device in mobile_devices}
    cache = candidate_cache if candidate_cache is not None else {}
    payments: dict[str, float] = {}
    all_md_ids = {device.md_id for device in mobile_devices}
    for winner_id in winners:
        winner_candidate = winning_candidates[winner_id]
        target_ids = set(winner_candidate.md_ids)
        if not target_ids:
            payments[winner_id] = 0.0
            continue
        remaining_bs_ids = {base_station.bs_id for base_station in base_stations if base_station.bs_id != winner_id}
        locally_unserved = set(all_md_ids)
        locally_served: set[str] = set()
        critical_payment = 0.0
        while target_ids - locally_served and remaining_bs_ids:
            candidates: dict[str, TRAIMCandidate] = {}
            for bs_id in list(remaining_bs_ids):
                candidate = _cached_assignment(mobile_devices, bs_by_id[bs_id], frozenset(locally_unserved), f_max, theta_max, cache)
                if candidate.cpr_denominator <= 0.0:
                    remaining_bs_ids.remove(bs_id)
                    continue
                candidates[bs_id] = candidate
            if not candidates:
                break
            substitute_id = min(
                candidates,
                key=lambda item: (
                    candidate_cost(bs_by_id[item], candidates[item], md_by_id) / candidates[item].cpr_denominator,
                    candidate_cost(bs_by_id[item], candidates[item], md_by_id),
                    item,
                ),
            )
            substitute = candidates[substitute_id]
            substitute_cpr = candidate_cost(bs_by_id[substitute_id], substitute, md_by_id) / substitute.cpr_denominator
            remaining_bs_ids.remove(substitute_id)
            locally_unserved.difference_update(substitute.md_ids)
            locally_served.update(substitute.md_ids)
            if target_ids.issubset(locally_served):
                critical_payment = substitute_cpr * winner_candidate.cpr_denominator
                break
        winner_cost = candidate_cost(bs_by_id[winner_id], winner_candidate, md_by_id)
        payments[winner_id] = max(critical_payment, winner_cost)
    return payments


def subchannels_required(device: MobileDevice, base_station: BaseStation) -> int:
    distance = max(_distance(device, base_station), 1.0)
    gain = distance ** -4.0
    snr = device.transmit_power_mw * gain / max(device.noise_mw, 1e-18)
    rate_per_channel = base_station.bandwidth_mbps * math.log2(1.0 + snr)
    if rate_per_channel <= 1e-9:
        return base_station.subchannels + 1
    return max(1, math.ceil(device.data_rate / rate_per_channel))


def normalized_resource_for_md(device: MobileDevice, phi: int, f_max: int, theta_max: int) -> float:
    return device.cpu_cores / f_max + phi / theta_max


def candidate_cost(base_station: BaseStation, candidate: TRAIMCandidate, md_by_id: dict[str, MobileDevice]) -> float:
    return sum(md_by_id[md_id].local_value * base_station.task_cost_ratio for md_id in candidate.md_ids)


def _cached_assignment(
    mobile_devices: list[MobileDevice],
    base_station: BaseStation,
    available_md_ids: frozenset[str],
    f_max: int,
    theta_max: int,
    cache: dict[tuple[str, frozenset[str]], TRAIMCandidate],
) -> TRAIMCandidate:
    cache_key = (base_station.bs_id, available_md_ids)
    if cache_key not in cache:
        cache[cache_key] = assign_mobile_devices(mobile_devices, base_station, set(available_md_ids), f_max, theta_max)
    return cache[cache_key]


def _distance(device: MobileDevice, base_station: BaseStation) -> float:
    return math.hypot(device.x - base_station.x, device.y - base_station.y)


def _scale_to_int(value: float, source_low: float, source_high: float, target_low: int, target_high: int) -> int:
    if source_high <= source_low:
        return target_low
    ratio = (value - source_low) / (source_high - source_low)
    scaled = target_low + ratio * (target_high - target_low)
    return min(target_high, max(target_low, int(round(scaled))))


def _numeric_suffix(identifier: str) -> int:
    return int(identifier.split("_")[-1])
