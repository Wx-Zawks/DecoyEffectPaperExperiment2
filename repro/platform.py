from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

from repro.dim import publish_dim_bundle
from repro.formulas import (
    attractiveness,
    delta_increment,
    estimate_preference,
    execution_cost,
    local_execution_cost,
    monitor_bid,
    profile_preference,
    task_choice_threshold,
    truthful_bid,
    update_preference,
)
from repro.models import Assignment, BidRecord, FogNode, PublishedBundle, RoundResult, Task, mean
from repro.pmmra import publish_pmmra_bundle
from repro.prm import build_group_push_bundles, classify_group
from repro.random_utils import build_rng
from repro.traim import build_traim_environment, candidate_cost, normalized_resource_for_md, run_traim


class Platform:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.rng = build_rng(config["seed"])

    def generate_tasks(self, task_count: int) -> list[Task]:
        tasks: list[Task] = []
        low_f, high_f = self.config["frequency_range"]
        low_reward, high_reward = self.config["reward_range"]
        low_time, high_time = self.config["time_cost_range"]
        for index in range(task_count):
            tasks.append(
                Task(
                    task_id=f"task_{index + 1}",
                    reward=round(self.rng.uniform(low_reward, high_reward), 6),
                    time_cost=round(self.rng.uniform(low_time, high_time), 6),
                    clock_frequency=round(self.rng.uniform(low_f, high_f), 6),
                )
            )
        return tasks

    def generate_nodes(self, node_count: int, preference_mean: float) -> list[FogNode]:
        nodes: list[FogNode] = []
        low_f, high_f = self.config["frequency_range"]
        std = self.config["preference_std"]
        for index in range(node_count):
            preference = self._draw_truncated_normal(preference_mean, std)
            nodes.append(
                FogNode(
                    node_id=f"node_{index + 1}",
                    clock_frequency=round(self.rng.uniform(low_f, high_f), 6),
                    preference=preference,
                    initial_preference=preference,
                    estimated_preference=preference,
                )
            )
        return nodes

    def prepare_environment(self, node_count: int, task_count: int, preference_mean: float) -> tuple[list[Task], list[FogNode]]:
        max_attempts = 256
        for _ in range(max_attempts):
            tasks = self.generate_tasks(task_count)
            nodes = self.generate_nodes(node_count, preference_mean)
            try:
                publish_dim_bundle(
                    tasks,
                    alpha=self.config["alpha"],
                    beta=self.config["beta"],
                    lambda_loss=self.config["lambda_loss"],
                    strategy="R",
                )
            except ValueError:
                continue
            return tasks, nodes
        raise RuntimeError("Failed to sample tasks with non-empty A/B classes after repeated attempts.")

    def compare_mechanisms(self, node_count: int, task_count: int, preference_mean: float) -> dict:
        tasks, nodes = self.prepare_environment(node_count, task_count, preference_mean)
        dim = self.simulate_dim(self.clone_tasks(tasks), self.clone_nodes(nodes), strategy="R")
        prm = self.simulate_prm(self.clone_tasks(tasks), self.clone_nodes(nodes))
        traim = self.simulate_traim(self.clone_tasks(tasks), self.clone_nodes(nodes))
        return {
            "task_count": task_count,
            "node_count": node_count,
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
            "DIM": dim,
            "PRM": prm,
            "TRAIM": traim,
        }

    def simulate_no_decoy(self, tasks: list[Task], nodes: list[FogNode], max_tasks_per_node: int | None = 2) -> dict:
        self._reset_nodes(nodes)
        bundle = publish_pmmra_bundle(tasks, self.config["alpha"], self.config["beta"], self.config["lambda_loss"])
        round_result = self._simulate_round(
            nodes=nodes,
            real_tasks=tasks,
            bundles={"DEFAULT": bundle},
            dynamic_update=False,
            round_index=0,
            max_tasks_per_node=max_tasks_per_node,
        )
        return self._aggregate_result("NO_DECOY", nodes, [round_result], bundle)

    def simulate_dim(self, tasks: list[Task], nodes: list[FogNode], strategy: str = "R", chi: float | None = None, gamma: float | None = None) -> dict:
        self._reset_nodes(nodes)
        bundle = publish_dim_bundle(
            tasks,
            alpha=self.config["alpha"],
            beta=self.config["beta"],
            lambda_loss=self.config["lambda_loss"],
            strategy=strategy,
            chi_override=chi,
            gamma_override=gamma,
        )
        round_result = self._simulate_round(
            nodes=nodes,
            real_tasks=tasks,
            bundles={"DEFAULT": bundle},
            dynamic_update=False,
            round_index=0,
            max_tasks_per_node=None,
        )
        return self._aggregate_result("DIM", nodes, [round_result], bundle)

    def simulate_prm(self, tasks: list[Task], nodes: list[FogNode]) -> dict:
        self._reset_nodes(nodes)
        rounds: list[RoundResult] = []
        initial_bundle = publish_dim_bundle(
            tasks,
            alpha=self.config["alpha"],
            beta=self.config["beta"],
            lambda_loss=self.config["lambda_loss"],
            strategy="R",
            release_round=0,
            label="initial",
        )
        current_leftover = self.clone_tasks(tasks)
        round_result = self._simulate_round(
            nodes=nodes,
            real_tasks=current_leftover,
            bundles={"DEFAULT": initial_bundle},
            dynamic_update=True,
            round_index=0,
            max_tasks_per_node=None,
        )
        rounds.append(round_result)
        self._update_estimated_preferences(nodes, round_result)
        current_leftover = self._collect_leftovers(current_leftover, round_result.leftover_task_ids)

        release_round = 1
        while current_leftover:
            try:
                base_bundle, push_bundles, threshold = build_group_push_bundles(
                    current_leftover,
                    alpha=self.config["alpha"],
                    beta=self.config["beta"],
                    lambda_loss=self.config["lambda_loss"],
                    release_round=release_round,
                )
            except ValueError:
                fallback_bundle = publish_pmmra_bundle(current_leftover, self.config["alpha"], self.config["beta"], self.config["lambda_loss"])
                round_result = self._simulate_round(
                    nodes=nodes,
                    real_tasks=current_leftover,
                    bundles={"DEFAULT": fallback_bundle},
                    dynamic_update=False,
                    round_index=release_round,
                    max_tasks_per_node=None,
                )
                rounds.append(round_result)
                self._update_estimated_preferences(nodes, round_result)
                if round_result.assigned_real_tasks == 0:
                    break
                current_leftover = self._collect_leftovers(current_leftover, round_result.leftover_task_ids)
                release_round += 1
                if release_round > self.config["task_count"]:
                    break
                continue
            for node in nodes:
                node.group = classify_group(node.estimated_preference, threshold)
            round_result = self._simulate_round(
                nodes=nodes,
                real_tasks=current_leftover,
                bundles=push_bundles,
                dynamic_update=True,
                round_index=release_round,
                max_tasks_per_node=None,
            )
            rounds.append(round_result)
            self._update_estimated_preferences(nodes, round_result)
            if round_result.assigned_real_tasks == 0:
                break
            current_leftover = self._collect_leftovers(current_leftover, round_result.leftover_task_ids)
            release_round += 1
            if release_round > self.config["task_count"]:
                break

        return self._aggregate_result("PRM", nodes, rounds, initial_bundle)

    def simulate_traim(self, tasks: list[Task], nodes: list[FogNode]) -> dict:
        self._reset_nodes(nodes)
        mobile_devices, base_stations = build_traim_environment(tasks, nodes, self.rng, self.config)
        outcome = run_traim(mobile_devices, base_stations)
        md_by_id = {device.md_id: device for device in mobile_devices}
        task_by_id = {task.task_id: task for task in tasks}
        bs_by_id = {base_station.bs_id: base_station for base_station in base_stations}
        bidder_node_ids = [bs_by_id[bs_id].node_id for bs_id in outcome.bidder_ids]
        f_max = max((base_station.cpu_cores for base_station in base_stations), default=1)
        theta_max = max((base_station.subchannels for base_station in base_stations), default=1)

        assignments: list[Assignment] = []
        bid_records: list[BidRecord] = []
        offloaded_time_costs: list[float] = []
        bid_values: list[float] = []
        payment_values: list[float] = []
        user_total_utility = 0.0

        for bs_id in outcome.winners:
            base_station = bs_by_id[bs_id]
            candidate = outcome.candidates[bs_id]
            payment = outcome.payments.get(bs_id, 0.0)
            winner_cost = candidate_cost(base_station, candidate, md_by_id)
            denominator = max(candidate.cpr_denominator, 1e-9)
            node = self._node_by_id(nodes, base_station.node_id)
            node.utility += payment - winner_cost
            for md_id in candidate.md_ids:
                device = md_by_id[md_id]
                phi = candidate.phi_by_md[md_id]
                resource_share = normalized_resource_for_md(device, phi, f_max, theta_max) / denominator
                bid_share = winner_cost * resource_share
                payment_share = payment * resource_share
                task = task_by_id[device.task_id]
                task.assigned_node_id = base_station.node_id
                task.transaction_price = payment_share
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        winner_node_id=base_station.node_id,
                        winning_bid=bid_share,
                        transaction_price=payment_share,
                        is_decoy=False,
                    )
                )
                bid_records.append(
                    BidRecord(
                        node_id=base_station.node_id,
                        task_id=task.task_id,
                        task_kind="MEC",
                        selected_kind="TRAIM",
                        task_time_cost=device.task_time_cost,
                        task_reward=device.task_reward,
                        reported_bid=bid_share,
                        truthful_bid=bid_share,
                        accepted_bid=bid_share,
                        execution_cost=winner_cost * resource_share,
                        satisfaction_threshold=0.0,
                        selected_score=candidate.cpr_denominator,
                        competitor_kind=None,
                        competitor_score=0.0,
                        truthful=True,
                        is_decoy=False,
                    )
                )
                bid_values.append(bid_share)
                payment_values.append(payment_share)
                offloaded_time_costs.append(device.task_time_cost)
                user_total_utility += device.local_value - payment_share

        served_task_ids = {assignment.task_id for assignment in assignments}
        round_result = RoundResult(
            round_index=0,
            bundle_labels={"DEFAULT": "TRAIM"},
            selection_counts={"BS": len(outcome.bidder_ids), "WINNER": len(outcome.winners)},
            display_selection_counts={"BS": len(outcome.bidder_ids), "WINNER": len(outcome.winners)},
            participants_real=len(outcome.bidder_ids),
            participant_node_ids=bidder_node_ids,
            assigned_real_tasks=len(assignments),
            offloaded_time_costs=offloaded_time_costs,
            positive_bid_count_real=len(bid_values),
            average_bid_real=mean(bid_values),
            average_transaction_price_real=mean(payment_values),
            user_total_utility=user_total_utility,
            assignments=assignments,
            bids=bid_records,
            leftover_task_ids=[task.task_id for task in tasks if task.task_id not in served_task_ids],
            estimated_preferences={node.node_id: node.preference for node in nodes},
            preference_updates={node.node_id: node.preference for node in nodes},
            group_counts={"BS": len(nodes)},
            truthful_cancellations=0,
        )
        social_cost = sum(candidate_cost(bs_by_id[bs_id], outcome.candidates[bs_id], md_by_id) for bs_id in outcome.winners)
        total_payment = sum(outcome.payments.values())
        return {
            "mechanism": "TRAIM",
            "participants": len(outcome.bidder_ids),
            "goal_participants": 0,
            "participant_ids": bidder_node_ids,
            "offloaded_tasks": len(assignments),
            "average_bid": mean(bid_values),
            "average_transaction_price": mean(payment_values),
            "user_total_utility": user_total_utility,
            "goal_selection_rate": 0.0,
            "goal_choice_threshold": None,
            "selection_counts": {"BS": len(outcome.bidder_ids), "WINNER": len(outcome.winners)},
            "offloaded_time_costs": offloaded_time_costs,
            "rounds": [self._serialize_round(round_result)],
            "bundle": {
                "strategy": "TRAIM",
                "chi": 0.0,
                "gamma": 0.0,
                "k_value": 0.0,
                "reward_factor_range": (0.0, 0.0),
                "time_factor_range": (0.0, 0.0),
                "positive_reward_lower": 0.0,
                "positive_time_lower": 0.0,
                "social_cost": round(social_cost, 6),
                "total_payment": round(total_payment, 6),
                "overpayment_ratio": round((total_payment - social_cost) / social_cost, 6) if social_cost > 0.0 else 0.0,
                "winning_bs_count": len(outcome.winners),
                "valid_bidder_count": len(outcome.bidder_ids),
                "unserved_md_count": len(outcome.unserved_md_ids),
            },
            "node_utilities": {node.node_id: round(node.utility, 6) for node in nodes},
            "individual_rationality": min((node.utility for node in nodes), default=0.0) >= -1e-9,
            "truthful_cancellations": 0,
        }

    def validate_properties(self) -> dict:
        validation_node_count = 30
        validation_task_count = self.config["task_count"]
        tasks, nodes = self.prepare_environment(validation_node_count, validation_task_count, 0.5)
        started_at = time.perf_counter()
        prm_result = self.simulate_prm(self.clone_tasks(tasks), self.clone_nodes(nodes))
        runtime_ms = (time.perf_counter() - started_at) * 1000.0

        truthful_demo_bundle = publish_dim_bundle(
            self.clone_tasks(tasks),
            alpha=self.config["alpha"],
            beta=self.config["beta"],
            lambda_loss=self.config["lambda_loss"],
            strategy="R",
        )
        demo_node = self.clone_nodes(nodes)[0]
        score_map = self._score_map(demo_node.preference, truthful_demo_bundle)
        selected_kind, selected_score = self._choose_kind(score_map, truthful_demo_bundle)
        other_scores = [score for kind, score in score_map.items() if kind != selected_kind]
        demo_task = truthful_demo_bundle.category_tasks[selected_kind][0]
        true_bid, _ = truthful_bid(demo_task, demo_node, selected_score, other_scores)
        honest_after_monitor, honest_cancelled = monitor_bid(true_bid, true_bid)
        inflated_after_monitor, inflated_cancelled = monitor_bid(true_bid + max(abs(true_bid) * 0.2, 1.0), true_bid)

        node_utilities = prm_result["node_utilities"]
        return {
            "individual_rationality": min(node_utilities.values(), default=0.0) >= -1e-9,
            "truthfulness": {
                "honest_bid_preserved": not honest_cancelled and math.isclose(honest_after_monitor, max(true_bid, 0.0), rel_tol=1e-9, abs_tol=1e-9),
                "inflated_bid_cancelled": inflated_cancelled and math.isclose(inflated_after_monitor, 0.0, abs_tol=1e-9),
            },
            "computational_efficiency": {
                "claimed_complexity": "O(IJlogJ)",
                "validation_runtime_ms": round(runtime_ms, 6),
            },
            "minimum_node_utility": round(min(node_utilities.values(), default=0.0), 6),
        }

    def clone_tasks(self, tasks: list[Task]) -> list[Task]:
        return [task.clone() for task in tasks]

    def clone_nodes(self, nodes: list[FogNode]) -> list[FogNode]:
        return [node.clone() for node in nodes]

    def _draw_truncated_normal(self, mean_value: float, std_value: float) -> float:
        for _ in range(64):
            candidate = self.rng.gauss(mean_value, std_value)
            if 0.0 <= candidate <= 1.0:
                return round(candidate, 6)
        return round(min(1.0, max(0.0, candidate)), 6)

    def _reset_nodes(self, nodes: list[FogNode]) -> None:
        for node in nodes:
            node.preference = node.initial_preference
            node.estimated_preference = node.initial_preference
            node.group = "UNCLASSIFIED"
            node.utility = 0.0
            node.bonus_utility = 0.0
            node.last_choice = None

    def _score_map(self, delta: float, bundle: PublishedBundle) -> dict[str, float]:
        score_map: dict[str, float] = {}
        for kind, profile in bundle.profiles.items():
            if bundle.category_tasks.get(kind):
                score_map[kind] = profile_preference(
                    delta,
                    profile,
                    bundle.reference_after,
                    self.config["alpha"],
                    self.config["beta"],
                    self.config["lambda_loss"],
                )
        return score_map

    def _choose_kind(self, score_map: dict[str, float], bundle: PublishedBundle) -> tuple[str, float]:
        ordered = sorted(score_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
        chosen_kind, chosen_score = ordered[0]
        if chosen_score <= 0.0 or not bundle.category_tasks.get(chosen_kind):
            return chosen_kind, chosen_score
        return chosen_kind, chosen_score

    def _simulate_round(
        self,
        nodes: list[FogNode],
        real_tasks: list[Task],
        bundles: dict[str, PublishedBundle],
        dynamic_update: bool,
        round_index: int,
        max_tasks_per_node: int | None,
    ) -> RoundResult:
        task_registry: dict[str, Task] = {task.task_id: task for task in real_tasks}
        bid_book: dict[str, list[BidRecord]] = defaultdict(list)
        selection_counts: Counter[str] = Counter()
        display_selection_counts: Counter[str] = Counter()
        participant_node_ids: set[str] = set()
        truthful_cancellations = 0
        estimated_preferences: dict[str, float] = {}
        preference_updates: dict[str, float] = {}
        group_counts: Counter[str] = Counter()
        positive_bid_values_real: list[float] = []

        for node in nodes:
            bundle = bundles.get(node.group, bundles.get("DEFAULT"))
            if bundle is None:
                continue
            group_counts[node.group] += 1
            pre_scores = self._score_map(node.preference, bundle)
            if dynamic_update and "A" in bundle.profiles and "B" in bundle.profiles:
                delta_y = delta_increment(
                    node.preference,
                    bundle.profiles["A"],
                    bundle.profiles["B"],
                    bundle.reference_after,
                    self.config["alpha"],
                    self.config["beta"],
                    self.config["lambda_loss"],
                )
                updated_delta = update_preference(node.preference, pre_scores.get("A", 0.0), pre_scores.get("B", 0.0), delta_y)
                node.preference = updated_delta
            preference_updates[node.node_id] = node.preference
            score_map = self._score_map(node.preference, bundle)
            if not score_map:
                estimated_preferences[node.node_id] = node.preference
                continue
            display_choice, display_score = self._choose_kind(score_map, bundle)
            node.last_choice = display_choice
            if display_score > 0.0:
                display_selection_counts[display_choice] += 1
            action_score_map = {kind: score for kind, score in score_map.items() if kind != "DECOY"}
            if not action_score_map:
                estimated_preferences[node.node_id] = node.preference
                continue
            if display_choice == "DECOY" and "A" in action_score_map:
                # Decoys are virtual reference options; a decoy-triggered action is redirected to the target class.
                chosen_kind, chosen_score = "A", action_score_map["A"]
            else:
                chosen_kind, chosen_score = self._choose_kind(action_score_map, bundle)
            if chosen_score <= 0.0 or not bundle.category_tasks.get(chosen_kind):
                estimated_preferences[node.node_id] = node.preference
                continue

            ordered_scores = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            competitor_kind = ordered_scores[1][0] if len(ordered_scores) > 1 else ordered_scores[0][0]
            competitor_score = ordered_scores[1][1] if len(ordered_scores) > 1 else ordered_scores[0][1]
            selection_counts[chosen_kind] += 1

            recorded_bid: BidRecord | None = None
            other_scores = [value for kind, value in score_map.items() if kind != chosen_kind]
            chosen_tasks = list(bundle.category_tasks[chosen_kind])
            if max_tasks_per_node is not None:
                chosen_tasks = chosen_tasks[:max_tasks_per_node]
            for task in chosen_tasks:
                if not task.is_decoy and task.task_id not in task_registry:
                    continue
                truthful_value, threshold = truthful_bid(task, node, chosen_score, other_scores)
                accepted_bid, was_cancelled = monitor_bid(truthful_value, truthful_value)
                truthful_cancellations += int(was_cancelled)
                if accepted_bid <= 0.0:
                    continue
                if not task.is_decoy:
                    participant_node_ids.add(node.node_id)
                    positive_bid_values_real.append(accepted_bid)
                effective_task = task_registry.get(task.task_id, task)
                task_registry.setdefault(effective_task.task_id, effective_task)
                bid_record = BidRecord(
                    node_id=node.node_id,
                    task_id=effective_task.task_id,
                    task_kind=effective_task.kind,
                    selected_kind=chosen_kind,
                    task_time_cost=effective_task.time_cost,
                    task_reward=effective_task.reward,
                    reported_bid=truthful_value,
                    truthful_bid=truthful_value,
                    accepted_bid=accepted_bid,
                    execution_cost=execution_cost(task, node),
                    satisfaction_threshold=threshold,
                    selected_score=chosen_score,
                    competitor_kind=competitor_kind,
                    competitor_score=competitor_score,
                    truthful=True,
                    is_decoy=effective_task.is_decoy,
                )
                bid_book[effective_task.task_id].append(bid_record)
                if recorded_bid is None:
                    recorded_bid = bid_record
            if recorded_bid is None:
                estimated_preferences[node.node_id] = node.preference
                continue
            selected_profile = bundle.profiles[chosen_kind]
            competitor_profile = bundle.profiles[recorded_bid.competitor_kind] if recorded_bid.competitor_kind else selected_profile
            estimated_preferences[node.node_id] = estimate_preference(
                recorded_bid.execution_cost - recorded_bid.truthful_bid,
                selected_profile,
                competitor_profile,
                bundle.reference_after,
                self.config["alpha"],
                self.config["beta"],
                self.config["lambda_loss"],
                node.preference,
            )

        assignments: list[Assignment] = []
        assigned_real_task_ids: set[str] = set()
        winning_bid_values: list[float] = []
        transaction_values_real: list[float] = []
        offloaded_time_costs: list[float] = []
        user_total_utility = 0.0

        for task_id, bids in bid_book.items():
            ordered_bids = sorted(bids, key=lambda item: item.accepted_bid)
            winner = ordered_bids[0]
            second_price = ordered_bids[1].accepted_bid if len(ordered_bids) > 1 else winner.accepted_bid
            task = task_registry[task_id]
            task.assigned_node_id = winner.node_id
            task.transaction_price = second_price
            assignments.append(
                Assignment(
                    task_id=task.task_id,
                    winner_node_id=winner.node_id,
                    winning_bid=winner.accepted_bid,
                    transaction_price=second_price,
                    is_decoy=task.is_decoy,
                )
            )
            winning_bid_values.append(winner.accepted_bid)
            winner_node = self._node_by_id(nodes, winner.node_id)
            winner_node.utility += second_price - winner.truthful_bid
            if not task.is_decoy:
                assigned_real_task_ids.add(task.task_id)
                transaction_values_real.append(second_price)
                offloaded_time_costs.append(task.time_cost)
                user_total_utility += local_execution_cost(task) - second_price

        bonus_nodes = self._bonus_nodes(bid_book, assignments)
        bonus_value = 0.0
        if winning_bid_values:
            average_winning_bid = mean(winning_bid_values)
            if average_winning_bid > 0.0:
                bonus_value = 0.05 * len(assignments) / average_winning_bid
        for node in nodes:
            if node.node_id in bonus_nodes:
                node.utility += bonus_value
                node.bonus_utility += bonus_value

        leftover_task_ids = [task.task_id for task in real_tasks if task.task_id not in assigned_real_task_ids]
        bundle_labels = {group: bundle.strategy for group, bundle in bundles.items()}
        return RoundResult(
            round_index=round_index,
            bundle_labels=bundle_labels,
            selection_counts=dict(selection_counts),
            display_selection_counts=dict(display_selection_counts),
            participants_real=len(participant_node_ids),
            participant_node_ids=sorted(participant_node_ids),
            assigned_real_tasks=len(assigned_real_task_ids),
            offloaded_time_costs=offloaded_time_costs,
            positive_bid_count_real=len(positive_bid_values_real),
            average_bid_real=mean(positive_bid_values_real),
            average_transaction_price_real=mean(transaction_values_real),
            user_total_utility=user_total_utility,
            assignments=assignments,
            bids=[bid for bids in bid_book.values() for bid in bids],
            leftover_task_ids=leftover_task_ids,
            estimated_preferences=estimated_preferences,
            preference_updates=preference_updates,
            group_counts=dict(group_counts),
            truthful_cancellations=truthful_cancellations,
        )

    def _aggregate_result(self, mechanism: str, nodes: list[FogNode], rounds: list[RoundResult], base_bundle: PublishedBundle) -> dict:
        participant_ids = sorted({node_id for round_result in rounds for node_id in round_result.participant_node_ids})
        average_bid = self._weighted_average(
            [(round_result.average_bid_real, round_result.positive_bid_count_real) for round_result in rounds]
        )
        average_price = self._weighted_average(
            [(round_result.average_transaction_price_real, round_result.assigned_real_tasks) for round_result in rounds]
        )
        offloaded_time_costs = [value for round_result in rounds for value in round_result.offloaded_time_costs]
        aggregate_selection_counts = dict(sum((Counter(round_result.selection_counts) for round_result in rounds), Counter()))
        goal_threshold = None
        if "A" in base_bundle.profiles and "B" in base_bundle.profiles:
            goal_threshold = task_choice_threshold(
                base_bundle.profiles["A"],
                base_bundle.profiles["B"],
                base_bundle.reference_after,
                self.config["alpha"],
                self.config["beta"],
                self.config["lambda_loss"],
            )
        return {
            "mechanism": mechanism,
            "participants": len(participant_ids),
            "goal_participants": aggregate_selection_counts.get("A", 0),
            "participant_ids": participant_ids,
            "offloaded_tasks": sum(round_result.assigned_real_tasks for round_result in rounds),
            "average_bid": average_bid,
            "average_transaction_price": average_price,
            "user_total_utility": sum(round_result.user_total_utility for round_result in rounds),
            "goal_selection_rate": rounds[0].selection_counts.get("A", 0) / len(nodes) if nodes else 0.0,
            "goal_choice_threshold": goal_threshold,
            "selection_counts": aggregate_selection_counts,
            "offloaded_time_costs": offloaded_time_costs,
            "rounds": [self._serialize_round(round_result) for round_result in rounds],
            "bundle": {
                "strategy": base_bundle.strategy,
                "chi": base_bundle.chi,
                "gamma": base_bundle.gamma,
                "k_value": base_bundle.k_value,
                "reward_factor_range": base_bundle.reward_factor_range,
                "time_factor_range": base_bundle.time_factor_range,
                "positive_reward_lower": base_bundle.positive_reward_lower,
                "positive_time_lower": base_bundle.positive_time_lower,
            },
            "node_utilities": {node.node_id: round(node.utility, 6) for node in nodes},
            "individual_rationality": min((node.utility for node in nodes), default=0.0) >= -1e-9,
            "truthful_cancellations": sum(round_result.truthful_cancellations for round_result in rounds),
        }

    def _weighted_average(self, values: list[tuple[float, int]]) -> float:
        weighted_sum = sum(value * weight for value, weight in values)
        weight_sum = sum(weight for _, weight in values)
        if weight_sum == 0:
            return 0.0
        return weighted_sum / weight_sum

    def _serialize_round(self, round_result: RoundResult) -> dict:
        return {
            "round_index": round_result.round_index,
            "bundle_labels": round_result.bundle_labels,
            "selection_counts": round_result.selection_counts,
            "display_selection_counts": round_result.display_selection_counts,
            "participants_real": round_result.participants_real,
            "participant_node_ids": round_result.participant_node_ids,
            "assigned_real_tasks": round_result.assigned_real_tasks,
            "offloaded_time_costs": round_result.offloaded_time_costs,
            "positive_bid_count_real": round_result.positive_bid_count_real,
            "average_bid_real": round(round_result.average_bid_real, 6),
            "average_transaction_price_real": round(round_result.average_transaction_price_real, 6),
            "user_total_utility": round(round_result.user_total_utility, 6),
            "assignments": [assignment.__dict__ for assignment in round_result.assignments],
            "bids": [bid.__dict__ for bid in round_result.bids],
            "leftover_task_ids": round_result.leftover_task_ids,
            "estimated_preferences": {key: round(value, 6) for key, value in round_result.estimated_preferences.items()},
            "preference_updates": {key: round(value, 6) for key, value in round_result.preference_updates.items()},
            "group_counts": round_result.group_counts,
            "truthful_cancellations": round_result.truthful_cancellations,
        }

    def _update_estimated_preferences(self, nodes: list[FogNode], round_result: RoundResult) -> None:
        for node in nodes:
            node.estimated_preference = round_result.estimated_preferences.get(node.node_id, node.preference)

    def _collect_leftovers(self, tasks: list[Task], leftover_ids: list[str]) -> list[Task]:
        registry = {task.task_id: task for task in tasks}
        return [registry[task_id].clone() for task_id in leftover_ids if task_id in registry]

    def _bonus_nodes(self, bid_book: dict[str, list[BidRecord]], assignments: list[Assignment]) -> set[str]:
        eligible = sorted({bid.node_id for bids in bid_book.values() for bid in bids if bid.truthful and bid.accepted_bid > 0.0})
        if not eligible:
            return set()
        count = max(1, round(len(eligible) * 0.05))
        return set(self.rng.sample(eligible, min(count, len(eligible))))

    def _node_by_id(self, nodes: list[FogNode], node_id: str) -> FogNode:
        for node in nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)
