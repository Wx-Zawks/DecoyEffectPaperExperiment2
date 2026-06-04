from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskKind = Literal["A", "B", "C", "DECOY", "RAW"]
NodeGroup = Literal["NPRN", "PRN", "PRIN", "UNCLASSIFIED"]


@dataclass
class Task:
    task_id: str
    reward: float
    time_cost: float
    clock_frequency: float
    kind: TaskKind = "RAW"
    normalized_reward: float = 0.0
    release_round: int = 0
    transaction_price: float = 0.0
    assigned_node_id: str | None = None
    is_decoy: bool = False

    def clone(self, **changes: object) -> "Task":
        data = {
            "task_id": self.task_id,
            "reward": self.reward,
            "time_cost": self.time_cost,
            "clock_frequency": self.clock_frequency,
            "kind": self.kind,
            "normalized_reward": self.normalized_reward,
            "release_round": self.release_round,
            "transaction_price": self.transaction_price,
            "assigned_node_id": self.assigned_node_id,
            "is_decoy": self.is_decoy,
        }
        data.update(changes)
        return Task(**data)


@dataclass
class FogNode:
    node_id: str
    clock_frequency: float
    preference: float
    initial_preference: float
    estimated_preference: float = 0.0
    group: NodeGroup = "UNCLASSIFIED"
    utility: float = 0.0
    bonus_utility: float = 0.0
    last_choice: str | None = None

    def clone(self, **changes: object) -> "FogNode":
        data = {
            "node_id": self.node_id,
            "clock_frequency": self.clock_frequency,
            "preference": self.preference,
            "initial_preference": self.initial_preference,
            "estimated_preference": self.estimated_preference,
            "group": self.group,
            "utility": self.utility,
            "bonus_utility": self.bonus_utility,
            "last_choice": self.last_choice,
        }
        data.update(changes)
        return FogNode(**data)


@dataclass
class ReferencePoint:
    reward: float
    time_cost: float


@dataclass
class ClassProfile:
    kind: TaskKind
    average_reward: float
    average_time_cost: float
    task_ids: list[str] = field(default_factory=list)
    task_count: int = 0

    def as_task(self, task_id: str) -> Task:
        return Task(
            task_id=task_id,
            reward=self.average_reward,
            time_cost=self.average_time_cost,
            clock_frequency=1.0,
            kind=self.kind,
            normalized_reward=self.average_reward,
            is_decoy=self.kind == "DECOY",
        )


@dataclass
class PublishedBundle:
    strategy: str
    chi: float
    gamma: float
    reference_before: ReferencePoint
    reference_after: ReferencePoint
    profiles: dict[str, ClassProfile]
    category_tasks: dict[str, list[Task]]
    k_value: float
    reward_factor_range: tuple[float, float]
    time_factor_range: tuple[float, float]
    positive_reward_lower: float
    positive_time_lower: float


@dataclass
class BidRecord:
    node_id: str
    task_id: str
    task_kind: str
    selected_kind: str
    task_time_cost: float
    task_reward: float
    reported_bid: float
    truthful_bid: float
    accepted_bid: float
    execution_cost: float
    satisfaction_threshold: float
    selected_score: float
    competitor_kind: str | None
    competitor_score: float
    truthful: bool
    is_decoy: bool


@dataclass
class Assignment:
    task_id: str
    winner_node_id: str
    winning_bid: float
    transaction_price: float
    is_decoy: bool


@dataclass
class RoundResult:
    round_index: int
    bundle_labels: dict[str, str]
    selection_counts: dict[str, int]
    display_selection_counts: dict[str, int]
    participants_real: int
    participant_node_ids: list[str]
    assigned_real_tasks: int
    offloaded_time_costs: list[float]
    positive_bid_count_real: int
    average_bid_real: float
    average_transaction_price_real: float
    user_total_utility: float
    assignments: list[Assignment]
    bids: list[BidRecord]
    leftover_task_ids: list[str]
    estimated_preferences: dict[str, float]
    preference_updates: dict[str, float]
    group_counts: dict[str, int]
    truthful_cancellations: int


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
