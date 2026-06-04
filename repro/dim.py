from __future__ import annotations

from repro.formulas import attractiveness, clamp, normalize_rewards
from repro.models import ClassProfile, PublishedBundle, ReferencePoint, Task, mean


STRATEGIES = ("F", "R", "RF")


def classify_tasks(tasks: list[Task]) -> tuple[dict[str, list[Task]], dict[str, ClassProfile], ReferencePoint]:
    normalize_rewards(tasks)
    real_tasks = [task for task in tasks if not task.is_decoy]
    reward_average = mean([task.normalized_reward for task in real_tasks])
    time_average = mean([task.time_cost for task in real_tasks])
    categories: dict[str, list[Task]] = {"A": [], "B": [], "C": []}
    for task in real_tasks:
        if task.normalized_reward > reward_average and task.time_cost > time_average:
            categories["A"].append(task.clone(kind="A"))
        elif task.normalized_reward < reward_average and task.time_cost < time_average:
            categories["B"].append(task.clone(kind="B"))
        else:
            categories["C"].append(task.clone(kind="C"))
    profiles = build_profiles(categories)
    return categories, profiles, ReferencePoint(reward_average, time_average)


def build_profiles(categories: dict[str, list[Task]]) -> dict[str, ClassProfile]:
    profiles: dict[str, ClassProfile] = {}
    for kind, tasks in categories.items():
        if not tasks:
            continue
        profiles[kind] = ClassProfile(
            kind=kind,  # type: ignore[arg-type]
            average_reward=mean([task.normalized_reward for task in tasks]),
            average_time_cost=mean([task.time_cost for task in tasks]),
            task_ids=[task.task_id for task in tasks],
            task_count=len(tasks),
        )
    return profiles


def effective_reward_lower(goal: ClassProfile, compete: ClassProfile) -> float:
    return max(0.0, (2.0 * compete.average_reward - goal.average_reward) / goal.average_reward)


def effective_time_upper(goal: ClassProfile, compete: ClassProfile) -> float:
    return max(1.0, (2.0 * goal.average_time_cost - compete.average_time_cost) / goal.average_time_cost)


def positive_reward_lower(goal: ClassProfile, compete: ClassProfile, reference_before: ReferencePoint) -> float:
    return max(0.0, (3.0 * reference_before.reward - (goal.average_reward + compete.average_reward)) / goal.average_reward)


def positive_time_lower(goal: ClassProfile, compete: ClassProfile, reference_before: ReferencePoint) -> float:
    return max(1.0, (3.0 * reference_before.time_cost - (goal.average_time_cost + compete.average_time_cost)) / goal.average_time_cost)


def reference_after(goal: ClassProfile, compete: ClassProfile, decoy: ClassProfile) -> ReferencePoint:
    return ReferencePoint(
        reward=mean([goal.average_reward, compete.average_reward, decoy.average_reward]),
        time_cost=mean([goal.average_time_cost, compete.average_time_cost, decoy.average_time_cost]),
    )


def compute_k(
    goal: ClassProfile,
    compete: ClassProfile,
    decoy: ClassProfile,
    reference_before_point: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> float:
    reference_after_point = reference_after(goal, compete, decoy)
    _, _, goal_after = attractiveness(
        goal.average_reward,
        goal.average_time_cost,
        reference_after_point,
        alpha,
        beta,
        lambda_loss,
    )
    _, _, compete_after = attractiveness(
        compete.average_reward,
        compete.average_time_cost,
        reference_after_point,
        alpha,
        beta,
        lambda_loss,
    )
    _, _, goal_before = attractiveness(
        goal.average_reward,
        goal.average_time_cost,
        reference_before_point,
        alpha,
        beta,
        lambda_loss,
    )
    _, _, compete_before = attractiveness(
        compete.average_reward,
        compete.average_time_cost,
        reference_before_point,
        alpha,
        beta,
        lambda_loss,
    )
    return goal_after - compete_after - (goal_before - compete_before)


def build_decoy_profile(goal: ClassProfile, chi: float, gamma: float) -> ClassProfile:
    return ClassProfile(
        kind="DECOY",
        average_reward=goal.average_reward * chi,
        average_time_cost=goal.average_time_cost * gamma,
        task_ids=[],
        task_count=1,
    )


def build_decoy_task(goal: ClassProfile, chi: float, gamma: float, release_round: int, label: str) -> Task:
    reward = goal.average_reward * chi
    time_cost = goal.average_time_cost * gamma
    return Task(
        task_id=f"decoy_{label}_r{release_round}",
        reward=reward,
        time_cost=time_cost,
        clock_frequency=1.0,
        kind="DECOY",
        normalized_reward=reward,
        release_round=release_round,
        is_decoy=True,
    )


def resolve_strategy_factors(
    strategy: str,
    goal: ClassProfile,
    compete: ClassProfile,
    reference_before_point: ReferencePoint,
    chi_override: float | None = None,
    gamma_override: float | None = None,
) -> tuple[float, float, tuple[float, float], tuple[float, float], float, float]:
    reward_lower = effective_reward_lower(goal, compete)
    reward_positive = positive_reward_lower(goal, compete, reference_before_point)
    time_lower = positive_time_lower(goal, compete, reference_before_point)
    time_upper = effective_time_upper(goal, compete)
    reward_range = (reward_lower, 1.0)
    time_range = (1.0, time_upper)
    almost_one = 1.0 - 1e-3

    if chi_override is not None or gamma_override is not None:
        chi = 1.0 if chi_override is None else chi_override
        gamma = time_upper if gamma_override is None else gamma_override
        return chi, gamma, reward_range, time_range, reward_positive, time_lower

    if strategy == "F":
        chi = 1.0
        gamma = 1.0 + 0.75 * (time_upper - 1.0)
    elif strategy == "RF":
        chi = reward_lower
        gamma = time_upper
    else:
        chi = 1.0
        gamma = time_upper
    return chi, gamma, reward_range, time_range, reward_positive, time_lower


def publish_dim_bundle(
    tasks: list[Task],
    alpha: float,
    beta: float,
    lambda_loss: float,
    strategy: str = "R",
    chi_override: float | None = None,
    gamma_override: float | None = None,
    release_round: int = 0,
    label: str = "dim",
) -> PublishedBundle:
    categories, profiles, reference_before_point = classify_tasks([task.clone() for task in tasks if not task.is_decoy])
    if "A" not in profiles or "B" not in profiles:
        raise ValueError("DIM requires both A and B task classes.")
    goal = profiles["A"]
    compete = profiles["B"]
    chi, gamma, reward_range, time_range, reward_positive, time_lower = resolve_strategy_factors(
        strategy,
        goal,
        compete,
        reference_before_point,
        chi_override,
        gamma_override,
    )
    decoy_profile = build_decoy_profile(goal, chi, gamma)
    decoy_task = build_decoy_task(goal, chi, gamma, release_round, label)
    profiles["DECOY"] = decoy_profile
    categories["DECOY"] = [decoy_task]
    reference_after_point = reference_after(goal, compete, decoy_profile)
    k_value = compute_k(goal, compete, decoy_profile, reference_before_point, alpha, beta, lambda_loss)
    return PublishedBundle(
        strategy=strategy,
        chi=chi,
        gamma=gamma,
        reference_before=reference_before_point,
        reference_after=reference_after_point,
        profiles=profiles,
        category_tasks=categories,
        k_value=k_value,
        reward_factor_range=reward_range,
        time_factor_range=time_range,
        positive_reward_lower=reward_positive,
        positive_time_lower=time_lower,
    )
