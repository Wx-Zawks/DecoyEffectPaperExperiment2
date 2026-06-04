from __future__ import annotations

import math

from repro.models import ClassProfile, FogNode, ReferencePoint, Task


EPSILON = 1e-9


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def normalize_rewards(tasks: list[Task]) -> None:
    """Equation (2): linearly scale rewards to the time-cost magnitude."""
    if not tasks:
        return
    raw_rewards = [task.reward for task in tasks]
    time_costs = [task.time_cost for task in tasks]
    reward_min = min(raw_rewards)
    reward_max = max(raw_rewards)
    time_min = min(time_costs)
    time_max = max(time_costs)
    if math.isclose(reward_max, reward_min):
        midpoint = (time_min + time_max) / 2.0
        for task in tasks:
            task.normalized_reward = midpoint
        return
    scale = (time_max - time_min) / (reward_max - reward_min)
    for task in tasks:
        task.normalized_reward = (task.reward - reward_min) * scale + time_min


def reward_attraction(value: float, reference: float, alpha: float, beta: float, lambda_loss: float) -> float:
    delta = value - reference
    if delta >= 0:
        return delta**alpha
    return -lambda_loss * ((-delta) ** beta)


def time_attraction(value: float, reference: float, alpha: float, beta: float, lambda_loss: float) -> float:
    delta = value - reference
    if delta >= 0:
        return -lambda_loss * (delta**beta)
    return (reference - value) ** alpha


def attractiveness(
    reward: float,
    time_cost: float,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> tuple[float, float, float]:
    reward_term = reward_attraction(reward, reference.reward, alpha, beta, lambda_loss)
    time_term = time_attraction(time_cost, reference.time_cost, alpha, beta, lambda_loss)
    return reward_term, time_term, reward_term + time_term


def profile_preference(
    delta: float,
    profile: ClassProfile,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> float:
    reward_term, time_term, _ = attractiveness(
        profile.average_reward,
        profile.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    return delta * reward_term + (1.0 - delta) * time_term


def task_choice_threshold(
    profile_a: ClassProfile,
    profile_b: ClassProfile,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> float:
    """Preference value where A and B have equal weighted attractiveness."""
    p_a, q_a, _ = attractiveness(
        profile_a.average_reward,
        profile_a.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    p_b, q_b, _ = attractiveness(
        profile_b.average_reward,
        profile_b.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    numerator = q_b - q_a
    denominator = (p_a - p_b) + numerator
    if abs(denominator) <= EPSILON:
        return 0.5
    return clamp(numerator / denominator, 0.0, 1.0)


def delta_increment(
    delta: float,
    profile_a: ClassProfile,
    profile_b: ClassProfile,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> float:
    """Equation (17)."""
    p_a, q_a, _ = attractiveness(profile_a.average_reward, profile_a.average_time_cost, reference, alpha, beta, lambda_loss)
    p_b, q_b, _ = attractiveness(profile_b.average_reward, profile_b.average_time_cost, reference, alpha, beta, lambda_loss)
    denominator = (1.0 - delta) * (q_b - q_a)
    if abs(denominator) <= EPSILON:
        return 0.0
    return delta * (p_a - p_b) / denominator


def update_preference(delta: float, w_a: float, w_b: float, delta_y: float) -> float:
    """Equation (16)."""
    if math.isclose(w_a, w_b, abs_tol=EPSILON) or delta <= 0.0 or delta >= 1.0:
        return clamp(delta, 0.0, 1.0)
    if w_a > w_b:
        return clamp(delta + delta_y, 0.0, 1.0)
    return clamp(delta - delta_y, 0.0, 1.0)


def satisfaction_threshold(selected_score: float, other_scores: list[float]) -> float:
    """Equation (20)."""
    if not other_scores:
        return max(selected_score, 0.0)
    return max(selected_score - other for other in other_scores)


def execution_cost(task: Task, node: FogNode) -> float:
    return task.time_cost / node.clock_frequency


def local_execution_cost(task: Task) -> float:
    return task.time_cost / task.clock_frequency


def truthful_bid(task: Task, node: FogNode, selected_score: float, other_scores: list[float]) -> tuple[float, float]:
    threshold = satisfaction_threshold(selected_score, other_scores)
    return execution_cost(task, node) - threshold, threshold


def monitor_bid(reported_bid: float, truthful_bid_value: float) -> tuple[float, bool]:
    """Lemma 4-4: cancel exaggerated bids discovered by the overseer."""
    if reported_bid - truthful_bid_value > EPSILON:
        return 0.0, True
    return max(reported_bid, 0.0), False


def estimate_preference(
    discount: float,
    selected_profile: ClassProfile,
    competitor_profile: ClassProfile,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
    fallback: float,
) -> float:
    """Equation (24): infer delta from the observed bid discount."""
    p_s, q_s, _ = attractiveness(
        selected_profile.average_reward,
        selected_profile.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    p_c, q_c, _ = attractiveness(
        competitor_profile.average_reward,
        competitor_profile.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    numerator = discount - (q_s - q_c)
    denominator = (p_s - p_c) - (q_s - q_c)
    if abs(denominator) <= EPSILON:
        return clamp(fallback, 0.0, 1.0)
    return clamp(numerator / denominator, 0.0, 1.0)


def preference_reversal_threshold(
    profile_a: ClassProfile,
    profile_b: ClassProfile,
    reference: ReferencePoint,
    alpha: float,
    beta: float,
    lambda_loss: float,
) -> float:
    """Theorem 4-1, derived from Equations (17)-(19)."""
    p_a, q_a, _ = attractiveness(
        profile_a.average_reward,
        profile_a.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    p_b, q_b, _ = attractiveness(
        profile_b.average_reward,
        profile_b.average_time_cost,
        reference,
        alpha,
        beta,
        lambda_loss,
    )
    q_gap = q_b - q_a
    p_gap = p_a - p_b
    if q_gap <= EPSILON:
        return 0.5
    a = q_gap
    b = -(1.5 * q_gap + p_gap)
    c = 0.5 * q_gap
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    lower_root = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return clamp(lower_root, 0.0, 0.5)
