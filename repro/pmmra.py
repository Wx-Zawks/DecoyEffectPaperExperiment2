from __future__ import annotations

from repro.dim import classify_tasks
from repro.models import PublishedBundle, ReferencePoint


def publish_pmmra_bundle(tasks: list, alpha: float, beta: float, lambda_loss: float) -> PublishedBundle:
    categories, profiles, reference_before = classify_tasks(tasks)
    return PublishedBundle(
        strategy="NO_DECOY",
        chi=0.0,
        gamma=0.0,
        reference_before=reference_before,
        reference_after=ReferencePoint(reference_before.reward, reference_before.time_cost),
        profiles=profiles,
        category_tasks=categories,
        k_value=0.0,
        reward_factor_range=(0.0, 0.0),
        time_factor_range=(0.0, 0.0),
        positive_reward_lower=0.0,
        positive_time_lower=0.0,
    )
