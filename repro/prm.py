from __future__ import annotations

from repro.dim import publish_dim_bundle
from repro.formulas import attractiveness, preference_reversal_threshold
from repro.models import NodeGroup, PublishedBundle


def classify_group(preference: float, lower_threshold: float) -> NodeGroup:
    if preference > 0.5:
        return "NPRN"
    if preference > lower_threshold:
        return "PRN"
    return "PRIN"


def build_group_push_bundles(
    tasks: list,
    alpha: float,
    beta: float,
    lambda_loss: float,
    release_round: int,
) -> tuple[PublishedBundle, dict[str, PublishedBundle], float]:
    base_bundle = publish_dim_bundle(
        tasks,
        alpha=alpha,
        beta=beta,
        lambda_loss=lambda_loss,
        strategy="R",
        release_round=release_round,
        label="nprn",
    )
    profile_a = base_bundle.profiles["A"]
    profile_b = base_bundle.profiles["B"]
    threshold = preference_reversal_threshold(
        profile_a,
        profile_b,
        base_bundle.reference_after,
        alpha,
        beta,
        lambda_loss,
    )

    prn_bundle = publish_dim_bundle(
        tasks,
        alpha=alpha,
        beta=beta,
        lambda_loss=lambda_loss,
        strategy="R",
        chi_override=1.0,
        gamma_override=base_bundle.positive_time_lower,
        release_round=release_round,
        label="prn",
    )
    prin_bundle = publish_dim_bundle(
        tasks,
        alpha=alpha,
        beta=beta,
        lambda_loss=lambda_loss,
        strategy="RF",
        chi_override=base_bundle.reward_factor_range[0],
        gamma_override=base_bundle.positive_time_lower,
        release_round=release_round,
        label="prin",
    )
    return base_bundle, {"NPRN": base_bundle, "PRN": prn_bundle, "PRIN": prin_bundle}, threshold


def profile_terms(bundle: PublishedBundle, alpha: float, beta: float, lambda_loss: float) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for kind, profile in bundle.profiles.items():
        result[kind] = attractiveness(
            profile.average_reward,
            profile.average_time_cost,
            bundle.reference_after,
            alpha,
            beta,
            lambda_loss,
        )
    return result
