from __future__ import annotations

import random


def build_rng(seed: int) -> random.Random:
    """Return a seeded random generator for reproducible experiments."""
    return random.Random(seed)
