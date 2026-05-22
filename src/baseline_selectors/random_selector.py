from __future__ import annotations

import random


def select_random(num_candidates: int, subset_budget: int, seed: int = 0) -> list[int]:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")
    if subset_budget <= 0:
        raise ValueError("subset_budget must be positive.")
    if subset_budget > num_candidates:
        raise ValueError("subset_budget cannot exceed num_candidates.")

    rng = random.Random(int(seed))
    return rng.sample(range(num_candidates), int(subset_budget))
