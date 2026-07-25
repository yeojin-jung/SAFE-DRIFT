from __future__ import annotations

import random


def select_replay(
    num_less_candidates: int,
    num_reference_candidates: int,
    subset_budget: int,
    k: float,
    seed: int = 0,
    with_replacement: bool = False,
) -> tuple[list[int], list[int]]:
    """Split a training budget between LESS-selected data and a random replay buffer.

    ``num_less_candidates`` is assumed to be the length of an already-ranked LESS
    subset (highest score first), so the LESS share is simply its first
    ``round(subset_budget * k / 100)`` indices. The remaining share is drawn
    uniformly at random from a reference/replay pool of size
    ``num_reference_candidates``, independent of the LESS ranking, which is the
    standard replay-based continual-learning recipe: mix task-relevant selected
    data with samples from previously-seen/reference data to curb forgetting.

    Returns (less_indices, reference_indices).
    """
    if subset_budget <= 0:
        raise ValueError("subset_budget must be positive.")
    if not (0.0 <= k <= 100.0):
        raise ValueError("k must be a percentage in [0, 100].")

    less_budget = min(num_less_candidates, round(subset_budget * k / 100.0))
    replay_budget = subset_budget - less_budget
    if replay_budget > num_reference_candidates and not with_replacement:
        raise ValueError(
            f"Need {replay_budget} replay examples but the reference pool only has "
            f"{num_reference_candidates}. Lower k, lower subset_budget, or pass with_replacement=True."
        )

    less_indices = list(range(less_budget))

    rng = random.Random(int(seed))
    if with_replacement:
        reference_indices = [rng.randrange(num_reference_candidates) for _ in range(replay_budget)]
    else:
        reference_indices = rng.sample(range(num_reference_candidates), replay_budget)

    return less_indices, reference_indices
