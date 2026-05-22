
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .safe_drift_update import (
    compute_update_costs,
    delta_theta_star_from_cost_norm_budgets,
    delta_theta_star_from_lambda_alpha,
)


@dataclass
class CandidateScore:
    index: int
    rank_score: float
    singleton_objective: float


@dataclass
class SafeSubsetSelectionResult:
    selected_indices: list[int]
    selected_candidates: list[Any]
    safe_update: dict[str, float | str | torch.Tensor]
    subset_update: torch.Tensor
    subset_gradient: torch.Tensor
    subset_budget: int
    learning_rate: float
    alpha: float
    geometry: str
    solver: str
    average_by_budget: bool
    shortlist_size: int
    objective_value: float
    reference_cost: float
    norm_cost: float
    predicted_target_gain: float
    candidate_scores: list[CandidateScore]
    shortlist_indices: list[int]


def _as_fisher_tensor(reference: torch.Tensor, fisher) -> torch.Tensor:
    fisher_t = torch.as_tensor(fisher, device=reference.device, dtype=reference.dtype)
    if fisher_t.ndim == 1:
        if fisher_t.shape != reference.shape:
            raise ValueError("Diagonal Fisher vector must match the update dimension.")
        return fisher_t
    if fisher_t.ndim == 2:
        if fisher_t.shape[0] != fisher_t.shape[1]:
            raise ValueError("Full Fisher matrix must be square.")
        if fisher_t.shape[0] != reference.numel():
            raise ValueError("Full Fisher matrix size must match the update dimension.")
        return fisher_t
    raise ValueError("fisher must be a 1D diagonal vector or a 2D square matrix.")


def _whiten_vectors(
    vectors: torch.Tensor,
    fisher,
    geometry: str,
    clamp_min: float = 1.0e-8,
) -> torch.Tensor:
    """
    Apply the square-root metric transform.

    If geometry == "euclidean", this is the identity.
    If geometry == "reference", this applies F_R^{1/2}.

    vectors can be shape [d] or [n, d].
    """
    if geometry not in {"euclidean", "reference"}:
        raise ValueError("geometry must be one of: euclidean, reference.")

    if geometry == "euclidean":
        return vectors

    fisher_t = _as_fisher_tensor(vectors[-1] if vectors.ndim == 2 else vectors, fisher)
    if fisher_t.ndim == 1:
        sqrt_fisher = fisher_t.clamp_min(clamp_min).sqrt()
        return vectors * sqrt_fisher if vectors.ndim == 1 else vectors * sqrt_fisher.unsqueeze(0)

    fisher_sym = 0.5 * (fisher_t + fisher_t.T)
    evals, evecs = torch.linalg.eigh(fisher_sym)
    evals = evals.clamp_min(clamp_min).sqrt()
    sqrt_fisher = (evecs * evals.unsqueeze(0)) @ evecs.T
    if vectors.ndim == 1:
        return sqrt_fisher @ vectors
    return vectors @ sqrt_fisher.T


def solve_budgeted_safe_update(
    target_gradient: torch.Tensor,
    fisher,
    alpha: float | None,
    cost_c: float | None = None,
    epsilon: float | None = None,
    clamp_min: float = 1.0e-8,
) -> dict[str, float | str | torch.Tensor]:
    """
    Compute the continuous safe update
        Delta* = -(1 / lambda) (F_R + alpha I)^{-1} g_T
    and choose lambda to satisfy the optional reference-cost and norm budgets.
    """
    if alpha is None:
        if cost_c is None or epsilon is None:
            raise ValueError("alpha=None requires both cost_c/rho and epsilon.")
        return delta_theta_star_from_cost_norm_budgets(
            target_grad=target_gradient,
            fisher=fisher,
            rho=float(cost_c),
            epsilon=float(epsilon),
            clamp_min=clamp_min,
        )

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if cost_c is not None and cost_c <= 0.0:
        raise ValueError("cost_c must be positive when provided.")
    if epsilon is not None and epsilon <= 0.0:
        raise ValueError("epsilon must be positive when provided.")

    base_solution = delta_theta_star_from_lambda_alpha(
        target_grad=target_gradient,
        fisher=fisher,
        lambda_value=1.0,
        alpha=alpha,
        clamp_min=clamp_min,
    )

    lambda_candidates = [1.0]
    if cost_c is not None:
        lambda_candidates.append(math.sqrt(base_solution["reference_cost"] / float(cost_c)))
    if epsilon is not None:
        norm_budget = 0.5 * float(epsilon) * float(epsilon)
        lambda_candidates.append(math.sqrt(base_solution["norm_cost"] / norm_budget))

    lambda_value = max(lambda_candidates)
    solution = delta_theta_star_from_lambda_alpha(
        target_grad=target_gradient,
        fisher=fisher,
        lambda_value=lambda_value,
        alpha=alpha,
        clamp_min=clamp_min,
    )
    solution["cost_c"] = None if cost_c is None else float(cost_c)
    solution["epsilon"] = None if epsilon is None else float(epsilon)
    return solution


def _prepare_dictionary(
    candidate_gradients: torch.Tensor,
    learning_rate: float,
    subset_budget: int,
    average_by_budget: bool,
) -> torch.Tensor:
    """
    Build the dictionary A = [a_1, ..., a_n] in row-major form [n, d].

    Each atom starts as the one-step update u_i = -eta g_i.
    This selector-side eta only scales the candidate update atoms. It does not
    change the continuous safe target update Delta*, which is solved upstream
    from (target_gradient, fisher, alpha, lambda).
    If average_by_budget is True, we absorb the 1/k factor into each atom so
    that the selected update is A^T w = (1/k) sum_{i:w_i=1} u_i.
    Otherwise the selected update is sum_{i:w_i=1} u_i.
    """
    atoms = -float(learning_rate) * candidate_gradients
    if average_by_budget:
        atoms = atoms / float(subset_budget)
    return atoms


def _compute_rank_scores(
    atoms_white: torch.Tensor,
    delta_white: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rank score for singleton atoms:
        score_j = <a_j, delta> - 0.5 ||a_j||^2
    which is equivalent to minimizing the singleton objective
        0.5 ||a_j - delta||^2
    up to a constant independent of j.
    """
    dot_to_target = atoms_white @ delta_white
    atom_norm_sq = torch.sum(atoms_white.square(), dim=1)
    rank_scores = dot_to_target - 0.5 * atom_norm_sq
    singleton_objectives = 0.5 * atom_norm_sq - dot_to_target
    return rank_scores, singleton_objectives


def _greedy_marginal_selection(
    atoms_white: torch.Tensor,
    delta_white: torch.Tensor,
    candidate_indices: torch.Tensor,
    subset_budget: int,
) -> list[int]:
    """
    Forward greedy on the whitened objective
        0.5 ||A w - delta||^2
    with exact cardinality |S| = subset_budget.

    If the current selected atom-sum is s = sum_{i in S} a_i, then adding j
    changes the objective by
        Delta J(j | S) = 0.5 ||a_j||^2 + <s, a_j> - <delta, a_j>.
    """
    if subset_budget > int(candidate_indices.numel()):
        raise ValueError("subset_budget cannot exceed the shortlist size.")

    atom_norm_sq = torch.sum(atoms_white.square(), dim=1)
    delta_dots = atoms_white @ delta_white

    remaining = candidate_indices.clone()
    selected: list[int] = []
    current_sum = torch.zeros_like(delta_white)

    for _ in range(subset_budget):
        if remaining.numel() == 0:
            raise RuntimeError("Ran out of shortlist candidates before filling the subset budget.")

        current_dots = atoms_white[remaining] @ current_sum
        marginals = 0.5 * atom_norm_sq[remaining] + current_dots - delta_dots[remaining]
        best_pos = int(torch.argmin(marginals).item())
        best_idx = int(remaining[best_pos].item())

        selected.append(best_idx)
        current_sum = current_sum + atoms_white[best_idx]

        if remaining.numel() == 1:
            remaining = remaining.new_empty((0,), dtype=remaining.dtype)
        else:
            mask = torch.ones_like(remaining, dtype=torch.bool)
            mask[best_pos] = False
            remaining = remaining[mask]

    return selected


def select_safe_subset_from_gradients(
    candidate_gradients: torch.Tensor,
    candidates: list[Any],
    target_gradient: torch.Tensor,
    fisher,
    subset_budget: int,
    alpha: float | None,
    learning_rate: float = 1.0,
    cost_c: float | None = None,
    epsilon: float | None = None,
    geometry: str = "reference",
    solver: str = "greedy_marginal",
    shortlist_size: int | None = None,
    average_by_budget: bool = True,
    preconditioner: torch.Tensor | None = None,
    clamp_min: float = 1.0e-8,
) -> SafeSubsetSelectionResult:
    """
    Select an exact-cardinality binary subset to match the continuous safe update.

    Supported solvers:
      - solver="rank":
          pick the top-k singleton atoms according to
              score_j = <a_j, delta>_G - 0.5 ||a_j||_G^2.
          This is the cheapest baseline and ignores pairwise interactions.

      - solver="greedy_marginal":
          first shortlist by the same rank score, then run forward greedy using
          the true marginal of the whitened quadratic objective. This is a
          practical approximation to the exact binary objective.

    Supported geometries:
      - geometry="euclidean":
            min_w 0.5 ||A w - Delta*||_2^2
      - geometry="reference":
            min_w 0.5 (A w - Delta*)^T F_R (A w - Delta*)
        implemented by whitening with F_R^{1/2}.

    If average_by_budget is True, atoms are u_i / k and the selected update is
        (1 / k) sum_{i in S} u_i.
    If average_by_budget is False, atoms are u_i and the selected update is
        sum_{i in S} u_i.
    """
    if candidate_gradients.ndim != 2:
        raise ValueError("candidate_gradients must be a 2D tensor [num_examples, gradient_dim].")
    if target_gradient.ndim != 1:
        raise ValueError("target_gradient must be a 1D tensor.")
    if candidate_gradients.shape[0] != len(candidates):
        raise ValueError("candidate_gradients must have one row per candidate.")
    if candidate_gradients.shape[1] != target_gradient.numel():
        raise ValueError("candidate_gradients width must match target_gradient dimension.")
    if subset_budget <= 0:
        raise ValueError("subset_budget must be positive.")
    if subset_budget > candidate_gradients.shape[0]:
        raise ValueError("subset_budget cannot exceed the number of candidates.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if geometry not in {"euclidean", "reference"}:
        raise ValueError("geometry must be one of: euclidean, reference.")
    if solver not in {"rank", "greedy_marginal"}:
        raise ValueError("solver must be one of: rank, greedy_marginal.")

    candidate_gradients = candidate_gradients.float()
    target_gradient = target_gradient.float()
    if preconditioner is not None:
        preconditioner = torch.as_tensor(
            preconditioner,
            device=candidate_gradients.device,
            dtype=candidate_gradients.dtype,
        )
        if preconditioner.ndim == 1:
            if preconditioner.numel() != target_gradient.numel():
                raise ValueError("1D preconditioner must have the same dimension as target_gradient.")
            candidate_gradients = candidate_gradients * preconditioner.unsqueeze(0)
        elif preconditioner.ndim == 2:
            if preconditioner.shape != (target_gradient.numel(), target_gradient.numel()):
                raise ValueError("2D preconditioner must have shape [gradient_dim, gradient_dim].")
            candidate_gradients = candidate_gradients @ preconditioner.T
        else:
            raise ValueError("preconditioner must be either a 1D diagonal vector or a 2D matrix.")

    safe_update = solve_budgeted_safe_update(
        target_gradient=target_gradient,
        fisher=fisher,
        alpha=alpha,
        cost_c=cost_c,
        epsilon=epsilon,
        clamp_min=clamp_min,
    )
    target_delta = safe_update["delta"]
    if not isinstance(target_delta, torch.Tensor):
        raise TypeError("safe_update['delta'] must be a torch.Tensor.")

    atoms = _prepare_dictionary(
        candidate_gradients=candidate_gradients,
        learning_rate=learning_rate,
        subset_budget=subset_budget,
        average_by_budget=average_by_budget,
    )

    atoms_white = _whiten_vectors(
        vectors=atoms,
        fisher=fisher,
        geometry=geometry,
        clamp_min=clamp_min,
    )
    delta_white = _whiten_vectors(
        vectors=target_delta,
        fisher=fisher,
        geometry=geometry,
        clamp_min=clamp_min,
    )

    rank_scores_t, singleton_objectives_t = _compute_rank_scores(
        atoms_white=atoms_white,
        delta_white=delta_white,
    )
    ranking = [
        CandidateScore(
            index=int(i),
            rank_score=float(rank_scores_t[i].item()),
            singleton_objective=float(singleton_objectives_t[i].item()),
        )
        for i in range(candidate_gradients.shape[0])
    ]
    ranking.sort(key=lambda row: row.rank_score, reverse=True)

    if shortlist_size is None:
        shortlist_size = candidate_gradients.shape[0]
    shortlist_size = max(int(shortlist_size), int(subset_budget))
    shortlist_size = min(shortlist_size, candidate_gradients.shape[0])

    shortlist_indices = [row.index for row in ranking[:shortlist_size]]
    shortlist_tensor = torch.tensor(
        shortlist_indices,
        device=candidate_gradients.device,
        dtype=torch.long,
    )

    if solver == "rank":
        selected_indices = shortlist_indices[:subset_budget]
    else:
        selected_indices = _greedy_marginal_selection(
            atoms_white=atoms_white,
            delta_white=delta_white,
            candidate_indices=shortlist_tensor,
            subset_budget=subset_budget,
        )

    selected_tensor = torch.tensor(
        selected_indices,
        device=candidate_gradients.device,
        dtype=torch.long,
    )
    selected_atoms = atoms[selected_tensor]
    subset_update = selected_atoms.sum(dim=0)

    if average_by_budget:
        # atoms already include the 1 / k factor, so subset_update is already
        # the final averaged update.
        subset_gradient = -subset_update / float(learning_rate)
    else:
        subset_gradient = -subset_update / float(learning_rate)

    final_costs = compute_update_costs(subset_update, fisher)
    predicted_target_gain = -torch.dot(target_gradient, subset_update)

    residual_white = atoms_white[selected_tensor].sum(dim=0) - delta_white
    objective_value = 0.5 * torch.dot(residual_white, residual_white)

    return SafeSubsetSelectionResult(
        selected_indices=selected_indices,
        selected_candidates=[candidates[idx] for idx in selected_indices],
        safe_update=safe_update,
        subset_update=subset_update,
        subset_gradient=subset_gradient,
        subset_budget=int(subset_budget),
        learning_rate=float(learning_rate),
        alpha=float(safe_update["alpha"]),
        geometry=geometry,
        solver=solver,
        average_by_budget=bool(average_by_budget),
        shortlist_size=int(shortlist_size),
        objective_value=float(objective_value.item()),
        reference_cost=float(final_costs["reference_cost"]),
        norm_cost=float(final_costs["norm_cost"]),
        predicted_target_gain=float(predicted_target_gain.item()),
        candidate_scores=ranking,
        shortlist_indices=shortlist_indices,
    )
