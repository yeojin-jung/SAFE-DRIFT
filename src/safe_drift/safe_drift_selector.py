
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
    selection_weights: list[float]
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
    solver_status: str
    optimization_trace: list[dict[str, float | int | str | bool | None]]


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
    alpha: float = 0.0,
    clamp_min: float = 1.0e-8,
) -> torch.Tensor:
    """
    Apply the square-root metric transform.

    If geometry == "euclidean", this is the identity.
    If geometry == "reference", this applies (F_R + alpha I)^{1/2}.

    vectors can be shape [d] or [n, d].
    """
    if geometry not in {"euclidean", "reference"}:
        raise ValueError("geometry must be one of: euclidean, reference.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    if geometry == "euclidean":
        return vectors

    fisher_t = _as_fisher_tensor(vectors[-1] if vectors.ndim == 2 else vectors, fisher)
    if fisher_t.ndim == 1:
        sqrt_fisher = (fisher_t.clamp_min(0.0) + float(alpha)).clamp_min(clamp_min).sqrt()
        return vectors * sqrt_fisher if vectors.ndim == 1 else vectors * sqrt_fisher.unsqueeze(0)

    fisher_sym = 0.5 * (fisher_t + fisher_t.T)
    evals, evecs = torch.linalg.eigh(fisher_sym)
    evals = (evals.clamp_min(0.0) + float(alpha)).clamp_min(clamp_min).sqrt()
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


def _constraint_tolerance(budget: float | None, tolerance: float) -> float:
    if budget is None:
        return 0.0
    return max(1.0e-12, abs(float(budget)) * float(tolerance))


def _trace_diagnostics(
    *,
    objective: float,
    reference_cost: float,
    norm_cost: float,
    reference_budget: float | None,
    norm_budget: float | None,
    feasibility_tolerance: float,
) -> dict[str, float | bool | None]:
    reference_tol = _constraint_tolerance(reference_budget, feasibility_tolerance)
    norm_tol = _constraint_tolerance(norm_budget, feasibility_tolerance)
    return {
        "approximation_error": math.sqrt(max(0.0, 2.0 * float(objective))),
        "approximation_error_squared": max(0.0, 2.0 * float(objective)),
        "reference_budget": reference_budget,
        "reference_budget_ratio": (
            None
            if reference_budget is None
            else float(reference_cost) / max(float(reference_budget), 1.0e-30)
        ),
        "reference_budget_satisfied": (
            None
            if reference_budget is None
            else bool(float(reference_cost) <= float(reference_budget) + reference_tol)
        ),
        "norm_budget": norm_budget,
        "norm_budget_ratio": (
            None if norm_budget is None else float(norm_cost) / max(float(norm_budget), 1.0e-30)
        ),
        "norm_budget_satisfied": (
            None
            if norm_budget is None
            else bool(float(norm_cost) <= float(norm_budget) + norm_tol)
        ),
    }


def _selected_prefix_trace(
    *,
    atoms: torch.Tensor,
    atoms_match: torch.Tensor,
    delta_match: torch.Tensor,
    atoms_reference: torch.Tensor,
    selected_indices: list[int],
    selection_weights: list[float],
    reference_budget: float | None,
    epsilon: float | None,
    feasibility_tolerance: float,
) -> list[dict[str, float | int | str | bool | None]]:
    """Audit each prefix of a selector that does not enforce constraints internally."""
    norm_budget = None if epsilon is None else 0.5 * float(epsilon) * float(epsilon)
    current_update = torch.zeros_like(atoms[0])
    current_match = torch.zeros_like(delta_match)
    current_reference = torch.zeros_like(atoms_reference[0])
    initial_objective = float(0.5 * torch.dot(delta_match, delta_match).item())
    trace: list[dict[str, float | int | str | bool | None]] = [
        {
            "iteration": 0,
            "action": "start",
            "selected_count": 0,
            "weight_sum": 0.0,
            "objective": initial_objective,
            "reference_cost": 0.0,
            "norm_cost": 0.0,
            **_trace_diagnostics(
                objective=initial_objective,
                reference_cost=0.0,
                norm_cost=0.0,
                reference_budget=reference_budget,
                norm_budget=norm_budget,
                feasibility_tolerance=feasibility_tolerance,
            ),
        }
    ]
    weight_sum = 0.0
    for iteration, (candidate_index, weight) in enumerate(
        zip(selected_indices, selection_weights, strict=True),
        start=1,
    ):
        weight = float(weight)
        weight_sum += weight
        current_update = current_update + weight * atoms[candidate_index]
        current_match = current_match + weight * atoms_match[candidate_index]
        current_reference = (
            current_reference + weight * atoms_reference[candidate_index]
        )
        residual = current_match - delta_match
        objective = float(0.5 * torch.dot(residual, residual).item())
        reference_cost = float(
            0.5 * torch.dot(current_reference, current_reference).item()
        )
        norm_cost = float(0.5 * torch.dot(current_update, current_update).item())
        trace.append(
            {
                "iteration": iteration,
                "action": "prefix_add",
                "candidate_index": int(candidate_index),
                "candidate_weight": weight,
                "selected_count": iteration,
                "weight_sum": weight_sum,
                "objective": objective,
                "reference_cost": reference_cost,
                "norm_cost": norm_cost,
                **_trace_diagnostics(
                    objective=objective,
                    reference_cost=reference_cost,
                    norm_cost=norm_cost,
                    reference_budget=reference_budget,
                    norm_budget=norm_budget,
                    feasibility_tolerance=feasibility_tolerance,
                ),
            }
        )
    return trace


def _constrained_greedy_selection(
    *,
    atoms: torch.Tensor,
    atoms_match: torch.Tensor,
    delta_match: torch.Tensor,
    atoms_reference: torch.Tensor,
    candidate_indices: torch.Tensor,
    subset_budget: int,
    reference_budget: float | None,
    epsilon: float | None,
    feasibility_tolerance: float,
    objective_tolerance: float,
    swap_passes: int,
) -> tuple[list[int], list[dict[str, float | int | str | bool | None]], str]:
    """Forward greedy with hard update budgets and an at-most-k cardinality."""
    if subset_budget > int(candidate_indices.numel()):
        raise ValueError("subset_budget cannot exceed the shortlist size.")
    if swap_passes < 0:
        raise ValueError("swap_passes must be non-negative.")

    norm_budget = None if epsilon is None else 0.5 * float(epsilon) * float(epsilon)
    reference_tol = _constraint_tolerance(reference_budget, feasibility_tolerance)
    norm_tol = _constraint_tolerance(norm_budget, feasibility_tolerance)

    selected: list[int] = []
    remaining = candidate_indices.clone()
    current_update = torch.zeros_like(atoms[0])
    current_match = torch.zeros_like(delta_match)
    current_reference = torch.zeros_like(atoms_reference[0])
    current_objective = float(0.5 * torch.dot(delta_match, delta_match).item())
    trace: list[dict[str, float | int | str | bool | None]] = [
        {
            "iteration": 0,
            "action": "start",
            "selected_count": 0,
            "objective": current_objective,
            "reference_cost": 0.0,
            "norm_cost": 0.0,
            **_trace_diagnostics(
                objective=current_objective,
                reference_cost=0.0,
                norm_cost=0.0,
                reference_budget=reference_budget,
                norm_budget=norm_budget,
                feasibility_tolerance=feasibility_tolerance,
            ),
        }
    ]
    status = "budget_reached"

    for iteration in range(1, subset_budget + 1):
        if remaining.numel() == 0:
            status = "shortlist_exhausted"
            break

        trial_match = current_match.unsqueeze(0) + atoms_match[remaining]
        residual = trial_match - delta_match.unsqueeze(0)
        objectives = 0.5 * torch.sum(residual.square(), dim=1)

        trial_reference = current_reference.unsqueeze(0) + atoms_reference[remaining]
        reference_costs = 0.5 * torch.sum(trial_reference.square(), dim=1)
        trial_updates = current_update.unsqueeze(0) + atoms[remaining]
        norm_costs = 0.5 * torch.sum(trial_updates.square(), dim=1)

        feasible = torch.ones_like(objectives, dtype=torch.bool)
        if reference_budget is not None:
            feasible &= reference_costs <= float(reference_budget) + reference_tol
        if norm_budget is not None:
            feasible &= norm_costs <= float(norm_budget) + norm_tol

        feasible_positions = torch.nonzero(feasible, as_tuple=False).flatten()
        if feasible_positions.numel() == 0:
            status = "no_feasible_addition"
            break

        feasible_objectives = objectives[feasible_positions]
        best_feasible_position = int(torch.argmin(feasible_objectives).item())
        best_pos = int(feasible_positions[best_feasible_position].item())
        best_objective = float(objectives[best_pos].item())
        if best_objective >= current_objective - float(objective_tolerance):
            status = "no_improving_feasible_addition"
            break

        best_idx = int(remaining[best_pos].item())
        selected.append(best_idx)
        current_update = trial_updates[best_pos]
        current_match = trial_match[best_pos]
        current_reference = trial_reference[best_pos]
        current_objective = best_objective
        trace.append(
            {
                "iteration": iteration,
                "action": "add",
                "candidate_index": best_idx,
                "selected_count": len(selected),
                "objective": current_objective,
                "reference_cost": float(reference_costs[best_pos].item()),
                "norm_cost": float(norm_costs[best_pos].item()),
                "feasible_candidate_count": int(feasible_positions.numel()),
                **_trace_diagnostics(
                    objective=current_objective,
                    reference_cost=float(reference_costs[best_pos].item()),
                    norm_cost=float(norm_costs[best_pos].item()),
                    reference_budget=reference_budget,
                    norm_budget=norm_budget,
                    feasibility_tolerance=feasibility_tolerance,
                ),
            }
        )

        mask = torch.ones_like(remaining, dtype=torch.bool)
        mask[best_pos] = False
        remaining = remaining[mask]
    else:
        status = "budget_reached"

    # A small feasible swap search repairs common forward-greedy mistakes
    # without changing the selected cardinality.
    for swap_pass in range(swap_passes):
        if not selected or remaining.numel() == 0:
            break
        best_swap: tuple[int, int, int, float, torch.Tensor, torch.Tensor, torch.Tensor, float, float] | None = None
        for selected_pos, removed_idx in enumerate(selected):
            base_update = current_update - atoms[removed_idx]
            base_match = current_match - atoms_match[removed_idx]
            base_reference = current_reference - atoms_reference[removed_idx]

            trial_match = base_match.unsqueeze(0) + atoms_match[remaining]
            residual = trial_match - delta_match.unsqueeze(0)
            objectives = 0.5 * torch.sum(residual.square(), dim=1)
            trial_reference = base_reference.unsqueeze(0) + atoms_reference[remaining]
            reference_costs = 0.5 * torch.sum(trial_reference.square(), dim=1)
            trial_updates = base_update.unsqueeze(0) + atoms[remaining]
            norm_costs = 0.5 * torch.sum(trial_updates.square(), dim=1)

            feasible = torch.ones_like(objectives, dtype=torch.bool)
            if reference_budget is not None:
                feasible &= reference_costs <= float(reference_budget) + reference_tol
            if norm_budget is not None:
                feasible &= norm_costs <= float(norm_budget) + norm_tol
            improving = feasible & (objectives < current_objective - float(objective_tolerance))
            positions = torch.nonzero(improving, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            local_pos = int(positions[torch.argmin(objectives[positions])].item())
            local_objective = float(objectives[local_pos].item())
            if best_swap is None or local_objective < best_swap[3]:
                best_swap = (
                    selected_pos,
                    removed_idx,
                    local_pos,
                    local_objective,
                    trial_updates[local_pos],
                    trial_match[local_pos],
                    trial_reference[local_pos],
                    float(reference_costs[local_pos].item()),
                    float(norm_costs[local_pos].item()),
                )

        if best_swap is None:
            break

        selected_pos, removed_idx, remaining_pos, current_objective, current_update, current_match, current_reference, ref_cost, norm_cost = best_swap
        added_idx = int(remaining[remaining_pos].item())
        selected[selected_pos] = added_idx
        remaining[remaining_pos] = removed_idx
        trace.append(
            {
                "iteration": len(trace),
                "action": "swap",
                "swap_pass": swap_pass + 1,
                "removed_candidate_index": removed_idx,
                "candidate_index": added_idx,
                "selected_count": len(selected),
                "objective": current_objective,
                "reference_cost": ref_cost,
                "norm_cost": norm_cost,
                **_trace_diagnostics(
                    objective=current_objective,
                    reference_cost=ref_cost,
                    norm_cost=norm_cost,
                    reference_budget=reference_budget,
                    norm_budget=norm_budget,
                    feasibility_tolerance=feasibility_tolerance,
                ),
            }
        )

    return selected, trace, status


def _radially_enforce_relaxed_budgets(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    atoms_reference: torch.Tensor,
    subset_budget: int,
    reference_budget: float | None,
    epsilon: float | None,
) -> torch.Tensor:
    """Remove small numerical violations while preserving box feasibility."""
    weights = weights.clamp(min=0.0, max=1.0)
    scales = [1.0]
    weight_sum = float(weights.sum().item())
    if weight_sum > float(subset_budget):
        scales.append(float(subset_budget) / weight_sum)

    reference_update = weights @ atoms_reference
    reference_cost = float(0.5 * torch.dot(reference_update, reference_update).item())
    if reference_budget is not None and reference_cost > float(reference_budget) and reference_cost > 0.0:
        scales.append(math.sqrt(float(reference_budget) / reference_cost))

    update = weights @ atoms
    norm_cost = float(0.5 * torch.dot(update, update).item())
    norm_budget = None if epsilon is None else 0.5 * float(epsilon) * float(epsilon)
    if norm_budget is not None and norm_cost > norm_budget and norm_cost > 0.0:
        scales.append(math.sqrt(norm_budget / norm_cost))

    scale = min(scales)
    if scale < 1.0:
        weights = weights * (scale * (1.0 - 1.0e-9))
    return weights


def _relaxed_selection(
    *,
    atoms: torch.Tensor,
    atoms_match: torch.Tensor,
    delta_match: torch.Tensor,
    atoms_reference: torch.Tensor,
    candidate_indices: torch.Tensor,
    subset_budget: int,
    reference_budget: float | None,
    epsilon: float | None,
    cvx_solver: str,
    weight_threshold: float,
) -> tuple[list[int], list[float], list[dict[str, float | int | str | bool | None]], str]:
    """Solve the continuous w in [0, 1] convex relaxation with CVXPY."""
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "solver='relaxed' requires cvxpy and clarabel; install requirements/lambda-train.txt."
        ) from exc

    shortlist = candidate_indices.detach().cpu()
    atoms_short = atoms[shortlist].detach().double().cpu()
    match_short = atoms_match[shortlist].detach().double().cpu()
    reference_short = atoms_reference[shortlist].detach().double().cpu()
    delta_np = delta_match.detach().double().cpu().numpy()

    num_candidates = int(shortlist.numel())
    weights_var = cp.Variable(num_candidates)
    match_update = match_short.numpy().T @ weights_var
    objective = cp.Minimize(0.5 * cp.sum_squares(match_update - delta_np))
    constraints = [
        weights_var >= 0.0,
        weights_var <= 1.0,
        cp.sum(weights_var) <= float(subset_budget),
    ]
    if reference_budget is not None:
        constraints.append(
            0.5 * cp.sum_squares(reference_short.numpy().T @ weights_var)
            <= float(reference_budget)
        )
    if epsilon is not None:
        constraints.append(cp.sum_squares(atoms_short.numpy().T @ weights_var) <= float(epsilon) ** 2)

    requested_solver = str(cvx_solver).upper()
    installed = {str(name).upper() for name in cp.installed_solvers()}
    if requested_solver not in installed:
        raise RuntimeError(
            f"Requested relaxed solver {requested_solver!r} is not installed. "
            f"Available CVXPY solvers: {sorted(installed)}"
        )

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=requested_solver, verbose=False)
    acceptable_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if problem.status not in acceptable_statuses or weights_var.value is None:
        raise RuntimeError(f"Relaxed SAFE selector failed with CVXPY status {problem.status!r}.")

    weights = torch.as_tensor(
        np.asarray(weights_var.value, dtype=np.float64),
        device=atoms_short.device,
        dtype=atoms_short.dtype,
    )
    weights = torch.where(weights > float(weight_threshold), weights, torch.zeros_like(weights))
    weights = _radially_enforce_relaxed_budgets(
        weights=weights,
        atoms=atoms_short,
        atoms_reference=reference_short,
        subset_budget=subset_budget,
        reference_budget=reference_budget,
        epsilon=epsilon,
    )
    support_positions = torch.nonzero(weights > 0.0, as_tuple=False).flatten()
    selected_indices = [int(shortlist[pos].item()) for pos in support_positions]
    selected_weights = [float(weights[pos].item()) for pos in support_positions]

    update = weights @ atoms_short
    reference_update = weights @ reference_short
    match_update_t = weights @ match_short
    objective_value = float(0.5 * torch.sum((match_update_t - delta_match).square()).item())
    reference_cost = float(0.5 * torch.dot(reference_update, reference_update).item())
    norm_cost = float(0.5 * torch.dot(update, update).item())
    norm_budget = None if epsilon is None else 0.5 * float(epsilon) * float(epsilon)
    trace: list[dict[str, float | int | str | bool | None]] = [
        {
            "iteration": 0,
            "action": "relaxed_solution",
            "selected_count": len(selected_indices),
            "weight_sum": float(weights.sum().item()),
            "objective": objective_value,
            "reference_cost": reference_cost,
            "norm_cost": norm_cost,
            "cvxpy_objective": float(problem.value),
            "cvxpy_status": str(problem.status),
            **_trace_diagnostics(
                objective=objective_value,
                reference_cost=reference_cost,
                norm_cost=norm_cost,
                reference_budget=reference_budget,
                norm_budget=norm_budget,
                feasibility_tolerance=1.0e-6,
            ),
        }
    ]
    return selected_indices, selected_weights, trace, str(problem.status)


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
    feasibility_tolerance: float = 1.0e-6,
    objective_tolerance: float = 1.0e-12,
    greedy_swap_passes: int = 1,
    relaxed_cvx_solver: str = "CLARABEL",
    relaxed_weight_threshold: float = 1.0e-8,
    clamp_min: float = 1.0e-8,
) -> SafeSubsetSelectionResult:
    """
    Select candidate updates to match the continuous safe update.

    Supported solvers:
      - solver="rank":
          pick the top-k singleton atoms according to
              score_j = <a_j, delta>_G - 0.5 ||a_j||_G^2.
          This is the cheapest baseline and ignores pairwise interactions.

      - solver="greedy_marginal":
          first shortlist by the same rank score, then run forward greedy using
          the true marginal of the whitened quadratic objective. This is a
          practical approximation to the exact binary objective.

      - solver="constrained_greedy":
          run at-most-k forward greedy, accepting only additions that satisfy
          the reference and norm budgets and improve the matching objective.

      - solver="relaxed":
          solve the convex relaxation with 0 <= w_i <= 1, sum_i w_i <= k, and
          the same reference and norm budgets. Fractional weights are returned.

    Supported geometries:
      - geometry="euclidean":
            min_w 0.5 ||A w - Delta*||_2^2
      - geometry="reference":
            min_w 0.5 (A w - Delta*)^T (F_R + alpha I) (A w - Delta*)
        implemented by whitening with (F_R + alpha I)^{1/2}.

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
    supported_solvers = {"rank", "greedy_marginal", "constrained_greedy", "relaxed"}
    if solver not in supported_solvers:
        raise ValueError(f"solver must be one of: {', '.join(sorted(supported_solvers))}.")
    if feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be non-negative.")
    if objective_tolerance < 0.0:
        raise ValueError("objective_tolerance must be non-negative.")
    if relaxed_weight_threshold < 0.0:
        raise ValueError("relaxed_weight_threshold must be non-negative.")

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
    resolved_alpha = float(safe_update["alpha"])

    atoms_white = _whiten_vectors(
        vectors=atoms,
        fisher=fisher,
        geometry=geometry,
        alpha=resolved_alpha if geometry == "reference" else 0.0,
        clamp_min=clamp_min,
    )
    delta_white = _whiten_vectors(
        vectors=target_delta,
        fisher=fisher,
        geometry=geometry,
        alpha=resolved_alpha if geometry == "reference" else 0.0,
        clamp_min=clamp_min,
    )
    atoms_reference = _whiten_vectors(
        vectors=atoms,
        fisher=fisher,
        geometry="reference",
        alpha=0.0,
        clamp_min=0.0,
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

    optimization_trace: list[dict[str, float | int | str | bool | None]] = []
    solver_status = "completed"
    if solver == "rank":
        selected_indices = shortlist_indices[:subset_budget]
        selection_weights = [1.0] * len(selected_indices)
    elif solver == "greedy_marginal":
        selected_indices = _greedy_marginal_selection(
            atoms_white=atoms_white,
            delta_white=delta_white,
            candidate_indices=shortlist_tensor,
            subset_budget=subset_budget,
        )
        selection_weights = [1.0] * len(selected_indices)
    elif solver == "constrained_greedy":
        selected_indices, optimization_trace, solver_status = _constrained_greedy_selection(
            atoms=atoms,
            atoms_match=atoms_white,
            delta_match=delta_white,
            atoms_reference=atoms_reference,
            candidate_indices=shortlist_tensor,
            subset_budget=subset_budget,
            reference_budget=cost_c,
            epsilon=epsilon,
            feasibility_tolerance=feasibility_tolerance,
            objective_tolerance=objective_tolerance,
            swap_passes=greedy_swap_passes,
        )
        selection_weights = [1.0] * len(selected_indices)
    else:
        selected_indices, selection_weights, optimization_trace, solver_status = _relaxed_selection(
            atoms=atoms,
            atoms_match=atoms_white,
            delta_match=delta_white,
            atoms_reference=atoms_reference,
            candidate_indices=shortlist_tensor,
            subset_budget=subset_budget,
            reference_budget=cost_c,
            epsilon=epsilon,
            cvx_solver=relaxed_cvx_solver,
            weight_threshold=relaxed_weight_threshold,
        )

    if solver in {"rank", "greedy_marginal"}:
        optimization_trace = _selected_prefix_trace(
            atoms=atoms,
            atoms_match=atoms_white,
            delta_match=delta_white,
            atoms_reference=atoms_reference,
            selected_indices=selected_indices,
            selection_weights=selection_weights,
            reference_budget=cost_c,
            epsilon=epsilon,
            feasibility_tolerance=feasibility_tolerance,
        )

    selected_tensor = torch.tensor(
        selected_indices,
        device=candidate_gradients.device,
        dtype=torch.long,
    )
    if selected_indices:
        selected_atoms = atoms[selected_tensor]
        selected_weights_t = torch.tensor(
            selection_weights,
            device=candidate_gradients.device,
            dtype=atoms.dtype,
        )
        subset_update = selected_weights_t @ selected_atoms
    else:
        subset_update = torch.zeros_like(target_delta)

    if average_by_budget:
        # atoms already include the 1 / k factor, so subset_update is already
        # the final averaged update.
        subset_gradient = -subset_update / float(learning_rate)
    else:
        subset_gradient = -subset_update / float(learning_rate)

    final_costs = compute_update_costs(subset_update, fisher)
    predicted_target_gain = -torch.dot(target_gradient, subset_update)

    if selected_indices:
        residual_white = selected_weights_t @ atoms_white[selected_tensor] - delta_white
    else:
        residual_white = -delta_white
    objective_value = 0.5 * torch.dot(residual_white, residual_white)

    return SafeSubsetSelectionResult(
        selected_indices=selected_indices,
        selected_candidates=[candidates[idx] for idx in selected_indices],
        selection_weights=selection_weights,
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
        solver_status=solver_status,
        optimization_trace=optimization_trace,
    )
