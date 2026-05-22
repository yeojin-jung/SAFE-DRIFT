from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# Standalone implementation of the closed-form local safe-drift update
#     Delta* = -(1 / lambda) (F_R + alpha I)^(-1) g
# where:
# - g is the target-side gradient,
# - F_R is the reference Fisher geometry,
# - alpha controls the direction,
# - lambda controls the scale.


@dataclass
class AlphaFromBudgetsResult:
    alpha: float
    status: str
    target_ratio: float
    achieved_ratio: float
    achieved_reference_cost: float
    rho: float
    epsilon: float
    phi_min: float
    phi_max: float
    iterations: int


def _coerce_fisher_like(target_grad: torch.Tensor, fisher) -> torch.Tensor:
    fisher_t = torch.as_tensor(fisher, device=target_grad.device, dtype=target_grad.dtype)
    if target_grad.ndim != 1:
        raise ValueError("target_grad must be a 1D tensor.")
    if fisher_t.ndim == 1:
        if fisher_t.shape != target_grad.shape:
            raise ValueError("Diagonal Fisher vector must have the same shape as target_grad.")
        return fisher_t
    if fisher_t.ndim == 2:
        if fisher_t.shape[0] != fisher_t.shape[1]:
            raise ValueError("Full Fisher matrix must be square.")
        if fisher_t.shape[0] != target_grad.numel():
            raise ValueError("Full Fisher matrix size must match target_grad dimension.")
        return fisher_t
    raise ValueError("fisher must be either a 1D vector or a 2D square matrix.")


def _apply_fisher(fisher: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if fisher.ndim == 1:
        return fisher * delta
    return fisher @ delta


# Solve (F_R + alpha I)^(-1) g for either a diagonal Fisher or a small dense
# full Fisher. For large LLM problems, the diagonal branch is the intended path.
def _solve_shifted_system(
    target_grad: torch.Tensor,
    fisher,
    shift: float,
    clamp_min: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shift < 0.0:
        raise ValueError("shift must be non-negative.")
    fisher_t = _coerce_fisher_like(target_grad, fisher)
    if fisher_t.ndim == 1:
        denom = fisher_t.clamp_min(clamp_min) + float(shift)
        return target_grad / denom, fisher_t

    eye = torch.eye(fisher_t.shape[0], device=fisher_t.device, dtype=fisher_t.dtype)
    return torch.linalg.solve(fisher_t + float(shift) * eye, target_grad), fisher_t


# Compute the realized reference cost and norm cost of an update.
def compute_update_costs(delta: torch.Tensor, fisher) -> dict[str, float]:
    fisher_t = _coerce_fisher_like(delta, fisher)
    reference_cost = 0.5 * torch.dot(delta, _apply_fisher(fisher_t, delta))
    norm_cost = 0.5 * torch.dot(delta, delta)
    return {
        "reference_cost": float(reference_cost.item()),
        "norm_cost": float(norm_cost.item()),
    }


def fisher_spectral_components(
    target_grad: torch.Tensor,
    fisher,
    clamp_min: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the spectral components needed to derive alpha from budgets.

    Returns:
        eigenvalues sigma_i and target coordinates c_i = v_i^T g_T.
    For a diagonal Fisher vector, the eigenbasis is the coordinate basis.
    For a dense Fisher matrix, we eigendecompose the symmetrized matrix.
    """
    fisher_t = _coerce_fisher_like(target_grad, fisher)
    if fisher_t.ndim == 1:
        eigenvalues = fisher_t.clamp_min(float(clamp_min))
        components = target_grad
        return eigenvalues, components

    fisher_sym = 0.5 * (fisher_t + fisher_t.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher_sym)
    eigenvalues = eigenvalues.clamp_min(float(clamp_min))
    components = eigenvectors.T @ target_grad
    return eigenvalues, components


def fisher_ratio_from_components(
    eigenvalues: torch.Tensor,
    components: torch.Tensor,
    alpha: float,
    clamp_min: float = 1.0e-30,
) -> float:
    """
    Compute
        phi(alpha) =
            sum_i sigma_i c_i^2 / (sigma_i + alpha)^2
            ------------------------------------------------
            sum_i c_i^2 / (sigma_i + alpha)^2
    using only Fisher eigenvalues sigma_i and target-gradient components c_i.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    sigma = eigenvalues.detach().float()
    c_sq = components.detach().float().square()
    if float(c_sq.sum().item()) <= 0.0:
        raise ValueError("target_grad has zero norm in the Fisher eigenbasis.")

    if math.isinf(float(alpha)):
        return float((sigma * c_sq).sum().div(c_sq.sum().clamp_min(clamp_min)).item())

    denom = (sigma + float(alpha)).square().clamp_min(float(clamp_min))
    weights = c_sq / denom
    return float((sigma * weights).sum().div(weights.sum().clamp_min(clamp_min)).item())


def choose_alpha_from_cost_norm_budgets(
    target_grad: torch.Tensor,
    fisher,
    rho: float,
    epsilon: float,
    alpha_min: float = 1.0e-8,
    alpha_hi_init: float = 1.0,
    alpha_max: float = 1.0e12,
    tolerance: float = 1.0e-6,
    max_iters: int = 100,
    clamp_min: float = 0.0,
) -> AlphaFromBudgetsResult:
    """
    Derive alpha from a reference-cost budget rho and norm budget epsilon.

    The normalized safe direction is
        Delta*(alpha) = -epsilon (F_R + alpha I)^(-1) g_T
                         / ||(F_R + alpha I)^(-1) g_T||_2.

    We choose alpha so that
        0.5 Delta*(alpha)^T F_R Delta*(alpha) = rho.

    In the eigenspace F_R = V diag(sigma_i) V^T and c_i = v_i^T g_T,
    this is equivalent to solving
        phi(alpha) = 2 rho / epsilon^2.
    """
    if rho <= 0.0:
        raise ValueError("rho / cost_c must be positive.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive for log-space bisection.")
    if alpha_hi_init <= 0.0:
        raise ValueError("alpha_hi_init must be positive.")
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min.")

    eigenvalues, components = fisher_spectral_components(
        target_grad=target_grad,
        fisher=fisher,
        clamp_min=clamp_min,
    )
    target_ratio = float(2.0 * float(rho) / (float(epsilon) * float(epsilon)))
    phi_min = fisher_ratio_from_components(eigenvalues, components, alpha_min)
    phi_max = fisher_ratio_from_components(eigenvalues, components, float("inf"))

    if target_ratio <= phi_min:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * phi_min
        return AlphaFromBudgetsResult(
            alpha=float(alpha_min),
            status="too_strict",
            target_ratio=target_ratio,
            achieved_ratio=phi_min,
            achieved_reference_cost=achieved_reference_cost,
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=0,
        )

    if target_ratio >= phi_max:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * phi_max
        return AlphaFromBudgetsResult(
            alpha=float(alpha_max),
            status="inactive",
            target_ratio=target_ratio,
            achieved_ratio=phi_max,
            achieved_reference_cost=achieved_reference_cost,
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=0,
        )

    lo = float(alpha_min)
    hi = max(float(alpha_hi_init), lo * 2.0)
    iterations = 0
    while fisher_ratio_from_components(eigenvalues, components, hi) < target_ratio:
        hi *= 2.0
        iterations += 1
        if hi >= alpha_max:
            hi = float(alpha_max)
            break

    hi_ratio = fisher_ratio_from_components(eigenvalues, components, hi)
    if hi_ratio < target_ratio:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * hi_ratio
        return AlphaFromBudgetsResult(
            alpha=float(hi),
            status="inactive",
            target_ratio=target_ratio,
            achieved_ratio=float(hi_ratio),
            achieved_reference_cost=float(achieved_reference_cost),
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=iterations,
        )

    best_alpha = hi
    best_ratio = hi_ratio
    for _ in range(max_iters):
        iterations += 1
        mid = math.sqrt(lo * hi)
        mid_ratio = fisher_ratio_from_components(eigenvalues, components, mid)
        best_alpha = mid
        best_ratio = mid_ratio

        relative_error = abs(mid_ratio - target_ratio) / max(abs(target_ratio), 1.0e-30)
        if relative_error <= tolerance:
            break
        if mid_ratio < target_ratio:
            lo = mid
        else:
            hi = mid

    achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * best_ratio
    return AlphaFromBudgetsResult(
        alpha=float(best_alpha),
        status="feasible",
        target_ratio=target_ratio,
        achieved_ratio=float(best_ratio),
        achieved_reference_cost=float(achieved_reference_cost),
        rho=float(rho),
        epsilon=float(epsilon),
        phi_min=phi_min,
        phi_max=phi_max,
        iterations=iterations,
    )


def choose_alpha_from_spectral_components(
    eigenvalues: torch.Tensor,
    components: torch.Tensor,
    rho: float,
    epsilon: float,
    alpha_min: float = 1.0e-8,
    alpha_hi_init: float = 1.0,
    alpha_max: float = 1.0e12,
    tolerance: float = 1.0e-6,
    max_iters: int = 100,
) -> AlphaFromBudgetsResult:
    """
    Derive alpha directly from spectral components.

    Inputs:
        eigenvalues: sigma_i with shape [d].
        components: c_i = v_i^T g_T with shape [d].

    This is the same solve as choose_alpha_from_cost_norm_budgets, but it does
    not require materializing F_R or recomputing an eigendecomposition.
    """
    if eigenvalues.ndim != 1 or components.ndim != 1:
        raise ValueError("eigenvalues and components must both be 1D tensors.")
    if eigenvalues.shape != components.shape:
        raise ValueError("eigenvalues and components must have the same shape.")
    if rho <= 0.0:
        raise ValueError("rho / cost_c must be positive.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if alpha_min <= 0.0:
        raise ValueError("alpha_min must be positive for log-space bisection.")
    if alpha_hi_init <= 0.0:
        raise ValueError("alpha_hi_init must be positive.")
    if alpha_max <= alpha_min:
        raise ValueError("alpha_max must be greater than alpha_min.")

    eigenvalues = eigenvalues.detach().float().clamp_min(0.0)
    components = components.detach().float()
    target_ratio = float(2.0 * float(rho) / (float(epsilon) * float(epsilon)))
    phi_min = fisher_ratio_from_components(eigenvalues, components, alpha_min)
    phi_max = fisher_ratio_from_components(eigenvalues, components, float("inf"))

    if target_ratio <= phi_min:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * phi_min
        return AlphaFromBudgetsResult(
            alpha=float(alpha_min),
            status="too_strict",
            target_ratio=target_ratio,
            achieved_ratio=phi_min,
            achieved_reference_cost=achieved_reference_cost,
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=0,
        )

    if target_ratio >= phi_max:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * phi_max
        return AlphaFromBudgetsResult(
            alpha=float(alpha_max),
            status="inactive",
            target_ratio=target_ratio,
            achieved_ratio=phi_max,
            achieved_reference_cost=achieved_reference_cost,
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=0,
        )

    lo = float(alpha_min)
    hi = max(float(alpha_hi_init), lo * 2.0)
    iterations = 0
    while fisher_ratio_from_components(eigenvalues, components, hi) < target_ratio:
        hi *= 2.0
        iterations += 1
        if hi >= alpha_max:
            hi = float(alpha_max)
            break

    hi_ratio = fisher_ratio_from_components(eigenvalues, components, hi)
    if hi_ratio < target_ratio:
        achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * hi_ratio
        return AlphaFromBudgetsResult(
            alpha=float(hi),
            status="inactive",
            target_ratio=target_ratio,
            achieved_ratio=float(hi_ratio),
            achieved_reference_cost=float(achieved_reference_cost),
            rho=float(rho),
            epsilon=float(epsilon),
            phi_min=phi_min,
            phi_max=phi_max,
            iterations=iterations,
        )

    best_alpha = hi
    best_ratio = hi_ratio
    for _ in range(max_iters):
        iterations += 1
        mid = math.sqrt(lo * hi)
        mid_ratio = fisher_ratio_from_components(eigenvalues, components, mid)
        best_alpha = mid
        best_ratio = mid_ratio

        relative_error = abs(mid_ratio - target_ratio) / max(abs(target_ratio), 1.0e-30)
        if relative_error <= tolerance:
            break
        if mid_ratio < target_ratio:
            lo = mid
        else:
            hi = mid

    achieved_reference_cost = 0.5 * float(epsilon) * float(epsilon) * best_ratio
    return AlphaFromBudgetsResult(
        alpha=float(best_alpha),
        status="feasible",
        target_ratio=target_ratio,
        achieved_ratio=float(best_ratio),
        achieved_reference_cost=float(achieved_reference_cost),
        rho=float(rho),
        epsilon=float(epsilon),
        phi_min=phi_min,
        phi_max=phi_max,
        iterations=iterations,
    )


def delta_theta_star_from_cost_norm_budgets(
    target_grad: torch.Tensor,
    fisher,
    rho: float,
    epsilon: float,
    alpha_min: float = 1.0e-8,
    alpha_hi_init: float = 1.0,
    alpha_max: float = 1.0e12,
    tolerance: float = 1.0e-6,
    max_iters: int = 100,
    clamp_min: float = 1.0e-8,
) -> dict[str, float | str | torch.Tensor]:
    """
    Compute Delta* by deriving alpha from (rho, epsilon) instead of tuning it.

    This fixes ||Delta*||_2 = epsilon and chooses alpha so that the reference
    Fisher cost matches rho when feasible.
    """
    alpha_result = choose_alpha_from_cost_norm_budgets(
        target_grad=target_grad,
        fisher=fisher,
        rho=rho,
        epsilon=epsilon,
        alpha_min=alpha_min,
        alpha_hi_init=alpha_hi_init,
        alpha_max=alpha_max,
        tolerance=tolerance,
        max_iters=max_iters,
        clamp_min=0.0,
    )
    preconditioned, fisher_t = _solve_shifted_system(
        target_grad=target_grad,
        fisher=fisher,
        shift=alpha_result.alpha,
        clamp_min=clamp_min,
    )
    preconditioned_norm = preconditioned.norm()
    if float(preconditioned_norm.item()) <= 0.0:
        raise ValueError("Cannot normalize a zero safe-update direction.")

    lambda_value = float(preconditioned_norm.item()) / float(epsilon)
    delta = -(float(epsilon) / preconditioned_norm) * preconditioned
    costs = compute_update_costs(delta, fisher_t)
    predicted_gain = -torch.dot(target_grad, delta)
    return {
        "delta": delta,
        "lambda": lambda_value,
        "alpha": float(alpha_result.alpha),
        "beta": lambda_value * float(alpha_result.alpha),
        "predicted_gain": float(predicted_gain.item()),
        "reference_cost": costs["reference_cost"],
        "norm_cost": costs["norm_cost"],
        "cost_c": float(rho),
        "epsilon": float(epsilon),
        "alpha_status": alpha_result.status,
        "alpha_target_ratio": alpha_result.target_ratio,
        "alpha_achieved_ratio": alpha_result.achieved_ratio,
        "alpha_phi_min": alpha_result.phi_min,
        "alpha_phi_max": alpha_result.phi_max,
        "alpha_iterations": alpha_result.iterations,
    }


# Compute Delta* for a fixed (lambda, alpha).
def delta_theta_star_from_lambda_alpha(
    target_grad: torch.Tensor,
    fisher,
    lambda_value: float,
    alpha: float,
    clamp_min: float = 1.0e-8,
) -> dict[str, float | torch.Tensor]:
    if lambda_value <= 0.0:
        raise ValueError("lambda_value must be positive.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    preconditioned, fisher_t = _solve_shifted_system(
        target_grad=target_grad,
        fisher=fisher,
        shift=float(alpha),
        clamp_min=clamp_min,
    )
    delta = -(1.0 / float(lambda_value)) * preconditioned
    costs = compute_update_costs(delta, fisher_t)
    predicted_gain = -torch.dot(target_grad, delta)
    return {
        "delta": delta,
        "lambda": float(lambda_value),
        "alpha": float(alpha),
        "beta": float(lambda_value) * float(alpha),
        "predicted_gain": float(predicted_gain.item()),
        "reference_cost": costs["reference_cost"],
        "norm_cost": costs["norm_cost"],
    }


# Compute the matched-gain update by solving lambda away for a desired local
# predicted target gain.
def matched_gain_delta_theta_star(
    target_grad: torch.Tensor,
    fisher,
    alpha: float,
    predicted_gain: float,
    clamp_min: float = 1.0e-8,
) -> dict[str, float | torch.Tensor]:
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if predicted_gain <= 0.0:
        raise ValueError("predicted_gain must be positive.")

    preconditioned, fisher_t = _solve_shifted_system(
        target_grad=target_grad,
        fisher=fisher,
        shift=float(alpha),
        clamp_min=clamp_min,
    )
    q_alpha = torch.dot(target_grad, preconditioned)
    lambda_value = q_alpha / float(predicted_gain)
    delta = -(float(predicted_gain) / float(q_alpha.item())) * preconditioned
    costs = compute_update_costs(delta, fisher_t)
    realized_gain = -torch.dot(target_grad, delta)
    return {
        "delta": delta,
        "alpha": float(alpha),
        "lambda": float(lambda_value.item()),
        "beta": float(alpha) * float(lambda_value.item()),
        "q_alpha": float(q_alpha.item()),
        "predicted_gain": float(realized_gain.item()),
        "reference_cost": costs["reference_cost"],
        "norm_cost": costs["norm_cost"],
    }
