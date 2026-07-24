from __future__ import annotations

import math
import statistics
from typing import Any

import torch


def random_subset_update_norm_calibration(
    candidate_gradients: torch.Tensor,
    *,
    subset_budget: int,
    learning_rate: float,
    preconditioner: torch.Tensor | None = None,
    sample_count: int = 64,
    seed: int = 42,
) -> tuple[float, dict[str, Any]]:
    gradients = candidate_gradients.detach().to(dtype=torch.float32, device="cpu")
    if gradients.ndim != 2:
        raise ValueError("candidate_gradients must have shape [num_candidates, feature_dim].")
    num_candidates, feature_dim = gradients.shape
    if not 0 < int(subset_budget) <= int(num_candidates):
        raise ValueError("subset_budget must be in [1, num_candidates].")
    if int(sample_count) <= 0:
        raise ValueError("sample_count must be positive.")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive and finite.")

    scale = None
    if preconditioner is not None:
        scale = preconditioner.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if int(scale.numel()) != int(feature_dim):
            raise ValueError("preconditioner width must match the candidate feature dimension.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    update_norms: list[float] = []
    for _ in range(int(sample_count)):
        indices = torch.randperm(int(num_candidates), generator=generator)[: int(subset_budget)]
        mean_gradient = gradients.index_select(0, indices).mean(dim=0)
        if scale is not None:
            mean_gradient = mean_gradient * scale
        update_norms.append(
            float(float(learning_rate) * torch.linalg.vector_norm(mean_gradient).item())
        )

    epsilon0 = float(statistics.median(update_norms))
    if not math.isfinite(epsilon0) or epsilon0 <= 0.0:
        raise ValueError("The calibrated median update norm is non-positive or non-finite.")
    return epsilon0, {
        "safe_epsilon_calibration_formula": (
            "median_A ||-(eta/S) P sum_{s in A} G_s||_2"
        ),
        "safe_epsilon_calibration_samples": int(sample_count),
        "safe_epsilon_calibration_seed": int(seed),
        "safe_epsilon_calibration_subset_budget": int(subset_budget),
        "safe_epsilon_calibration_candidate_count": int(num_candidates),
        "safe_epsilon_calibration_feature_dim": int(feature_dim),
        "safe_epsilon_calibration_learning_rate": float(learning_rate),
        "safe_epsilon_calibration_uses_preconditioner": scale is not None,
        "safe_epsilon_calibration_update_norm_min": float(min(update_norms)),
        "safe_epsilon_calibration_update_norm_mean": float(
            sum(update_norms) / len(update_norms)
        ),
        "safe_epsilon_calibration_update_norm_median": epsilon0,
        "safe_epsilon_calibration_update_norm_max": float(max(update_norms)),
        "safe_epsilon_calibration_update_norms": update_norms,
    }


def coupled_reference_budget(
    *,
    target_gradient: torch.Tensor,
    reference_fisher: torch.Tensor,
    epsilon: float,
    gamma: float,
    preconditioner: torch.Tensor | None = None,
) -> tuple[float, dict[str, Any]]:
    target = target_gradient.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    fisher = reference_fisher.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if int(target.numel()) != int(fisher.numel()):
        raise ValueError("target_gradient and reference_fisher must have the same width.")
    if preconditioner is not None:
        scale = preconditioner.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if int(scale.numel()) != int(target.numel()):
            raise ValueError("preconditioner width must match the target feature dimension.")
        target = target * scale
        direction_source = "negative_normalized_preconditioned_target_gradient"
    else:
        direction_source = "negative_normalized_target_gradient"

    target_norm = float(torch.linalg.vector_norm(target).item())
    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError("Cannot calibrate rho from a zero or non-finite target direction.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be positive and finite.")
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("gamma must be positive and finite.")

    v0 = -target / target_norm
    q0 = float(torch.dot(v0, fisher * v0).item())
    if not math.isfinite(q0) or q0 <= 0.0:
        raise ValueError("Reference curvature q0 is non-positive or non-finite.")
    rho = 0.5 * float(gamma) * q0 * float(epsilon) ** 2
    return float(rho), {
        "safe_cost_gamma": float(gamma),
        "safe_reference_direction_source": direction_source,
        "safe_reference_direction_norm": target_norm,
        "safe_reference_curvature_q0": q0,
        "safe_effective_cost_c": float(rho),
        "safe_effective_cost_c_formula": "0.5 * gamma * q0 * epsilon^2",
    }
