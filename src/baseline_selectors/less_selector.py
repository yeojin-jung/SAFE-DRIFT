from __future__ import annotations

import torch
import torch.nn.functional as F


def score_less(
    candidate_gradients: torch.Tensor,
    target_gradient: torch.Tensor,
    preconditioner: torch.Tensor | None = None,
    similarity: str = "dot",
) -> torch.Tensor:
    if candidate_gradients.ndim != 2:
        raise ValueError("candidate_gradients must be a 2D tensor [num_examples, gradient_dim].")
    if target_gradient.ndim != 1:
        raise ValueError("target_gradient must be a 1D tensor.")
    if candidate_gradients.shape[1] != target_gradient.numel():
        raise ValueError("Gradient dimension mismatch between candidates and target_gradient.")
    if similarity not in {"dot", "cosine"}:
        raise ValueError("similarity must be either 'dot' or 'cosine'.")

    candidate_gradients = candidate_gradients.float()
    target_gradient = target_gradient.float()

    if preconditioner is None:
        weighted_gradients = candidate_gradients
    else:
        preconditioner = torch.as_tensor(
            preconditioner,
            device=candidate_gradients.device,
            dtype=candidate_gradients.dtype,
        )
        if preconditioner.ndim == 1:
            if preconditioner.numel() != target_gradient.numel():
                raise ValueError("1D preconditioner must have the same dimension as target_gradient.")
            weighted_gradients = candidate_gradients * preconditioner.unsqueeze(0)
        elif preconditioner.ndim == 2:
            if preconditioner.shape != (target_gradient.numel(), target_gradient.numel()):
                raise ValueError("2D preconditioner must have shape [gradient_dim, gradient_dim].")
            weighted_gradients = candidate_gradients @ preconditioner.T
        else:
            raise ValueError("preconditioner must be either a 1D diagonal vector or a 2D matrix.")

    if similarity == "cosine":
        weighted_gradients = F.normalize(weighted_gradients, dim=1)
        target_gradient = F.normalize(target_gradient, dim=0)

    return weighted_gradients @ target_gradient


def select_less(
    candidate_gradients: torch.Tensor,
    target_gradient: torch.Tensor,
    subset_budget: int,
    preconditioner: torch.Tensor | None = None,
    similarity: str = "dot",
) -> list[int]:
    if subset_budget <= 0:
        raise ValueError("subset_budget must be positive.")
    if candidate_gradients.shape[0] < subset_budget:
        raise ValueError("subset_budget cannot exceed the number of candidate examples.")

    scores = score_less(
        candidate_gradients=candidate_gradients,
        target_gradient=target_gradient,
        preconditioner=preconditioner,
        similarity=similarity,
    )
    return scores.topk(subset_budget).indices.cpu().tolist()
