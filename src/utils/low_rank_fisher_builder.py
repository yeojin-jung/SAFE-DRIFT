"""Utilities for building the reference-side low-rank Fisher geometry and
the residualize-first task/candidate projection described in low_rank.tex.

Core objects implemented here
-----------------------------
1. Reference Fisher from reference-only scores / loss gradients:
       F_R_hat = (1 / N_R) sum_k s_k s_k^T
   using the dual Gram matrix
       K_R_dual = (1 / N_R) S S^T.

2. Reference low-rank basis:
       U_R = S^T V_R (N_R Lambda_R)^(-1/2)
       F_R_hat^(K_R) = U_R Lambda_R U_R^T.

3. Ridge-regularized safe direction:
       h_{alpha,R} = (F_R_hat^(K_R) + alpha I)^(-1) g_T
                    = U_R (Lambda_R + alpha I)^(-1) U_R^T g_T
                      + alpha^(-1) (I - U_R U_R^T) g_T.

4. Residualized task/candidate low-rank structure:
       G_TC     = [g_T, g_1, ..., g_{N_C}]
       G_tilde  = (I - U_R U_R^T) G_TC
       U_add    = low_rank_basis(G_tilde)

5. Common working basis:
       U_K = [U_R, U_add].

Implementation notes
--------------------
- In practice, d should be the number of *trainable* parameters only. For LLMs,
  this usually means LoRA / PEFT parameters rather than full model parameters.
- Reference gradients should be per-example gradients of the reference NLL
  (or scores, up to sign). Candidate gradients are typically per-example task
  gradients. The target gradient g_T is usually an average over a target anchor
  set.
- For causal language modeling, use sequence NLL *sum* rather than the default
  mean-over-tokens loss if you want gradients to correspond to the full-example
  log probability.
- This file implements the mathematically preferred "residualize first, then
  low-rank" construction for the added task block. The older "raw U_T first,
  residualize later" route is intentionally removed.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

Batch = Dict[str, torch.Tensor]
LossClosure = Callable[[nn.Module, Batch], torch.Tensor]


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class ReferenceLowRank:
    """Reference-side low-rank Fisher approximation.

    Attributes
    ----------
    U_R:
        Parameter-space basis, shape [d, K_R].
    Lambda_R:
        Retained reference eigenvalues used by the Fisher geometry, shape [K_R].
        These may be shrinkage-corrected when ``shrinkage_method`` is not
        ``"none"``.
    Lambda_R_raw:
        Retained raw sample reference eigenvalues before shrinkage, shape [K_R].
    evals_full:
        All nonnegative dual eigenvalues in descending order.
    V_top:
        Retained top dual eigenvectors, shape [N_R, K_R].
    K_R:
        Retained rank.
    alpha:
        Ridge parameter used for ridge-relevance rank selection.
    delta:
        Tolerance used in cumulative ridge relevance rank selection.
    """

    U_R: torch.Tensor
    Lambda_R: torch.Tensor
    Lambda_R_raw: torch.Tensor
    evals_full: torch.Tensor
    V_top: torch.Tensor
    K_R: int
    alpha: float
    delta: float
    shrinkage_method: str = "none"
    shrinkage_gamma: Optional[float] = None
    shrinkage_valid_mask: Optional[torch.Tensor] = None
    eigensolver: str = "full"


@dataclass
class SafeDirection:
    """Ridge-regularized safe direction in the reference basis."""

    h_alpha_R: torch.Tensor
    b_T_R: torch.Tensor
    g_T_perp_R: torch.Tensor


@dataclass
class TaskCandidateLowRank:
    """Residualized task/candidate low-rank structure.

    The stored basis ``U_T`` spans the added task subspace obtained by first
    removing the reference component and then performing the low-rank
    decomposition. The retained eigenvalues ``Omega_T`` come from the
    residualized task/candidate dual Gram matrix and are used only for rank
    selection and diagnostics; the reduced reference metric remains isotropic
    on this added block.

    Notes
    -----
    The field names keep the old ``U_T`` / ``K_T`` convention for compatibility,
    but mathematically this object represents the residualized added task basis.
    """

    U_T: torch.Tensor
    Omega_T: torch.Tensor
    evals_full: torch.Tensor
    K_T: int
    eigensolver: str = "full"

    @property
    def U_add(self) -> torch.Tensor:
        return self.U_T

    @property
    def K_add(self) -> int:
        return self.K_T

    @property
    def Omega_add(self) -> torch.Tensor:
        return self.Omega_T



@dataclass
class CommonBasis:
    """Combined working basis U_K = [U_R, U_add]."""

    U_K: torch.Tensor
    U_add: torch.Tensor
    M_diag: torch.Tensor
    K: int
    K_add: int


@dataclass
class LowRankSafeInputs:
    """Selector-ready tensors plus the low-rank objects that produced them."""

    candidate_features: torch.Tensor
    target_feature: torch.Tensor
    target_features: Optional[torch.Tensor]
    reference_fisher: torch.Tensor
    reference_grad_rows: torch.Tensor
    g_T: torch.Tensor
    candidate_grads: torch.Tensor
    reference_low_rank: ReferenceLowRank
    task_candidate_low_rank: Optional[TaskCandidateLowRank]
    common_basis: CommonBasis
    artifact_path: Optional[str] = None


# -----------------------------------------------------------------------------
# Basic tensor helpers
# -----------------------------------------------------------------------------


def move_batch_to_device(batch: Batch, device: torch.device | str) -> Batch:
    out: Batch = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def batch_size(batch: Batch) -> int:
    for v in batch.values():
        if torch.is_tensor(v):
            return int(v.shape[0])
    raise ValueError("Could not infer batch size from batch dictionary.")


def slice_batch(batch: Batch, index: int) -> Batch:
    """Extract a single-example batch from a collated batch."""
    out: Batch = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[index : index + 1]
        else:
            out[k] = v
    return out


def collect_trainable_parameters(
    model: nn.Module,
    requires_grad_only: bool = True,
    name_filter: Optional[Callable[[str, nn.Parameter], bool]] = None,
) -> Tuple[List[str], List[nn.Parameter]]:
    """Collect parameters to include in gradient vectors.

    For large models, this should usually be restricted to LoRA / PEFT params.
    """
    names: List[str] = []
    params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if requires_grad_only and not p.requires_grad:
            continue
        if name_filter is not None and not name_filter(name, p):
            continue
        names.append(name)
        params.append(p)
    if not params:
        raise ValueError("No parameters selected.")
    return names, params


def _flatten_grad_list(
    params: Sequence[nn.Parameter], grads: Sequence[Optional[torch.Tensor]]
) -> torch.Tensor:
    flat: List[torch.Tensor] = []
    for p, g in zip(params, grads):
        if g is None:
            flat.append(torch.zeros_like(p, memory_format=torch.contiguous_format).reshape(-1))
        else:
            flat.append(g.detach().reshape(-1))
    return torch.cat(flat, dim=0)


# -----------------------------------------------------------------------------
# Example loss closures
# -----------------------------------------------------------------------------


def hf_default_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Use model(**batch).loss.

    Suitable for many HuggingFace tasks, but note that for CausalLM this is
    usually a mean-over-tokens loss, not full-example NLL.
    """
    outputs = model(**batch)
    if not hasattr(outputs, "loss") or outputs.loss is None:
        raise ValueError("Model outputs do not contain `.loss`.")
    return outputs.loss


def causal_lm_sequence_nll(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Sequence negative log-likelihood summed over valid next-token positions.

    This is usually the right loss if you want per-example gradients to match
    the full-example score / NLL geometry in the paper.

    Expected keys in batch:
      - input_ids: [B, T]
      - labels:    [B, T] with -100 marking ignored positions
      - attention_mask: optional [B, T]
    """
    required = {"input_ids", "labels"}
    missing = required - set(batch.keys())
    if missing:
        raise ValueError(f"causal_lm_sequence_nll missing batch keys: {missing}")

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask", None),
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]

    safe_labels = labels.masked_fill(labels == -100, 0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = (labels != -100).to(token_logp.dtype)

    # Negative log-likelihood of the full example (sum, not mean).
    nll = -(token_logp * valid_mask).sum()
    return nll


# -----------------------------------------------------------------------------
# Gradient extraction
# -----------------------------------------------------------------------------


@torch.no_grad()
def parameter_dimension(params: Sequence[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def grad_vector_from_batch(
    model: nn.Module,
    batch: Batch,
    params: Sequence[nn.Parameter],
    loss_fn: LossClosure,
) -> torch.Tensor:
    """Compute a flattened gradient vector for one batch.

    The returned tensor is detached and lives on the same device as the model.
    """
    loss = loss_fn(model, batch)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    return _flatten_grad_list(params, grads)


def per_example_gradient_matrix(
    model: nn.Module,
    data_iter: Iterable[Batch],
    params: Sequence[nn.Parameter],
    loss_fn: LossClosure,
    device: torch.device | str,
    max_examples: Optional[int] = None,
    store_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute a matrix of per-example gradient vectors.

    Returns
    -------
    G:
        Shape [N, d], where each row is one example gradient vector.

    Notes
    -----
    - This function loops over examples. It is simple and faithful to the paper,
      but slow for large datasets. If needed, replace with functorch/vmap,
      BackPACK, or other per-sample-gradient tooling.
    - Gradients are moved to CPU by default (store_dtype on CPU) to avoid GPU
      memory blowup.
    """
    model.eval()
    rows: List[torch.Tensor] = []
    seen = 0

    for batch in data_iter:
        batch = move_batch_to_device(batch, device)
        B = batch_size(batch)
        for i in range(B):
            if max_examples is not None and seen >= max_examples:
                break
            ex = slice_batch(batch, i)
            g = grad_vector_from_batch(model, ex, params, loss_fn)
            rows.append(g.detach().to("cpu", dtype=store_dtype))
            seen += 1
        if max_examples is not None and seen >= max_examples:
            break

    if not rows:
        raise ValueError("No gradients were collected.")
    return torch.stack(rows, dim=0)  # [N, d]


def mean_gradient_over_examples(
    model: nn.Module,
    data_iter: Iterable[Batch],
    params: Sequence[nn.Parameter],
    loss_fn: LossClosure,
    device: torch.device | str,
    max_examples: Optional[int] = None,
    store_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Average gradient over examples, suitable for g_T on a target anchor set."""
    G = per_example_gradient_matrix(
        model=model,
        data_iter=data_iter,
        params=params,
        loss_fn=loss_fn,
        device=device,
        max_examples=max_examples,
        store_dtype=store_dtype,
    )
    return G.mean(dim=0)


# -----------------------------------------------------------------------------
# Reference-side low-rank Fisher
# -----------------------------------------------------------------------------


def choose_rank_by_ridge_relevance(
    evals_desc: torch.Tensor,
    alpha: float,
    delta: float = 0.01,
    max_rank: Optional[int] = None,
    min_rank: int = 1,
    eps: float = 1e-12,
) -> int:
    """Choose K_R from descending eigenvalues using cumulative ridge relevance.

    The paper's alpha-dependent relevance score is
        r_j(alpha) = lambda_j / (lambda_j + alpha).
    We choose the smallest K such that
        sum_{j<=K} r_j / sum_j r_j >= 1 - delta.
    """
    evals = evals_desc[evals_desc > eps]
    if evals.numel() == 0:
        return 0
    if max_rank is not None:
        evals = evals[:max_rank]

    alpha_t = torch.as_tensor(alpha, dtype=evals.dtype, device=evals.device)
    relevance = evals / (evals + alpha_t)
    total = relevance.sum()
    if float(total) <= eps:
        return min(min_rank, int(evals.numel()))

    cumulative = torch.cumsum(relevance, dim=0) / total
    threshold = 1.0 - delta
    idx = int(torch.searchsorted(cumulative, torch.tensor(threshold, device=cumulative.device), right=False).item())
    K = idx + 1
    K = max(K, min_rank)
    K = min(K, int(evals.numel()))
    return K


def stein_shrink_reference_eigenvalues(
    sample_eigenvalues: torch.Tensor,
    gamma: float,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Stein-loss spiked-covariance eigenvalue shrinker.

    The formula assumes the usual white-noise spiked covariance normalization.
    Eigenvalues below the BBP edge do not produce a real separated spike; for
    those entries we keep the raw sample eigenvalue and mark the shrinkage mask
    as invalid so the artifact metadata is explicit.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive for Stein shrinkage.")
    if sample_eigenvalues.ndim != 1:
        raise ValueError("sample_eigenvalues must be a 1D tensor.")

    lambdas = sample_eigenvalues.to(dtype=torch.float64)
    gamma_t = torch.as_tensor(float(gamma), dtype=torch.float64, device=lambdas.device)
    edge = (1.0 + torch.sqrt(gamma_t)) ** 2
    discriminant = (lambdas - gamma_t - 1.0) ** 2 - 4.0 * gamma_t
    valid = (lambdas > edge) & (discriminant > eps)

    shrunk = lambdas.clone()
    if bool(valid.any()):
        ell = 0.5 * (
            lambdas[valid]
            - gamma_t
            - 1.0
            + torch.sqrt(torch.clamp(discriminant[valid], min=0.0))
        )
        c2 = (1.0 - gamma_t / torch.clamp(ell**2, min=eps)) / (
            1.0 + gamma_t / torch.clamp(ell, min=eps)
        )
        c2 = torch.clamp(c2, min=0.0, max=1.0)
        denominator = c2 / torch.clamp(ell, min=eps) + (1.0 - c2)
        shrunk[valid] = 1.0 / torch.clamp(denominator, min=eps)

    return shrunk.to(sample_eigenvalues.dtype), valid.to(torch.bool)


def _top_eigenpairs_symmetric(
    matrix: torch.Tensor,
    *,
    requested_rank: Optional[int],
    eps: float = 1e-10,
    lobpcg_tol: float = 1e-5,
    lobpcg_niter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Return descending positive eigenpairs of a symmetric matrix.

    When ``requested_rank`` is provided, only the top-k eigenpairs are needed.
    We use LOBPCG in that case to avoid materializing the full eigenspectrum.
    Auto-rank selection still passes ``requested_rank=None`` and therefore uses
    full ``eigh``, because it needs the complete spectrum.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")

    n = int(matrix.shape[0])
    use_full = requested_rank is None
    if requested_rank is not None:
        requested_rank = int(requested_rank)
        if requested_rank <= 0:
            return (
                torch.empty(0, dtype=matrix.dtype, device=matrix.device),
                torch.empty((n, 0), dtype=matrix.dtype, device=matrix.device),
                "empty",
            )
        requested_rank = min(requested_rank, n)
        # PyTorch LOBPCG's stable path requires n >= 3k. For small matrices or
        # very large fixed ranks, full eigh is safer and often faster anyway.
        use_full = requested_rank >= n or n < max(6, 3 * requested_rank)

    if use_full:
        evals, evecs = torch.linalg.eigh(matrix)
        eigensolver = "full_eigh"
    else:
        try:
            init = torch.randn(
                n,
                int(requested_rank),
                dtype=matrix.dtype,
                device=matrix.device,
            )
            evals, evecs = torch.lobpcg(
                matrix,
                k=int(requested_rank),
                X=init,
                largest=True,
                niter=lobpcg_niter,
                tol=lobpcg_tol,
            )
            eigensolver = "lobpcg_topk"
        except Exception:
            # Keep the build robust: LOBPCG can be sensitive to degeneracy or
            # backend details, while full eigh preserves the previous behavior.
            evals, evecs = torch.linalg.eigh(matrix)
            eigensolver = "full_eigh_fallback"

    order = torch.argsort(evals, descending=True)
    if requested_rank is not None and eigensolver.startswith("full_eigh"):
        order = order[: int(requested_rank)]
    evals = torch.clamp(evals[order], min=0.0)
    evecs = evecs[:, order]

    positive = evals > eps
    return evals[positive], evecs[:, positive], eigensolver


def build_reference_low_rank(
    reference_grad_rows: torch.Tensor,
    alpha: float,
    K_R: Optional[int] = None,
    delta: float = 0.01,
    max_rank: Optional[int] = None,
    eigenvalue_shrinkage: str = "none",
    eps: float = 1e-10,
    compute_in_float64: bool = True,
) -> ReferenceLowRank:
    """Build the reference-only low-rank Fisher approximation.

    Parameters
    ----------
    reference_grad_rows:
        Tensor S of shape [N_R, d], where each row is a reference score vector
        s_k^{ref} or equivalently a reference loss gradient g_k^R (sign does not
        matter for the Fisher outer product).
    K_R:
        Optional explicit retained rank. When omitted, the rank is selected by
        cumulative ridge relevance using ``alpha`` and ``delta``. ``max_rank``
        remains an upper bound in either mode.
    """
    if reference_grad_rows.ndim != 2:
        raise ValueError("reference_grad_rows must have shape [N_R, d].")
    if eigenvalue_shrinkage not in {"none", "stein"}:
        raise ValueError("eigenvalue_shrinkage must be one of: none, stein.")

    S = reference_grad_rows
    N_R, d = S.shape
    work_dtype = torch.float64 if compute_in_float64 else S.dtype
    S_work = S.to(work_dtype)

    # K_R^{dual} = (1 / N_R) S S^T
    K_dual = (S_work @ S_work.T) / float(N_R)

    requested_eigen_rank: Optional[int] = None
    if K_R is not None:
        if K_R <= 0:
            raise ValueError("K_R must be positive when provided.")
        requested_eigen_rank = int(K_R)
        if max_rank is not None:
            requested_eigen_rank = min(requested_eigen_rank, int(max_rank))

    evals_pos, evecs_pos, eigensolver = _top_eigenpairs_symmetric(
        K_dual,
        requested_rank=requested_eigen_rank,
        eps=eps,
    )

    if evals_pos.numel() == 0:
        raise ValueError("Reference dual Gram matrix appears numerically rank-0.")

    if K_R is not None:
        K_R = min(int(K_R), int(evals_pos.numel()))
        if max_rank is not None:
            K_R = min(K_R, int(max_rank))
    else:
        K_R = choose_rank_by_ridge_relevance(
            evals_desc=evals_pos,
            alpha=alpha,
            delta=delta,
            max_rank=max_rank,
            min_rank=1,
            eps=eps,
        )
        if max_rank is not None:
            K_R = min(K_R, int(max_rank))
    if K_R <= 0:
        raise ValueError("Resolved K_R must be positive.")

    Lambda_R_raw = evals_pos[:K_R].clone()
    Lambda_R = Lambda_R_raw.clone()
    V_top = evecs_pos[:, :K_R].clone()

    # U_R is recovered from the raw sample eigenvalues. Shrinkage only changes
    # the Fisher eigenvalues used downstream, not the empirical eigenvectors.
    denom = torch.sqrt(torch.as_tensor(float(N_R), dtype=work_dtype, device=S_work.device) * Lambda_R_raw)
    U_R = (S_work.T @ V_top) / denom.unsqueeze(0)

    shrinkage_gamma: Optional[float] = None
    shrinkage_valid_mask: Optional[torch.Tensor] = None
    if eigenvalue_shrinkage == "stein":
        shrinkage_gamma = float(d) / float(N_R)
        Lambda_R, shrinkage_valid_mask = stein_shrink_reference_eigenvalues(
            sample_eigenvalues=Lambda_R_raw,
            gamma=shrinkage_gamma,
        )

    # In exact arithmetic the columns of U_R are orthonormal. We therefore keep
    # this construction directly, rather than applying an additional QR step
    # that would rotate the basis relative to Lambda_R.
    return ReferenceLowRank(
        U_R=U_R.to(torch.float32),
        Lambda_R=Lambda_R.to(torch.float32),
        Lambda_R_raw=Lambda_R_raw.to(torch.float32),
        evals_full=evals_pos.to(torch.float32),
        V_top=V_top.to(torch.float32),
        K_R=K_R,
        alpha=float(alpha),
        delta=float(delta),
        shrinkage_method=eigenvalue_shrinkage,
        shrinkage_gamma=shrinkage_gamma,
        shrinkage_valid_mask=None if shrinkage_valid_mask is None else shrinkage_valid_mask.cpu(),
        eigensolver=eigensolver,
    )


def approximate_reference_fisher_from_low_rank(ref_lr: ReferenceLowRank) -> torch.Tensor:
    """Materialize F_R^(K_R) = U_R diag(Lambda_R) U_R^T.

    Warning: this is [d, d] and should only be used for debugging with small d.
    """
    U_R = ref_lr.U_R
    Lambda_R = ref_lr.Lambda_R
    return (U_R * Lambda_R.unsqueeze(0)) @ U_R.T


# -----------------------------------------------------------------------------
# Safe direction in the reference basis
# -----------------------------------------------------------------------------


def decompose_target_gradient_in_reference_basis(
    g_T: torch.Tensor,
    U_R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (b_T^R, g_{T,perp}^R)."""
    b_T_R = U_R.T @ g_T
    g_T_perp_R = g_T - (U_R @ b_T_R)
    return b_T_R, g_T_perp_R


def reference_safe_direction(
    g_T: torch.Tensor,
    U_R: torch.Tensor,
    Lambda_R: torch.Tensor,
    alpha: float,
) -> SafeDirection:
    """Compute h_{alpha,R} = (F_R^(K_R) + alpha I)^(-1) g_T.

    Uses the decomposition from the paper:
      h_{alpha,R} = U_R (Lambda_R + alpha I)^(-1) b_T^R + alpha^{-1} g_{T,perp}^R.
    """
    alpha_t = torch.as_tensor(alpha, dtype=g_T.dtype, device=g_T.device)
    b_T_R, g_T_perp_R = decompose_target_gradient_in_reference_basis(g_T, U_R)
    coeff = b_T_R / (Lambda_R.to(g_T.device, g_T.dtype) + alpha_t)
    h = U_R.to(g_T.device, g_T.dtype) @ coeff + g_T_perp_R / alpha_t
    return SafeDirection(h_alpha_R=h, b_T_R=b_T_R, g_T_perp_R=g_T_perp_R)


def exact_dual_safe_direction(
    reference_grad_rows: torch.Tensor,
    g_T: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Exact dual formula before truncation.

    Implements
      (F_R + alpha I)^(-1) g_T
      = alpha^{-1} g_T
        - alpha^{-1} S^T (alpha N_R I + S S^T)^(-1) S g_T.
    """
    S = reference_grad_rows
    N_R = S.shape[0]
    dtype = torch.float64
    S64 = S.to(dtype)
    g64 = g_T.to(dtype)
    A = alpha * N_R * torch.eye(N_R, dtype=dtype, device=S64.device) + S64 @ S64.T
    rhs = S64 @ g64
    sol = torch.linalg.solve(A, rhs)
    out = (g64 / alpha) - (S64.T @ sol) / alpha
    return out.to(torch.float32)


def normalized_safe_update(
    h_alpha_R: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    norm = torch.linalg.norm(h_alpha_R)
    if float(norm) == 0.0:
        raise ValueError("Safe direction has zero norm.")
    return -float(epsilon) * h_alpha_R / norm


# -----------------------------------------------------------------------------
# Task/candidate structure and common basis
# -----------------------------------------------------------------------------


def choose_rank_by_cumulative_energy(
    evals_desc: torch.Tensor,
    delta: float = 0.01,
    max_rank: Optional[int] = None,
    min_rank: int = 0,
    eps: float = 1e-12,
) -> int:
    """Choose a rank from descending eigenvalues using cumulative energy.

    This is used for the residualized task/candidate spectrum. Unlike the
    reference-side Fisher rank, the added task rank is not chosen relative to
    ``alpha``; instead we retain enough residualized task energy to explain at
    least a ``1 - delta`` fraction of the total.
    """
    evals = evals_desc[evals_desc > eps]
    if evals.numel() == 0:
        return 0
    if max_rank is not None:
        evals = evals[:max_rank]
    total = evals.sum()
    if float(total) <= eps:
        return min(min_rank, int(evals.numel()))

    cumulative = torch.cumsum(evals, dim=0) / total
    threshold = 1.0 - delta
    idx = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(threshold, device=cumulative.device),
            right=False,
        ).item()
    )
    K = idx + 1
    K = max(K, min_rank)
    K = min(K, int(evals.numel()))
    return K


def residualize_columns(U_R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Project matrix columns onto the orthogonal complement of span(U_R)."""
    return X - U_R @ (U_R.T @ X)


def residualize_rows(U_R: torch.Tensor, X_rows: torch.Tensor) -> torch.Tensor:
    """Project matrix rows onto the orthogonal complement of span(U_R)."""
    return X_rows - (X_rows @ U_R) @ U_R.T


def _resolve_added_task_rank(
    *,
    K_R: int,
    K_add: Optional[int] = None,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
) -> Optional[int]:
    """Resolve the residualized added-task rank from user-facing arguments.

    ``K_add`` is the preferred argument. ``K_T`` and ``K_common`` are accepted
    as compatibility aliases:
      - ``K_T`` is interpreted as the residualized added-task rank.
      - ``K_common`` implies ``K_add = K_common - K_R``.
    """
    resolved = K_add
    if K_T is not None:
        if resolved is not None and int(K_T) != int(resolved):
            raise ValueError("K_add and K_T disagree.")
        resolved = int(K_T)
    if resolved is not None and resolved < 0:
        raise ValueError("K_add must be nonnegative.")

    if K_common is not None:
        K_common_int = int(K_common)
        if K_common_int < K_R:
            raise ValueError(f"K_common must satisfy K_common >= K_R={K_R}.")
        implied = K_common_int - K_R
        if resolved is not None and int(resolved) != implied:
            raise ValueError("K_common is inconsistent with K_add / K_T.")
        resolved = implied

    return None if resolved is None else int(resolved)


def build_task_candidate_low_rank(
    g_T: torch.Tensor,
    candidate_grads: torch.Tensor,
    U_R: torch.Tensor,
    K_add: Optional[int] = None,
    *,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
    delta: float = 0.01,
    max_rank: Optional[int] = None,
    eps: float = 1e-10,
    compute_in_float64: bool = True,
) -> TaskCandidateLowRank:
    """Build the residualized task/candidate basis.

    This implements the mathematically preferred ``residualize first, then
    low-rank`` construction:
        G_TC = [g_T, g_1, ..., g_{N_C}]
        G_tilde = (I - U_R U_R^T) G_TC
        U_add   = low_rank_basis(G_tilde)

    Parameters
    ----------
    g_T:
        Target gradient vector, shape [d].
    candidate_grads:
        Candidate gradient matrix, shape [d, N_C].
    U_R:
        Reference low-rank basis, shape [d, K_R].
    K_add:
        Optional explicit added-task rank after residualization. When omitted,
        the rank is chosen automatically from the residualized task spectrum
        using cumulative energy and ``delta``.
    """
    if candidate_grads.ndim != 2:
        raise ValueError("candidate_grads must have shape [d, N_C].")
    if g_T.ndim != 1:
        raise ValueError("g_T must have shape [d].")
    if g_T.shape[0] != candidate_grads.shape[0]:
        raise ValueError("g_T and candidate_grads must have the same parameter dimension.")
    if U_R.ndim != 2 or U_R.shape[0] != g_T.shape[0]:
        raise ValueError("U_R must have shape [d, K_R] with the same parameter dimension as g_T.")

    resolved_rank = _resolve_added_task_rank(
        K_R=int(U_R.shape[1]),
        K_add=K_add,
        K_T=K_T,
        K_common=K_common,
    )
    if resolved_rank == 0:
        return TaskCandidateLowRank(
            U_T=torch.zeros((g_T.shape[0], 0), dtype=torch.float32),
            Omega_T=torch.zeros((0,), dtype=torch.float32),
            evals_full=torch.zeros((0,), dtype=torch.float32),
            K_T=0,
            eigensolver="empty",
        )

    G_TC = torch.cat([g_T[:, None], candidate_grads], dim=1)
    n_cols = G_TC.shape[1]

    work_dtype = torch.float64 if compute_in_float64 else G_TC.dtype
    G = G_TC.to(work_dtype)
    U = U_R.to(work_dtype)
    G_tilde = residualize_columns(U, G)

    K_dual = (G_tilde.T @ G_tilde) / float(n_cols)
    requested_eigen_rank = resolved_rank
    if requested_eigen_rank is not None and max_rank is not None:
        requested_eigen_rank = min(int(requested_eigen_rank), int(max_rank))

    evals, evecs, eigensolver = _top_eigenpairs_symmetric(
        K_dual,
        requested_rank=requested_eigen_rank,
        eps=eps,
    )

    if resolved_rank is None:
        resolved_rank = choose_rank_by_cumulative_energy(
            evals_desc=evals,
            delta=delta,
            max_rank=max_rank,
            min_rank=0,
            eps=eps,
        )
    else:
        if max_rank is not None:
            resolved_rank = min(int(resolved_rank), int(max_rank))

    if evals.numel() == 0 or resolved_rank == 0:
        return TaskCandidateLowRank(
            U_T=torch.zeros((g_T.shape[0], 0), dtype=torch.float32),
            Omega_T=torch.zeros((0,), dtype=torch.float32),
            evals_full=evals.to(torch.float32),
            K_T=0,
            eigensolver=eigensolver,
        )

    resolved_rank = min(int(resolved_rank), int(evals.numel()))
    Omega_T = evals[:resolved_rank].clone()
    V_top = evecs[:, :resolved_rank].clone()
    denom = torch.sqrt(
        torch.as_tensor(float(n_cols), dtype=work_dtype, device=G.device) * Omega_T
    )
    U_add = (G_tilde @ V_top) / denom.unsqueeze(0)

    # The task-side block is isotropic under the reduced ridge geometry, so an
    # orthonormal basis spanning the same subspace is sufficient.
    U_add = orthonormalize_columns(U_add.to(torch.float32))
    K_eff = int(U_add.shape[1])

    return TaskCandidateLowRank(
        U_T=U_add,
        Omega_T=Omega_T[:K_eff].to(torch.float32),
        evals_full=evals.to(torch.float32),
        K_T=K_eff,
        eigensolver=eigensolver,
    )


def build_task_candidate_low_rank_from_rows(
    g_T: torch.Tensor,
    candidate_grad_rows: torch.Tensor,
    U_R: torch.Tensor,
    K_add: Optional[int] = None,
    *,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
    delta: float = 0.01,
    max_rank: Optional[int] = None,
    eps: float = 1e-10,
    compute_in_float64: bool = True,
) -> TaskCandidateLowRank:
    """Row-oriented version of the residualized task/candidate construction.

    This computes the same residualize-first basis as
    ``build_task_candidate_low_rank`` while avoiding a separate materialization
    of the column-major candidate matrix ``[d, N_C]``.
    """
    if candidate_grad_rows.ndim != 2:
        raise ValueError("candidate_grad_rows must have shape [N_C, d].")
    if g_T.ndim != 1:
        raise ValueError("g_T must have shape [d].")
    if g_T.shape[0] != candidate_grad_rows.shape[1]:
        raise ValueError("g_T and candidate_grad_rows must have the same parameter dimension.")
    if U_R.ndim != 2 or U_R.shape[0] != g_T.shape[0]:
        raise ValueError("U_R must have shape [d, K_R] with the same parameter dimension as g_T.")

    resolved_rank = _resolve_added_task_rank(
        K_R=int(U_R.shape[1]),
        K_add=K_add,
        K_T=K_T,
        K_common=K_common,
    )
    if resolved_rank == 0:
        return TaskCandidateLowRank(
            U_T=torch.zeros((g_T.shape[0], 0), dtype=torch.float32),
            Omega_T=torch.zeros((0,), dtype=torch.float32),
            evals_full=torch.zeros((0,), dtype=torch.float32),
            K_T=0,
            eigensolver="empty",
        )

    n_candidates = int(candidate_grad_rows.shape[0])
    n_cols = n_candidates + 1
    work_dtype = torch.float64 if compute_in_float64 else candidate_grad_rows.dtype

    C = candidate_grad_rows.to(work_dtype)
    g = g_T.to(work_dtype)
    U = U_R.to(work_dtype)

    C_tilde = residualize_rows(U, C)
    g_tilde = g - U @ (U.T @ g)

    K_dual = torch.empty((n_cols, n_cols), dtype=work_dtype, device=C.device)
    K_dual[0, 0] = torch.dot(g_tilde, g_tilde)
    cross = C_tilde @ g_tilde
    K_dual[0, 1:] = cross
    K_dual[1:, 0] = cross
    K_dual[1:, 1:] = C_tilde @ C_tilde.T
    K_dual /= float(n_cols)

    requested_eigen_rank = resolved_rank
    if requested_eigen_rank is not None and max_rank is not None:
        requested_eigen_rank = min(int(requested_eigen_rank), int(max_rank))

    evals, evecs, eigensolver = _top_eigenpairs_symmetric(
        K_dual,
        requested_rank=requested_eigen_rank,
        eps=eps,
    )

    if resolved_rank is None:
        resolved_rank = choose_rank_by_cumulative_energy(
            evals_desc=evals,
            delta=delta,
            max_rank=max_rank,
            min_rank=0,
            eps=eps,
        )
    else:
        if max_rank is not None:
            resolved_rank = min(int(resolved_rank), int(max_rank))

    if evals.numel() == 0 or resolved_rank == 0:
        return TaskCandidateLowRank(
            U_T=torch.zeros((g_T.shape[0], 0), dtype=torch.float32),
            Omega_T=torch.zeros((0,), dtype=torch.float32),
            evals_full=evals.to(torch.float32),
            K_T=0,
            eigensolver=eigensolver,
        )

    resolved_rank = min(int(resolved_rank), int(evals.numel()))
    Omega_T = evals[:resolved_rank].clone()
    V_top = evecs[:, :resolved_rank].clone()
    denom = torch.sqrt(
        torch.as_tensor(float(n_cols), dtype=work_dtype, device=C.device) * Omega_T
    )

    U_add = C_tilde.T @ V_top[1:, :]
    for col_idx in range(resolved_rank):
        U_add[:, col_idx].add_(g_tilde, alpha=float(V_top[0, col_idx].item()))
    U_add /= denom.unsqueeze(0)

    U_add = orthonormalize_columns(U_add.to(torch.float32))
    K_eff = int(U_add.shape[1])

    return TaskCandidateLowRank(
        U_T=U_add,
        Omega_T=Omega_T[:K_eff].to(torch.float32),
        evals_full=evals.to(torch.float32),
        K_T=K_eff,
        eigensolver=eigensolver,
    )


def orthonormalize_columns(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if X.numel() == 0:
        return X
    Q, R = torch.linalg.qr(X, mode="reduced")
    diag = torch.abs(torch.diag(R))
    keep = diag > eps
    if keep.numel() == 0:
        return X[:, :0]
    return Q[:, keep]


def build_common_basis(
    U_R: torch.Tensor,
    Lambda_R: torch.Tensor,
    U_T: torch.Tensor,
    alpha: float,
) -> CommonBasis:
    """Build U_K = [U_R, U_add] from a residualized task/candidate basis.

    ``U_T`` is assumed to already come from the residualize-first construction.
    We perform one final orthogonalization pass only as a numerical safeguard.
    """
    if U_R.ndim != 2:
        raise ValueError("U_R must have shape [d, K_R].")
    if U_T.ndim != 2:
        raise ValueError("U_T must have shape [d, K_add].")
    if U_R.shape[0] != U_T.shape[0]:
        raise ValueError("U_R and U_T must share the same parameter dimension.")

    residual = U_T - U_R @ (U_R.T @ U_T)
    U_add = orthonormalize_columns(residual)
    K_add = int(U_add.shape[1])

    if K_add > 0:
        U_K = torch.cat([U_R, U_add], dim=1)
    else:
        U_K = U_R

    M_diag = torch.cat(
        [
            Lambda_R,
            torch.zeros(K_add, dtype=Lambda_R.dtype, device=Lambda_R.device),
        ],
        dim=0,
    )
    M_diag = M_diag + float(alpha)

    return CommonBasis(
        U_K=U_K,
        U_add=U_add,
        M_diag=M_diag,
        K=U_K.shape[1],
        K_add=K_add,
    )


def project_gradient_matrix_to_basis(G: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Project gradient matrix into basis coordinates.

    Parameters
    ----------
    G:
        Shape [d, n] or [d].
    U:
        Shape [d, K].
    Returns
    -------
    Shape [K, n] or [K].
    """
    if G.ndim == 1:
        return U.T @ G
    return U.T @ G


def common_basis_safe_coefficients(
    g_T: torch.Tensor,
    common_basis: CommonBasis,
) -> torch.Tensor:
    """Compute c_{alpha,K} = M_{alpha,K}^{-1} b_T in the common basis."""
    U_K = common_basis.U_K.to(g_T.device, g_T.dtype)
    M_diag = common_basis.M_diag.to(g_T.device, g_T.dtype)
    b_T = U_K.T @ g_T
    return b_T / M_diag


def geometry_aware_selector_solution(
    A_K: torch.Tensor,
    c_alpha_K: torch.Tensor,
    M_diag: torch.Tensor,
    l2_reg: float = 0.0,
) -> torch.Tensor:
    """Solve the weighted least-squares selector problem.

    Minimizes
        0.5 (A_K w - c)^T diag(M_diag) (A_K w - c) + 0.5 l2_reg ||w||_2^2.
    """
    sqrt_M = torch.sqrt(M_diag)
    WA = sqrt_M[:, None] * A_K
    Wc = sqrt_M * c_alpha_K
    H = WA.T @ WA
    if l2_reg > 0:
        H = H + l2_reg * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    rhs = WA.T @ Wc
    return torch.linalg.solve(H, rhs)


def _empty_common_basis_from_reference(
    ref_lr: ReferenceLowRank,
    alpha: float,
) -> CommonBasis:
    return CommonBasis(
        U_K=ref_lr.U_R,
        U_add=ref_lr.U_R[:, :0],
        M_diag=ref_lr.Lambda_R + float(alpha),
        K=ref_lr.K_R,
        K_add=0,
    )


def project_candidate_rows_chunked(
    candidate_grad_rows: torch.Tensor,
    U_K: torch.Tensor,
    *,
    chunk_rows: int = 512,
) -> torch.Tensor:
    """Project candidate rows into the common basis in chunks.

    This avoids materializing the full output of ``candidate_grad_rows @ U_K``
    as one large temporary tensor.
    """
    if candidate_grad_rows.ndim != 2:
        raise ValueError("candidate_grad_rows must have shape [N_C, d].")
    if U_K.ndim != 2:
        raise ValueError("U_K must have shape [d, K].")
    if candidate_grad_rows.shape[1] != U_K.shape[0]:
        raise ValueError("candidate_grad_rows width must match U_K height.")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive.")

    num_rows = int(candidate_grad_rows.shape[0])
    out = torch.empty((num_rows, int(U_K.shape[1])), dtype=torch.float32, device="cpu")
    for start in range(0, num_rows, int(chunk_rows)):
        end = min(num_rows, start + int(chunk_rows))
        out[start:end].copy_(candidate_grad_rows[start:end] @ U_K)
    return out


def build_low_rank_safe_inputs_from_gradient_rows(
    reference_grad_rows: torch.Tensor,
    g_T: torch.Tensor,
    candidate_grad_rows: torch.Tensor,
    alpha: float,
    *,
    K_R: Optional[int] = None,
    delta: float = 0.01,
    max_reference_rank: Optional[int] = None,
    reference_eigenvalue_shrinkage: str = "none",
    K_add: Optional[int] = None,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
    auto_task_rank: bool = False,
    delta_task: float = 0.01,
    max_task_rank: Optional[int] = None,
    artifact_path: str | Path | None = None,
    include_full_gradients: bool = False,
    include_basis: bool = True,
    task_candidate_compute_in_float64: bool = False,
    candidate_projection_chunk_rows: int = 512,
) -> LowRankSafeInputs:
    """Build selector-ready low-rank tensors from row-oriented gradients.

    This uses the residualize-first task construction and avoids materializing
    the large column-major candidate matrix ``[d, N_C]``.
    """
    if reference_grad_rows.ndim != 2:
        raise ValueError("reference_grad_rows must have shape [N_R, d].")
    if g_T.ndim != 1:
        raise ValueError("g_T must have shape [d].")
    if candidate_grad_rows.ndim != 2:
        raise ValueError("candidate_grad_rows must have shape [N_C, d].")
    if reference_grad_rows.shape[1] != g_T.numel():
        raise ValueError("reference_grad_rows width must match g_T dimension.")
    if candidate_grad_rows.shape[1] != g_T.numel():
        raise ValueError("candidate_grad_rows width must match g_T dimension.")

    reference_grad_rows = reference_grad_rows.detach().to(dtype=torch.float32, device="cpu")
    g_T = g_T.detach().to(dtype=torch.float32, device="cpu")
    candidate_grad_rows = candidate_grad_rows.detach().to(dtype=torch.float32, device="cpu")

    ref_lr = build_reference_low_rank(
        reference_grad_rows=reference_grad_rows,
        alpha=alpha,
        K_R=K_R,
        delta=delta,
        max_rank=max_reference_rank,
        eigenvalue_shrinkage=reference_eigenvalue_shrinkage,
    )

    requested_k_add = _resolve_added_task_rank(
        K_R=ref_lr.K_R,
        K_add=K_add,
        K_T=K_T,
        K_common=K_common,
    )

    tc_lr: Optional[TaskCandidateLowRank] = None
    if auto_task_rank or (requested_k_add is not None and requested_k_add > 0):
        tc_lr = build_task_candidate_low_rank_from_rows(
            g_T=g_T,
            candidate_grad_rows=candidate_grad_rows,
            U_R=ref_lr.U_R,
            K_add=requested_k_add,
            delta=delta_task,
            max_rank=max_task_rank,
            compute_in_float64=task_candidate_compute_in_float64,
        )
        common_basis = build_common_basis(
            U_R=ref_lr.U_R,
            Lambda_R=ref_lr.Lambda_R,
            U_T=tc_lr.U_T,
            alpha=alpha,
        )
    else:
        common_basis = _empty_common_basis_from_reference(ref_lr=ref_lr, alpha=alpha)

    U_K = common_basis.U_K.to(dtype=torch.float32, device="cpu")
    candidate_features = project_candidate_rows_chunked(
        candidate_grad_rows,
        U_K,
        chunk_rows=candidate_projection_chunk_rows,
    )
    target_feature = U_K.T @ g_T

    reference_fisher = torch.cat(
        [
            ref_lr.Lambda_R.to(dtype=torch.float32, device="cpu"),
            torch.zeros(common_basis.K_add, dtype=torch.float32),
        ],
        dim=0,
    )

    if include_full_gradients:
        reference_rows_for_output = reference_grad_rows
        candidate_grads_for_output = candidate_grad_rows.T
        g_T_for_output = g_T
    else:
        reference_rows_for_output = torch.empty(0, dtype=torch.float32)
        candidate_grads_for_output = torch.empty(0, dtype=torch.float32)
        g_T_for_output = torch.empty(0, dtype=torch.float32)

    # Release the large dense gradient matrices as soon as the projected
    # features are materialized. They are not needed downstream unless the
    # caller explicitly asked to keep full gradients.
    if not include_full_gradients:
        del reference_grad_rows
        del candidate_grad_rows
        del g_T
        gc.collect()

    out = LowRankSafeInputs(
        candidate_features=candidate_features,
        target_feature=target_feature,
        target_features=None,
        reference_fisher=reference_fisher,
        reference_grad_rows=reference_rows_for_output,
        g_T=g_T_for_output,
        candidate_grads=candidate_grads_for_output,
        reference_low_rank=ref_lr,
        task_candidate_low_rank=tc_lr,
        common_basis=common_basis,
    )

    if artifact_path is not None:
        save_low_rank_safe_artifacts(
            out,
            artifact_path,
            include_full_gradients=include_full_gradients,
            include_basis=include_basis,
        )
        out.artifact_path = str(artifact_path)

    return out


def _gradient_rows_from_examples(
    model: nn.Module,
    tokenizer,
    examples: Sequence[dict[str, Any]],
    *,
    device: torch.device | str,
    max_seq_len: int,
    use_lora: bool = True,
    max_examples: Optional[int] = None,
    show_progress: bool = True,
    desc: str = "Computing full gradients",
) -> torch.Tensor:
    if max_examples is None:
        subset = list(examples)
    else:
        subset = list(examples[: max(0, int(max_examples))])
    if not subset:
        raise ValueError(f"No examples were provided for: {desc}")

    from .extract_gradients import compute_per_example_gradient

    model.eval()
    iterator = tqdm(subset, desc=desc) if show_progress else subset
    rows: List[torch.Tensor] = []
    for example in iterator:
        gradient = compute_per_example_gradient(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=max_seq_len,
            use_lora=use_lora,
        )
        rows.append(gradient.detach().to(dtype=torch.float32, device="cpu"))
    return torch.stack(rows, dim=0)


def project_examples_to_basis(
    model: nn.Module,
    tokenizer,
    examples: Sequence[dict[str, Any]],
    basis: torch.Tensor,
    *,
    device: torch.device | str,
    max_seq_len: int,
    use_lora: bool = True,
    show_progress: bool = True,
    desc: str = "Projecting gradients into low-rank basis",
) -> torch.Tensor:
    """Project per-example gradients into an existing low-rank basis.

    This avoids materializing the full candidate gradient matrix when the basis
    has already been learned from a smaller subset.
    """
    if basis.ndim != 2:
        raise ValueError("basis must have shape [d, K].")
    if not examples:
        raise ValueError(f"No examples were provided for: {desc}")

    from .extract_gradients import compute_per_example_gradient

    basis_cpu = basis.detach().to(dtype=torch.float32, device="cpu")
    num_examples = len(examples)
    out = torch.empty((num_examples, int(basis_cpu.shape[1])), dtype=torch.float32, device="cpu")

    model.eval()
    iterator = tqdm(enumerate(examples), total=num_examples, desc=desc) if show_progress else enumerate(examples)
    for row_idx, example in iterator:
        gradient = compute_per_example_gradient(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=max_seq_len,
            use_lora=use_lora,
        )
        projected = gradient.detach().to(dtype=torch.float32, device="cpu") @ basis_cpu
        out[row_idx].copy_(projected)
    return out


def compute_low_rank_safe_inputs_from_examples(
    model: nn.Module,
    tokenizer,
    candidates: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    references: Sequence[dict[str, Any]],
    *,
    alpha: float,
    device: torch.device | str,
    max_seq_len: int,
    K_R: Optional[int] = None,
    delta: float = 0.01,
    max_reference_rank: Optional[int] = None,
    reference_eigenvalue_shrinkage: str = "none",
    K_add: Optional[int] = None,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
    auto_task_rank: bool = False,
    delta_task: float = 0.01,
    max_task_rank: Optional[int] = None,
    use_lora: bool = True,
    max_reference_examples: Optional[int] = None,
    max_target_examples: Optional[int] = None,
    max_candidate_examples: Optional[int] = None,
    projection_examples: Optional[Sequence[dict[str, Any]]] = None,
    save_target_features: bool = False,
    show_progress: bool = True,
    artifact_path: str | Path | None = None,
    include_full_gradients: bool = False,
    include_basis: bool = True,
    candidate_projection_chunk_rows: int = 512,
) -> LowRankSafeInputs:
    """Example-driven entry point for the residualize-first low-rank pipeline."""
    reference_grad_rows = _gradient_rows_from_examples(
        model=model,
        tokenizer=tokenizer,
        examples=references,
        device=device,
        max_seq_len=max_seq_len,
        use_lora=use_lora,
        max_examples=max_reference_examples,
        show_progress=show_progress,
        desc="Computing reference full gradients",
    )
    target_rows = _gradient_rows_from_examples(
        model=model,
        tokenizer=tokenizer,
        examples=targets,
        device=device,
        max_seq_len=max_seq_len,
        use_lora=use_lora,
        max_examples=max_target_examples,
        show_progress=show_progress,
        desc="Computing target full gradients",
    )
    g_T = target_rows.mean(dim=0)

    candidate_rows = _gradient_rows_from_examples(
        model=model,
        tokenizer=tokenizer,
        examples=candidates,
        device=device,
        max_seq_len=max_seq_len,
        use_lora=use_lora,
        max_examples=max_candidate_examples,
        show_progress=show_progress,
        desc="Computing candidate full gradients",
    )

    out = build_low_rank_safe_inputs_from_gradient_rows(
        reference_grad_rows=reference_grad_rows,
        g_T=g_T,
        candidate_grad_rows=candidate_rows,
        alpha=alpha,
        K_R=K_R,
        delta=delta,
        max_reference_rank=max_reference_rank,
        reference_eigenvalue_shrinkage=reference_eigenvalue_shrinkage,
        K_add=K_add,
        K_T=K_T,
        K_common=K_common,
        auto_task_rank=auto_task_rank,
        delta_task=delta_task,
        max_task_rank=max_task_rank,
        artifact_path=artifact_path,
        include_full_gradients=include_full_gradients,
        include_basis=include_basis,
        task_candidate_compute_in_float64=False,
        candidate_projection_chunk_rows=candidate_projection_chunk_rows,
    )

    del reference_grad_rows
    del candidate_rows
    del g_T
    gc.collect()

    if projection_examples is not None:
        out.candidate_features = project_examples_to_basis(
            model=model,
            tokenizer=tokenizer,
            examples=projection_examples,
            basis=out.common_basis.U_K,
            device=device,
            max_seq_len=max_seq_len,
            use_lora=use_lora,
            show_progress=show_progress,
            desc="Projecting full candidate gradients into low-rank basis",
        )
    if save_target_features:
        U_K = out.common_basis.U_K.to(dtype=torch.float32, device="cpu")
        out.target_features = target_rows.to(dtype=torch.float32, device="cpu") @ U_K
    else:
        out.target_features = None
    del target_rows
    gc.collect()
    return out


def save_low_rank_safe_artifacts(
    safe_inputs: LowRankSafeInputs,
    artifact_path: str | Path,
    *,
    include_full_gradients: bool = False,
    include_basis: bool = True,
) -> None:
    """Save low-rank selector tensors and reproducibility artifacts."""
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ref_lr = safe_inputs.reference_low_rank
    common = safe_inputs.common_basis
    payload: dict[str, Any] = {
        "candidate_features": safe_inputs.candidate_features.cpu(),
        "target_feature": safe_inputs.target_feature.cpu(),
        "target_features": None if safe_inputs.target_features is None else safe_inputs.target_features.cpu(),
        "reference_fisher": safe_inputs.reference_fisher.cpu(),
        "reference_low_rank": None,
        "task_candidate_low_rank": None,
        "common_basis": None,
        "metadata": {
            "candidate_features_shape": tuple(safe_inputs.candidate_features.shape),
            "target_feature_shape": tuple(safe_inputs.target_feature.shape),
            "target_features_shape": None
            if safe_inputs.target_features is None
            else tuple(safe_inputs.target_features.shape),
            "reference_fisher_shape": tuple(safe_inputs.reference_fisher.shape),
            "resolved_common_rank": common.K,
            "resolved_reference_rank": ref_lr.K_R,
            "reference_eigensolver": ref_lr.eigensolver,
            "reference_eigenvalue_shrinkage": ref_lr.shrinkage_method,
            "reference_shrinkage_gamma": ref_lr.shrinkage_gamma,
            "reference_shrinkage_valid_count": None
            if ref_lr.shrinkage_valid_mask is None
            else int(ref_lr.shrinkage_valid_mask.sum().item()),
            "resolved_added_task_rank": common.K_add,
            "resolved_task_rank": common.K_add,  # compatibility alias
            "task_eigensolver": None
            if safe_inputs.task_candidate_low_rank is None
            else safe_inputs.task_candidate_low_rank.eigensolver,
            "task_basis_construction": "residualize_first",
            "reference_fisher_representation": "diagonal_in_common_reference_eigenbasis",
            "includes_full_gradients": bool(include_full_gradients),
            "includes_basis": bool(include_basis),
        },
    }
    if include_basis:
        payload["reference_low_rank"] = {
            "U_R": ref_lr.U_R.cpu(),
            "Lambda_R": ref_lr.Lambda_R.cpu(),
            "Lambda_R_raw": ref_lr.Lambda_R_raw.cpu(),
            "evals_full": ref_lr.evals_full.cpu(),
            "V_top": ref_lr.V_top.cpu(),
            "K_R": ref_lr.K_R,
            "alpha": ref_lr.alpha,
            "delta": ref_lr.delta,
            "shrinkage_method": ref_lr.shrinkage_method,
            "shrinkage_gamma": ref_lr.shrinkage_gamma,
            "shrinkage_valid_mask": None
            if ref_lr.shrinkage_valid_mask is None
            else ref_lr.shrinkage_valid_mask.cpu(),
            "eigensolver": ref_lr.eigensolver,
        }
        payload["common_basis"] = {
            "U_K": common.U_K.cpu(),
            "U_add": common.U_add.cpu(),
            "M_diag": common.M_diag.cpu(),
            "K": common.K,
            "K_add": common.K_add,
        }
        if safe_inputs.task_candidate_low_rank is not None:
            tc_lr = safe_inputs.task_candidate_low_rank
            payload["task_candidate_low_rank"] = {
                "U_T": tc_lr.U_T.cpu(),
                "U_add": tc_lr.U_add.cpu(),
                "Omega_T": tc_lr.Omega_T.cpu(),
                "Omega_add": tc_lr.Omega_add.cpu(),
                "evals_full": tc_lr.evals_full.cpu(),
                "K_T": tc_lr.K_T,
                "K_add": tc_lr.K_add,
                "construction": "residualize_first",
                "eigensolver": tc_lr.eigensolver,
            }
    if include_full_gradients:
        payload["full_gradients"] = {
            "reference_grad_rows": safe_inputs.reference_grad_rows.cpu(),
            "g_T": safe_inputs.g_T.cpu(),
            "candidate_grads": safe_inputs.candidate_grads.cpu(),
        }
    torch.save(payload, path)


# -----------------------------------------------------------------------------
# End-to-end convenience wrapper
# -----------------------------------------------------------------------------


@dataclass
class LowRankBuildOutput:
    reference_grad_rows: torch.Tensor
    reference_low_rank: ReferenceLowRank
    g_T: torch.Tensor
    candidate_grads: torch.Tensor
    safe_direction: SafeDirection
    task_candidate_low_rank: Optional[TaskCandidateLowRank] = None
    common_basis: Optional[CommonBasis] = None


def build_low_rank_objects(
    model: nn.Module,
    params: Sequence[nn.Parameter],
    reference_loader: Iterable[Batch],
    target_loader: Iterable[Batch],
    candidate_loader: Iterable[Batch],
    alpha: float,
    device: torch.device | str,
    reference_loss_fn: LossClosure = causal_lm_sequence_nll,
    target_loss_fn: LossClosure = causal_lm_sequence_nll,
    candidate_loss_fn: LossClosure = causal_lm_sequence_nll,
    K_R: Optional[int] = None,
    delta: float = 0.01,
    max_reference_rank: Optional[int] = None,
    reference_eigenvalue_shrinkage: str = "none",
    K_add: Optional[int] = None,
    K_T: Optional[int] = None,
    K_common: Optional[int] = None,
    auto_task_rank: bool = False,
    delta_task: float = 0.01,
    max_task_rank: Optional[int] = None,
    max_reference_examples: Optional[int] = None,
    max_target_examples: Optional[int] = None,
    max_candidate_examples: Optional[int] = None,
) -> LowRankBuildOutput:
    """Complete pipeline matching the residualize-first construction."""
    # 1) Reference rows S_R^{row}
    reference_grad_rows = per_example_gradient_matrix(
        model=model,
        data_iter=reference_loader,
        params=params,
        loss_fn=reference_loss_fn,
        device=device,
        max_examples=max_reference_examples,
    )

    # 2) Reference low-rank Fisher basis U_R, Lambda_R
    ref_lr = build_reference_low_rank(
        reference_grad_rows=reference_grad_rows,
        alpha=alpha,
        K_R=K_R,
        delta=delta,
        max_rank=max_reference_rank,
        eigenvalue_shrinkage=reference_eigenvalue_shrinkage,
    )

    # 3) Target gradient g_T (average over target anchor examples)
    g_T = mean_gradient_over_examples(
        model=model,
        data_iter=target_loader,
        params=params,
        loss_fn=target_loss_fn,
        device=device,
        max_examples=max_target_examples,
    )

    # 4) Candidate gradients g_i as columns [d, N_C]
    cand_rows = per_example_gradient_matrix(
        model=model,
        data_iter=candidate_loader,
        params=params,
        loss_fn=candidate_loss_fn,
        device=device,
        max_examples=max_candidate_examples,
    )
    candidate_grads = cand_rows.T.contiguous()

    # 5) Safe direction h_{alpha,R}
    safe_dir = reference_safe_direction(
        g_T=g_T,
        U_R=ref_lr.U_R,
        Lambda_R=ref_lr.Lambda_R,
        alpha=alpha,
    )

    out = LowRankBuildOutput(
        reference_grad_rows=reference_grad_rows,
        reference_low_rank=ref_lr,
        g_T=g_T,
        candidate_grads=candidate_grads,
        safe_direction=safe_dir,
    )

    requested_k_add = _resolve_added_task_rank(
        K_R=ref_lr.K_R,
        K_add=K_add,
        K_T=K_T,
        K_common=K_common,
    )

    if auto_task_rank or (requested_k_add is not None and requested_k_add > 0):
        tc_lr = build_task_candidate_low_rank(
            g_T=g_T,
            candidate_grads=candidate_grads,
            U_R=ref_lr.U_R,
            K_add=requested_k_add,
            delta=delta_task,
            max_rank=max_task_rank,
        )
        out.task_candidate_low_rank = tc_lr
        out.common_basis = build_common_basis(
            U_R=ref_lr.U_R,
            Lambda_R=ref_lr.Lambda_R,
            U_T=tc_lr.U_T,
            alpha=alpha,
        )
    elif requested_k_add == 0:
        out.common_basis = _empty_common_basis_from_reference(ref_lr=ref_lr, alpha=alpha)

    return out


# -----------------------------------------------------------------------------
# Example usage sketch
# -----------------------------------------------------------------------------


EXAMPLE_USAGE = r"""
# Example sketch (pseudo-usage):
#
# from transformers import AutoModelForCausalLM
# from peft import get_peft_model, LoraConfig
#
# model = AutoModelForCausalLM.from_pretrained(...).to(device)
# model = get_peft_model(model, LoraConfig(...))
# _, params = collect_trainable_parameters(model, requires_grad_only=True)
#
# out = build_low_rank_objects(
#     model=model,
#     params=params,
#     reference_loader=reference_loader,   # reference prompts
#     target_loader=target_loader,         # target anchor set
#     candidate_loader=candidate_loader,   # candidate pool
#     alpha=1e-2,
#     device=device,
#     reference_loss_fn=causal_lm_sequence_nll,
#     target_loss_fn=causal_lm_sequence_nll,
#     candidate_loss_fn=causal_lm_sequence_nll,
#     delta=0.01,
#     auto_task_rank=True,
#     delta_task=0.01,
#     max_task_rank=64,
# )
#
# U_R = out.reference_low_rank.U_R
# Lambda_R = out.reference_low_rank.Lambda_R
# g_T = out.g_T
# G_C = out.candidate_grads
# h_alpha_R = out.safe_direction.h_alpha_R
#
# if out.common_basis is not None:
#     U_K = out.common_basis.U_K
#     A_K = (U_K.T @ G_C).T
#     b_K = U_K.T @ g_T
#     c_K = b_K / out.common_basis.M_diag
"""


if __name__ == "__main__":
    print(__doc__)
    print(EXAMPLE_USAGE)
