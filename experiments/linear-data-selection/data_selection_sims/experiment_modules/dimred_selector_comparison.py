from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ..core import project_to_ball, solve_case4_budgeted_update, solve_damped
from ..plotting import apply_style, save_figure
from ._shared import ensure_output_dir, write_frame, write_json

Array = np.ndarray


@dataclass(frozen=True)
class DimRedSelectorConfig:
    d: int = 200
    r: int = 20
    n_candidates: int = 800
    n_target: int = 32
    n_reference: int = 300
    k_select: int = 40
    sigma2: float = 1.0
    spike_high: float = 25.0
    spike_low: float = 2.0
    tail_eig: float = 0.05
    spectrum_decay: str = "geometric"
    powerlaw_exponent: float = 1.0
    target_cov_scale: float = 1.0
    target_r: int | None = None
    target_spike_high: float | None = None
    target_spike_low: float | None = None
    target_tail_eig: float | None = None
    target_spectrum_decay: str | None = None
    target_powerlaw_exponent: float | None = None
    candidate_noise: float = 1.0
    alpha_sig: float = 0.25
    epsilon: float = 1.0
    alpha: float = 1.0
    rho: float | None = None
    alpha_min: float = 1.0e-8
    alpha_max: float = 1.0e8
    eta: float | None = None
    shortlist_mult: int = 10
    seeds: int = 5
    base_seed: int = 0
    regimes: tuple[str, ...] = (
        "reference_aligned",
        "residual_cheap",
        "adversarial_omitted",
    )
    score_rank_pairs: tuple[tuple[int, int], ...] = ((5, 5), (10, 10), (20, 20))


@dataclass(frozen=True)
class LinearProblem:
    d: int
    sigma2: float
    U_R: Array
    eigvals_R: Array
    F_R: Array
    Sigma_R: Array
    U_T: Array
    eigvals_T: Array
    Sigma_T: Array
    beta_T: Array
    g_T: Array
    G_candidates: Array
    candidate_labels: Array
    target_grads: Array
    g_T_hat: Array
    X_reference: Array
    F_R_hat: Array


METHOD_LABELS = {
    "safe-full": "SAFE full",
    "safe-lowrank": "SAFE low-rank",
    "safe-reference-only": "SAFE reference-only",
    "safe-task-only": "SAFE task-only",
    "safe-random-k": "SAFE random-K",
    "safe-diagonal": "SAFE diagonal",
    "baseline-less": "LESS",
    "baseline-random": "random",
    "baseline-prismatic": "PRISM",
}

METHOD_COLORS = {
    "safe-full": "#e41a1c",
    "safe-lowrank": "#377eb8",
    "safe-reference-only": "#984ea3",
    "safe-task-only": "#4daf4a",
    "safe-random-k": "#ff7f00",
    "safe-diagonal": "#a65628",
    "baseline-less": "#17becf",
    "baseline-random": "#8c564b",
    "baseline-prismatic": "#bcbd22",
}


def regime_offset(regime: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(regime)) % 997


def random_orthonormal(
    dimension: int,
    rank: int,
    rng: np.random.Generator,
    avoid: Array | None = None,
) -> Array:
    if rank <= 0:
        return np.zeros((dimension, 0), dtype=float)
    if avoid is not None and avoid.size:
        rank = min(rank, dimension - avoid.shape[1])
    matrix = rng.normal(size=(dimension, rank + 8))
    if avoid is not None and avoid.size:
        matrix = matrix - avoid @ (avoid.T @ matrix)
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return np.asarray(basis[:, :rank], dtype=float)


def top_eig(matrix: Array, rank: int, tol: float = 1.0e-12) -> tuple[Array, Array]:
    dimension = matrix.shape[0]
    if rank <= 0:
        return np.zeros((dimension, 0), dtype=float), np.zeros(0, dtype=float)
    rank = min(rank, dimension)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > tol
    return np.asarray(eigvecs[:, keep][:, :rank], dtype=float), np.asarray(eigvals[keep][:rank], dtype=float)


def orthonormalize_columns(matrix: Array, tol: float = 1.0e-10) -> Array:
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    basis, upper = np.linalg.qr(matrix, mode="reduced")
    keep = np.abs(np.diag(upper)) > tol
    return np.asarray(basis[:, keep], dtype=float)


def complete_basis_piece(existing_basis: Array, candidates: Array, rank: int) -> Array:
    if rank <= 0:
        return np.zeros((candidates.shape[0], 0), dtype=float)
    residual = candidates.copy()
    if existing_basis.size:
        residual = residual - existing_basis @ (existing_basis.T @ residual)
    basis = orthonormalize_columns(residual)
    return np.asarray(basis[:, : min(rank, basis.shape[1])], dtype=float)


def sym_psd_from_eig(basis: Array, eigvals: Array) -> Array:
    return np.asarray((basis * eigvals[None, :]) @ basis.T, dtype=float)


def matvec_cov_from_eig(standard_normal_rows: Array, basis: Array, eigvals: Array) -> Array:
    return np.asarray((standard_normal_rows * np.sqrt(eigvals)[None, :]) @ basis.T, dtype=float)


def cosine(vector_a: Array, vector_b: Array, eps: float = 1.0e-12) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a < eps or norm_b < eps:
        return float("nan")
    return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))


def make_spiked_eigs(
    dimension: int,
    rank: int,
    spike_high: float,
    spike_low: float,
    tail_eig: float,
    *,
    spectrum_decay: str = "geometric",
    powerlaw_exponent: float = 1.0,
) -> Array:
    rank = min(rank, dimension)
    if rank <= 0:
        return np.full(dimension, tail_eig, dtype=float)
    if rank == 1:
        spikes = np.array([spike_high], dtype=float)
    elif spectrum_decay == "geometric":
        spikes = np.geomspace(spike_high, spike_low, rank)
    elif spectrum_decay == "linear":
        spikes = np.linspace(spike_high, spike_low, rank)
    elif spectrum_decay == "powerlaw":
        indices = np.arange(1, rank + 1, dtype=float)
        raw = indices ** (-float(powerlaw_exponent))
        raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1.0e-12)
        spikes = spike_low + (spike_high - spike_low) * raw
    else:
        raise ValueError("spectrum_decay must be one of: geometric, linear, powerlaw.")
    if dimension > rank:
        return np.concatenate([spikes, np.full(dimension - rank, tail_eig, dtype=float)])
    return np.asarray(spikes, dtype=float)


def make_beta(
    regime: str,
    basis: Array,
    config: DimRedSelectorConfig,
    rng: np.random.Generator,
) -> Array:
    dimension = basis.shape[0]
    rank = min(config.r, dimension)

    def _normed(vector: Array) -> Array:
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm < 1.0e-12:
            return vector
        return vector / vector_norm

    if regime == "reference_aligned":
        coefficients = np.geomspace(1.0, 0.2, max(rank, 1))
        return _normed(basis[:, :rank] @ coefficients[:rank])

    if regime == "residual_cheap":
        ref_coefficients = rng.normal(size=max(rank, 1))
        ref_direction = _normed(basis[:, :rank] @ ref_coefficients[:rank]) if rank > 0 else np.zeros(dimension, dtype=float)
        if rank < dimension:
            cheap_index = rank + int(0.25 * max(dimension - rank - 1, 0))
            cheap_direction = basis[:, cheap_index]
        else:
            cheap_direction = random_orthonormal(dimension, 1, rng, avoid=basis[:, :rank])[:, 0]
        vector = math.sqrt(config.alpha_sig) * ref_direction + math.sqrt(max(1.0 - config.alpha_sig, 0.0)) * cheap_direction
        return _normed(vector)

    if regime == "adversarial_omitted":
        cutoff_like = max(1, min(rank - 1, config.score_rank_pairs[len(config.score_rank_pairs) // 2][0]))
        kept_index = 0
        omitted_index = min(cutoff_like + 1, dimension - 1)
        vector = math.sqrt(0.65) * basis[:, kept_index] + math.sqrt(0.35) * basis[:, omitted_index]
        return _normed(vector)

    raise ValueError(f"Unknown regime: {regime}")


def sample_rows_from_diag_cov(
    count: int,
    basis: Array,
    eigvals: Array,
    rng: np.random.Generator,
) -> Array:
    standard_normal = rng.normal(size=(count, len(eigvals)))
    return matvec_cov_from_eig(standard_normal, basis, eigvals)


def sample_linear_gradients(
    count: int,
    basis: Array,
    eigvals_cov: Array,
    beta_target: Array,
    sigma2: float,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> tuple[Array, Array, Array]:
    features = sample_rows_from_diag_cov(count, basis, eigvals_cov, rng)
    noise = rng.normal(scale=math.sqrt(sigma2) * noise_scale, size=count)
    responses = features @ beta_target + noise
    gradients = -(responses[:, None] * features) / sigma2
    return features, responses, gradients


def sample_candidate_pool(
    count: int,
    basis: Array,
    dimension: int,
    rank: int,
    beta_target: Array,
    sigma2: float,
    rng: np.random.Generator,
    noise_scale: float,
) -> tuple[Array, Array]:
    labels = rng.choice(3, size=count, p=np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))
    features = np.zeros((count, dimension), dtype=float)
    rank = min(rank, dimension)
    tail_start = min(rank, dimension - 1)

    for component in range(3):
        indices = np.where(labels == component)[0]
        if len(indices) == 0:
            continue
        if component == 0:
            diagonal = np.full(dimension, 0.15, dtype=float)
            diagonal[: max(rank, 1)] = 4.0
        elif component == 1:
            diagonal = np.full(dimension, 0.15, dtype=float)
            diagonal[tail_start:] = 4.0
        else:
            diagonal = np.ones(dimension, dtype=float)
        features[indices] = sample_rows_from_diag_cov(len(indices), basis, diagonal, rng)

    noise = rng.normal(scale=math.sqrt(sigma2) * noise_scale, size=count)
    responses = features @ beta_target + noise
    gradients = -(responses[:, None] * features) / sigma2
    return np.asarray(gradients, dtype=float), labels


def build_problem(config: DimRedSelectorConfig, seed: int, regime: str) -> LinearProblem:
    rng = np.random.default_rng(seed)
    dimension = config.d
    rank = min(config.r, dimension)

    basis_reference = random_orthonormal(dimension, dimension, rng)
    eigvals_reference = make_spiked_eigs(
        dimension,
        rank,
        config.spike_high,
        config.spike_low,
        config.tail_eig,
        spectrum_decay=config.spectrum_decay,
        powerlaw_exponent=config.powerlaw_exponent,
    )
    sigma_reference = sym_psd_from_eig(basis_reference, eigvals_reference)
    fisher_reference = sigma_reference / config.sigma2

    if config.target_spectrum_decay is None:
        basis_target = np.eye(dimension, dtype=float)
        eigvals_target = np.full(dimension, config.target_cov_scale, dtype=float)
        sigma_target = np.eye(dimension, dtype=float) * config.target_cov_scale
    else:
        target_rank = min(config.target_r if config.target_r is not None else config.r, dimension)
        target_spike_high = (
            config.target_spike_high if config.target_spike_high is not None else config.spike_high
        ) * config.target_cov_scale
        target_spike_low = (
            config.target_spike_low if config.target_spike_low is not None else config.spike_low
        ) * config.target_cov_scale
        target_tail_eig = (
            config.target_tail_eig if config.target_tail_eig is not None else config.tail_eig
        ) * config.target_cov_scale
        target_powerlaw_exponent = (
            config.target_powerlaw_exponent
            if config.target_powerlaw_exponent is not None
            else config.powerlaw_exponent
        )
        basis_target = random_orthonormal(dimension, dimension, rng)
        eigvals_target = make_spiked_eigs(
            dimension,
            target_rank,
            target_spike_high,
            target_spike_low,
            target_tail_eig,
            spectrum_decay=config.target_spectrum_decay,
            powerlaw_exponent=target_powerlaw_exponent,
        )
        sigma_target = sym_psd_from_eig(basis_target, eigvals_target)

    beta_target = make_beta(regime, basis_reference, config, rng)
    target_gradient = -(sigma_target @ beta_target) / config.sigma2

    candidate_gradients, candidate_labels = sample_candidate_pool(
        config.n_candidates,
        basis_reference,
        dimension,
        rank,
        beta_target,
        config.sigma2,
        rng,
        config.candidate_noise,
    )
    _, _, target_grads = sample_linear_gradients(
        config.n_target,
        basis_target,
        eigvals_target,
        beta_target,
        config.sigma2,
        rng,
        noise_scale=1.0,
    )
    target_gradient_hat = target_grads.mean(axis=0)
    features_reference = sample_rows_from_diag_cov(config.n_reference, basis_reference, eigvals_reference, rng)
    fisher_reference_hat = (features_reference.T @ features_reference) / (config.n_reference * config.sigma2)

    return LinearProblem(
        d=dimension,
        sigma2=config.sigma2,
        U_R=basis_reference,
        eigvals_R=eigvals_reference,
        F_R=fisher_reference,
        Sigma_R=sigma_reference,
        U_T=basis_target,
        eigvals_T=eigvals_target,
        Sigma_T=sigma_target,
        beta_T=beta_target,
        g_T=target_gradient,
        G_candidates=candidate_gradients,
        candidate_labels=candidate_labels,
        target_grads=target_grads,
        g_T_hat=target_gradient_hat,
        X_reference=features_reference,
        F_R_hat=fisher_reference_hat,
    )


def drift(delta: Array, fisher_reference: Array) -> float:
    return 0.5 * float(delta.T @ fisher_reference @ delta)


def exact_target_gain(delta: Array, sigma_target: Array, beta_target: Array, sigma2: float) -> float:
    return float((delta.T @ sigma_target @ beta_target - 0.5 * delta.T @ sigma_target @ delta) / sigma2)


def first_order_gain(delta: Array, target_gradient: Array) -> float:
    return float(-target_gradient.T @ delta)


def phi_alpha_from_eig(alpha: float, eigvals: Array, gradient_coords: Array) -> float:
    denominator_terms = (gradient_coords**2) / (eigvals + alpha) ** 2
    denominator = float(denominator_terms.sum())
    if denominator <= 1.0e-30:
        return 0.0
    return float((eigvals * denominator_terms).sum() / denominator)


def solve_alpha_for_budget(
    fisher_reference: Array,
    target_gradient: Array,
    epsilon: float,
    rho: float,
    alpha_min: float,
    alpha_max: float,
) -> float:
    eigvals, eigvecs = np.linalg.eigh(0.5 * (fisher_reference + fisher_reference.T))
    eigvals = np.maximum(eigvals, 0.0)
    gradient_coords = eigvecs.T @ target_gradient
    target_ratio = 2.0 * rho / max(epsilon * epsilon, 1.0e-30)
    lo = alpha_min
    hi = alpha_max
    phi_lo = phi_alpha_from_eig(lo, eigvals, gradient_coords)
    phi_hi = phi_alpha_from_eig(hi, eigvals, gradient_coords)
    if target_ratio <= min(phi_lo, phi_hi):
        return lo if phi_lo <= phi_hi else hi
    if target_ratio >= max(phi_lo, phi_hi):
        return hi if phi_hi >= phi_lo else lo
    increasing = phi_hi > phi_lo
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        phi_mid = phi_alpha_from_eig(mid, eigvals, gradient_coords)
        if (phi_mid < target_ratio and increasing) or (phi_mid > target_ratio and not increasing):
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def get_alpha(config: DimRedSelectorConfig, fisher_reference: Array, target_gradient: Array) -> float:
    if config.rho is not None:
        return solve_alpha_for_budget(
            fisher_reference=fisher_reference,
            target_gradient=target_gradient,
            epsilon=config.epsilon,
            rho=config.rho,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
        )
    return float(config.alpha)


def build_lowrank_basis(
    fisher_basis: Array,
    candidate_gradients: Array,
    target_gradient: Array,
    k_reference: int,
    k_task: int,
    *,
    use_reference: bool = True,
    use_task: bool = True,
    tol: float = 1.0e-10,
) -> tuple[Array, dict[str, Array]]:
    dimension = fisher_basis.shape[0]
    basis_reference = np.zeros((dimension, 0), dtype=float)
    eigvals_reference = np.zeros(0, dtype=float)
    if use_reference and k_reference > 0:
        basis_reference, eigvals_reference = top_eig(fisher_basis, k_reference, tol=tol)

    basis_task = np.zeros((dimension, 0), dtype=float)
    task_spectrum = np.zeros(0, dtype=float)
    if use_task and k_task > 0:
        stacked_gradients = np.vstack([target_gradient[None, :], candidate_gradients])
        residual_gradients = (
            stacked_gradients - (stacked_gradients @ basis_reference) @ basis_reference.T
            if basis_reference.size
            else stacked_gradients
        )
        try:
            _, singular_values, vt = np.linalg.svd(residual_gradients, full_matrices=False)
        except np.linalg.LinAlgError:
            jittered = residual_gradients + 1.0e-12 * np.random.default_rng(0).normal(size=residual_gradients.shape)
            _, singular_values, vt = np.linalg.svd(jittered, full_matrices=False)
        keep = singular_values > tol
        singular_values = singular_values[keep]
        vt = vt[keep]
        task_raw = vt[:k_task].T
        basis_task = complete_basis_piece(basis_reference, task_raw, k_task)
        task_spectrum = (singular_values[: basis_task.shape[1]] ** 2) / max(stacked_gradients.shape[0], 1)

    basis = np.hstack([basis_reference, basis_task]) if basis_reference.size or basis_task.size else np.zeros((dimension, 0), dtype=float)
    basis = orthonormalize_columns(basis)
    return basis, {
        "U_R": basis_reference,
        "lambda_R": eigvals_reference,
        "U_T": basis_task,
        "omega_T": task_spectrum,
    }


def calibrate_eta(candidate_gradients: Array, subset_budget: int, epsilon: float) -> float:
    norms = np.linalg.norm(candidate_gradients, axis=1)
    median_norm = float(np.median(norms))
    if median_norm < 1.0e-12:
        return epsilon
    return epsilon * math.sqrt(max(subset_budget, 1)) / median_norm


class _NumpyTorchCompat:
    float64 = np.float64

    @staticmethod
    def as_tensor(value, dtype=None):
        resolved_dtype = np.float64 if dtype is None else dtype
        return np.asarray(value, dtype=resolved_dtype)


class _NumpyTensorProxy:
    def __init__(self, value: Array | float):
        self._value = np.asarray(value, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._value, dtype=float)

    def __float__(self) -> float:
        array = np.asarray(self._value, dtype=float)
        return float(array.reshape(-1)[0])


@dataclass(frozen=True)
class _NumpySelectionResult:
    selected_indices: np.ndarray
    safe_update: dict[str, object]
    alpha: float
    predicted_target_gain: float
    objective_value: float


def _solve_budgeted_safe_update_numpy(
    *,
    target_gradient: Array,
    fisher: Array,
    alpha: float | None,
    cost_c: float | None,
    epsilon: float,
) -> dict[str, object]:
    target_gradient = np.atleast_1d(np.asarray(target_gradient, dtype=float))
    fisher = np.asarray(fisher, dtype=float)
    fisher = np.diag(fisher) if fisher.ndim == 1 else np.atleast_2d(fisher)
    if cost_c is None:
        if alpha is None:
            raise ValueError("alpha must be provided when cost_c is None.")
        delta = np.atleast_1d(project_to_ball(solve_damped(fisher, target_gradient, float(alpha)), float(epsilon)))
        resolved_alpha = float(alpha)
    else:
        solution = solve_case4_budgeted_update(
            fisher=fisher,
            gradient=target_gradient,
            rho=float(cost_c),
            radius=float(epsilon),
        )
        delta = np.atleast_1d(np.asarray(solution.update, dtype=float))
        resolved_alpha = float(solution.alpha)
    return {
        "delta": _NumpyTensorProxy(delta),
        "alpha": resolved_alpha,
        "reference_cost": 0.5 * float(delta @ fisher @ delta),
        "norm_cost": float(np.linalg.norm(delta)),
    }


def _select_safe_subset_from_gradients_numpy(
    *,
    candidate_gradients: Array,
    candidates,
    target_gradient: Array,
    fisher: Array,
    subset_budget: int,
    alpha: float | None,
    learning_rate: float,
    cost_c: float | None,
    epsilon: float,
    geometry: str,
    solver: str,
    shortlist_size: int | None,
    average_by_budget: bool,
) -> _NumpySelectionResult:
    del candidates, solver

    candidate_gradients = np.asarray(candidate_gradients, dtype=float)
    if candidate_gradients.ndim == 1:
        candidate_gradients = candidate_gradients[:, None]
    target_gradient = np.atleast_1d(np.asarray(target_gradient, dtype=float))
    fisher = np.asarray(fisher, dtype=float)
    fisher = np.diag(fisher) if fisher.ndim == 1 else np.atleast_2d(fisher)
    safe_update = _solve_budgeted_safe_update_numpy(
        target_gradient=target_gradient,
        fisher=fisher,
        alpha=alpha,
        cost_c=cost_c,
        epsilon=epsilon,
    )
    desired_update = np.atleast_1d(safe_update["delta"].numpy())
    scale = (-float(learning_rate) / float(subset_budget)) if average_by_budget else -float(learning_rate)
    candidate_updates = scale * candidate_gradients

    if geometry == "reference":
        transformed_updates = candidate_updates @ fisher.T
        transformed_target = fisher @ desired_update
    else:
        transformed_updates = candidate_updates
        transformed_target = desired_update

    scores = transformed_updates @ transformed_target
    candidate_indices = np.argsort(scores)[::-1]
    if shortlist_size is not None:
        candidate_indices = candidate_indices[: max(int(subset_budget), int(shortlist_size))]

    selected: list[int] = []
    selected_set: set[int] = set()
    aggregate = np.zeros_like(desired_update)
    best_value = float("inf")

    def objective(next_aggregate: Array) -> float:
        delta = next_aggregate - desired_update
        if geometry == "reference":
            return 0.5 * float(delta @ fisher @ delta)
        return 0.5 * float(delta @ delta)

    while len(selected) < int(subset_budget):
        best_index = None
        best_trial_value = float("inf")
        for index in candidate_indices:
            index_int = int(index)
            if index_int in selected_set:
                continue
            trial_aggregate = aggregate + candidate_updates[index_int]
            value = objective(trial_aggregate)
            if value < best_trial_value:
                best_trial_value = value
                best_index = index_int
        if best_index is None:
            break
        selected.append(best_index)
        selected_set.add(best_index)
        aggregate = aggregate + candidate_updates[best_index]
        best_value = best_trial_value

    selected_indices = np.sort(np.asarray(selected, dtype=int))
    subset_update = candidate_updates[selected_indices].sum(axis=0) if selected_indices.size else np.zeros(candidate_gradients.shape[1], dtype=float)
    predicted_target_gain = float((-target_gradient) @ subset_update)
    return _NumpySelectionResult(
        selected_indices=selected_indices,
        safe_update=safe_update,
        alpha=float(safe_update["alpha"]),
        predicted_target_gain=predicted_target_gain,
        objective_value=float(best_value if np.isfinite(best_value) else 0.0),
    )


def _select_less_numpy(
    *,
    candidate_gradients: Array,
    target_gradient: Array,
    subset_budget: int,
    preconditioner: Array | None = None,
    similarity: str = "dot",
) -> np.ndarray:
    gradients = np.asarray(candidate_gradients, dtype=float)
    if preconditioner is not None:
        preconditioner_array = np.asarray(preconditioner, dtype=float)
        if preconditioner_array.ndim == 1:
            gradients = gradients * preconditioner_array[None, :]
        else:
            gradients = gradients @ preconditioner_array.T
    target = -np.asarray(target_gradient, dtype=float)
    if similarity == "cosine":
        target_norm = max(float(np.linalg.norm(target)), 1.0e-12)
        scores = np.array(
            [float(gradient @ target) / max(float(np.linalg.norm(gradient)) * target_norm, 1.0e-12) for gradient in gradients],
            dtype=float,
        )
    else:
        scores = gradients @ target
    return np.argsort(scores)[-int(subset_budget) :]


def _select_random_numpy(total_candidates: int, subset_budget: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.choice(int(total_candidates), size=int(subset_budget), replace=False)


def _select_prismatic_numpy(
    *,
    candidate_gradients: Array,
    subset_budget: int,
    cluster_ratio: float = 0.1,
    sparsity: float = 0.5,
    num_iters: int = 20,
    method: str = "hard",
    seed: int = 0,
) -> np.ndarray:
    del cluster_ratio, sparsity, num_iters, method
    rng = np.random.default_rng(int(seed))
    gradients = np.asarray(candidate_gradients, dtype=float)
    norms = np.linalg.norm(gradients, axis=1)
    normalized = np.array(
        [gradient / max(float(np.linalg.norm(gradient)), 1.0e-12) for gradient in gradients],
        dtype=float,
    )
    first = int(np.argmax(norms))
    selected = [first]
    available = set(range(gradients.shape[0]))
    available.remove(first)
    while len(selected) < int(subset_budget) and available:
        best_index = None
        best_score = -np.inf
        for index in available:
            similarities = np.abs(normalized[selected] @ normalized[index])
            novelty = 1.0 - float(np.max(similarities)) if similarities.size else 1.0
            score = novelty + 1.0e-6 * float(rng.random())
            if score > best_score:
                best_score = score
                best_index = int(index)
        selected.append(best_index)
        available.remove(best_index)
    return np.asarray(selected, dtype=int)


@lru_cache(maxsize=1)
def _load_main_selectors():
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.append(repo_root_str)

    try:
        import torch
    except ModuleNotFoundError as exc:
        del exc
        return (
            _NumpyTorchCompat(),
            _select_safe_subset_from_gradients_numpy,
            _solve_budgeted_safe_update_numpy,
            _select_less_numpy,
            _select_prismatic_numpy,
            _select_random_numpy,
        )

    from selector_methods.baseline_selectors import (
        select_less,
        select_prismatic,
        select_random,
    )
    from selector_methods.safe_selector import (
        select_safe_subset_from_gradients,
        solve_budgeted_safe_update,
    )

    return torch, select_safe_subset_from_gradients, solve_budgeted_safe_update, select_less, select_prismatic, select_random


def _solve_oracle_update(
    target_gradient: Array,
    fisher,
    config: DimRedSelectorConfig,
) -> dict[str, float | Array]:
    torch, _, solve_budgeted_safe_update, _, _, _ = _load_main_selectors()
    solution = solve_budgeted_safe_update(
        target_gradient=torch.as_tensor(target_gradient, dtype=torch.float64),
        fisher=torch.as_tensor(fisher, dtype=torch.float64),
        alpha=None if config.rho is not None else float(config.alpha),
        cost_c=None if config.rho is None else float(config.rho),
        epsilon=float(config.epsilon),
    )
    return {
        "delta": np.asarray(solution["delta"].detach().cpu().numpy(), dtype=float),
        "alpha": float(solution["alpha"]),
        "reference_cost": float(solution["reference_cost"]),
        "norm_cost": float(solution["norm_cost"]),
    }


def _subset_update_from_indices(
    candidate_gradients: Array,
    selected_indices: np.ndarray,
    learning_rate: float,
    subset_budget: int,
) -> Array:
    if len(selected_indices) == 0:
        return np.zeros(candidate_gradients.shape[1], dtype=float)
    return np.asarray(
        (-float(learning_rate) / float(subset_budget)) * candidate_gradients[selected_indices].sum(axis=0),
        dtype=float,
    )


def _run_safe_selector_variant(
    candidate_gradients: Array,
    target_gradient: Array,
    fisher,
    config: DimRedSelectorConfig,
    *,
    learning_rate: float,
    basis: Array | None = None,
    method: str,
    basis_variant: str,
) -> dict[str, float | int | str]:
    torch, select_safe_subset_from_gradients, _, _, _, _ = _load_main_selectors()

    gradients_for_selection = candidate_gradients if basis is None else candidate_gradients @ basis
    target_for_selection = target_gradient if basis is None else basis.T @ target_gradient
    fisher_for_selection = fisher if basis is None else basis.T @ fisher @ basis
    shortlist_size = min(candidate_gradients.shape[0], max(config.k_select, config.shortlist_mult * config.k_select))

    selection_result = select_safe_subset_from_gradients(
        candidate_gradients=torch.as_tensor(gradients_for_selection, dtype=torch.float64),
        candidates=list(range(candidate_gradients.shape[0])),
        target_gradient=torch.as_tensor(target_for_selection, dtype=torch.float64),
        fisher=torch.as_tensor(fisher_for_selection, dtype=torch.float64),
        subset_budget=int(config.k_select),
        alpha=None if config.rho is not None else float(config.alpha),
        learning_rate=float(learning_rate),
        cost_c=None if config.rho is None else float(config.rho),
        epsilon=float(config.epsilon),
        geometry="reference",
        solver="greedy_marginal",
        shortlist_size=shortlist_size,
        average_by_budget=True,
    )
    selected_indices = np.sort(np.asarray(selection_result.selected_indices, dtype=int))
    applied_update = _subset_update_from_indices(candidate_gradients, selected_indices, learning_rate, config.k_select)
    selector_delta = np.asarray(selection_result.safe_update["delta"].detach().cpu().numpy(), dtype=float)
    full_selector_delta = selector_delta if basis is None else basis @ selector_delta
    return {
        "method": method,
        "method_family": "safe_selector",
        "basis_variant": basis_variant,
        "solver": "greedy_marginal",
        "geometry": "reference",
        "selected_count": int(len(selected_indices)),
        "K_actual": int(candidate_gradients.shape[1] if basis is None else basis.shape[1]),
        "selector_alpha": float(selection_result.alpha),
        "selector_reference_cost": float(selection_result.safe_update["reference_cost"]),
        "selector_norm_cost": float(selection_result.safe_update["norm_cost"]),
        "selector_predicted_target_gain": float(selection_result.predicted_target_gain),
        "selector_objective_value": float(selection_result.objective_value),
        "selected_indices": " ".join(str(index) for index in selected_indices.tolist()),
        "update": applied_update,
        "selector_delta": full_selector_delta,
    }


def _run_baseline_variants(
    candidate_gradients: Array,
    target_gradient: Array,
    config: DimRedSelectorConfig,
    *,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float | int | str]]:
    torch, _, _, select_less, select_prismatic, select_random = _load_main_selectors()
    records: list[dict[str, float | int | str]] = []

    baseline_indices = {
        "baseline-less": np.sort(
            np.asarray(
                select_less(
                    candidate_gradients=torch.as_tensor(candidate_gradients, dtype=torch.float64),
                    target_gradient=torch.as_tensor(target_gradient, dtype=torch.float64),
                    subset_budget=int(config.k_select),
                    preconditioner=None,
                    similarity="dot",
                ),
                dtype=int,
            )
        ),
        "baseline-random": np.sort(np.asarray(select_random(candidate_gradients.shape[0], int(config.k_select), seed=int(seed)), dtype=int)),
        "baseline-prismatic": np.sort(
            np.asarray(
                select_prismatic(
                    candidate_gradients=torch.as_tensor(candidate_gradients, dtype=torch.float64),
                    subset_budget=int(config.k_select),
                    cluster_ratio=0.1,
                    sparsity=0.5,
                    num_iters=20,
                    method="hard",
                    seed=int(seed),
                ),
                dtype=int,
            )
        ),
    }

    for method, selected_indices in baseline_indices.items():
        records.append(
            {
                "method": method,
                "method_family": "baseline_selectors",
                "basis_variant": "none",
                "solver": {
                    "baseline-less": "less(dot)",
                    "baseline-random": "random",
                    "baseline-prismatic": "prismatic(hard)",
                }[method],
                "geometry": "euclidean",
                "selected_count": int(len(selected_indices)),
                "K_actual": int(candidate_gradients.shape[1]),
                "selector_alpha": float("nan"),
                "selector_reference_cost": float("nan"),
                "selector_norm_cost": float("nan"),
                "selector_predicted_target_gain": float("nan"),
                "selector_objective_value": float("nan"),
                "selected_indices": " ".join(str(index) for index in selected_indices.tolist()),
                "update": _subset_update_from_indices(candidate_gradients, selected_indices, learning_rate, config.k_select),
                "selector_delta": np.full(candidate_gradients.shape[1], np.nan, dtype=float),
            }
        )
    return records


def _evaluate_method(
    record: dict[str, float | int | str],
    problem: LinearProblem,
    *,
    regime: str,
    seed: int,
    k_reference: int,
    k_task: int,
    k_requested: int,
    learning_rate: float,
    oracle_delta: Array,
    oracle_alpha: float,
    rho: float | None,
) -> dict[str, float | int | str]:
    update = np.asarray(record.pop("update"), dtype=float)
    selector_delta = np.asarray(record.pop("selector_delta"), dtype=float)
    gain = exact_target_gain(update, problem.Sigma_T, problem.beta_T, problem.sigma2)
    first_order = first_order_gain(update, problem.g_T)
    reference_drift = drift(update, problem.F_R)
    return {
        **record,
        "experiment": "dimred-selector-comparison",
        "regime": regime,
        "seed": seed,
        "K_R": int(k_reference),
        "K_T": int(k_task),
        "K_requested": int(k_requested),
        "eta": float(learning_rate),
        "oracle_alpha": float(oracle_alpha),
        "gain": float(gain),
        "first_order_gain": float(first_order),
        "drift": float(reference_drift),
        "drift_budget_ratio": float(reference_drift / rho) if rho is not None and rho > 0.0 else float("nan"),
        "cos_to_full_oracle": cosine(update, oracle_delta),
        "dir_error": 1.0 - cosine(update, oracle_delta) if not np.isnan(cosine(update, oracle_delta)) else float("nan"),
        "norm": float(np.linalg.norm(update)),
        "selector_oracle_cos": cosine(selector_delta, oracle_delta),
    }


def evaluate_selector_bundle(
    config: DimRedSelectorConfig,
    problem: LinearProblem,
    *,
    seed: int,
    regime: str,
    k_reference: int,
    k_task: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed + 12345)
    learning_rate = config.eta if config.eta is not None else calibrate_eta(problem.G_candidates, config.k_select, config.epsilon)
    oracle_solution = _solve_oracle_update(problem.g_T, problem.F_R, config)
    oracle_delta = np.asarray(oracle_solution["delta"], dtype=float)
    oracle_alpha = float(oracle_solution["alpha"])
    k_requested = int(k_reference + k_task)

    records: list[dict[str, float | int | str]] = []
    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            problem.F_R,
            config,
            learning_rate=learning_rate,
            basis=None,
            method="safe-full",
            basis_variant="full",
        )
    )

    basis_lowrank, _ = build_lowrank_basis(problem.F_R, problem.G_candidates, problem.g_T, k_reference, k_task)
    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            problem.F_R,
            config,
            learning_rate=learning_rate,
            basis=basis_lowrank,
            method="safe-lowrank",
            basis_variant="reference+task",
        )
    )

    basis_reference, _ = build_lowrank_basis(
        problem.F_R,
        problem.G_candidates,
        problem.g_T,
        k_reference,
        0,
        use_reference=True,
        use_task=False,
    )
    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            problem.F_R,
            config,
            learning_rate=learning_rate,
            basis=basis_reference,
            method="safe-reference-only",
            basis_variant="reference-only",
        )
    )

    basis_task, _ = build_lowrank_basis(
        problem.F_R,
        problem.G_candidates,
        problem.g_T,
        0,
        k_requested,
        use_reference=False,
        use_task=True,
    )
    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            problem.F_R,
            config,
            learning_rate=learning_rate,
            basis=basis_task,
            method="safe-task-only",
            basis_variant="task-only",
        )
    )

    basis_random = random_orthonormal(problem.d, min(k_requested, problem.d), rng)
    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            problem.F_R,
            config,
            learning_rate=learning_rate,
            basis=basis_random,
            method="safe-random-k",
            basis_variant="random-k",
        )
    )

    records.append(
        _run_safe_selector_variant(
            problem.G_candidates,
            problem.g_T,
            np.diag(problem.F_R).copy(),
            config,
            learning_rate=learning_rate,
            basis=None,
            method="safe-diagonal",
            basis_variant="diagonal",
        )
    )
    records.extend(
        _run_baseline_variants(
            problem.G_candidates,
            problem.g_T,
            config,
            learning_rate=learning_rate,
            seed=seed,
        )
    )

    evaluated = [
        _evaluate_method(
            record,
            problem,
            regime=regime,
            seed=seed,
            k_reference=k_reference,
            k_task=k_task,
            k_requested=k_requested,
            learning_rate=learning_rate,
            oracle_delta=oracle_delta,
            oracle_alpha=oracle_alpha,
            rho=config.rho,
        )
        for record in records
    ]
    return evaluated


def _sem_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _summarize_by_method(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, subframe in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_row = {column: value for column, value in zip(group_columns, keys)}
        rows.append(
            {
                **key_row,
                "num_runs": int(subframe["seed"].nunique()),
                "gain_mean": float(subframe["gain"].mean()),
                "gain_sem": _sem_or_zero(subframe["gain"]),
                "drift_mean": float(subframe["drift"].mean()),
                "drift_sem": _sem_or_zero(subframe["drift"]),
                "cos_to_full_oracle_mean": float(subframe["cos_to_full_oracle"].mean()),
                "cos_to_full_oracle_sem": _sem_or_zero(subframe["cos_to_full_oracle"]),
                "selector_oracle_cos_mean": float(subframe["selector_oracle_cos"].mean()),
                "selector_oracle_cos_sem": _sem_or_zero(subframe["selector_oracle_cos"]),
                "first_order_gain_mean": float(subframe["first_order_gain"].mean()),
                "first_order_gain_sem": _sem_or_zero(subframe["first_order_gain"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_pareto_summary(summary: pd.DataFrame, output_path: Path, regime: str, k_reference: int, k_task: int) -> None:
    if summary.empty:
        return
    apply_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.6, 5.2))
    for _, row in summary.iterrows():
        method = str(row["method"])
        color = METHOD_COLORS.get(method, "#333333")
        axis.errorbar(
            float(row["drift_mean"]),
            float(row["gain_mean"]),
            xerr=float(row["drift_sem"]),
            yerr=float(row["gain_sem"]),
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            alpha=0.6,
            capsize=2.5,
            zorder=2,
        )
        axis.scatter(
            float(row["drift_mean"]),
            float(row["gain_mean"]),
            s=78,
            color=color,
            zorder=3,
        )
        axis.annotate(
            METHOD_LABELS.get(method, method),
            xy=(float(row["drift_mean"]), float(row["gain_mean"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )
    axis.set_xlabel("Mean true reference drift")
    axis.set_ylabel("Mean true target gain (symlog)")
    axis.set_title(f"{regime}: selector tradeoff (K_R={k_reference}, K_T={k_task})")
    if (summary["drift_mean"] > 0).all():
        axis.set_xscale("log")
    axis.set_yscale("symlog", linthresh=0.1)
    axis.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, zorder=1)
    save_figure(figure, str(output_path))


def _plot_regret_by_rank(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    methods: list[str] | None = None,
    title: str = "Selector absolute regret across low-rank working bases",
    y_scale: str = "linear",
    symlog_linthresh: float = 0.1,
) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    group_columns = ["regime", "seed", "K_R", "K_T"]
    safe_full = plot_frame.loc[plot_frame["method"] == "safe-full", group_columns + ["gain"]].rename(columns={"gain": "safe_full_gain"})
    plot_frame = plot_frame.merge(safe_full, on=group_columns, how="left")
    plot_frame["absolute_regret_vs_safe_full"] = plot_frame["safe_full_gain"] - plot_frame["gain"]
    if methods is not None:
        plot_frame = plot_frame.loc[plot_frame["method"].isin(methods)].copy()

    apply_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.6, 5.8))
    figure.subplots_adjust(bottom=0.32)
    for method, subframe in plot_frame.groupby("method", sort=False):
        aggregated = subframe.groupby("K_requested")["absolute_regret_vs_safe_full"].agg(["mean", "sem"]).reset_index()
        axis.errorbar(
            aggregated["K_requested"],
            aggregated["mean"],
            yerr=aggregated["sem"].fillna(0.0),
            marker="o",
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#333333"),
        )
    axis.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Requested rank K_R + K_T")
    axis.set_ylabel("Absolute regret vs SAFE full" + (" (symlog)" if y_scale == "symlog" else ""))
    axis.set_title(title)
    if y_scale == "symlog":
        axis.set_yscale("symlog", linthresh=symlog_linthresh)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8,
    )
    save_figure(figure, str(output_path))


def run_dimred_selector_comparison_experiment(
    output_root: str | Path,
    seed: int = 1,
    *,
    config: DimRedSelectorConfig | None = None,
) -> dict:
    config = DimRedSelectorConfig(base_seed=seed) if config is None else config
    output_dir = ensure_output_dir(Path(output_root) / "08_dimred_selector_comparison")

    rows: list[dict[str, float | int | str]] = []
    for regime in config.regimes:
        for seed_index in range(config.seeds):
            local_seed = config.base_seed + 3000 * seed_index + regime_offset(regime)
            problem = build_problem(config, local_seed, regime)
            for k_reference, k_task in config.score_rank_pairs:
                rows.extend(
                    evaluate_selector_bundle(
                        config,
                        problem,
                        seed=local_seed,
                        regime=regime,
                        k_reference=k_reference,
                        k_task=k_task,
                    )
                )

    frame = pd.DataFrame(rows)
    write_frame(frame, output_dir / "subset_selector_package_comparison.csv")

    summary_by_rank = _summarize_by_method(frame, ["regime", "K_R", "K_T", "method", "method_family", "basis_variant"])
    summary_overall = _summarize_by_method(frame, ["regime", "method", "method_family", "basis_variant"])
    write_frame(summary_by_rank, output_dir / "subset_selector_package_summary_by_rank.csv")
    write_frame(summary_overall, output_dir / "subset_selector_package_summary_overall.csv")

    for (regime, k_reference, k_task), subframe in summary_by_rank.groupby(["regime", "K_R", "K_T"], sort=False):
        _plot_pareto_summary(
            subframe,
            output_dir / f"subset_pareto_{regime}_KR{k_reference}_KT{k_task}.png",
            regime=str(regime),
            k_reference=int(k_reference),
            k_task=int(k_task),
        )

    rank_sensitive_methods = [
        "safe-full",
        "safe-lowrank",
        "safe-reference-only",
        "safe-task-only",
        "safe-random-k",
    ]
    _plot_regret_by_rank(
        frame,
        output_dir / "relative_regret_by_rank.png",
        title="Selector absolute regret across all methods",
        y_scale="symlog",
        symlog_linthresh=0.05,
    )
    _plot_regret_by_rank(
        frame,
        output_dir / "absolute_regret_by_rank_safe_only.png",
        methods=rank_sensitive_methods,
        title="Absolute regret across rank-sensitive SAFE variants",
    )

    write_json(
        {
            "seed": seed,
            "config": asdict(config),
            "methods": METHOD_LABELS,
        },
        output_dir / "config.json",
    )

    figure_paths = [
        str(output_dir / "relative_regret_by_rank.png"),
        str(output_dir / "absolute_regret_by_rank_safe_only.png"),
    ]
    for regime in config.regimes:
        for k_reference, k_task in config.score_rank_pairs:
            figure_paths.append(str(output_dir / f"subset_pareto_{regime}_KR{k_reference}_KT{k_task}.png"))

    return {
        "name": "High-dimensional selector comparison",
        "directory": str(output_dir),
        "tables": [
            str(output_dir / "subset_selector_package_comparison.csv"),
            str(output_dir / "subset_selector_package_summary_by_rank.csv"),
            str(output_dir / "subset_selector_package_summary_overall.csv"),
        ],
        "figures": figure_paths,
    }


__all__ = [
    "DimRedSelectorConfig",
    "run_dimred_selector_comparison_experiment",
]
