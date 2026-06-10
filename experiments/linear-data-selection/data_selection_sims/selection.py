from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (
    Array,
    Case4BudgetedUpdate,
    GaussianTeacherStudentModel,
    cosine_similarity,
    normalize,
    solve_case4_budgeted_update,
    sorted_eigendecomposition,
)


@dataclass(frozen=True)
class CandidatePool:
    features: Array
    responses: Array
    updates: Array
    groups: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.features))


@dataclass(frozen=True)
class SharedSelectorMethodResult:
    selected_indices: np.ndarray
    update: Array
    solver: str
    geometry: str
    alpha: float
    alpha_status: str
    shortlist_size: int
    objective_value: float
    predicted_target_gain: float
    reference_cost: float
    norm_cost: float


def _group_covariance(direction: Array, orthogonal: Array, major: float, minor: float) -> Array:
    basis = np.column_stack([normalize(direction), normalize(orthogonal)])
    return basis @ np.diag([major, minor]) @ basis.T


def build_candidate_pool(
    model: GaussianTeacherStudentModel,
    samples_per_group: int,
    rng: np.random.Generator,
) -> CandidatePool:
    if model.dimension != 2:
        raise ValueError("The low-dimensional candidate pool is defined only for dimension 2.")

    _, eigenvectors = sorted_eigendecomposition(model.sigma_reference)
    high_curvature = eigenvectors[:, 0]
    low_curvature = eigenvectors[:, 1]
    beta_direction = normalize(model.beta_target)
    orthogonal_target = np.array([-beta_direction[1], beta_direction[0]], dtype=float)

    group_specs = [
        (
            "useful-expensive",
            1.9 * high_curvature + 0.3 * low_curvature,
            _group_covariance(high_curvature, low_curvature, major=0.55, minor=0.09),
        ),
        (
            "useful-cheap",
            1.4 * low_curvature + 0.65 * beta_direction,
            _group_covariance(low_curvature, high_curvature, major=0.48, minor=0.08),
        ),
        (
            "distractor",
            1.6 * orthogonal_target,
            _group_covariance(orthogonal_target, beta_direction, major=0.4, minor=0.14),
        ),
    ]

    features: list[Array] = []
    responses: list[Array] = []
    groups: list[str] = []
    for group_name, mean, covariance in group_specs:
        group_features, group_responses = model.sample_target_dataset(
            samples_per_group,
            rng=rng,
            covariance=covariance,
            mean=mean,
        )
        features.append(group_features)
        responses.append(group_responses)
        groups.extend([group_name] * samples_per_group)

    features_array = np.vstack(features)
    responses_array = np.concatenate(responses)
    updates = (responses_array[:, None] * features_array) / (model.sigma_noise**2)
    return CandidatePool(
        features=features_array,
        responses=responses_array,
        updates=updates,
        groups=np.asarray(groups, dtype=object),
    )


def solve_shared_safe_update(
    target_gradient: Array,
    fisher_reference: Array,
    rho: float,
    radius: float,
) -> Case4BudgetedUpdate:
    return solve_case4_budgeted_update(
        fisher=np.asarray(fisher_reference, dtype=float),
        gradient=np.asarray(target_gradient, dtype=float),
        rho=float(rho),
        radius=float(radius),
    )


def select_random(pool: CandidatePool, selection_size: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(pool.size, size=selection_size, replace=False))


def select_alignment_only(pool: CandidatePool, selection_size: int, target_update: Array) -> np.ndarray:
    scores = pool.updates @ target_update
    return np.argsort(scores)[-selection_size:]


def select_diversity_only(pool: CandidatePool, selection_size: int) -> np.ndarray:
    normalized_updates = np.vstack([normalize(update) for update in pool.updates])
    norms = np.linalg.norm(pool.updates, axis=1)
    first = int(np.argmax(norms))
    selected = [first]

    while len(selected) < selection_size:
        best_score = -np.inf
        best_index = None
        for index in range(pool.size):
            if index in selected:
                continue
            similarities = [abs(cosine_similarity(normalized_updates[index], normalized_updates[item])) for item in selected]
            novelty = 1.0 - max(similarities)
            score = novelty + 0.05 * norms[index]
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(int(best_index))
    return np.sort(np.asarray(selected, dtype=int))


def select_reference_aware(
    pool: CandidatePool,
    selection_size: int,
    target_update: Array,
    fisher_reference: Array,
    step_size: float,
    drift_penalty: float,
    norm_penalty: float,
) -> np.ndarray:
    chosen: list[int] = []
    selected = np.zeros(pool.size, dtype=bool)
    scale = step_size / selection_size
    current_update = np.zeros_like(target_update)

    for _ in range(selection_size):
        best_score = -np.inf
        best_index = None
        for index, example_update in enumerate(pool.updates):
            if selected[index]:
                continue
            contribution = scale * example_update
            incremental_gain = float(target_update @ contribution)
            incremental_drift = float(current_update @ fisher_reference @ contribution + 0.5 * contribution @ fisher_reference @ contribution)
            incremental_norm = float(current_update @ contribution + 0.5 * contribution @ contribution)
            score = incremental_gain - drift_penalty * incremental_drift - norm_penalty * incremental_norm
            if score > best_score:
                best_score = score
                best_index = index
        chosen.append(int(best_index))
        selected[best_index] = True
        current_update = current_update + scale * pool.updates[best_index]

    return np.asarray(chosen, dtype=int)


def _resolve_metric_matrix(
    updates: Array,
    target_update: Array,
    geometry: str,
    fisher_reference: Array | None = None,
) -> tuple[Array, Array]:
    if geometry == "reference" and fisher_reference is not None:
        fisher = np.asarray(fisher_reference, dtype=float)
        return updates @ fisher.T, fisher @ target_update
    return updates, target_update


def select_shared_safe_subset(
    pool: CandidatePool,
    selection_size: int,
    target_gradient: Array,
    fisher_reference: Array,
    step_size: float,
    rho: float,
    radius: float,
    *,
    solver: str,
    geometry: str = "reference",
    shortlist_size: int | None = None,
    average_by_budget: bool = True,
) -> SharedSelectorMethodResult:
    safe_update = solve_shared_safe_update(target_gradient, fisher_reference, rho, radius)
    desired_update = np.asarray(safe_update.update, dtype=float)
    contribution_scale = float(step_size) / float(selection_size) if average_by_budget else float(step_size)
    candidate_updates = contribution_scale * np.asarray(pool.updates, dtype=float)
    transformed_updates, transformed_target = _resolve_metric_matrix(
        candidate_updates,
        desired_update,
        geometry,
        np.asarray(fisher_reference, dtype=float),
    )
    scores = transformed_updates @ transformed_target
    candidate_indices = np.argsort(scores)[::-1]
    if shortlist_size is not None:
        candidate_indices = candidate_indices[: max(selection_size, int(shortlist_size))]

    selected: list[int] = []
    selected_set: set[int] = set()
    aggregate = np.zeros_like(desired_update)
    fisher_reference_array = np.asarray(fisher_reference, dtype=float)

    def objective(next_aggregate: Array) -> float:
        delta = next_aggregate - desired_update
        if geometry == "reference":
            return 0.5 * float(delta @ fisher_reference_array @ delta)
        return 0.5 * float(delta @ delta)

    best_value = objective(aggregate)
    while len(selected) < selection_size:
        best_index = None
        best_trial_value = np.inf
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
            remaining = [idx for idx in range(pool.size) if idx not in selected_set]
            if not remaining:
                break
            best_index = int(remaining[0])
            best_trial_value = objective(aggregate + candidate_updates[best_index])
        selected.append(best_index)
        selected_set.add(best_index)
        aggregate = aggregate + candidate_updates[best_index]
        best_value = best_trial_value

    selected_indices = np.sort(np.asarray(selected, dtype=int))
    subset_update = build_update(pool, selected_indices, contribution_scale)
    reference_cost = 0.5 * float(subset_update @ fisher_reference_array @ subset_update)
    norm_cost = float(np.linalg.norm(subset_update))
    predicted_target_gain = float(-np.asarray(target_gradient, dtype=float) @ subset_update)
    return SharedSelectorMethodResult(
        selected_indices=selected_indices,
        update=subset_update,
        solver=str(solver),
        geometry=str(geometry),
        alpha=float(safe_update.alpha),
        alpha_status="analytic_case4",
        shortlist_size=int(len(candidate_indices)),
        objective_value=float(best_value),
        predicted_target_gain=predicted_target_gain,
        reference_cost=reference_cost,
        norm_cost=norm_cost,
    )


def select_baseline_random_subset(pool: CandidatePool, selection_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(pool.size, size=int(selection_size), replace=False).astype(int))


def select_baseline_less_subset(
    pool: CandidatePool,
    selection_size: int,
    target_gradient: Array,
    similarity: str = "dot",
    preconditioner: Array | None = None,
) -> np.ndarray:
    updates = np.asarray(pool.updates, dtype=float)
    if preconditioner is not None:
        preconditioner_array = np.asarray(preconditioner, dtype=float)
        if preconditioner_array.ndim == 1:
            updates = updates * preconditioner_array[None, :]
        else:
            updates = updates @ preconditioner_array.T

    target = -np.asarray(target_gradient, dtype=float)
    if similarity == "cosine":
        scores = np.array([cosine_similarity(update, target) for update in updates], dtype=float)
    else:
        scores = updates @ target
    selected = np.argsort(scores)[-int(selection_size) :]
    return np.sort(selected.astype(int))


def select_baseline_prismatic_subset(
    pool: CandidatePool,
    selection_size: int,
    seed: int,
    cluster_ratio: float = 0.5,
    sparsity: float = 0.1,
    num_iters: int = 10,
    method: str = "hard",
) -> np.ndarray:
    del cluster_ratio, sparsity, num_iters, method
    rng = np.random.default_rng(int(seed))
    updates = np.asarray(pool.updates, dtype=float)
    norms = np.linalg.norm(updates, axis=1)
    normalized = np.array([normalize(update) for update in updates], dtype=float)
    first = int(np.argmax(norms))
    selected = [first]
    available = set(range(pool.size))
    available.remove(first)

    while len(selected) < int(selection_size) and available:
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
    return np.sort(np.asarray(selected, dtype=int))


def build_update(pool: CandidatePool, indices: np.ndarray, step_size: float) -> Array:
    return step_size * pool.updates[indices].mean(axis=0)
