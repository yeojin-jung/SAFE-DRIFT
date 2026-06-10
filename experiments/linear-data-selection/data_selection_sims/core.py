from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray


@dataclass(frozen=True)
class Case4BudgetedUpdate:
    update: Array
    alpha: float
    reference_cost: float
    norm: float


def rotation_matrix(angle_degrees: float) -> Array:
    radians = np.deg2rad(angle_degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=float)


def make_rotated_covariance(eigenvalues, angle_degrees: float) -> Array:
    eigenvalues = np.asarray(list(eigenvalues), dtype=float)
    rotation = rotation_matrix(angle_degrees)
    return rotation @ np.diag(eigenvalues) @ rotation.T


def normalize(vector: Array, eps: float = 1.0e-12) -> Array:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity(left: Array, right: Array, eps: float = 1.0e-12) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm < eps or right_norm < eps:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def angle_of(vector: Array) -> float:
    return float(np.rad2deg(np.arctan2(vector[1], vector[0])))


def angle_between(left: Array, right: Array) -> float:
    cosine = np.clip(cosine_similarity(left, right), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def sorted_eigendecomposition(matrix: Array) -> tuple[Array, Array]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def solve_damped(fisher: Array, gradient: Array, damping: float) -> Array:
    system = fisher + damping * np.eye(fisher.shape[0], dtype=float)
    return np.linalg.solve(system, -gradient)


def _case4_profile(eigenvalues: Array, coefficients: Array, alpha: float, eps: float = 1.0e-12) -> Array:
    profile = np.zeros_like(coefficients)
    positive_mask = eigenvalues > eps
    null_mask = ~positive_mask

    if np.isinf(alpha):
        return coefficients.copy()

    if alpha <= eps:
        if np.any(np.abs(coefficients[null_mask]) > eps):
            profile[null_mask] = coefficients[null_mask]
            return profile
        profile[positive_mask] = coefficients[positive_mask] / eigenvalues[positive_mask]
        return profile

    profile = coefficients / (eigenvalues + alpha)
    return profile


def case4_reference_cost(
    fisher: Array,
    gradient: Array,
    alpha: float,
    radius: float,
    eps: float = 1.0e-12,
) -> float:
    eigenvalues, eigenvectors = sorted_eigendecomposition(fisher)
    coefficients = eigenvectors.T @ gradient
    profile = _case4_profile(eigenvalues, coefficients, alpha, eps=eps)
    denominator = float(np.dot(profile, profile))
    if denominator <= eps:
        return 0.0
    numerator = float(np.dot(eigenvalues, profile**2))
    return 0.5 * (radius**2) * numerator / denominator


def solve_case4_budgeted_update(
    fisher: Array,
    gradient: Array,
    rho: float,
    radius: float,
    *,
    eps: float = 1.0e-12,
    tol: float = 1.0e-10,
    max_iter: int = 100,
) -> Case4BudgetedUpdate:
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm <= eps:
        zero = np.zeros_like(gradient)
        return Case4BudgetedUpdate(update=zero, alpha=np.inf, reference_cost=0.0, norm=0.0)

    eigenvalues, eigenvectors = sorted_eigendecomposition(fisher)
    coefficients = eigenvectors.T @ gradient

    def profile_for(alpha_value: float) -> Array:
        return _case4_profile(eigenvalues, coefficients, alpha_value, eps=eps)

    def cost_for(alpha_value: float) -> float:
        profile = profile_for(alpha_value)
        denominator = float(np.dot(profile, profile))
        if denominator <= eps:
            return 0.0
        numerator = float(np.dot(eigenvalues, profile**2))
        return 0.5 * (radius**2) * numerator / denominator

    rho_min = cost_for(0.0)
    rho_max = cost_for(np.inf)

    if rho <= rho_min + tol:
        alpha = 0.0
    elif rho >= rho_max - tol:
        alpha = np.inf
    else:
        alpha_low = 0.0
        alpha_high = 1.0
        while cost_for(alpha_high) < rho and alpha_high < 1.0e12:
            alpha_high *= 2.0
        for _ in range(max_iter):
            alpha_mid = 0.5 * (alpha_low + alpha_high)
            if cost_for(alpha_mid) < rho:
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid
        alpha = alpha_high

    direction = eigenvectors @ profile_for(alpha)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= eps:
        update = np.zeros_like(gradient)
    else:
        update = -radius * direction / direction_norm
    return Case4BudgetedUpdate(
        update=update,
        alpha=alpha,
        reference_cost=0.5 * float(update @ fisher @ update),
        norm=float(np.linalg.norm(update)),
    )


def project_to_ball(vector: Array, radius: float | None) -> Array:
    if radius is None:
        return np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= radius or norm <= 1.0e-12:
        return np.asarray(vector, dtype=float)
    return np.asarray((radius / norm) * vector, dtype=float)


def empirical_target_gradient(features: Array, responses: Array, sigma_noise: float) -> Array:
    return -np.mean(responses[:, None] * features, axis=0) / (sigma_noise**2)


def empirical_fisher(features: Array, sigma_noise: float) -> Array:
    return (features.T @ features) / (features.shape[0] * (sigma_noise**2))


def diagonal_approximation(matrix: Array) -> Array:
    return np.diag(np.diag(matrix))


def low_rank_approximation(matrix: Array, rank: int, isotropic_floor: float = 0.0) -> Array:
    eigenvalues, eigenvectors = sorted_eigendecomposition(matrix)
    rank = max(0, min(rank, len(eigenvalues)))
    retained = np.zeros_like(eigenvalues)
    if rank > 0:
        retained[:rank] = eigenvalues[:rank]
    approximation = eigenvectors @ np.diag(retained) @ eigenvectors.T
    if isotropic_floor > 0.0:
        approximation = approximation + isotropic_floor * np.eye(matrix.shape[0], dtype=float)
    return approximation


def shrinkage_estimate(matrix: Array, alpha: float) -> Array:
    dimension = matrix.shape[0]
    tau = float(np.trace(matrix) / dimension)
    return (1.0 - alpha) * matrix + alpha * tau * np.eye(dimension, dtype=float)


def random_orthonormal_matrix(dimension: int, rng: np.random.Generator) -> Array:
    random_matrix = rng.normal(size=(dimension, dimension))
    orthonormal, _ = np.linalg.qr(random_matrix)
    return orthonormal


def spiked_covariance(
    dimension: int,
    rank: int,
    spike_values,
    epsilon: float,
    rng: np.random.Generator,
    align_to_coordinates: bool = False,
) -> Array:
    if len(spike_values) != rank:
        raise ValueError("rank and spike_values length must match")
    diagonal = np.concatenate([np.asarray(spike_values, dtype=float), epsilon * np.ones(dimension - rank)])
    if align_to_coordinates:
        return np.diag(diagonal)
    rotation = random_orthonormal_matrix(dimension, rng)
    return rotation @ np.diag(diagonal) @ rotation.T


@dataclass(frozen=True)
class GaussianTeacherStudentModel:
    sigma_noise: float
    sigma_reference: Array
    sigma_target: Array
    beta_target: Array

    @property
    def dimension(self) -> int:
        return int(self.beta_target.shape[0])

    @property
    def target_gradient(self) -> Array:
        return -(self.sigma_target @ self.beta_target) / (self.sigma_noise**2)

    @property
    def fisher_reference(self) -> Array:
        return self.sigma_reference / (self.sigma_noise**2)

    @property
    def fisher_target(self) -> Array:
        return self.sigma_target / (self.sigma_noise**2)

    @property
    def target_update(self) -> Array:
        return -self.target_gradient

    def oracle_update(self, damping: float, radius: float | None = None) -> Array:
        update = solve_damped(self.fisher_reference, self.target_gradient, damping)
        return project_to_ball(update, radius)

    def case4_update(self, rho: float, radius: float) -> Case4BudgetedUpdate:
        return solve_case4_budgeted_update(self.fisher_reference, self.target_gradient, rho, radius)

    def target_improvement(self, update: Array) -> float:
        linear_term = -float(self.target_gradient @ update)
        curvature_term = 0.5 * float(update @ self.fisher_target @ update)
        return linear_term - curvature_term

    def reference_drift(self, update: Array) -> float:
        return 0.5 * float(update @ self.fisher_reference @ update)

    def sample_features(
        self,
        sample_size: int,
        covariance: Array,
        rng: np.random.Generator,
        mean: Array | None = None,
    ) -> Array:
        if mean is None:
            mean = np.zeros(covariance.shape[0], dtype=float)
        return rng.multivariate_normal(mean, covariance, size=sample_size)

    def sample_target_dataset(
        self,
        sample_size: int,
        rng: np.random.Generator,
        covariance: Array | None = None,
        mean: Array | None = None,
    ) -> tuple[Array, Array]:
        covariance = self.sigma_target if covariance is None else covariance
        features = self.sample_features(sample_size, covariance, rng, mean=mean)
        noise = rng.normal(scale=self.sigma_noise, size=sample_size)
        responses = features @ self.beta_target + noise
        return features, responses

    def sample_reference_dataset(
        self,
        sample_size: int,
        rng: np.random.Generator,
        covariance: Array | None = None,
        mean: Array | None = None,
    ) -> Array:
        covariance = self.sigma_reference if covariance is None else covariance
        return self.sample_features(sample_size, covariance, rng, mean=mean)


def empirical_target_improvement(update: Array, features: Array, responses: Array, sigma_noise: float) -> float:
    baseline = np.mean((responses - 0.0) ** 2) / (2.0 * (sigma_noise**2))
    updated = np.mean((responses - features @ update) ** 2) / (2.0 * (sigma_noise**2))
    return float(baseline - updated)


def empirical_reference_drift(update: Array, reference_features: Array, sigma_noise: float) -> float:
    predictions = reference_features @ update
    return float(np.mean(predictions**2) / (2.0 * (sigma_noise**2)))
