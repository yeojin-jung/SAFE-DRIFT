from __future__ import annotations

import json
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd

from data_selection_sims.core import (
    GaussianTeacherStudentModel,
    normalize,
    solve_damped,
    sorted_eigendecomposition,
    spiked_covariance,
)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def color_map(values) -> list[tuple[float, float, float, float]]:
    cmap = cm.viridis(np.linspace(0.1, 0.9, len(values)))
    return [tuple(color) for color in cmap]


def scale_fixed_mu_update_to_budget(
    model: GaussianTeacherStudentModel,
    damping: float,
    drift_budget: float,
    norm_radius: float,
) -> np.ndarray:
    base_update = solve_damped(model.fisher_reference, model.target_gradient, float(damping))
    fisher_norm = float(np.sqrt(max(base_update @ model.fisher_reference @ base_update, 0.0)))
    euclidean_norm = float(np.linalg.norm(base_update))
    scale = 1.0
    if fisher_norm > 0.0:
        scale = min(scale, float(drift_budget) / fisher_norm)
    if euclidean_norm > 0.0:
        scale = min(scale, float(norm_radius) / euclidean_norm)
    return np.asarray(scale * base_update, dtype=float)


def make_spiked_model(
    dimension: int,
    rank: int,
    rng: np.random.Generator,
    *,
    align_to_coordinates: bool = False,
) -> GaussianTeacherStudentModel:
    spike_values = np.geomspace(12.0, 2.0, rank) if rank > 1 else np.array([12.0], dtype=float)
    sigma_reference = spiked_covariance(
        dimension=dimension,
        rank=rank,
        spike_values=spike_values,
        epsilon=0.2,
        rng=rng,
        align_to_coordinates=align_to_coordinates,
    )
    basis, _ = sorted_eigendecomposition(sigma_reference)
    beta_target = normalize(basis[:, 0] if basis.size else rng.normal(size=dimension))
    return GaussianTeacherStudentModel(
        sigma_noise=1.0,
        sigma_reference=sigma_reference,
        sigma_target=np.eye(dimension, dtype=float),
        beta_target=np.asarray(beta_target, dtype=float),
    )
