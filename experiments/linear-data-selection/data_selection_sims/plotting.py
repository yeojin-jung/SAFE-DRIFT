from __future__ import annotations

from collections.abc import Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from data_selection_sims.core import Array, normalize, sorted_eigendecomposition


GROUP_COLORS = {
    "useful-expensive": "#d55e00",
    "useful-cheap": "#0072b2",
    "distractor": "#4d4d4d",
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
    "random": "#8c564b",
    "alignment-only": "#5e3c99",
    "diversity-only": "#1b9e77",
    "reference-aware": "#d95f02",
    "selector-rank": "#e41a1c",
    "selector-greedy": "#377eb8",
    "selector-rank-euclidean": "#4daf4a",
    "selector-greedy-euclidean": "#984ea3",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "font.size": 10,
        }
    )


def save_figure(figure, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def add_arrow(
    axis,
    vector: Array,
    color: str,
    label: str | None = None,
    *,
    start: Array | None = None,
    linewidth: float = 2.0,
    alpha: float = 0.95,
    zorder: float = 3.0,
) -> None:
    start_vec = np.zeros_like(vector, dtype=float) if start is None else np.asarray(start, dtype=float)
    delta = np.asarray(vector, dtype=float)
    axis.arrow(
        float(start_vec[0]),
        float(start_vec[1]),
        float(delta[0]),
        float(delta[1]),
        color=color,
        width=0.006,
        head_width=0.05,
        length_includes_head=True,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )


def add_gain_lines(axis, gradient: Array, levels: Iterable[float], x_limits: tuple[float, float], color: str) -> None:
    grad = np.asarray(gradient, dtype=float)
    x_values = np.linspace(float(x_limits[0]), float(x_limits[1]), 256)
    if abs(float(grad[1])) < 1.0e-12:
        axis.axvline(0.0, color=color, linewidth=0.8, alpha=0.35)
        return
    for level in levels:
        y_values = (float(level) - grad[0] * x_values) / grad[1]
        axis.plot(x_values, y_values, color=color, linewidth=0.8, alpha=0.22)


def add_drift_contours(
    axis,
    fisher: Array,
    levels: Iterable[float],
    color: str,
    *,
    linestyle: str = "--",
    label: str | None = None,
    alpha: float = 0.45,
) -> None:
    eigenvalues, eigenvectors = sorted_eigendecomposition(np.asarray(fisher, dtype=float))
    if eigenvalues.size == 0:
        return
    angles = np.linspace(0.0, 2.0 * np.pi, 256)
    unit_circle = np.vstack([np.cos(angles), np.sin(angles)])
    for index, level in enumerate(levels):
        axis_lengths = np.sqrt(np.maximum(2.0 * float(level) / np.maximum(eigenvalues, 1.0e-12), 0.0))
        ellipse = eigenvectors @ (np.diag(axis_lengths) @ unit_circle)
        axis.plot(
            ellipse[0],
            ellipse[1],
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            alpha=alpha,
            label=label if index == 0 else None,
        )


def add_norm_ball(axis, radius: float, color: str) -> None:
    circle = plt.Circle((0.0, 0.0), float(radius), fill=False, linestyle="--", linewidth=1.0, color=color, alpha=0.5)
    axis.add_patch(circle)


def add_eigen_directions(axis, fisher: Array, scale: float) -> None:
    eigenvalues, eigenvectors = sorted_eigendecomposition(np.asarray(fisher, dtype=float))
    if eigenvectors.size == 0:
        return
    for idx in range(min(2, int(eigenvectors.shape[1]))):
        add_arrow(axis, float(scale) * eigenvectors[:, idx], "#999999", label=None, linewidth=1.0, alpha=0.55)


def set_equal_axes(axis, limit: float, *, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_xlim(-float(limit), float(limit))
    axis.set_ylim(-float(limit), float(limit))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def scatter_with_labels(
    axis,
    x_values: Iterable[float],
    y_values: Iterable[float],
    labels: Iterable[str],
    colors: Iterable[str],
    *,
    marker_size: float = 50.0,
) -> None:
    for x_value, y_value, label, color in zip(x_values, y_values, labels, colors, strict=False):
        axis.scatter(float(x_value), float(y_value), s=marker_size, color=color, label=label)


def draw_update_cloud(axis, updates: Array, color: str, *, alpha: float = 0.18, linewidth: float = 0.9) -> None:
    for update in np.asarray(updates, dtype=float):
        add_arrow(axis, update, color, linewidth=linewidth, alpha=alpha, zorder=1.5)


def add_reference_axes(axis, directions: Array, colors: Iterable[str], *, scale: float = 1.0) -> None:
    for direction, color in zip(np.asarray(directions, dtype=float).T, colors, strict=False):
        add_arrow(axis, normalize(direction) * float(scale), color, linewidth=1.2, alpha=0.7)
