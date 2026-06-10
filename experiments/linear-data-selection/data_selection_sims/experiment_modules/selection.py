from __future__ import annotations

from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ..core import (
    GaussianTeacherStudentModel,
    cosine_similarity,
    empirical_reference_drift,
    empirical_target_improvement,
    make_rotated_covariance,
    normalize,
)
from ..plotting import GROUP_COLORS, METHOD_COLORS, add_arrow, apply_style, save_figure
from ..selection import (
    build_candidate_pool,
    build_update,
    select_alignment_only,
    select_baseline_less_subset,
    select_baseline_prismatic_subset,
    select_baseline_random_subset,
    select_diversity_only,
    select_random,
    select_reference_aware,
    select_shared_safe_subset,
)
from ._shared import ensure_output_dir, write_frame, write_json


FORBIDDEN_AXIS_COLOR = "#c0392b"
SAFE_AXIS_COLOR = "#9a9a9a"


def _annotate_scatter_points_without_overlap(
    axis,
    rows: pd.DataFrame,
    *,
    x_key: str,
    y_key: str,
    label_map: dict[str, str],
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    y_scale: str = "linear",
    preferred_offsets: dict[str, list[tuple[int, int]] | tuple[int, int]] | None = None,
) -> None:
    del x_limits, y_limits, y_scale
    preferred_offsets = preferred_offsets or {}
    usage_counter: defaultdict[str, int] = defaultdict(int)
    for _, row in rows.iterrows():
        method_name = str(row["method"])
        label = label_map.get(method_name, method_name)
        method_offsets = preferred_offsets.get(method_name, (10, 10))
        if isinstance(method_offsets, list):
            offset = method_offsets[min(usage_counter[method_name], len(method_offsets) - 1)]
        else:
            offset = method_offsets
        usage_counter[method_name] += 1
        base_method_name = method_name.removesuffix("-euclidean")
        color = METHOD_COLORS.get(method_name, METHOD_COLORS.get(base_method_name, "#333333"))
        axis.annotate(
            label,
            xy=(float(row[x_key]), float(row[y_key])),
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=color,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )


def _build_model(beta_target: np.ndarray | None) -> GaussianTeacherStudentModel:
    if beta_target is None:
        beta_target = normalize(np.array([1.0, 0.45], dtype=float))
    else:
        beta_target = normalize(np.asarray(beta_target, dtype=float))
    sigma_reference = make_rotated_covariance([6.0, 0.45], 32.0)
    sigma_target = np.array([[1.2, 0.1], [0.1, 0.85]], dtype=float)
    return GaussianTeacherStudentModel(
        sigma_noise=1.0,
        sigma_reference=sigma_reference,
        sigma_target=sigma_target,
        beta_target=beta_target,
    )


def _group_counts(pool_groups: np.ndarray, selected_indices: np.ndarray) -> dict[str, int]:
    selected_groups = pool_groups[selected_indices]
    return {
        "selected_useful_expensive": int(np.sum(selected_groups == "useful-expensive")),
        "selected_useful_cheap": int(np.sum(selected_groups == "useful-cheap")),
        "selected_distractor": int(np.sum(selected_groups == "distractor")),
    }


def _evaluate_method(
    *,
    method: str,
    method_family: str,
    solver: str,
    geometry: str,
    selected_indices: np.ndarray,
    update: np.ndarray,
    pool_groups: np.ndarray,
    model: GaussianTeacherStudentModel,
    target_holdout: tuple[np.ndarray, np.ndarray],
    reference_holdout: np.ndarray,
    oracle_update: np.ndarray,
) -> dict[str, float | int | str]:
    target_features, target_responses = target_holdout
    counts = _group_counts(pool_groups, selected_indices)
    target_alignment = float(-model.target_gradient @ update)
    return {
        "method": method,
        "method_family": method_family,
        "solver": solver,
        "geometry": geometry,
        "selection_size": int(len(selected_indices)),
        **counts,
        "target_alignment_exact": target_alignment,
        "target_improvement_exact": float(model.target_improvement(update)),
        "reference_drift_exact": float(model.reference_drift(update)),
        "target_improvement_holdout": float(
            empirical_target_improvement(update, target_features, target_responses, model.sigma_noise)
        ),
        "reference_drift_holdout": float(
            empirical_reference_drift(update, reference_holdout, model.sigma_noise)
        ),
        "cosine_to_oracle": float(cosine_similarity(update, oracle_update)),
    }


def _plot_tradeoff(metrics_frame: pd.DataFrame, output_path: Path) -> None:
    apply_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 5.0))
    axis.axhline(1.0, color=FORBIDDEN_AXIS_COLOR, linestyle="--", linewidth=1.4, alpha=0.85)
    for _, row in metrics_frame.iterrows():
        method_name = str(row["method"])
        base_method_name = method_name.removesuffix("-euclidean")
        color = METHOD_COLORS.get(method_name, METHOD_COLORS.get(base_method_name, "#333333"))
        marker = "o" if str(row["geometry"]) == "reference" else "s"
        axis.scatter(
            float(row["target_improvement_holdout"]),
            float(row["reference_drift_holdout"]),
            s=72,
            color=color,
            marker=marker,
            alpha=0.9,
        )
    x_values = metrics_frame["target_improvement_holdout"].to_numpy(dtype=float)
    y_values = metrics_frame["reference_drift_holdout"].to_numpy(dtype=float)
    x_limits = (float(x_values.min()) - 0.15, float(x_values.max()) + 0.15)
    y_limits = (max(1.0e-4, float(y_values.min()) * 0.65), float(y_values.max()) * 1.5)
    axis.set_xlim(*x_limits)
    axis.set_yscale("log")
    axis.set_ylim(*y_limits)
    _annotate_scatter_points_without_overlap(
        axis,
        metrics_frame,
        x_key="target_improvement_holdout",
        y_key="reference_drift_holdout",
        label_map={method: method for method in metrics_frame["method"]},
        x_limits=x_limits,
        y_limits=y_limits,
        y_scale="log",
    )
    axis.set_title("2D discrete selection")
    axis.set_xlabel("Held-out target improvement")
    axis.set_ylabel("Held-out reference drift (log scale)")
    save_figure(figure, str(output_path))


def _plot_method_updates(pool, updates_by_method: dict[str, np.ndarray], output_path: Path) -> None:
    apply_style()
    figure, axis = plt.subplots(1, 1, figsize=(6.5, 6.0))
    axis.axhline(0.0, color=SAFE_AXIS_COLOR, linewidth=0.8)
    axis.axvline(0.0, color=SAFE_AXIS_COLOR, linewidth=0.8)
    for group_name, color in GROUP_COLORS.items():
        mask = pool.groups == group_name
        axis.scatter(
            pool.updates[mask, 0],
            pool.updates[mask, 1],
            s=22,
            color=color,
            alpha=0.45,
            label=group_name,
        )
    for method, update in updates_by_method.items():
        color = METHOD_COLORS.get(method, METHOD_COLORS.get(method.removesuffix("-euclidean"), "#333333"))
        add_arrow(axis, np.asarray(update, dtype=float), color, label=method, linewidth=1.4, alpha=0.9, zorder=4.0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Selection method updates")
    axis.set_xlabel("Update coordinate 1")
    axis.set_ylabel("Update coordinate 2")
    axis.legend(fontsize=7.5, loc="upper right", frameon=True)
    save_figure(figure, str(output_path))


def run_selection_experiment(
    output_root: Path,
    seed: int = 1,
    *,
    beta_target: np.ndarray | None = None,
    output_dir_name: str = "02_selection",
    experiment_name: str = "2D discrete selection",
) -> dict:
    rng = np.random.default_rng(seed)
    output_dir = ensure_output_dir(Path(output_root) / output_dir_name)
    model = _build_model(beta_target)
    pool = build_candidate_pool(model, samples_per_group=12, rng=rng)
    selection_size = 9
    step_size = 0.95
    rho = 1.0
    radius = 1.0

    oracle = model.case4_update(rho=rho, radius=radius)
    oracle_update = np.asarray(oracle.update, dtype=float)

    target_holdout = model.sample_target_dataset(sample_size=4096, rng=rng)
    reference_holdout = model.sample_reference_dataset(sample_size=4096, rng=rng)

    methods: list[tuple[str, str, str, str, np.ndarray, np.ndarray]] = []

    random_indices = select_random(pool, selection_size, rng)
    methods.append(
        (
            "random",
            "heuristic",
            "random",
            "euclidean",
            np.sort(np.asarray(random_indices, dtype=int)),
            build_update(pool, np.sort(np.asarray(random_indices, dtype=int)), step_size),
        )
    )

    alignment_indices = np.sort(np.asarray(select_alignment_only(pool, selection_size, model.target_update), dtype=int))
    methods.append(
        (
            "alignment-only",
            "heuristic",
            "alignment",
            "euclidean",
            alignment_indices,
            build_update(pool, alignment_indices, step_size),
        )
    )

    diversity_indices = select_diversity_only(pool, selection_size)
    methods.append(
        (
            "diversity-only",
            "heuristic",
            "diversity",
            "euclidean",
            diversity_indices,
            build_update(pool, diversity_indices, step_size),
        )
    )

    reference_aware_indices = np.asarray(
        select_reference_aware(
            pool,
            selection_size,
            model.target_update,
            model.fisher_reference,
            step_size,
            drift_penalty=1.2,
            norm_penalty=0.2,
        ),
        dtype=int,
    )
    methods.append(
        (
            "reference-aware",
            "heuristic",
            "reference-aware",
            "reference",
            reference_aware_indices,
            build_update(pool, reference_aware_indices, step_size / selection_size),
        )
    )

    baseline_random_indices = select_baseline_random_subset(pool, selection_size, seed=seed + 101)
    methods.append(
        (
            "baseline-random",
            "baseline_selectors",
            "random",
            "euclidean",
            baseline_random_indices,
            build_update(pool, baseline_random_indices, step_size),
        )
    )

    baseline_less_indices = select_baseline_less_subset(pool, selection_size, model.target_gradient, similarity="dot")
    methods.append(
        (
            "baseline-less",
            "baseline_selectors",
            "less(dot)",
            "euclidean",
            baseline_less_indices,
            build_update(pool, baseline_less_indices, step_size),
        )
    )

    baseline_less_cosine_indices = select_baseline_less_subset(
        pool,
        selection_size,
        model.target_gradient,
        similarity="cosine",
    )
    methods.append(
        (
            "baseline-less-cosine",
            "baseline_selectors",
            "less(cosine)",
            "euclidean",
            baseline_less_cosine_indices,
            build_update(pool, baseline_less_cosine_indices, step_size),
        )
    )

    baseline_prismatic_indices = select_baseline_prismatic_subset(pool, selection_size, seed=seed + 202)
    methods.append(
        (
            "baseline-prismatic",
            "baseline_selectors",
            "prismatic(hard)",
            "euclidean",
            baseline_prismatic_indices,
            build_update(pool, baseline_prismatic_indices, step_size),
        )
    )

    for method_name, solver_name, geometry_name, shortlist_size in [
        ("selector-rank", "rank", "reference", selection_size),
        ("selector-greedy", "greedy", "reference", 3 * selection_size),
        ("selector-rank-euclidean", "rank", "euclidean", selection_size),
        ("selector-greedy-euclidean", "greedy", "euclidean", 3 * selection_size),
    ]:
        selector_result = select_shared_safe_subset(
            pool,
            selection_size,
            model.target_gradient,
            model.fisher_reference,
            step_size,
            rho,
            radius,
            solver=solver_name,
            geometry=geometry_name,
            shortlist_size=shortlist_size,
            average_by_budget=True,
        )
        methods.append(
            (
                method_name,
                "selector_methods",
                solver_name,
                geometry_name,
                selector_result.selected_indices,
                selector_result.update,
            )
        )

    metrics_records: list[dict[str, float | int | str]] = []
    updates_by_method: dict[str, np.ndarray] = {}
    for method_name, method_family, solver_name, geometry_name, selected_indices, update in methods:
        selected_indices = np.asarray(selected_indices, dtype=int)
        update = np.asarray(update, dtype=float)
        updates_by_method[method_name] = update
        metrics_records.append(
            _evaluate_method(
                method=method_name,
                method_family=method_family,
                solver=solver_name,
                geometry=geometry_name,
                selected_indices=selected_indices,
                update=update,
                pool_groups=pool.groups,
                model=model,
                target_holdout=target_holdout,
                reference_holdout=reference_holdout,
                oracle_update=oracle_update,
            )
        )

    metrics_frame = pd.DataFrame(metrics_records).sort_values(
        ["method_family", "geometry", "reference_drift_holdout", "target_improvement_holdout"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    holdout_frame = metrics_frame[
        [
            "method",
            "method_family",
            "solver",
            "geometry",
            "target_improvement_holdout",
            "reference_drift_holdout",
            "cosine_to_oracle",
        ]
    ].copy()

    write_frame(metrics_frame, output_dir / "single_pool_metrics.csv")
    write_frame(holdout_frame, output_dir / "holdout_tradeoff_metrics.csv")
    _plot_tradeoff(holdout_frame, output_dir / "selection_tradeoff.png")
    _plot_method_updates(pool, updates_by_method, output_dir / "selection_methods.png")
    write_json(
        {
            "seed": seed,
            "experiment_name": experiment_name,
            "output_dir_name": output_dir_name,
            "rho": rho,
            "radius": radius,
            "selection_size": selection_size,
            "step_size": step_size,
            "beta_target": np.asarray(model.beta_target, dtype=float).tolist(),
        },
        output_dir / "config.json",
    )

    return {
        "name": experiment_name,
        "directory": str(output_dir),
        "tables": [
            str(output_dir / "single_pool_metrics.csv"),
            str(output_dir / "holdout_tradeoff_metrics.csv"),
        ],
        "figures": [
            str(output_dir / "selection_tradeoff.png"),
            str(output_dir / "selection_methods.png"),
        ],
    }
