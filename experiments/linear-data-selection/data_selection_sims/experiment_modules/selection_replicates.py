from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from ._shared import ensure_output_dir, write_frame, write_json
from .selection import _annotate_scatter_points_without_overlap, run_selection_experiment
from ..plotting import METHOD_COLORS, apply_style, save_figure


_GROUP_KEYS = ["method", "method_family", "solver", "geometry"]
_METHOD_LABELS = {
    "random": "random",
    "alignment-only": "alignment-only",
    "diversity-only": "diversity-only",
    "reference-aware": "reference-aware",
    "baseline-random": "baseline random",
    "baseline-less": "baseline LESS",
    "baseline-less-cosine": "baseline LESS (cosine)",
    "baseline-prismatic": "baseline Prismatic",
    "selector-rank": "SAFE (rank-based selector)",
    "selector-greedy": "SAFE (greedy selector)",
    "selector-rank-euclidean": "SAFE (rank-based, euclidean selector)",
    "selector-greedy-euclidean": "SAFE (greedy euclidean selector)",
}
_PREFERRED_LABEL_OFFSETS = {
    "selector-rank-euclidean": (22, 54),
    "baseline-prismatic": [(12, 10), (16, 16), (18, -2)],
}
_ORTHOGONAL_TARGET_LABEL_LAYOUT = {
    "baseline-prismatic": (0.02, 0.89, "left"),
    "baseline-random": (0.30, 0.33, "left"),
    "baseline-less-cosine": (0.88, 0.47, "left"),
    "baseline-less": (0.88, 0.92, "center"),
    "selector-rank-euclidean": (0.24, 0.70, "left"),
    "selector-rank": (0.42, 0.60, "left"),
    "selector-greedy-euclidean": (0.50, 0.56, "left"),
    "selector-greedy": (0.50, 0.48, "left"),
}
_ORTHOGONAL_TARGET_HOLDOUT_LABEL_LAYOUT = {
    **_ORTHOGONAL_TARGET_LABEL_LAYOUT,
    "baseline-less-cosine": (0.85, 0.19, "center"),
}
_ORTHOGONAL_TARGET_ALIGNMENT_LABEL_LAYOUT = {
    **_ORTHOGONAL_TARGET_LABEL_LAYOUT,
    "baseline-less-cosine": (0.67, 0.19, "center"),
}


def _std_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def _sem_or_zero(values: pd.Series) -> float:
    if len(values) == 0:
        return float("nan")
    return _std_or_zero(values) / float(np.sqrt(len(values)))


def _summarize_holdout_tradeoff(frame: pd.DataFrame, drift_budget: float) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(_GROUP_KEYS, sort=False, dropna=False):
        target = group["target_improvement_holdout"]
        drift = group["reference_drift_holdout"]
        cosine = group["cosine_to_oracle"]
        records.append(
            {
                "method": keys[0],
                "method_family": keys[1],
                "solver": keys[2],
                "geometry": keys[3],
                "num_runs": int(group["replicate_index"].nunique()),
                "target_improvement_holdout_mean": float(target.mean()),
                "target_improvement_holdout_std": _std_or_zero(target),
                "target_improvement_holdout_sem": _sem_or_zero(target),
                "reference_drift_holdout_mean": float(drift.mean()),
                "reference_drift_holdout_std": _std_or_zero(drift),
                "reference_drift_holdout_sem": _sem_or_zero(drift),
                "cosine_to_oracle_mean": float(cosine.mean()),
                "cosine_to_oracle_std": _std_or_zero(cosine),
                "cosine_to_oracle_sem": _sem_or_zero(cosine),
                "holdout_budget_satisfaction_rate": float((drift <= drift_budget).mean()),
            }
        )
    summary = pd.DataFrame(records)
    return summary.sort_values(
        ["method_family", "reference_drift_holdout_mean", "target_improvement_holdout_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _summarize_full_metrics(frame: pd.DataFrame, drift_budget: float) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(_GROUP_KEYS, sort=False, dropna=False):
        target_alignment = group["target_alignment_exact"]
        target_exact = group["target_improvement_exact"]
        drift_exact = group["reference_drift_exact"]
        target_holdout = group["target_improvement_holdout"]
        drift_holdout = group["reference_drift_holdout"]
        cosine = group["cosine_to_oracle"]
        expensive = group["selected_useful_expensive"]
        cheap = group["selected_useful_cheap"]
        distractor = group["selected_distractor"]
        records.append(
            {
                "method": keys[0],
                "method_family": keys[1],
                "solver": keys[2],
                "geometry": keys[3],
                "num_runs": int(group["replicate_index"].nunique()),
                "selected_useful_expensive_mean": float(expensive.mean()),
                "selected_useful_expensive_std": _std_or_zero(expensive),
                "selected_useful_cheap_mean": float(cheap.mean()),
                "selected_useful_cheap_std": _std_or_zero(cheap),
                "selected_distractor_mean": float(distractor.mean()),
                "selected_distractor_std": _std_or_zero(distractor),
                "target_alignment_exact_mean": float(target_alignment.mean()),
                "target_alignment_exact_std": _std_or_zero(target_alignment),
                "target_improvement_exact_mean": float(target_exact.mean()),
                "target_improvement_exact_std": _std_or_zero(target_exact),
                "reference_drift_exact_mean": float(drift_exact.mean()),
                "reference_drift_exact_std": _std_or_zero(drift_exact),
                "target_improvement_holdout_mean": float(target_holdout.mean()),
                "target_improvement_holdout_std": _std_or_zero(target_holdout),
                "reference_drift_holdout_mean": float(drift_holdout.mean()),
                "reference_drift_holdout_std": _std_or_zero(drift_holdout),
                "cosine_to_oracle_mean": float(cosine.mean()),
                "cosine_to_oracle_std": _std_or_zero(cosine),
                "exact_budget_satisfaction_rate": float((drift_exact <= drift_budget).mean()),
                "holdout_budget_satisfaction_rate": float((drift_holdout <= drift_budget).mean()),
            }
        )
    summary = pd.DataFrame(records)
    return summary.sort_values(
        ["method_family", "reference_drift_holdout_mean", "target_improvement_holdout_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _summarize_alignment_tradeoff(frame: pd.DataFrame, drift_budget: float) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for keys, group in frame.groupby(_GROUP_KEYS, sort=False, dropna=False):
        alignment = group["target_alignment_exact"]
        drift = group["reference_drift_holdout"]
        cosine = group["cosine_to_oracle"]
        records.append(
            {
                "method": keys[0],
                "method_family": keys[1],
                "solver": keys[2],
                "geometry": keys[3],
                "num_runs": int(group["replicate_index"].nunique()),
                "target_alignment_exact_mean": float(alignment.mean()),
                "target_alignment_exact_std": _std_or_zero(alignment),
                "target_alignment_exact_sem": _sem_or_zero(alignment),
                "reference_drift_holdout_mean": float(drift.mean()),
                "reference_drift_holdout_std": _std_or_zero(drift),
                "reference_drift_holdout_sem": _sem_or_zero(drift),
                "cosine_to_oracle_mean": float(cosine.mean()),
                "cosine_to_oracle_std": _std_or_zero(cosine),
                "cosine_to_oracle_sem": _sem_or_zero(cosine),
                "holdout_budget_satisfaction_rate": float((drift <= drift_budget).mean()),
            }
        )
    summary = pd.DataFrame(records)
    return summary.sort_values(
        ["method_family", "reference_drift_holdout_mean", "target_alignment_exact_mean"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def _plot_average_tradeoff(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    drift_budget: float,
    x_mean_key: str,
    x_sem_key: str,
    x_label: str,
    title: str,
    preferred_offsets: dict[str, list[tuple[int, int]] | tuple[int, int]] | None = None,
    label_layout: dict[str, tuple[float, float, str]] | None = None,
) -> None:
    plotted_frame = frame.loc[frame["method_family"] != "experiment"].reset_index(drop=True)
    apply_style()
    figure, axis = plt.subplots(1, 1, figsize=(7.4, 5.2))
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.9)
    axis.axvline(0.0, color="#d9d9d9", linewidth=0.8, zorder=0)

    x_values = plotted_frame[x_mean_key].to_numpy(dtype=float)
    x_errors = plotted_frame[x_sem_key].to_numpy(dtype=float)
    y_values = plotted_frame["reference_drift_holdout_mean"].to_numpy(dtype=float)
    y_errors = plotted_frame["reference_drift_holdout_sem"].to_numpy(dtype=float)

    x_min = float(np.min(x_values - x_errors))
    x_max = float(np.max(x_values + x_errors))
    x_span = max(x_max - x_min, 1.0)
    x_limits = (x_min - 0.08 * x_span, x_max + 0.10 * x_span)

    y_lower = np.maximum(1.0e-6, y_values - y_errors)
    y_upper = y_values + y_errors
    y_limits = (min(max(1.0e-6, float(y_lower.min()) * 0.72), drift_budget * 0.88), float(y_upper.max()) * 1.22)

    for _, row in plotted_frame.iterrows():
        method_name = str(row["method"])
        base_method_name = method_name.removesuffix("-euclidean")
        color = METHOD_COLORS.get(method_name, METHOD_COLORS.get(base_method_name, "#333333"))
        geometry = str(row["geometry"])
        marker_style = "o" if geometry == "reference" else "s"

        axis.errorbar(
            float(row[x_mean_key]),
            float(row["reference_drift_holdout_mean"]),
            xerr=float(row[x_sem_key]),
            yerr=float(row["reference_drift_holdout_sem"]),
            fmt="none",
            ecolor=color,
            elinewidth=1.05,
            alpha=0.55,
            capsize=2.5,
            zorder=2,
        )
        axis.scatter(
            float(row[x_mean_key]),
            float(row["reference_drift_holdout_mean"]),
            s=92,
            marker=marker_style,
            facecolors=color,
            edgecolors=color,
            linewidths=1.9,
            alpha=0.92,
            zorder=4,
        )

    axis.set_xlim(*x_limits)
    axis.set_yscale("log")
    axis.set_ylim(*y_limits)
    axis.axhline(drift_budget, color="#c0392b", linestyle="--", linewidth=1.6, alpha=0.95, zorder=6)
    if label_layout is None:
        _annotate_scatter_points_without_overlap(
            axis,
            plotted_frame,
            x_key=x_mean_key,
            y_key="reference_drift_holdout_mean",
            label_map=_METHOD_LABELS,
            x_limits=x_limits,
            y_limits=y_limits,
            y_scale="log",
            preferred_offsets=_PREFERRED_LABEL_OFFSETS if preferred_offsets is None else preferred_offsets,
        )
    else:
        for _, row in plotted_frame.iterrows():
            method_name = str(row["method"])
            if method_name not in label_layout:
                continue
            x_value = float(row[x_mean_key])
            y_value = float(row["reference_drift_holdout_mean"])
            label_text = _METHOD_LABELS.get(method_name, method_name)
            base_method_name = method_name.removesuffix("-euclidean")
            color = METHOD_COLORS.get(method_name, METHOD_COLORS.get(base_method_name, "#333333"))
            x_axes, y_axes, ha = label_layout[method_name]
            axis.annotate(
                label_text,
                xy=(x_value, y_value),
                xycoords="data",
                xytext=(x_axes, y_axes),
                textcoords="axes fraction",
                ha=ha,
                va="center",
                fontsize=8.5,
                color=color,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 0.95,
                    "alpha": 0.70,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=5,
            )
    axis.set_title(f"{title} ({int(plotted_frame['num_runs'].max())} runs)")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean held-out reference drift (log scale)")
    save_figure(figure, str(output_path))


def run_selection_replicates_experiment(
    output_root: Path,
    seed: int = 1,
    *,
    num_runs: int = 20,
    beta_target: np.ndarray | None = None,
    selection_output_dir_name: str = "02_selection",
    replicates_output_dir_name: str = "02_selection_replicates",
    experiment_name: str = "2D discrete selection replicates",
    preferred_label_offsets: dict[str, list[tuple[int, int]] | tuple[int, int]] | None = None,
    label_layout: dict[str, tuple[float, float, str]] | None = None,
    holdout_label_layout: dict[str, tuple[float, float, str]] | None = None,
    alignment_label_layout: dict[str, tuple[float, float, str]] | None = None,
) -> dict:
    output_dir = ensure_output_dir(Path(output_root) / replicates_output_dir_name)
    replicates_root = ensure_output_dir(output_dir / "replicates")

    holdout_frames: list[pd.DataFrame] = []
    full_metric_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, int | str]] = []

    for replicate_index in range(1, num_runs + 1):
        run_seed = seed + replicate_index - 1
        replicate_root = replicates_root / f"run_{replicate_index:02d}"
        run_selection_experiment(
            output_root=replicate_root,
            seed=run_seed,
            beta_target=beta_target,
            output_dir_name=selection_output_dir_name,
            experiment_name=experiment_name.replace(" replicates", ""),
        )

        replicate_dir = replicate_root / selection_output_dir_name
        holdout_frame = pd.read_csv(replicate_dir / "holdout_tradeoff_metrics.csv")
        holdout_frame["replicate_index"] = replicate_index
        holdout_frame["seed"] = run_seed
        holdout_frames.append(holdout_frame)

        full_metric_frame = pd.read_csv(replicate_dir / "single_pool_metrics.csv")
        full_metric_frame["replicate_index"] = replicate_index
        full_metric_frame["seed"] = run_seed
        full_metric_frames.append(full_metric_frame)

        run_records.append(
            {
                "replicate_index": replicate_index,
                "seed": run_seed,
                "directory": str(replicate_dir),
            }
        )

    all_holdout = pd.concat(holdout_frames, ignore_index=True)
    all_full_metrics = pd.concat(full_metric_frames, ignore_index=True)

    average_holdout = _summarize_holdout_tradeoff(all_holdout, drift_budget=1.0)
    average_full_metrics = _summarize_full_metrics(all_full_metrics, drift_budget=1.0)
    average_alignment = _summarize_alignment_tradeoff(all_full_metrics, drift_budget=1.0)

    write_frame(all_holdout, output_dir / "holdout_tradeoff_metrics_all_runs.csv")
    write_frame(all_full_metrics, output_dir / "single_pool_metrics_all_runs.csv")
    write_frame(average_holdout, output_dir / "average_holdout_tradeoff_metrics.csv")
    write_frame(average_full_metrics, output_dir / "average_single_pool_metrics.csv")
    write_frame(average_alignment, output_dir / "average_alignment_tradeoff_metrics.csv")
    _plot_average_tradeoff(
        average_holdout,
        output_dir / "average_holdout_tradeoff_scatter.png",
        drift_budget=1.0,
        x_mean_key="target_improvement_holdout_mean",
        x_sem_key="target_improvement_holdout_sem",
        x_label="Mean held-out target improvement",
        title="Average Held-Out Target Improvement vs Average Held-Out Reference Drift",
        preferred_offsets=preferred_label_offsets,
        label_layout=holdout_label_layout if holdout_label_layout is not None else label_layout,
    )
    _plot_average_tradeoff(
        average_alignment,
        output_dir / "average_alignment_tradeoff_scatter.png",
        drift_budget=1.0,
        x_mean_key="target_alignment_exact_mean",
        x_sem_key="target_alignment_exact_sem",
        x_label=r"Mean alignment $-g_T^\top \Delta\theta$",
        title=r"Average Alignment $-g_T^\top \Delta\theta$ vs Average Held-Out Reference Drift",
        preferred_offsets=preferred_label_offsets,
        label_layout=alignment_label_layout if alignment_label_layout is not None else label_layout,
    )
    write_json(
        {
            "seed": seed,
            "num_runs": num_runs,
            "beta_target": None if beta_target is None else np.asarray(beta_target, dtype=float).tolist(),
            "drift_budget": 1.0,
            "replicates": run_records,
        },
        output_dir / "config.json",
    )

    return {
        "name": experiment_name,
        "directory": str(output_dir),
        "tables": [
            str(output_dir / "average_holdout_tradeoff_metrics.csv"),
            str(output_dir / "average_alignment_tradeoff_metrics.csv"),
            str(output_dir / "average_single_pool_metrics.csv"),
            str(output_dir / "holdout_tradeoff_metrics_all_runs.csv"),
            str(output_dir / "single_pool_metrics_all_runs.csv"),
        ],
        "figures": [
            str(output_dir / "average_holdout_tradeoff_scatter.png"),
            str(output_dir / "average_alignment_tradeoff_scatter.png"),
        ],
    }


def run_selection_replicates_orthogonal_target_experiment(
    output_root: Path,
    seed: int = 1,
    *,
    num_runs: int = 20,
) -> dict:
    beta_target = np.array([-np.sin(np.pi / 6.0), np.cos(np.pi / 6.0)], dtype=float)
    return run_selection_replicates_experiment(
        output_root=output_root,
        seed=seed,
        num_runs=num_runs,
        beta_target=beta_target,
        selection_output_dir_name="02_selection_orthogonal_target",
        replicates_output_dir_name="02_selection_orthogonal_target_replicates",
        experiment_name="2D discrete selection replicates (orthogonal target)",
        holdout_label_layout=_ORTHOGONAL_TARGET_HOLDOUT_LABEL_LAYOUT,
        alignment_label_layout=_ORTHOGONAL_TARGET_ALIGNMENT_LABEL_LAYOUT,
    )
