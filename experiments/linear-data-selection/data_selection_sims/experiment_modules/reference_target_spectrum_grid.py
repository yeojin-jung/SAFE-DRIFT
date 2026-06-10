from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..plotting import apply_style, save_figure
from ._shared import ensure_output_dir, write_frame, write_json
from .dimred_selector_comparison import (
    DimRedSelectorConfig,
    METHOD_COLORS,
    METHOD_LABELS,
    run_dimred_selector_comparison_experiment,
)


@dataclass(frozen=True)
class ReferenceTargetSpectrumGridConfig:
    dimred_config: DimRedSelectorConfig
    reference_decays: tuple[str, ...] = ("geometric", "linear", "powerlaw")
    target_decays: tuple[str, ...] = ("geometric", "linear", "powerlaw")
    reference_powerlaw_exponents: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    target_powerlaw_exponents: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    comparison_reference_powerlaw_exponent: float = 1.0
    comparison_target_powerlaw_exponent: float = 1.0


def _sem_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _setting_label(decay: str, powerlaw_exponent: float | None) -> str:
    if decay == "powerlaw":
        exponent_text = "na" if powerlaw_exponent is None else f"{float(powerlaw_exponent):.2f}".rstrip("0").rstrip(".")
        return f"powerlaw-q{exponent_text}"
    return str(decay)


def _setting_title(prefix: str, decay: str, powerlaw_exponent: float | None) -> str:
    if decay == "powerlaw":
        return f"{prefix} powerlaw (q={float(powerlaw_exponent):.2f})"
    return f"{prefix} {decay}"


def _family_settings(decays: tuple[str, ...], comparison_powerlaw_exponent: float) -> list[tuple[str, str, float | None]]:
    settings: list[tuple[str, str, float | None]] = []
    for decay in decays:
        exponent = float(comparison_powerlaw_exponent) if decay == "powerlaw" else None
        settings.append((_setting_label(decay, exponent), decay, exponent))
    return settings


def _powerlaw_settings(exponents: tuple[float, ...]) -> list[tuple[str, str, float]]:
    return [(_setting_label("powerlaw", exponent), "powerlaw", float(exponent)) for exponent in exponents]


def _make_pair_config(
    base: DimRedSelectorConfig,
    *,
    reference_decay: str,
    reference_powerlaw_exponent: float | None,
    target_decay: str,
    target_powerlaw_exponent: float | None,
) -> DimRedSelectorConfig:
    return DimRedSelectorConfig(
        **{
            **asdict(base),
            "spectrum_decay": str(reference_decay),
            "powerlaw_exponent": (
                float(reference_powerlaw_exponent)
                if reference_powerlaw_exponent is not None
                else base.powerlaw_exponent
            ),
            "target_spectrum_decay": str(target_decay),
            "target_powerlaw_exponent": target_powerlaw_exponent,
        }
    )


def _augment_frame(
    frame: pd.DataFrame,
    *,
    reference_setting: str,
    reference_decay: str,
    reference_powerlaw_exponent: float | None,
    target_setting: str,
    target_decay: str,
    target_powerlaw_exponent: float | None,
) -> pd.DataFrame:
    augmented = frame.copy()
    augmented["reference_setting"] = reference_setting
    augmented["reference_decay"] = reference_decay
    augmented["reference_powerlaw_exponent"] = (
        float(reference_powerlaw_exponent) if reference_powerlaw_exponent is not None else float("nan")
    )
    augmented["target_setting"] = target_setting
    augmented["target_decay"] = target_decay
    augmented["target_powerlaw_exponent"] = (
        float(target_powerlaw_exponent) if target_powerlaw_exponent is not None else float("nan")
    )
    return augmented


def _add_absolute_regret(frame: pd.DataFrame) -> pd.DataFrame:
    plot_frame = frame.copy()
    group_columns = ["reference_setting", "target_setting", "regime", "seed", "K_R", "K_T"]
    safe_full = plot_frame.loc[
        plot_frame["method"] == "safe-full",
        group_columns + ["gain"],
    ].rename(columns={"gain": "safe_full_gain"})
    plot_frame = plot_frame.merge(safe_full, on=group_columns, how="left")
    plot_frame["absolute_regret_vs_safe_full"] = plot_frame["safe_full_gain"] - plot_frame["gain"]
    return plot_frame


def _summarize_grid(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for keys, subframe in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_row = {column: value for column, value in zip(group_columns, keys)}
        rows.append(
            {
                **key_row,
                "num_instances": int(len(subframe)),
                "num_runs": int(subframe["seed"].nunique()),
                "absolute_regret_mean": float(subframe["absolute_regret_vs_safe_full"].mean()),
                "absolute_regret_sem": _sem_or_zero(subframe["absolute_regret_vs_safe_full"]),
                "gain_mean": float(subframe["gain"].mean()),
                "gain_sem": _sem_or_zero(subframe["gain"]),
                "drift_mean": float(subframe["drift"].mean()),
                "drift_sem": _sem_or_zero(subframe["drift"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_grid(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    row_order: list[str],
    col_order: list[str],
    row_title_map: dict[str, str],
    col_title_map: dict[str, str],
    methods: list[str],
    figure_title: str,
    metric_mean_column: str,
    metric_sem_column: str,
    y_axis_label: str,
    y_scale: str = "linear",
    symlog_linthresh: float = 0.05,
    horizontal_line: float | None = None,
) -> None:
    if summary.empty:
        return
    plot_frame = summary.loc[summary["method"].isin(methods)].copy()
    available_rows = [value for value in row_order if value in set(plot_frame["reference_setting"].tolist())]
    available_cols = [value for value in col_order if value in set(plot_frame["target_setting"].tolist())]
    if not available_rows or not available_cols:
        return

    apply_style()
    figure, axes = plt.subplots(
        len(available_rows),
        len(available_cols),
        figsize=(5.1 * len(available_cols), 4.1 * len(available_rows)),
        sharex=True,
        sharey=True,
    )
    if len(available_rows) == 1 and len(available_cols) == 1:
        axes = [[axes]]
    elif len(available_rows) == 1:
        axes = [list(axes)]
    elif len(available_cols) == 1:
        axes = [[axis] for axis in axes]
    else:
        axes = [list(row) for row in axes]

    figure.subplots_adjust(bottom=0.18, wspace=0.16, hspace=0.22)

    for row_index, reference_setting in enumerate(available_rows):
        for col_index, target_setting in enumerate(available_cols):
            axis = axes[row_index][col_index]
            subframe = plot_frame.loc[
                (plot_frame["reference_setting"] == reference_setting)
                & (plot_frame["target_setting"] == target_setting)
            ]
            for method, method_frame in subframe.groupby("method", sort=False):
                axis.errorbar(
                    method_frame["K_requested"],
                    method_frame[metric_mean_column],
                    yerr=method_frame[metric_sem_column].fillna(0.0),
                    marker="o",
                    color=METHOD_COLORS.get(method, "#333333"),
                    linewidth=1.5,
                    markersize=4.0,
                    capsize=2.2,
                    label=METHOD_LABELS.get(method, method),
                )
            if horizontal_line is not None:
                axis.axhline(horizontal_line, color="#999999", linestyle="--", linewidth=1.0)
            if y_scale == "symlog":
                axis.set_yscale("symlog", linthresh=symlog_linthresh)
            elif y_scale == "log":
                axis.set_yscale("log")
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(col_title_map.get(target_setting, target_setting))
            if col_index == 0:
                axis.set_ylabel(
                    y_axis_label + "\n"
                    + row_title_map.get(reference_setting, reference_setting),
                    fontsize=9,
                )
            if row_index == len(available_rows) - 1:
                axis.set_xlabel("Requested rank K_R + K_T")

    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, fontsize=8)
    figure.suptitle(figure_title, y=0.99)
    save_figure(figure, str(output_path))


def run_reference_target_spectrum_grid(
    output_root: str | Path,
    *,
    seed: int = 1,
    config: ReferenceTargetSpectrumGridConfig,
) -> dict:
    output_dir = ensure_output_dir(Path(output_root) / "10_reference_target_spectrum_grid")
    base = config.dimred_config

    family_reference_settings = _family_settings(config.reference_decays, config.comparison_reference_powerlaw_exponent)
    family_target_settings = _family_settings(config.target_decays, config.comparison_target_powerlaw_exponent)
    powerlaw_reference_settings = _powerlaw_settings(config.reference_powerlaw_exponents)
    powerlaw_target_settings = _powerlaw_settings(config.target_powerlaw_exponents)

    unique_pairs: dict[str, tuple[str, str, float | None, str, str, float | None]] = {}
    for ref_label, ref_decay, ref_exp in family_reference_settings:
        for tgt_label, tgt_decay, tgt_exp in family_target_settings:
            pair_key = f"ref-{ref_label}__target-{tgt_label}"
            unique_pairs[pair_key] = (ref_label, ref_decay, ref_exp, tgt_label, tgt_decay, tgt_exp)
    for ref_label, ref_decay, ref_exp in powerlaw_reference_settings:
        for tgt_label, tgt_decay, tgt_exp in powerlaw_target_settings:
            pair_key = f"ref-{ref_label}__target-{tgt_label}"
            unique_pairs[pair_key] = (ref_label, ref_decay, ref_exp, tgt_label, tgt_decay, tgt_exp)

    frames: list[pd.DataFrame] = []
    manifests: dict[str, dict] = {}

    for pair_key, (ref_label, ref_decay, ref_exp, tgt_label, tgt_decay, tgt_exp) in unique_pairs.items():
        pair_config = _make_pair_config(
            base,
            reference_decay=ref_decay,
            reference_powerlaw_exponent=ref_exp,
            target_decay=tgt_decay,
            target_powerlaw_exponent=tgt_exp,
        )
        pair_root = output_dir / pair_key
        manifest = run_dimred_selector_comparison_experiment(
            output_root=pair_root,
            seed=seed,
            config=pair_config,
        )
        manifests[pair_key] = manifest
        frame = pd.read_csv(Path(manifest["tables"][0]))
        frames.append(
            _augment_frame(
                frame,
                reference_setting=ref_label,
                reference_decay=ref_decay,
                reference_powerlaw_exponent=ref_exp,
                target_setting=tgt_label,
                target_decay=tgt_decay,
                target_powerlaw_exponent=tgt_exp,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined_with_regret = _add_absolute_regret(combined)
    write_frame(combined_with_regret, output_dir / "combined_reference_target_runs.csv")

    summary = _summarize_grid(
        combined_with_regret,
        [
            "reference_setting",
            "reference_decay",
            "reference_powerlaw_exponent",
            "target_setting",
            "target_decay",
            "target_powerlaw_exponent",
            "method",
            "K_requested",
        ],
    )
    summary_by_regime = _summarize_grid(
        combined_with_regret,
        [
            "reference_setting",
            "reference_decay",
            "reference_powerlaw_exponent",
            "target_setting",
            "target_decay",
            "target_powerlaw_exponent",
            "regime",
            "method",
            "K_requested",
        ],
    )
    write_frame(summary, output_dir / "reference_target_regret_summary.csv")
    write_frame(summary_by_regime, output_dir / "reference_target_regret_summary_by_regime.csv")

    row_titles_family = {
        label: _setting_title("reference", decay, exponent)
        for label, decay, exponent in family_reference_settings
    }
    col_titles_family = {
        label: _setting_title("target", decay, exponent)
        for label, decay, exponent in family_target_settings
    }
    row_titles_powerlaw = {
        label: _setting_title("reference", decay, exponent)
        for label, decay, exponent in powerlaw_reference_settings
    }
    col_titles_powerlaw = {
        label: _setting_title("target", decay, exponent)
        for label, decay, exponent in powerlaw_target_settings
    }

    family_summary = summary.loc[
        summary["reference_setting"].isin([label for label, _, _ in family_reference_settings])
        & summary["target_setting"].isin([label for label, _, _ in family_target_settings])
    ].copy()
    powerlaw_summary = summary.loc[
        summary["reference_setting"].isin([label for label, _, _ in powerlaw_reference_settings])
        & summary["target_setting"].isin([label for label, _, _ in powerlaw_target_settings])
    ].copy()

    all_methods = list(METHOD_LABELS.keys())
    safe_methods = [
        "safe-full",
        "safe-lowrank",
        "safe-reference-only",
        "safe-task-only",
        "safe-random-k",
        "safe-diagonal",
    ]

    _plot_metric_grid(
        family_summary,
        output_dir / "absolute_regret_reference_vs_target_decay_all_methods.png",
        row_order=[label for label, _, _ in family_reference_settings],
        col_order=[label for label, _, _ in family_target_settings],
        row_title_map=row_titles_family,
        col_title_map=col_titles_family,
        methods=all_methods,
        figure_title="Absolute regret vs SAFE full: reference vs target decay families",
        metric_mean_column="absolute_regret_mean",
        metric_sem_column="absolute_regret_sem",
        y_axis_label="Absolute regret vs SAFE full (symlog)",
        y_scale="symlog",
        symlog_linthresh=0.05,
        horizontal_line=0.0,
    )
    _plot_metric_grid(
        family_summary,
        output_dir / "absolute_regret_reference_vs_target_decay_safe_methods.png",
        row_order=[label for label, _, _ in family_reference_settings],
        col_order=[label for label, _, _ in family_target_settings],
        row_title_map=row_titles_family,
        col_title_map=col_titles_family,
        methods=safe_methods,
        figure_title="Absolute regret vs SAFE full: reference vs target decay families (SAFE variants)",
        metric_mean_column="absolute_regret_mean",
        metric_sem_column="absolute_regret_sem",
        y_axis_label="Absolute regret vs SAFE full (symlog)",
        y_scale="symlog",
        symlog_linthresh=0.05,
        horizontal_line=0.0,
    )
    _plot_metric_grid(
        powerlaw_summary,
        output_dir / "absolute_regret_reference_vs_target_powerlaw_all_methods.png",
        row_order=[label for label, _, _ in powerlaw_reference_settings],
        col_order=[label for label, _, _ in powerlaw_target_settings],
        row_title_map=row_titles_powerlaw,
        col_title_map=col_titles_powerlaw,
        methods=all_methods,
        figure_title="Absolute regret vs SAFE full: reference vs target powerlaw exponents",
        metric_mean_column="absolute_regret_mean",
        metric_sem_column="absolute_regret_sem",
        y_axis_label="Absolute regret vs SAFE full (symlog)",
        y_scale="symlog",
        symlog_linthresh=0.05,
        horizontal_line=0.0,
    )
    _plot_metric_grid(
        powerlaw_summary,
        output_dir / "absolute_regret_reference_vs_target_powerlaw_safe_methods.png",
        row_order=[label for label, _, _ in powerlaw_reference_settings],
        col_order=[label for label, _, _ in powerlaw_target_settings],
        row_title_map=row_titles_powerlaw,
        col_title_map=col_titles_powerlaw,
        methods=safe_methods,
        figure_title="Absolute regret vs SAFE full: reference vs target powerlaw exponents (SAFE variants)",
        metric_mean_column="absolute_regret_mean",
        metric_sem_column="absolute_regret_sem",
        y_axis_label="Absolute regret vs SAFE full (symlog)",
        y_scale="symlog",
        symlog_linthresh=0.05,
        horizontal_line=0.0,
    )
    _plot_metric_grid(
        family_summary,
        output_dir / "reference_drift_reference_vs_target_decay_all_methods.png",
        row_order=[label for label, _, _ in family_reference_settings],
        col_order=[label for label, _, _ in family_target_settings],
        row_title_map=row_titles_family,
        col_title_map=col_titles_family,
        methods=all_methods,
        figure_title="Reference drift: reference vs target decay families",
        metric_mean_column="drift_mean",
        metric_sem_column="drift_sem",
        y_axis_label="Mean reference drift (log)",
        y_scale="log",
        horizontal_line=base.rho if base.rho is not None else None,
    )
    _plot_metric_grid(
        family_summary,
        output_dir / "reference_drift_reference_vs_target_decay_safe_methods.png",
        row_order=[label for label, _, _ in family_reference_settings],
        col_order=[label for label, _, _ in family_target_settings],
        row_title_map=row_titles_family,
        col_title_map=col_titles_family,
        methods=safe_methods,
        figure_title="Reference drift: reference vs target decay families (SAFE variants)",
        metric_mean_column="drift_mean",
        metric_sem_column="drift_sem",
        y_axis_label="Mean reference drift (log)",
        y_scale="log",
        horizontal_line=base.rho if base.rho is not None else None,
    )
    _plot_metric_grid(
        powerlaw_summary,
        output_dir / "reference_drift_reference_vs_target_powerlaw_all_methods.png",
        row_order=[label for label, _, _ in powerlaw_reference_settings],
        col_order=[label for label, _, _ in powerlaw_target_settings],
        row_title_map=row_titles_powerlaw,
        col_title_map=col_titles_powerlaw,
        methods=all_methods,
        figure_title="Reference drift: reference vs target powerlaw exponents",
        metric_mean_column="drift_mean",
        metric_sem_column="drift_sem",
        y_axis_label="Mean reference drift (log)",
        y_scale="log",
        horizontal_line=base.rho if base.rho is not None else None,
    )
    _plot_metric_grid(
        powerlaw_summary,
        output_dir / "reference_drift_reference_vs_target_powerlaw_safe_methods.png",
        row_order=[label for label, _, _ in powerlaw_reference_settings],
        col_order=[label for label, _, _ in powerlaw_target_settings],
        row_title_map=row_titles_powerlaw,
        col_title_map=col_titles_powerlaw,
        methods=safe_methods,
        figure_title="Reference drift: reference vs target powerlaw exponents (SAFE variants)",
        metric_mean_column="drift_mean",
        metric_sem_column="drift_sem",
        y_axis_label="Mean reference drift (log)",
        y_scale="log",
        horizontal_line=base.rho if base.rho is not None else None,
    )

    write_json(
        {
            "seed": seed,
            "grid_config": {
                "dimred_config": asdict(config.dimred_config),
                "reference_decays": list(config.reference_decays),
                "target_decays": list(config.target_decays),
                "reference_powerlaw_exponents": list(config.reference_powerlaw_exponents),
                "target_powerlaw_exponents": list(config.target_powerlaw_exponents),
                "comparison_reference_powerlaw_exponent": config.comparison_reference_powerlaw_exponent,
                "comparison_target_powerlaw_exponent": config.comparison_target_powerlaw_exponent,
            },
            "setting_manifests": manifests,
        },
        output_dir / "config.json",
    )

    return {
        "name": "Reference-target spectrum grid",
        "directory": str(output_dir),
        "tables": [
            str(output_dir / "combined_reference_target_runs.csv"),
            str(output_dir / "reference_target_regret_summary.csv"),
            str(output_dir / "reference_target_regret_summary_by_regime.csv"),
        ],
        "figures": [
            str(output_dir / "absolute_regret_reference_vs_target_decay_all_methods.png"),
            str(output_dir / "absolute_regret_reference_vs_target_decay_safe_methods.png"),
            str(output_dir / "absolute_regret_reference_vs_target_powerlaw_all_methods.png"),
            str(output_dir / "absolute_regret_reference_vs_target_powerlaw_safe_methods.png"),
            str(output_dir / "reference_drift_reference_vs_target_decay_all_methods.png"),
            str(output_dir / "reference_drift_reference_vs_target_decay_safe_methods.png"),
            str(output_dir / "reference_drift_reference_vs_target_powerlaw_all_methods.png"),
            str(output_dir / "reference_drift_reference_vs_target_powerlaw_safe_methods.png"),
        ],
    }


__all__ = [
    "ReferenceTargetSpectrumGridConfig",
    "run_reference_target_spectrum_grid",
]
