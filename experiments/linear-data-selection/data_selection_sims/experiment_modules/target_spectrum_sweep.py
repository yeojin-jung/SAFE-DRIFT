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
class TargetSpectrumSweepConfig:
    dimred_config: DimRedSelectorConfig
    target_decays: tuple[str, ...] = ("geometric", "linear", "powerlaw")
    target_powerlaw_exponents: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    comparison_powerlaw_exponent: float = 1.0


def _sem_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _setting_label(target_decay: str, target_powerlaw_exponent: float | None) -> str:
    if target_decay == "powerlaw":
        exponent_text = "na" if target_powerlaw_exponent is None else f"{float(target_powerlaw_exponent):.2f}".rstrip("0").rstrip(".")
        return f"powerlaw-q{exponent_text}"
    return str(target_decay)


def _setting_title(target_decay: str, target_powerlaw_exponent: float | None) -> str:
    if target_decay == "powerlaw":
        return f"target powerlaw (q={float(target_powerlaw_exponent):.2f})"
    return f"target {target_decay}"


def _build_setting_configs(config: TargetSpectrumSweepConfig) -> list[tuple[str, DimRedSelectorConfig]]:
    settings: list[tuple[str, DimRedSelectorConfig]] = []
    base = config.dimred_config
    for target_decay in config.target_decays:
        if target_decay == "powerlaw":
            for exponent in config.target_powerlaw_exponents:
                settings.append(
                    (
                        _setting_label(target_decay, exponent),
                        DimRedSelectorConfig(
                            **{
                                **asdict(base),
                                "target_spectrum_decay": "powerlaw",
                                "target_powerlaw_exponent": float(exponent),
                            }
                        ),
                    )
                )
        else:
            settings.append(
                (
                    _setting_label(target_decay, None),
                    DimRedSelectorConfig(
                        **{
                            **asdict(base),
                            "target_spectrum_decay": str(target_decay),
                            "target_powerlaw_exponent": None,
                        }
                    ),
                )
            )
    return settings


def _augment_frame(frame: pd.DataFrame, *, setting_name: str, target_decay: str, target_powerlaw_exponent: float | None) -> pd.DataFrame:
    augmented = frame.copy()
    augmented["target_setting"] = setting_name
    augmented["target_decay"] = target_decay
    augmented["target_powerlaw_exponent"] = (
        float(target_powerlaw_exponent) if target_powerlaw_exponent is not None else float("nan")
    )
    return augmented


def _add_absolute_regret(frame: pd.DataFrame) -> pd.DataFrame:
    plot_frame = frame.copy()
    group_columns = ["target_setting", "regime", "seed", "K_R", "K_T"]
    safe_full = plot_frame.loc[plot_frame["method"] == "safe-full", group_columns + ["gain"]].rename(columns={"gain": "safe_full_gain"})
    plot_frame = plot_frame.merge(safe_full, on=group_columns, how="left")
    plot_frame["absolute_regret_vs_safe_full"] = plot_frame["safe_full_gain"] - plot_frame["gain"]
    return plot_frame


def _summarize_regret(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
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


def _plot_faceted_regret(
    summary: pd.DataFrame,
    output_path: Path,
    facet_order: list[str],
    *,
    title_map: dict[str, str],
    methods: list[str] | None = None,
    figure_title: str,
) -> None:
    if summary.empty:
        return
    plot_frame = summary.copy()
    if methods is not None:
        plot_frame = plot_frame.loc[plot_frame["method"].isin(methods)].copy()
    facet_values = [value for value in facet_order if value in set(plot_frame["facet_value"].tolist())]
    if not facet_values:
        return

    apply_style()
    figure, axes = plt.subplots(1, len(facet_values), figsize=(5.8 * len(facet_values), 5.4), sharey=True)
    if len(facet_values) == 1:
        axes = [axes]
    figure.subplots_adjust(bottom=0.30, wspace=0.20)

    for axis, facet_value in zip(axes, facet_values):
        subframe = plot_frame.loc[plot_frame["facet_value"] == facet_value]
        for method, method_frame in subframe.groupby("method", sort=False):
            axis.errorbar(
                method_frame["K_requested"],
                method_frame["absolute_regret_mean"],
                yerr=method_frame["absolute_regret_sem"].fillna(0.0),
                marker="o",
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method, "#333333"),
                capsize=2.5,
                linewidth=1.8,
                markersize=4.8,
            )
        axis.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
        axis.set_yscale("symlog", linthresh=0.05)
        axis.set_title(title_map.get(facet_value, facet_value))
        axis.set_xlabel("Requested rank K_R + K_T")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Absolute regret vs SAFE full (symlog)")
    figure.suptitle(figure_title, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=8)
    save_figure(figure, str(output_path))


def run_target_spectrum_sweep(
    output_root: str | Path,
    *,
    seed: int = 1,
    config: TargetSpectrumSweepConfig,
) -> dict:
    output_dir = ensure_output_dir(Path(output_root) / "09_target_spectrum_sweep")
    setting_frames: list[pd.DataFrame] = []
    manifests: dict[str, dict] = {}

    for setting_name, setting_config in _build_setting_configs(config):
        setting_root = output_dir / setting_name
        manifest = run_dimred_selector_comparison_experiment(
            output_root=setting_root,
            seed=seed,
            config=setting_config,
        )
        manifests[setting_name] = manifest
        comparison_path = Path(manifest["tables"][0])
        frame = pd.read_csv(comparison_path)
        setting_frames.append(
            _augment_frame(
                frame,
                setting_name=setting_name,
                target_decay=str(setting_config.target_spectrum_decay),
                target_powerlaw_exponent=setting_config.target_powerlaw_exponent,
            )
        )

    combined = pd.concat(setting_frames, ignore_index=True)
    combined_with_regret = _add_absolute_regret(combined)
    write_frame(combined_with_regret, output_dir / "combined_target_spectrum_runs.csv")

    by_setting = _summarize_regret(
        combined_with_regret,
        ["target_setting", "target_decay", "target_powerlaw_exponent", "method", "K_requested"],
    )
    by_setting_and_regime = _summarize_regret(
        combined_with_regret,
        ["target_setting", "target_decay", "target_powerlaw_exponent", "regime", "method", "K_requested"],
    )
    write_frame(by_setting, output_dir / "target_spectrum_regret_summary.csv")
    write_frame(by_setting_and_regime, output_dir / "target_spectrum_regret_summary_by_regime.csv")

    decay_comparison = by_setting.loc[
        (
            (by_setting["target_decay"] == "geometric")
            | (by_setting["target_decay"] == "linear")
            | (
                (by_setting["target_decay"] == "powerlaw")
                & (by_setting["target_powerlaw_exponent"].round(8) == round(config.comparison_powerlaw_exponent, 8))
            )
        )
    ].copy()
    decay_comparison["facet_value"] = decay_comparison["target_decay"].astype(str)
    decay_titles = {
        "geometric": "target geometric",
        "linear": "target linear",
        "powerlaw": f"target powerlaw (q={config.comparison_powerlaw_exponent:g})",
    }

    powerlaw_only = by_setting.loc[by_setting["target_decay"] == "powerlaw"].copy()
    powerlaw_only["facet_value"] = powerlaw_only["target_setting"].astype(str)
    powerlaw_titles = {
        _setting_label("powerlaw", exponent): _setting_title("powerlaw", exponent)
        for exponent in config.target_powerlaw_exponents
    }

    all_methods = list(METHOD_LABELS.keys())
    safe_methods = [
        "safe-full",
        "safe-lowrank",
        "safe-reference-only",
        "safe-task-only",
        "safe-random-k",
        "safe-diagonal",
    ]

    _plot_faceted_regret(
        decay_comparison,
        output_dir / "absolute_regret_by_target_decay_all_methods.png",
        ["geometric", "linear", "powerlaw"],
        title_map=decay_titles,
        methods=all_methods,
        figure_title="Absolute regret vs SAFE full across target decay families",
    )
    _plot_faceted_regret(
        decay_comparison,
        output_dir / "absolute_regret_by_target_decay_safe_methods.png",
        ["geometric", "linear", "powerlaw"],
        title_map=decay_titles,
        methods=safe_methods,
        figure_title="Absolute regret vs SAFE full across target decay families (SAFE variants)",
    )
    _plot_faceted_regret(
        powerlaw_only,
        output_dir / "absolute_regret_by_target_powerlaw_exponent_all_methods.png",
        [_setting_label("powerlaw", exponent) for exponent in config.target_powerlaw_exponents],
        title_map=powerlaw_titles,
        methods=all_methods,
        figure_title="Absolute regret vs SAFE full across target powerlaw exponents",
    )
    _plot_faceted_regret(
        powerlaw_only,
        output_dir / "absolute_regret_by_target_powerlaw_exponent_safe_methods.png",
        [_setting_label("powerlaw", exponent) for exponent in config.target_powerlaw_exponents],
        title_map=powerlaw_titles,
        methods=safe_methods,
        figure_title="Absolute regret vs SAFE full across target powerlaw exponents (SAFE variants)",
    )

    write_json(
        {
            "seed": seed,
            "sweep_config": {
                "dimred_config": asdict(config.dimred_config),
                "target_decays": list(config.target_decays),
                "target_powerlaw_exponents": list(config.target_powerlaw_exponents),
                "comparison_powerlaw_exponent": config.comparison_powerlaw_exponent,
            },
            "setting_manifests": manifests,
        },
        output_dir / "config.json",
    )

    return {
        "name": "Target spectrum sweep",
        "directory": str(output_dir),
        "tables": [
            str(output_dir / "combined_target_spectrum_runs.csv"),
            str(output_dir / "target_spectrum_regret_summary.csv"),
            str(output_dir / "target_spectrum_regret_summary_by_regime.csv"),
        ],
        "figures": [
            str(output_dir / "absolute_regret_by_target_decay_all_methods.png"),
            str(output_dir / "absolute_regret_by_target_decay_safe_methods.png"),
            str(output_dir / "absolute_regret_by_target_powerlaw_exponent_all_methods.png"),
            str(output_dir / "absolute_regret_by_target_powerlaw_exponent_safe_methods.png"),
        ],
    }


__all__ = [
    "TargetSpectrumSweepConfig",
    "run_target_spectrum_sweep",
]
