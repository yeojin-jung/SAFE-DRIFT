#!/usr/bin/env python3
"""Render comparison plots from aggregated SAFE-DRIFT result summaries."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


EXPERIMENT_TITLES = {
    "setting2_medqa_medmcqa_truthfulqa_bbq": "Setting 2: MedMCQA candidates -> MedQA target",
    "setting3_empathetic_esconv_gsm8k": "Setting 3: Empathetic Dialogues candidates -> ESConv target",
}

METRIC_LABELS = {
    "target_primary": "Target score",
    "reference_primary": "Reference / drift score",
    "target_delta_primary": "Target delta vs base",
    "reference_delta_primary": "Reference / drift delta vs base",
    "target_medqa_accuracy": "MedQA accuracy",
    "ood_medical_reference_primary_score": "TruthfulQA/BBQ primary score",
    "ood_bbq_accuracy": "BBQ accuracy",
    "ood_truthfulqa_token_f1": "TruthfulQA token F1",
    "target_esconv_token_f1": "ESConv token F1",
    "target_esconv_rouge_l_f1": "ESConv ROUGE-L F1",
    "ood_gsm8k_accuracy": "GSM8K accuracy",
    "reference_validation_gsm8k_accuracy": "GSM8K validation accuracy",
    "final_validation_loss": "Validation loss",
    "final_test_loss": "Target test loss",
    "candidate_validation_loss": "Candidate validation loss",
    "reference_validation_loss": "Reference validation loss",
    "final_model_weight_delta_l2_norm": "Full weight delta L2 norm",
    "final_trainable_parameter_delta_l2_norm": "Trainable delta L2 norm",
    "entanglement_rho_E_selected_mean": "Entanglement rho_E selected mean",
    "entanglement_rho_P_selected_mean": "Entanglement rho_P selected mean",
    "entanglement_reference_load_selected_mean": "Reference load selected mean",
    "entanglement_target_alignment_selected_ideal_fraction": "Target-alignment ideal fraction",
    "entanglement_damped_alignment_selected_ideal_fraction": "Damped-alignment ideal fraction",
}

METRICS_BY_EXPERIMENT = {
    "setting2_medqa_medmcqa_truthfulqa_bbq": [
        "target_primary",
        "reference_primary",
        "target_delta_primary",
        "reference_delta_primary",
        "ood_bbq_accuracy",
        "ood_truthfulqa_token_f1",
        "final_validation_loss",
        "final_test_loss",
        "candidate_validation_loss",
        "reference_validation_loss",
        "final_model_weight_delta_l2_norm",
        "final_trainable_parameter_delta_l2_norm",
        "entanglement_rho_E_selected_mean",
        "entanglement_rho_P_selected_mean",
        "entanglement_reference_load_selected_mean",
        "entanglement_target_alignment_selected_ideal_fraction",
        "entanglement_damped_alignment_selected_ideal_fraction",
    ],
    "setting3_empathetic_esconv_gsm8k": [
        "target_primary",
        "target_esconv_rouge_l_f1",
        "reference_primary",
        "reference_validation_gsm8k_accuracy",
        "target_delta_primary",
        "reference_delta_primary",
        "final_validation_loss",
        "final_test_loss",
        "candidate_validation_loss",
        "reference_validation_loss",
        "final_model_weight_delta_l2_norm",
        "final_trainable_parameter_delta_l2_norm",
        "entanglement_rho_E_selected_mean",
        "entanglement_rho_P_selected_mean",
        "entanglement_reference_load_selected_mean",
        "entanglement_target_alignment_selected_ideal_fraction",
        "entanglement_damped_alignment_selected_ideal_fraction",
    ],
}

CONSTRAINT_COLORS = {
    "feasible": "#2ca02c",
    "inactive": "#d62728",
    "violated": "#f2c744",
    "": "#777777",
}

OPT_MARKERS = {
    "adam": "o",
    "sgd": "s",
    "": "D",
}

BASELINE_STYLES = {
    "Base model": ("#222222", "-"),
    "random": ("#7f7f7f", "--"),
    "less": ("#9467bd", "--"),
    "dsir": ("#8c564b", "--"),
    "prismatic": ("#17becf", "--"),
}


def slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"/", " ", "-", "_"}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "plot"


def fmt_epsilon(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value:g}"


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def method_palette(methods: list[str]) -> dict[str, str]:
    preferred = {
        "SAFE reference / adam": "#1f77b4",
        "SAFE reference / sgd": "#8ecae6",
        "SAFE euclidean / adam": "#e66101",
        "SAFE euclidean / sgd": "#fdb863",
        "SAFE rank-solver / adam": "#1b9e77",
        "SAFE rank-solver / sgd": "#80cdc1",
    }
    palette = {}
    fallback = sns.color_palette("tab10", n_colors=max(10, len(methods)))
    fallback_index = 0
    for method in methods:
        if method in preferred:
            palette[method] = preferred[method]
        else:
            palette[method] = fallback[fallback_index % len(fallback)]
            fallback_index += 1
    return palette


def safe_frame(df: pd.DataFrame, experiment: str, include_rank: bool) -> pd.DataFrame:
    sub = df.loc[
        (df["experiment"] == experiment)
        & (df["complete"] == True)
        & (df["kind"] == "safe")
    ].copy()
    if not include_rank:
        variant = sub["variant"].fillna("").astype(str)
        sub = sub.loc[
            ~sub["geometry"].eq("rank")
            & ~variant.str.contains("random_sketch", case=False, regex=False)
        ].copy()
    return sub


def baseline_frame(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    return df.loc[
        (df["experiment"] == experiment)
        & (df["complete"] == True)
        & (df["kind"].isin(["base", "baseline"]))
    ].copy()


def plot_rho_sweep(
    df: pd.DataFrame,
    experiment: str,
    metric: str,
    out_dir: Path,
    *,
    include_rank: bool,
) -> None:
    sub = safe_frame(df, experiment, include_rank=include_rank)
    sub = sub.loc[sub[metric].notna() & sub["rho"].notna() & sub["epsilon"].notna()].copy()
    if sub.empty:
        return
    baselines = baseline_frame(df, experiment)
    baselines = baselines.loc[baselines[metric].notna()].copy()
    epsilons = sorted(sub["epsilon"].dropna().unique())
    ncols = min(3, len(epsilons))
    nrows = int(math.ceil(len(epsilons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), sharey=False)
    axes_arr = np.asarray(axes).reshape(-1)
    methods = sorted(sub["method_label"].dropna().unique())
    palette = method_palette(methods)

    for axis, epsilon in zip(axes_arr, epsilons):
        part = sub.loc[sub["epsilon"] == epsilon].copy()
        for method in methods:
            method_frame = part.loc[part["method_label"] == method].sort_values("rho")
            if method_frame.empty:
                continue
            color = palette[method]
            opt = str(method_frame["optimizer"].dropna().iloc[0]) if method_frame["optimizer"].notna().any() else ""
            marker = OPT_MARKERS.get(opt, "D")
            axis.plot(
                method_frame["rho"],
                method_frame[metric],
                color=color,
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
            )
            for _, row in method_frame.iterrows():
                status = str(row.get("constraint_status_for_plot") or "")
                axis.scatter(
                    row["rho"],
                    row[metric],
                    color=color,
                    marker=marker,
                    s=54,
                    linewidth=1.5,
                    edgecolor=CONSTRAINT_COLORS.get(status, "#777777"),
                    zorder=3,
                )
        for _, row in baselines.iterrows():
            label = str(row["method_label"])
            color, style = BASELINE_STYLES.get(label, ("#999999", "--"))
            axis.axhline(row[metric], color=color, linestyle=style, linewidth=1.0, alpha=0.65)
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.18)
        axis.set_title(f"epsilon = {fmt_epsilon(epsilon)}")
        axis.set_xlabel("rho")
        axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    for axis in axes_arr[len(epsilons):]:
        axis.axis("off")

    method_handles = []
    for method in methods:
        optimizer = ""
        if " / adam" in method:
            optimizer = "adam"
        elif " / sgd" in method:
            optimizer = "sgd"
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=palette[method],
                marker=OPT_MARKERS.get(optimizer, "D"),
                linewidth=1.5,
                label=method,
            )
        )
    constraint_handles = [
        Line2D([0], [0], color="white", marker="o", markerfacecolor="white",
               markeredgecolor=color, markeredgewidth=1.8, linewidth=0, label=status or "unknown")
        for status, color in CONSTRAINT_COLORS.items()
        if status
    ]
    baseline_handles = [
        Line2D([0], [0], color=color, linestyle=style, linewidth=1.2, label=label)
        for label, (color, style) in BASELINE_STYLES.items()
        if label in set(baselines["method_label"].tolist())
    ]
    fig.suptitle(
        f"{EXPERIMENT_TITLES.get(experiment, experiment)}\n{METRIC_LABELS.get(metric, metric)}",
        y=1.02,
        fontsize=13,
    )
    fig.legend(
        handles=method_handles + baseline_handles + constraint_handles,
        loc="lower center",
        ncol=min(4, len(method_handles) + len(baseline_handles) + len(constraint_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    rank_tag = "all_safe" if include_rank else "ref_euc"
    save(fig, out_dir / f"{slugify(experiment)}__rho__{rank_tag}__{slugify(metric)}")


def plot_target_drift_grid(
    df: pd.DataFrame,
    experiment: str,
    out_dir: Path,
    *,
    include_rank: bool,
) -> None:
    metrics = ["target_primary", "reference_primary"]
    if any(metric not in df.columns for metric in metrics):
        return
    sub = safe_frame(df, experiment, include_rank=include_rank)
    sub = sub.loc[
        sub["rho"].notna()
        & sub["epsilon"].notna()
        & (sub[metrics[0]].notna() | sub[metrics[1]].notna())
    ].copy()
    if sub.empty:
        return
    baselines = baseline_frame(df, experiment)
    epsilons = sorted(sub["epsilon"].dropna().unique())
    methods = sorted(sub["method_label"].dropna().unique())
    palette = method_palette(methods)
    fig, axes = plt.subplots(
        len(epsilons),
        2,
        figsize=(11.5, max(3.4, 2.9 * len(epsilons))),
        squeeze=False,
        sharex=False,
    )

    for row_index, epsilon in enumerate(epsilons):
        part = sub.loc[sub["epsilon"] == epsilon].copy()
        for col_index, metric in enumerate(metrics):
            axis = axes[row_index, col_index]
            metric_part = part.loc[part[metric].notna()].copy()
            if metric_part.empty:
                axis.axis("off")
                continue
            for method in methods:
                method_frame = metric_part.loc[metric_part["method_label"] == method].sort_values("rho")
                if method_frame.empty:
                    continue
                color = palette[method]
                opt = (
                    str(method_frame["optimizer"].dropna().iloc[0])
                    if method_frame["optimizer"].notna().any()
                    else ""
                )
                marker = OPT_MARKERS.get(opt, "D")
                axis.plot(
                    method_frame["rho"],
                    method_frame[metric],
                    color=color,
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=2,
                )
                for _, row in method_frame.iterrows():
                    status = str(row.get("constraint_status_for_plot") or "")
                    axis.scatter(
                        row["rho"],
                        row[metric],
                        color=color,
                        marker=marker,
                        s=54,
                        linewidth=1.5,
                        edgecolor=CONSTRAINT_COLORS.get(status, "#777777"),
                        zorder=3,
                    )
            metric_baselines = baselines.loc[baselines[metric].notna()].copy()
            for _, row in metric_baselines.iterrows():
                label = str(row["method_label"])
                color, style = BASELINE_STYLES.get(label, ("#999999", "--"))
                axis.axhline(row[metric], color=color, linestyle=style, linewidth=1.0, alpha=0.65)
            axis.set_xscale("log")
            axis.grid(True, which="both", alpha=0.18)
            axis.set_xlabel("rho")
            axis.set_ylabel(METRIC_LABELS.get(metric, metric))
            title = "Target" if col_index == 0 else "Drift / reference"
            axis.set_title(f"{title}\nepsilon = {fmt_epsilon(epsilon)}")

    method_handles = []
    for method in methods:
        optimizer = ""
        if " / adam" in method:
            optimizer = "adam"
        elif " / sgd" in method:
            optimizer = "sgd"
        method_handles.append(
            Line2D(
                [0],
                [0],
                color=palette[method],
                marker=OPT_MARKERS.get(optimizer, "D"),
                linewidth=1.5,
                label=method,
            )
        )
    baseline_labels = set(baselines["method_label"].tolist())
    baseline_handles = [
        Line2D([0], [0], color=color, linestyle=style, linewidth=1.2, label=label)
        for label, (color, style) in BASELINE_STYLES.items()
        if label in baseline_labels
    ]
    constraint_handles = [
        Line2D(
            [0],
            [0],
            color="white",
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.8,
            linewidth=0,
            label=status or "unknown",
        )
        for status, color in CONSTRAINT_COLORS.items()
        if status
    ]
    rank_tag = "all_safe" if include_rank else "ref_euc"
    fig.suptitle(
        f"{EXPERIMENT_TITLES.get(experiment, experiment)}\nTarget performance and corresponding drift",
        y=1.01,
        fontsize=13,
    )
    fig.legend(
        handles=method_handles + baseline_handles + constraint_handles,
        loc="lower center",
        ncol=min(4, len(method_handles) + len(baseline_handles) + len(constraint_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    save(fig, out_dir / f"{slugify(experiment)}__target_vs_drift__{rank_tag}")


def plot_method_distribution(df: pd.DataFrame, experiment: str, metric: str, out_dir: Path) -> None:
    sub = df.loc[
        (df["experiment"] == experiment)
        & (df["complete"] == True)
        & (df[metric].notna())
    ].copy()
    if sub.empty:
        return
    order = [
        "Base model",
        "random",
        "less",
        "dsir",
        "prismatic",
        "SAFE reference / adam",
        "SAFE reference / sgd",
        "SAFE euclidean / adam",
        "SAFE euclidean / sgd",
        "SAFE rank-solver / adam",
        "SAFE rank-solver / sgd",
    ]
    order = [x for x in order if x in set(sub["method_label"].tolist())]
    fig, axis = plt.subplots(figsize=(max(9, 0.8 * len(order)), 4.8))
    sns.boxplot(
        data=sub,
        x="method_label",
        y=metric,
        order=order,
        color="#d9d9d9",
        width=0.55,
        fliersize=0,
        ax=axis,
    )
    sns.stripplot(
        data=sub,
        x="method_label",
        y=metric,
        order=order,
        hue="optimizer",
        dodge=False,
        jitter=0.22,
        size=4.2,
        alpha=0.75,
        palette={"adam": "#1f77b4", "sgd": "#ff7f0e", "": "#555555"},
        ax=axis,
    )
    axis.set_title(f"{EXPERIMENT_TITLES.get(experiment, experiment)}\n{METRIC_LABELS.get(metric, metric)}")
    axis.set_xlabel("")
    axis.set_ylabel(METRIC_LABELS.get(metric, metric))
    axis.grid(axis="y", alpha=0.2)
    axis.tick_params(axis="x", labelrotation=35)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, title="optimizer", frameon=False, loc="best")
    fig.tight_layout()
    save(fig, out_dir / f"{slugify(experiment)}__method_distribution__{slugify(metric)}")


def plot_constraint_counts(df: pd.DataFrame, experiment: str, out_dir: Path) -> None:
    sub = safe_frame(df, experiment, include_rank=True)
    sub = sub.loc[sub["constraint_status_for_plot"].notna()].copy()
    if sub.empty:
        return
    counts = (
        sub.groupby(["method_label", "constraint_status_for_plot"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    order = sorted(counts["method_label"].unique())
    fig, axis = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(order))
    x = np.arange(len(order))
    for status in ["feasible", "inactive", "violated", ""]:
        vals = []
        for method in order:
            match = counts.loc[
                (counts["method_label"] == method)
                & (counts["constraint_status_for_plot"].fillna("") == status),
                "count",
            ]
            vals.append(float(match.iloc[0]) if not match.empty else 0.0)
        axis.bar(
            x,
            vals,
            bottom=bottom,
            color=CONSTRAINT_COLORS.get(status, "#777777"),
            label=status or "unknown",
        )
        bottom += np.asarray(vals)
    axis.set_xticks(x)
    axis.set_xticklabels(order, rotation=35, ha="right")
    axis.set_ylabel("completed SAFE runs")
    axis.set_title(f"{EXPERIMENT_TITLES.get(experiment, experiment)}\nSAFE constraint cases")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir / f"{slugify(experiment)}__constraint_case_counts")


def plot_missing_counts(df: pd.DataFrame, out_dir: Path) -> None:
    missing = df.loc[df["complete"] == False].copy()
    if missing.empty:
        return
    missing["label"] = missing["experiment"].map(EXPERIMENT_TITLES).fillna(missing["experiment"])
    counts = missing.groupby(["label", "source"]).size().reset_index(name="missing")
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    sns.barplot(data=counts, x="label", y="missing", hue="source", ax=axis)
    axis.set_xlabel("")
    axis.set_ylabel("missing / unsubmitted runs")
    axis.set_title("Incomplete comparison runs")
    axis.tick_params(axis="x", labelrotation=20)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir / "incomplete_run_counts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    df = pd.read_csv(args.input)
    bool_map = {"True": True, "False": False, True: True, False: False}
    df["complete"] = df["complete"].map(bool_map).fillna(False)
    for col in [
        "rho",
        "epsilon",
        *METRIC_LABELS.keys(),
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["method_label", "optimizer", "geometry", "constraint_status_for_plot", "kind"]:
        if col in df:
            df[col] = df[col].fillna("")

    for experiment, metrics in METRICS_BY_EXPERIMENT.items():
        plot_target_drift_grid(df, experiment, args.output_dir, include_rank=False)
        plot_target_drift_grid(df, experiment, args.output_dir, include_rank=True)
        for metric in metrics:
            if metric not in df.columns or df[metric].dropna().empty:
                continue
            plot_rho_sweep(df, experiment, metric, args.output_dir, include_rank=False)
            plot_rho_sweep(df, experiment, metric, args.output_dir, include_rank=True)
            plot_method_distribution(df, experiment, metric, args.output_dir)
        plot_constraint_counts(df, experiment, args.output_dir)
    plot_missing_counts(df, args.output_dir)


if __name__ == "__main__":
    main()
