#!/usr/bin/env python3
"""Plot target/drift rho sweeps averaged over completed seeds."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_TITLES = {
    "setting2_medqa_medmcqa_truthfulqa_bbq": "Setting 2: MedMCQA candidates -> MedQA target",
    "setting3_empathetic_esconv_gsm8k": "Setting 3: Empathetic Dialogues candidates -> ESConv target",
}

METRIC_TITLES = {
    "target_primary": "Target",
    "reference_primary": "Reference score",
    "target_delta_primary": "Target delta vs base",
    "reference_delta_primary": "Reference delta vs base",
    "absolute_reference_delta_primary": "Absolute reference drift |reference - base|",
}

PRIMARY_METRICS = ["target_primary", "reference_primary"]
DELTA_METRICS = ["target_delta_primary", "reference_delta_primary"]
ABSOLUTE_DRIFT_METRIC = "absolute_reference_delta_primary"
ALL_METRICS = PRIMARY_METRICS + DELTA_METRICS + [ABSOLUTE_DRIFT_METRIC]

ACCURACY_DELTA_SPECS = {
    "setting2_medqa_medmcqa_truthfulqa_bbq": {
        "source": "target_delta_primary",
        "metric": "medqa_accuracy_delta_pp",
        "title": "MedQA accuracy delta vs base (pp)",
    },
    "setting3_empathetic_esconv_gsm8k": {
        "source": "reference_delta_primary",
        "metric": "gsm8k_accuracy_delta_pp",
        "title": "GSM8K accuracy delta vs base (pp)",
    },
}

METHOD_ORDER = [
    "SAFE reference / adam",
    "SAFE reference / sgd",
    "SAFE euclidean / adam",
    "SAFE euclidean / sgd",
]

METHOD_COLORS = {
    "SAFE reference / adam": "#1f77b4",
    "SAFE reference / sgd": "#8ecae6",
    "SAFE euclidean / adam": "#e66101",
    "SAFE euclidean / sgd": "#fdb863",
}

METHOD_MARKERS = {
    "adam": "circle",
    "sgd": "square",
}

BASELINE_ORDER = ["Base model", "random", "less", "dsir", "prismatic"]

BASELINE_STYLES = {
    "Base model": ("#222222", (10, 6)),
    "random": ("#7f7f7f", (7, 5)),
    "less": ("#9467bd", (3, 4)),
    "dsir": ("#8c564b", (7, 5)),
    "prismatic": ("#17becf", (7, 5)),
}

CONSTRAINT_COLORS = {
    "feasible": "#2ca02c",
    "active": "#2ca02c",
    "inactive": "#d62728",
    "violated": "#f2c744",
    "mixed": "#777777",
    "unknown": "#999999",
}

CONSTRAINT_LABELS = {
    "feasible": "active / feasible",
    "inactive": "inactive",
    "violated": "violated",
    "mixed": "mixed across seeds",
    "unknown": "unknown",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT = load_font(16)
FONT_SMALL = load_font(13)
FONT_TINY = load_font(11)
FONT_TITLE = load_font(26, bold=True)
FONT_SUBTITLE = load_font(18, bold=True)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def slugify(value: str) -> str:
    cleaned = []
    for char in str(value).lower():
        if char.isalnum():
            cleaned.append(char)
        elif char in {"/", " ", "-", "_", "."}:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "plot"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return ""
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return f"{value:g}"


def parse_seed(row: pd.Series) -> float:
    if "seed" in row and pd.notna(row["seed"]):
        try:
            return int(row["seed"])
        except (TypeError, ValueError):
            pass
    for col in ["name", "manifest_path", "summary_path"]:
        value = str(row.get(col, ""))
        match = re.search(r"seed(\d+)", value)
        if match:
            return int(match.group(1))
    return np.nan


def combine_constraint_status(values: pd.Series) -> str:
    statuses = {
        str(value).strip().lower()
        for value in values.dropna().tolist()
        if str(value).strip()
    }
    if not statuses:
        return "unknown"
    if "violated" in statuses:
        return "violated"
    if "inactive" in statuses:
        return "inactive"
    if statuses.issubset({"feasible", "active"}):
        return "feasible"
    return "mixed"


def prepare_safe_rows(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bool_map = {"True": True, "False": False, True: True, False: False}
    data["complete"] = data["complete"].map(bool_map).fillna(False)
    data["seed"] = data.apply(parse_seed, axis=1)
    for col in ["rho", "epsilon", *ALL_METRICS]:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[ABSOLUTE_DRIFT_METRIC] = data["reference_delta_primary"].abs()
    for col in ["kind", "variant", "geometry", "optimizer", "method_label", "constraint_status_for_plot"]:
        data[col] = data[col].fillna("").astype(str)

    safe = data.loc[
        (data["complete"] == True)
        & (data["kind"] == "safe")
        & data["seed"].notna()
        & data["rho"].notna()
        & data["epsilon"].notna()
        & (data["target_primary"].notna() | data["reference_primary"].notna())
    ].copy()
    variant = safe["variant"].fillna("").astype(str)
    safe = safe.loc[
        ~safe["geometry"].eq("rank")
        & ~variant.str.contains("random_sketch", case=False, regex=False)
    ].copy()

    keys = [
        "experiment",
        "seed",
        "method_label",
        "geometry",
        "optimizer",
        "epsilon",
        "rho",
    ]
    rows = []
    for key_values, part in safe.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values))
        for metric in ALL_METRICS:
            row[metric] = part[metric].mean()
        row["constraint_status_for_plot"] = combine_constraint_status(part["constraint_status_for_plot"])
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bool_map = {"True": True, "False": False, True: True, False: False}
    data["complete"] = data["complete"].map(bool_map).fillna(False)
    data["seed"] = data.apply(parse_seed, axis=1)
    for col in ALL_METRICS:
        if col not in data.columns:
            data[col] = np.nan
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[ABSOLUTE_DRIFT_METRIC] = data["reference_delta_primary"].abs()
    data["kind"] = data["kind"].fillna("").astype(str)
    data["method_label"] = data["method_label"].fillna("").astype(str)
    return data.loc[
        (data["complete"] == True)
        & data["kind"].isin(["base", "baseline"])
        & data["method_label"].isin(BASELINE_ORDER)
        & data["seed"].notna()
        & (data["target_primary"].notna() | data["reference_primary"].notna())
    ].copy()


def averaged_summary(seed_rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "epsilon", "rho", "method_label", "geometry", "optimizer"]
    grouped = seed_rows.groupby(keys, dropna=False)
    rows = []
    for keys_values, part in grouped:
        row = dict(zip(keys, keys_values))
        seeds = sorted(int(x) for x in part["seed"].dropna().unique())
        row["seeds"] = ",".join(str(x) for x in seeds)
        row["n_seeds"] = len(seeds)
        for metric in ALL_METRICS:
            vals = part[metric].dropna()
            row[f"{metric}_mean"] = vals.mean() if not vals.empty else np.nan
            row[f"{metric}_std"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{metric}_n"] = len(vals)
        row["constraint_status_for_plot"] = combine_constraint_status(part["constraint_status_for_plot"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["experiment", "epsilon", "method_label", "rho"])


def baseline_summary(base_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (experiment, method_label), part in base_rows.groupby(["experiment", "method_label"], dropna=False):
        row = {"experiment": experiment, "method_label": method_label}
        seeds = sorted(int(x) for x in part["seed"].dropna().unique())
        row["seeds"] = ",".join(str(x) for x in seeds)
        row["n_seeds"] = len(seeds)
        for metric in ALL_METRICS:
            vals = part[metric].dropna()
            row[f"{metric}_mean"] = vals.mean() if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_accuracy_delta_metrics(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
) -> None:
    for experiment, spec in ACCURACY_DELTA_SPECS.items():
        source = spec["source"]
        metric = spec["metric"]
        METRIC_TITLES[metric] = spec["title"]
        summary_mask = summary["experiment"] == experiment
        baseline_mask = baseline["experiment"] == experiment
        for suffix in ["mean", "std"]:
            source_col = f"{source}_{suffix}"
            metric_col = f"{metric}_{suffix}"
            summary[metric_col] = np.nan
            summary.loc[summary_mask, metric_col] = (
                100.0 * summary.loc[summary_mask, source_col]
            )
        summary[f"{metric}_n"] = np.nan
        summary.loc[summary_mask, f"{metric}_n"] = summary.loc[
            summary_mask, f"{source}_n"
        ]
        baseline[f"{metric}_mean"] = np.nan
        baseline.loc[baseline_mask, f"{metric}_mean"] = (
            100.0 * baseline.loc[baseline_mask, f"{source}_mean"]
        )


def value_domain(summary: pd.DataFrame, baseline: pd.DataFrame, experiment: str, metric: str) -> tuple[float, float]:
    vals = summary.loc[summary["experiment"] == experiment, f"{metric}_mean"].dropna().tolist()
    base_vals = baseline.loc[baseline["experiment"] == experiment, f"{metric}_mean"].dropna().tolist()
    vals.extend(base_vals)
    if not vals:
        return 0.0, 1.0
    ymin = min(vals)
    ymax = max(vals)
    if math.isclose(ymin, ymax):
        pad = max(0.05, abs(ymin) * 0.1)
    else:
        pad = 0.08 * (ymax - ymin)
    if metric == ABSOLUTE_DRIFT_METRIC:
        return 0.0, ymax + pad
    return ymin - pad, ymax + pad


def draw_pattern_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    fill: tuple[int, int, int],
    *,
    pattern: tuple[int, int] | None = None,
    width: int = 2,
) -> None:
    x0, y0, x1, y1 = xy
    if pattern is None:
        draw.line(xy, fill=fill, width=width)
        return
    dash, gap = pattern
    dx = x1 - x0
    dy = y1 - y0
    length = max(1.0, math.hypot(dx, dy))
    steps = int(length // (dash + gap)) + 1
    for i in range(steps):
        start = i * (dash + gap)
        end = min(start + dash, length)
        if start >= length:
            break
        sx = x0 + dx * (start / length)
        sy = y0 + dy * (start / length)
        ex = x0 + dx * (end / length)
        ey = y0 + dy * (end / length)
        draw.line((sx, sy, ex, ey), fill=fill, width=width)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    color: tuple[int, int, int],
    marker: str,
    *,
    outline: tuple[int, int, int] = (35, 35, 35),
    outline_width: int = 2,
    radius: int = 6,
) -> None:
    r = radius
    if marker == "square":
        draw.rectangle((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=outline_width)
    else:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=outline_width)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    experiment: str,
    epsilon: float,
    metric: str,
    y_domain: tuple[float, float],
    rho_values: list[float],
    method_order: list[str],
) -> None:
    left, top, right, bottom = rect
    width = right - left
    height = bottom - top
    plot_left = left + 68
    plot_top = top + 34
    plot_right = right - 18
    plot_bottom = bottom - 48
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    bg = (255, 255, 255)
    grid = (225, 229, 234)
    axis = (45, 45, 45)
    text = (35, 35, 35)
    draw.rectangle((left, top, right, bottom), fill=bg)
    draw.text((left + 8, top + 4), f"{METRIC_TITLES[metric]} | epsilon = {fmt_num(epsilon)}", font=FONT_SUBTITLE, fill=text)

    ymin, ymax = y_domain
    log_min = math.log10(min(rho_values))
    log_max = math.log10(max(rho_values))

    def x_pos(rho: float) -> float:
        if math.isclose(log_min, log_max):
            return (plot_left + plot_right) / 2
        return plot_left + (math.log10(rho) - log_min) / (log_max - log_min) * plot_width

    def y_pos(value: float) -> float:
        if math.isclose(ymin, ymax):
            return (plot_top + plot_bottom) / 2
        return plot_bottom - (value - ymin) / (ymax - ymin) * plot_height

    for tick in np.linspace(ymin, ymax, 5):
        y = y_pos(float(tick))
        draw.line((plot_left, y, plot_right, y), fill=grid, width=1)
        label = f"{tick:.3g}"
        tw, th = text_size(draw, label, FONT_TINY)
        draw.text((plot_left - tw - 8, y - th / 2), label, font=FONT_TINY, fill=(90, 90, 90))

    for rho in rho_values:
        x = x_pos(rho)
        draw.line((x, plot_top, x, plot_bottom), fill=(238, 240, 243), width=1)
        label = fmt_num(rho)
        tw, _ = text_size(draw, label, FONT_TINY)
        draw.text((x - tw / 2, plot_bottom + 8), label, font=FONT_TINY, fill=(90, 90, 90))

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis, width=2)
    xlabel = "rho"
    tw, _ = text_size(draw, xlabel, FONT_SMALL)
    draw.text((plot_left + (plot_width - tw) / 2, bottom - 18), xlabel, font=FONT_SMALL, fill=text)

    baseline_part = baseline.loc[
        (baseline["experiment"] == experiment)
        & baseline[f"{metric}_mean"].notna()
    ].copy()
    for method in BASELINE_ORDER:
        rows = baseline_part.loc[baseline_part["method_label"] == method]
        if rows.empty:
            continue
        color_hex, pattern = BASELINE_STYLES.get(method, ("#888888", (7, 5)))
        y = y_pos(float(rows.iloc[0][f"{metric}_mean"]))
        draw_pattern_line(
            draw,
            (plot_left, y, plot_right, y),
            fill=hex_to_rgb(color_hex),
            pattern=pattern,
            width=2,
        )

    part = data.loc[
        (data["experiment"] == experiment)
        & np.isclose(data["epsilon"].astype(float), float(epsilon))
        & data[f"{metric}_mean"].notna()
    ].copy()
    for method in method_order:
        method_frame = part.loc[part["method_label"] == method].sort_values("rho")
        if method_frame.empty:
            continue
        color = hex_to_rgb(METHOD_COLORS[method])
        points = [(x_pos(float(row["rho"])), y_pos(float(row[f"{metric}_mean"]))) for _, row in method_frame.iterrows()]
        if len(points) > 1:
            draw.line(points, fill=color, width=3, joint="curve")
        marker = METHOD_MARKERS.get(str(method_frame["optimizer"].iloc[0]), "circle")
        for (x, y), (_, row) in zip(points, method_frame.iterrows()):
            std = row.get(f"{metric}_std")
            if pd.notna(std):
                y_low = y_pos(float(row[f"{metric}_mean"] - std))
                y_high = y_pos(float(row[f"{metric}_mean"] + std))
                draw.line((x, y_low, x, y_high), fill=color, width=1)
                draw.line((x - 4, y_low, x + 4, y_low), fill=color, width=1)
                draw.line((x - 4, y_high, x + 4, y_high), fill=color, width=1)
            status = str(row.get("constraint_status_for_plot") or "unknown")
            outline = hex_to_rgb(CONSTRAINT_COLORS.get(status, CONSTRAINT_COLORS["unknown"]))
            draw_marker(draw, x, y, color, marker, outline=outline, outline_width=3)


def draw_legend(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    max_width: int,
    method_order: list[str],
) -> int:
    current_x = x
    current_y = y
    for method in method_order:
        color = hex_to_rgb(METHOD_COLORS[method])
        draw.line((current_x, current_y + 8, current_x + 28, current_y + 8), fill=color, width=3)
        marker = "square" if method.endswith("/ sgd") else "circle"
        draw_marker(draw, current_x + 14, current_y + 8, color, marker, outline=(35, 35, 35), outline_width=1, radius=5)
        label = method
        draw.text((current_x + 36, current_y), label, font=FONT_SMALL, fill=(40, 40, 40))
        tw, _ = text_size(draw, label, FONT_SMALL)
        current_x += 36 + tw + 26

    current_x = x
    current_y += 24
    for method in BASELINE_ORDER:
        color_hex, pattern = BASELINE_STYLES[method]
        label = method
        tw, _ = text_size(draw, label, FONT_SMALL)
        item_width = 36 + tw + 26
        if current_x > x and current_x + item_width > x + max_width:
            current_x = x
            current_y += 22
        draw_pattern_line(
            draw,
            (current_x, current_y + 8, current_x + 30, current_y + 8),
            fill=hex_to_rgb(color_hex),
            pattern=pattern,
            width=2,
        )
        draw.text((current_x + 38, current_y), label, font=FONT_SMALL, fill=(40, 40, 40))
        current_x += item_width

    current_x = x
    current_y += 26
    for status in ["feasible", "inactive", "violated", "mixed", "unknown"]:
        label = CONSTRAINT_LABELS[status]
        outline = hex_to_rgb(CONSTRAINT_COLORS[status])
        tw, _ = text_size(draw, label, FONT_SMALL)
        item_width = 24 + tw + 26
        if current_x > x and current_x + item_width > x + max_width:
            current_x = x
            current_y += 22
        draw_marker(
            draw,
            current_x + 8,
            current_y + 8,
            (255, 255, 255),
            "circle",
            outline=outline,
            outline_width=3,
            radius=6,
        )
        draw.text((current_x + 22, current_y), label, font=FONT_SMALL, fill=(40, 40, 40))
        current_x += item_width
    return current_y + 24


def render_grid(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    experiment: str,
    out_path: Path,
    *,
    epsilons: list[float] | None = None,
    optimizer: str | None = None,
    metrics: list[str] | None = None,
    subtitle: str | None = None,
) -> None:
    if metrics is None:
        metrics = PRIMARY_METRICS
    if len(metrics) not in {1, 2}:
        raise ValueError("render_grid expects one or two metrics")
    exp_summary = summary.loc[summary["experiment"] == experiment].copy()
    if optimizer is not None:
        exp_summary = exp_summary.loc[exp_summary["optimizer"] == optimizer].copy()
    if exp_summary.empty:
        return
    if epsilons is None:
        epsilons = sorted(float(x) for x in exp_summary["epsilon"].dropna().unique())
    if not epsilons:
        return
    rho_values = sorted(float(x) for x in exp_summary["rho"].dropna().unique())
    y_domains = {
        metric: value_domain(summary, baseline, experiment, metric)
        for metric in metrics
    }
    present_methods = set(exp_summary["method_label"].dropna())
    method_order = [method for method in METHOD_ORDER if method in present_methods]
    panel_w = 1020 if len(metrics) == 1 else 620
    panel_h = 275
    margin_x = 46
    title_h = 88
    row_gap = 24
    col_gap = 26
    legend_h = 160 if len(metrics) == 1 else 118
    width = margin_x * 2 + panel_w * len(metrics) + col_gap * (len(metrics) - 1)
    height = title_h + len(epsilons) * panel_h + max(0, len(epsilons) - 1) * row_gap + legend_h
    image = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(image)
    title = EXPERIMENT_TITLES.get(experiment, experiment)
    if optimizer is not None:
        title += f" | {optimizer.upper()} optimizer"
    draw.text((margin_x, 24), title, font=FONT_TITLE, fill=(25, 25, 25))
    if subtitle is None:
        if metrics == DELTA_METRICS:
            subtitle = "Delta from the matching base model; rho is the x-axis, rows are epsilon values"
        else:
            subtitle = "Mean over completed seeds; rho is the x-axis, rows are epsilon values, columns are target and reference"
    draw.text((margin_x, 56), subtitle, font=FONT_SMALL, fill=(85, 85, 85))
    y = title_h
    for epsilon in epsilons:
        for col_index, metric in enumerate(metrics):
            x = margin_x + col_index * (panel_w + col_gap)
            draw_panel(
                draw,
                (x, y, x + panel_w, y + panel_h),
                summary,
                baseline,
                experiment,
                epsilon,
                metric,
                y_domains[metric],
                rho_values,
                method_order,
            )
        y += panel_h + row_gap
    legend_y = height - legend_h + 18
    note_y = draw_legend(draw, margin_x, legend_y, width - 2 * margin_x, method_order)
    note = "Marker outline: active/feasible, inactive, violated, mixed, or unknown constraint status. Error bars show +/-1 SD when more than one seed is available."
    draw.text((margin_x, note_y + 2), note, font=FONT_TINY, fill=(100, 100, 100))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        action="append",
        choices=sorted(EXPERIMENT_TITLES),
        help="Experiment(s) to plot. Defaults to all known experiments in the input.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    safe = prepare_safe_rows(df)
    base = prepare_baseline_rows(df)
    summary = averaged_summary(safe)
    baseline = baseline_summary(base)
    add_accuracy_delta_metrics(summary, baseline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "average_seed_target_drift_by_rho.csv"
    summary.to_csv(summary_path, index=False)
    baseline.to_csv(args.output_dir / "average_seed_baselines.csv", index=False)

    experiments = args.experiment or [
        exp for exp in EXPERIMENT_TITLES if exp in set(summary["experiment"].dropna())
    ]
    for experiment in experiments:
        exp_summary = summary.loc[summary["experiment"] == experiment]
        if exp_summary.empty:
            continue
        render_grid(
            summary,
            baseline,
            experiment,
            args.output_dir / f"{slugify(experiment)}__avg_seed_target_drift_by_epsilon__ref_euc.png",
        )
        for epsilon in sorted(float(x) for x in exp_summary["epsilon"].dropna().unique()):
            render_grid(
                summary,
                baseline,
                experiment,
                args.output_dir
                / f"{slugify(experiment)}__avg_seed_target_drift__eps_{slugify(fmt_num(epsilon))}__ref_euc.png",
                epsilons=[epsilon],
            )
        optimizer_dir = args.output_dir / "by_optimizer"
        for optimizer in sorted(str(x) for x in exp_summary["optimizer"].dropna().unique()):
            render_grid(
                summary,
                baseline,
                experiment,
                optimizer_dir
                / (
                    f"{slugify(experiment)}__optimizer_{slugify(optimizer)}"
                    "__avg_seed_target_drift_by_epsilon__ref_euc.png"
                ),
                optimizer=optimizer,
            )
            render_grid(
                summary,
                baseline,
                experiment,
                optimizer_dir
                / (
                    f"{slugify(experiment)}__optimizer_{slugify(optimizer)}"
                    "__avg_seed_delta_vs_base_by_epsilon__ref_euc.png"
                ),
                optimizer=optimizer,
                metrics=DELTA_METRICS,
            )
            accuracy_spec = ACCURACY_DELTA_SPECS.get(experiment)
            if accuracy_spec is not None:
                render_grid(
                    summary,
                    baseline,
                    experiment,
                    optimizer_dir
                    / (
                        f"{slugify(experiment)}__optimizer_{slugify(optimizer)}"
                        "__avg_seed_accuracy_delta_pp_by_epsilon__ref_euc.png"
                    ),
                    optimizer=optimizer,
                    metrics=[accuracy_spec["metric"]],
                    subtitle=(
                        "Percentage-point accuracy change from the matching base model; "
                        "rho is the x-axis, rows are epsilon values"
                    ),
                )
                render_grid(
                    summary,
                    baseline,
                    experiment,
                    optimizer_dir
                    / (
                        f"{slugify(experiment)}__optimizer_{slugify(optimizer)}"
                        "__avg_seed_accuracy_delta_and_absolute_drift_by_epsilon"
                        "__ref_euc.png"
                    ),
                    optimizer=optimizer,
                    metrics=[
                        accuracy_spec["metric"],
                        ABSOLUTE_DRIFT_METRIC,
                    ],
                    subtitle=(
                        "Signed accuracy delta is on the left; absolute reference "
                        "drift is on the right; rho is the x-axis"
                    ),
                )
            render_grid(
                summary,
                baseline,
                experiment,
                optimizer_dir
                / (
                    f"{slugify(experiment)}__optimizer_{slugify(optimizer)}"
                    "__avg_seed_absolute_reference_drift_by_epsilon__ref_euc.png"
                ),
                optimizer=optimizer,
                metrics=[ABSOLUTE_DRIFT_METRIC],
                subtitle=(
                    "Absolute reference-score change from the matching base model; "
                    "lower is better, rho is the x-axis, rows are epsilon values"
                ),
            )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
