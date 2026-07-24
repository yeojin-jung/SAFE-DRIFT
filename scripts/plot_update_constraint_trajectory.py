#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        step = row.get("step", row.get("selected_count", row.get("iteration")))
        if value is None or step is None:
            continue
        try:
            xs.append(float(step))
            ys.append(float(value))
        except (TypeError, ValueError):
            continue
    return xs, ys


def positive(values: list[float], floor: float = 1.0e-16) -> list[float]:
    return [max(float(value), floor) for value in values]


def plot_trajectory(
    run_dir: Path,
    *,
    output_prefix: Path | None = None,
    dpi: int = 220,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    run_dir = run_dir.resolve()
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary = load_json(summary_path)
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    selector_trace = [
        row for row in selection.get("safe_optimization_trace", []) if isinstance(row, dict)
    ]
    optimizer_rows = [
        row
        for row in load_jsonl(metrics_path)
        if row.get("step") is not None
        and (
            row.get("incremental_reference_cost") is not None
            or row.get("cumulative_reference_cost") is not None
        )
    ]
    rho = summary.get("safe_rho")
    epsilon = summary.get("safe_training_epsilon", summary.get("safe_epsilon"))
    norm_budget = None if epsilon is None else 0.5 * float(epsilon) ** 2
    running_reference_cost = 0.0
    for row in optimizer_rows:
        if row.get("incremental_reference_cost") is None:
            row["incremental_reference_cost"] = row.get("delta_theta_reference_cost")
        incremental_reference = row.get("incremental_reference_cost")
        if incremental_reference is not None:
            running_reference_cost += float(incremental_reference)
            row.setdefault("sum_incremental_reference_cost", running_reference_cost)
            if rho is not None:
                row.setdefault(
                    "incremental_reference_budget_ratio",
                    float(incremental_reference) / max(float(rho), 1.0e-30),
                )
        if row.get("cumulative_reference_cost") is not None and rho is not None:
            row.setdefault(
                "cumulative_reference_budget_ratio",
                float(row["cumulative_reference_cost"]) / max(float(rho), 1.0e-30),
            )
        if row.get("incremental_norm_cost") is None and row.get("delta_theta_norm") is not None:
            row["incremental_norm_cost"] = 0.5 * float(row["delta_theta_norm"]) ** 2
        if row.get("cumulative_norm_cost") is None and row.get("cumulative_delta_theta_norm") is not None:
            row["cumulative_norm_cost"] = 0.5 * float(row["cumulative_delta_theta_norm"]) ** 2
        if norm_budget is not None:
            if row.get("incremental_norm_cost") is not None:
                row.setdefault(
                    "incremental_norm_budget_ratio",
                    float(row["incremental_norm_cost"]) / max(norm_budget, 1.0e-30),
                )
            if row.get("cumulative_norm_cost") is not None:
                row.setdefault(
                    "cumulative_norm_budget_ratio",
                    float(row["cumulative_norm_cost"]) / max(norm_budget, 1.0e-30),
                )

    output_prefix = output_prefix or (run_dir / "update_constraint_trajectory")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_prefix.with_name(output_prefix.name + "_selector.csv"), selector_trace)
    write_csv(output_prefix.with_name(output_prefix.name + "_optimizer.csv"), optimizer_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    navy = "#183B56"
    rust = "#C84A2F"
    teal = "#23807B"
    gold = "#D39B2A"

    ax = axes[0, 0]
    x, y = numeric(selector_trace, "approximation_error")
    if x:
        ax.plot(x, positive(y), color=navy, marker="o", markersize=3, linewidth=1.8)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "No selector-prefix trace", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Selected-set approximation")
    ax.set_xlabel("Selected support size")
    ax.set_ylabel(r"$\|\Delta(w)-\Delta^\star\|_{F_R+\alpha I}$")

    ax = axes[0, 1]
    for key, label, color in (
        ("reference_budget_ratio", r"Fisher cost / $\rho$", rust),
        ("norm_budget_ratio", r"Norm cost / $(\epsilon^2/2)$", teal),
    ):
        x, y = numeric(selector_trace, key)
        if x:
            ax.plot(x, positive(y), color=color, marker="o", markersize=3, linewidth=1.6, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1, label="constraint boundary")
    ax.set_yscale("log")
    ax.set_title("Selector-prefix feasibility")
    ax.set_xlabel("Selected support size")
    ax.set_ylabel("Budget ratio")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for key, label, color in (
        ("incremental_reference_budget_ratio", r"Step Fisher / $\rho$", rust),
        ("incremental_norm_budget_ratio", r"Step norm / $(\epsilon^2/2)$", teal),
    ):
        x, y = numeric(optimizer_rows, key)
        if x:
            ax.plot(x, positive(y), color=color, linewidth=1.5, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title("Each optimizer update")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Budget ratio")
    projection_ax = ax.twinx()
    projection_x, projection_y = numeric(optimizer_rows, "trajectory_projection_scale")
    if projection_x:
        projection_ax.plot(
            projection_x,
            projection_y,
            color=navy,
            linewidth=1.5,
            alpha=0.8,
            label="Accepted proposal scale",
        )
        projection_ax.set_ylim(-0.03, 1.03)
        projection_ax.set_ylabel("Projection scale", color=navy)
        projection_ax.tick_params(axis="y", colors=navy)
    handles, labels = ax.get_legend_handles_labels()
    projection_handles, projection_labels = projection_ax.get_legend_handles_labels()
    ax.legend(
        handles + projection_handles,
        labels + projection_labels,
        frameon=False,
        fontsize=8,
    )

    ax = axes[1, 1]
    for key, label, color, linestyle, alpha in (
        (
            "cumulative_reference_budget_ratio",
            r"Accepted Fisher / $\rho$",
            rust,
            "-",
            1.0,
        ),
        (
            "proposed_cumulative_reference_budget_ratio",
            r"Proposed Fisher / $\rho$",
            rust,
            "--",
            0.65,
        ),
        (
            "cumulative_norm_budget_ratio",
            r"Accepted norm / $(\epsilon^2/2)$",
            teal,
            "-",
            1.0,
        ),
        (
            "proposed_cumulative_norm_budget_ratio",
            r"Proposed norm / $(\epsilon^2/2)$",
            teal,
            "--",
            0.65,
        ),
        (
            "sum_incremental_reference_cost",
            "Sum of step Fisher costs",
            gold,
            ":",
            0.85,
        ),
    ):
        x, y = numeric(optimizer_rows, key)
        if not x:
            continue
        if key == "sum_incremental_reference_cost":
            rho = summary.get("safe_rho")
            if rho is None:
                continue
            y = [value / max(float(rho), 1.0e-30) for value in y]
        ax.plot(
            x,
            positive(y),
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            linewidth=1.5,
            label=label,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title("Whole training trajectory")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Budget ratio")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(True, which="both", linewidth=0.35, alpha=0.3)

    title = str(summary.get("selection_method") or selection.get("selector") or run_dir.name)
    constraint_mode = str(summary.get("safe_training_constraint_mode") or "none")
    fig.suptitle(
        f"SAFE-DRIFT update audit: {title} ({constraint_mode})",
        fontsize=14,
    )
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    result = {
        "run_dir": str(run_dir),
        "selector_trace_points": len(selector_trace),
        "optimizer_steps": len(optimizer_rows),
        "safe_training_constraint_mode": constraint_mode,
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "selector_csv": str(
            output_prefix.with_name(output_prefix.name + "_selector.csv").resolve()
        ),
        "optimizer_csv": str(
            output_prefix.with_name(output_prefix.name + "_optimizer.csv").resolve()
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected-set approximation and per-step/cumulative SAFE constraint checks."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            plot_trajectory(
                args.run_dir,
                output_prefix=args.output_prefix,
                dpi=args.dpi,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
