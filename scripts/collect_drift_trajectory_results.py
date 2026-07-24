#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = 0.5 * ((position + 1) + end)
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def spearman(left: Iterable[Any], right: Iterable[Any]) -> float | None:
    pairs = []
    for left_value, right_value in zip(left, right):
        try:
            pair = (float(left_value), float(right_value))
        except (TypeError, ValueError):
            continue
        if math.isfinite(pair[0]) and math.isfinite(pair[1]):
            pairs.append(pair)
    if len(pairs) < 3:
        return None
    left_ranks = average_ranks([pair[0] for pair in pairs])
    right_ranks = average_ranks([pair[1] for pair in pairs])
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((rank - left_mean) ** 2 for rank in left_ranks))
    right_scale = math.sqrt(sum((rank - right_mean) ** 2 for rank in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def scalar_metrics(summary: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if key.startswith(prefixes) and isinstance(value, (int, float, str, bool))
    }


def load_run(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    selection_method = summary.get("selection_method")
    selection_family = (
        str(selection_method).split("_cost", 1)[0]
        if selection_method is not None
        else None
    )
    points = [
        point
        for point in summary.get("trajectory_points", [])
        if isinstance(point, dict)
    ]
    final_point = points[-1] if points else {}
    row: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "run_output_dir": summary.get("output_dir"),
        "seed": summary.get("seed"),
        "selection_method": selection_method,
        "selection_family": selection_family,
        "selection_preconditioner": summary.get("selection_preconditioner"),
        "safe_geometry": selection.get("safe_geometry"),
        "safe_solver": selection.get("safe_solver"),
        "safe_solver_status": selection.get("safe_solver_status"),
        "safe_selected_count": selection.get("safe_selected_count"),
        "safe_is_noop": selection.get("safe_is_noop"),
        "safe_selection_weight_sum": selection.get("safe_selection_weight_sum"),
        "safe_objective_value": selection.get("safe_objective_value"),
        "safe_rho": summary.get("safe_rho"),
        "safe_epsilon": summary.get("safe_epsilon"),
        "safe_epsilon_multiplier": selection.get("safe_epsilon_multiplier"),
        "safe_epsilon_calibration_base": selection.get(
            "safe_epsilon_calibration_base"
        ),
        "safe_training_epsilon": summary.get("safe_training_epsilon"),
        "safe_training_epsilon_scale_by_steps": summary.get(
            "safe_training_epsilon_scale_by_steps",
            False,
        ),
        "safe_training_epsilon_scale_horizon": summary.get(
            "safe_training_epsilon_scale_horizon"
        ),
        "safe_training_constraint_mode": summary.get(
            "safe_training_constraint_mode",
            "none",
        ),
        "safe_cost_beta": selection.get("safe_cost_beta"),
        "safe_cost_gamma": selection.get("safe_cost_gamma"),
        "safe_reference_curvature_q0": selection.get(
            "safe_reference_curvature_q0"
        ),
        "safe_unconstrained_target_reference_cost": selection.get(
            "safe_unconstrained_target_reference_cost"
        ),
        "safe_effective_cost_c": selection.get("safe_effective_cost_c"),
        "reference_composition_name": selection.get(
            "reference_composition_name",
            summary.get("reference_composition_name"),
        ),
        "reference_composition_domains": ",".join(
            str(domain)
            for domain in (
                selection.get(
                    "reference_composition_domains",
                    summary.get("reference_composition_domains") or [],
                )
                or []
            )
        ),
        "kl_regularization_lambda": summary.get("kl_regularization_lambda"),
        "selected_predicted_fisher_cost": summary.get("selected_predicted_fisher_cost"),
        "selected_predicted_norm_cost": summary.get("selected_predicted_norm_cost"),
        "trained_lora_fisher_cost": summary.get("trained_lora_fisher_cost"),
        "trajectory_projection_steps": summary.get("trajectory_projection_steps"),
        "trajectory_projection_active_steps": summary.get(
            "trajectory_projection_active_steps"
        ),
        "trajectory_projection_active_fraction": summary.get(
            "trajectory_projection_active_fraction"
        ),
        "trajectory_projection_mean_scale": summary.get(
            "trajectory_projection_mean_scale"
        ),
        "trajectory_projection_min_scale": summary.get(
            "trajectory_projection_min_scale"
        ),
        "realized_heldout_kl": final_point.get("realized_heldout_kl"),
        "target_loss_degradation": final_point.get("target_loss_degradation"),
        "target_performance_delta": final_point.get("target_performance_delta"),
        "target_medqa_accuracy": final_point.get("target_medqa_accuracy"),
        "target_medqa_accuracy_delta": final_point.get("target_medqa_accuracy_delta"),
        "reference_performance_degradation": final_point.get("reference_performance_degradation"),
        "instruction_performance_degradation": final_point.get("instruction_performance_degradation"),
        "estimated_sft_flops": summary.get("estimated_sft_flops"),
        "estimated_selection_flops": summary.get("estimated_selection_flops"),
        "estimated_kl_regularization_flops": summary.get("estimated_kl_regularization_flops"),
        "estimated_trajectory_diagnostic_flops": summary.get("estimated_trajectory_diagnostic_flops"),
        "estimated_trajectory_generation_flops": summary.get("estimated_trajectory_generation_flops"),
        "estimated_total_flops": summary.get("estimated_total_flops"),
        "estimated_total_flops_including_selection": summary.get(
            "estimated_total_flops_including_selection"
        ),
        "train_wallclock_seconds": summary.get("train_wallclock_seconds"),
        "completed_steps": summary.get("completed_steps"),
    }
    correlations = summary.get("trajectory_correlations") or {}
    for key, value in correlations.items():
        row[f"trajectory_{key}"] = value
    row.update(
        scalar_metrics(
            summary,
            (
                "target_",
                "reference_",
                "base_target_",
                "base_reference_",
                "ood_",
                "base_ood_",
                "extra_eval_",
                "base_extra_eval_",
            ),
        )
    )
    trajectory_rows = []
    for point in points:
        trajectory_rows.append(
            {
                "summary_path": str(summary_path.resolve()),
                "seed": summary.get("seed"),
                "selection_method": selection_method,
                "selection_family": selection_family,
                "selection_preconditioner": summary.get("selection_preconditioner"),
                "safe_geometry": selection.get("safe_geometry"),
                "safe_solver": selection.get("safe_solver"),
                "safe_rho": summary.get("safe_rho"),
                "safe_epsilon": summary.get("safe_epsilon"),
                "safe_epsilon_multiplier": selection.get(
                    "safe_epsilon_multiplier"
                ),
                "safe_epsilon_calibration_base": selection.get(
                    "safe_epsilon_calibration_base"
                ),
                "safe_training_epsilon": summary.get("safe_training_epsilon"),
                "safe_training_epsilon_scale_by_steps": summary.get(
                    "safe_training_epsilon_scale_by_steps",
                    False,
                ),
                "safe_training_constraint_mode": summary.get(
                    "safe_training_constraint_mode",
                    "none",
                ),
                "safe_cost_beta": selection.get("safe_cost_beta"),
                "safe_cost_gamma": selection.get("safe_cost_gamma"),
                "safe_reference_curvature_q0": selection.get(
                    "safe_reference_curvature_q0"
                ),
                "reference_composition_name": selection.get(
                    "reference_composition_name",
                    summary.get("reference_composition_name"),
                ),
                "kl_regularization_lambda": summary.get("kl_regularization_lambda"),
                **point,
            }
        )
    return row, trajectory_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped_correlations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows = [row for row in rows if row.get("safe_rho") is not None]
    groups: dict[tuple[Any, Any, Any, Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in safe_rows:
        groups[
            (
                row.get("selection_family"),
                row.get("selection_preconditioner"),
                row.get("safe_geometry"),
                row.get("safe_solver"),
                (
                    row.get("safe_epsilon_multiplier")
                    if row.get("safe_epsilon_multiplier") is not None
                    else row.get("safe_epsilon")
                ),
                row.get("safe_training_constraint_mode", "none"),
            )
        ].append(row)
    results = []
    for (
        method,
        preconditioner,
        geometry,
        solver,
        epsilon_grid_value,
        constraint_mode,
    ), group in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        results.append(
            {
                "selection_family": method,
                "selection_preconditioner": preconditioner,
                "safe_geometry": geometry,
                "safe_solver": solver,
                "safe_epsilon_grid_value": epsilon_grid_value,
                "safe_epsilon_is_multiplier": any(
                    row.get("safe_epsilon_multiplier") is not None
                    for row in group
                ),
                "safe_training_constraint_mode": constraint_mode,
                "run_count": len(group),
                "spearman_actual_kl_vs_rho": spearman(
                    [row.get("realized_heldout_kl") for row in group],
                    [row.get("safe_rho") for row in group],
                ),
                "spearman_actual_kl_vs_target_degradation": spearman(
                    [row.get("realized_heldout_kl") for row in group],
                    [row.get("target_loss_degradation") for row in group],
                ),
                "spearman_actual_kl_vs_medqa_accuracy": spearman(
                    [row.get("realized_heldout_kl") for row in group],
                    [row.get("target_medqa_accuracy") for row in group],
                ),
                "spearman_actual_kl_vs_reference_degradation": spearman(
                    [row.get("realized_heldout_kl") for row in group],
                    [row.get("reference_performance_degradation") for row in group],
                ),
                "spearman_actual_kl_vs_instruction_degradation": spearman(
                    [row.get("realized_heldout_kl") for row in group],
                    [row.get("instruction_performance_degradation") for row in group],
                ),
                "spearman_selected_predicted_vs_trained_fisher_cost": spearman(
                    [row.get("selected_predicted_fisher_cost") for row in group],
                    [row.get("trained_lora_fisher_cost") for row in group],
                ),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect trajectory-wide SAFE-DRIFT diagnostics.")
    parser.add_argument("roots", nargs="+", help="Output roots recursively searched for summary.json.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_paths = sorted(
        {
            path.resolve()
            for root_raw in args.roots
            for path in Path(root_raw).rglob("summary.json")
        }
    )
    run_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for summary_path in summary_paths:
        try:
            run_row, point_rows = load_run(summary_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"summary_path": str(summary_path), "error": str(exc)})
            continue
        if run_row.get("trained_lora_fisher_cost") is None:
            continue
        run_rows.append(run_row)
        trajectory_rows.extend(point_rows)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "run_level_diagnostics.csv", run_rows)
    write_csv(output_dir / "trajectory_points.csv", trajectory_rows)
    correlations = grouped_correlations(run_rows)
    write_csv(output_dir / "cross_run_correlations.csv", correlations)
    payload = {
        "summary_files_scanned": len(summary_paths),
        "runs_collected": len(run_rows),
        "trajectory_points_collected": len(trajectory_rows),
        "errors": errors,
        "cross_run_correlations": correlations,
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
