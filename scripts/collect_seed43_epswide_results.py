#!/usr/bin/env python3
"""Collect completed seed43 epsilon-wide manifest results as a compact CSV."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path


ROOT = Path("outputs/cluster_manifests/seed43_refeuc_epswide_d95da37")
SOURCE = os.environ.get("SAFE_DRIFT_RESULT_SOURCE", "seed43_current_20260629")
COLUMNS = [
    "source",
    "array_index",
    "experiment",
    "complete",
    "name",
    "kind",
    "selector",
    "variant",
    "geometry",
    "optimizer",
    "rho",
    "epsilon",
    "method_label",
    "target_primary",
    "reference_primary",
    "target_delta_primary",
    "reference_delta_primary",
    "constraint_status_for_plot",
    "manifest_path",
    "summary_path",
]

SETTINGS = {
    "setting2_run_commands.jsonl": {
        "experiment": "setting2_medqa_medmcqa_truthfulqa_bbq",
        "target": "target_medqa_accuracy",
        "target_alt": "medqa_accuracy",
        "reference": "ood_medical_reference_primary_score",
        "target_delta": "target_delta_medqa_accuracy",
        "reference_delta": "ood_delta_medical_reference_primary_score",
        "base_target": "base_target_medqa_accuracy",
        "base_reference": "base_ood_medical_reference_primary_score",
    },
    "setting3_run_commands.jsonl": {
        "experiment": "setting3_empathetic_esconv_gsm8k",
        "target": "target_esconv_token_f1",
        "target_alt": "esconv_token_f1",
        "reference": "ood_gsm8k_accuracy",
        "reference_alt": "reference_gsm8k_accuracy",
        "target_delta": "target_delta_esconv_token_f1",
        "reference_delta": "ood_delta_gsm8k_accuracy",
        "base_target": "base_target_esconv_token_f1",
        "base_reference": "base_ood_gsm8k_accuracy",
    },
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def manifest_complete(data: dict) -> bool:
    runs = data.get("runs")
    return isinstance(runs, list) and bool(runs) and all(
        int(run.get("train_return_code", 1)) == 0 for run in runs
    )


def summary_for_manifest(data: dict) -> tuple[Path | None, dict | None]:
    runs = data.get("runs") or []
    if not runs:
        return None, None
    out = runs[0].get("run_output_dir")
    if not out:
        return None, None
    summary_path = Path(out) / "summary.json"
    if not summary_path.exists():
        return None, None
    return summary_path, read_json(summary_path)


def first_present(data: dict, *keys: str | None):
    for key in keys:
        if key and key in data and data[key] is not None:
            return data[key]
    return ""


def command_text(item: dict) -> str:
    return " ".join(map(str, item.get("command", [])))


def parse_geometry(item: dict) -> str:
    variant = str(item.get("variant") or "")
    if "reference" in variant:
        return "reference"
    if "euclidean" in variant:
        return "euclidean"
    if "rank" in variant:
        return "rank"
    match = re.search(r"--safe-geometry\s+(\S+)", command_text(item))
    return match.group(1) if match else ""


def parse_optimizer(item: dict) -> str:
    variant = str(item.get("variant") or item.get("selector") or "")
    if "sgd" in variant:
        return "sgd"
    if "adam" in variant or item.get("kind") in {"baseline", "base"}:
        return "adam"
    match = re.search(r"--selector-preconditioner\s+(\S+)", command_text(item))
    return match.group(1) if match else ""


def method_label(kind: str, selector: str, geometry: str, optimizer: str) -> str:
    if kind == "base":
        return "Base model"
    if kind == "baseline":
        return selector or "baseline"
    if kind == "safe":
        return f"SAFE {geometry} / {optimizer}".strip()
    return selector or kind or ""


def constraint_status(data: dict) -> str:
    runs = data.get("runs") or []
    if not runs:
        return ""
    selection = runs[0].get("selection") or {}
    alpha = str(selection.get("safe_alpha_status") or "").lower()
    case = str(selection.get("safe_constraint_case") or "").lower()
    if any(
        selection.get(key) is False
        for key in [
            "safe_continuous_reference_budget_satisfied",
            "safe_continuous_epsilon_budget_satisfied",
        ]
    ):
        return "violated"
    if alpha == "inactive" or "inactive" in case:
        return "inactive"
    if alpha in {"feasible", "active"} or "feasible" in case:
        return "feasible"
    return alpha or "unknown"


def main() -> None:
    rows = []
    base_rows = {}
    for filename, spec in SETTINGS.items():
        items = [
            json.loads(line)
            for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index, item in enumerate(items):
            manifest_path = Path(item["done_file"])
            if not manifest_path.exists():
                continue
            try:
                manifest = read_json(manifest_path)
            except Exception:
                continue
            if not manifest_complete(manifest):
                continue
            summary_path, summary = summary_for_manifest(manifest)
            if not summary:
                continue

            experiment = spec["experiment"]
            if experiment not in base_rows:
                base_rows[experiment] = {
                    "source": SOURCE,
                    "array_index": "base",
                    "experiment": experiment,
                    "complete": True,
                    "name": f"seed43/{experiment}/base_model",
                    "kind": "base",
                    "selector": "base_model",
                    "variant": "base_model",
                    "geometry": "",
                    "optimizer": "",
                    "rho": "",
                    "epsilon": "",
                    "method_label": "Base model",
                    "target_primary": first_present(summary, spec["base_target"]),
                    "reference_primary": first_present(summary, spec["base_reference"]),
                    "target_delta_primary": 0.0,
                    "reference_delta_primary": 0.0,
                    "constraint_status_for_plot": "",
                    "manifest_path": str(manifest_path.resolve()),
                    "summary_path": str(summary_path.parent if summary_path else ""),
                }

            kind = str(item.get("kind") or "")
            selector = str(item.get("selector") or ("safe" if kind == "safe" else ""))
            variant = str(item.get("variant") or selector)
            geometry = parse_geometry(item)
            optimizer = parse_optimizer(item)
            rows.append(
                {
                    "source": SOURCE,
                    "array_index": index,
                    "experiment": experiment,
                    "complete": True,
                    "name": item.get("name", ""),
                    "kind": kind,
                    "selector": selector,
                    "variant": variant,
                    "geometry": geometry,
                    "optimizer": optimizer,
                    "rho": item.get("rho", ""),
                    "epsilon": item.get("epsilon", ""),
                    "method_label": method_label(kind, selector, geometry, optimizer),
                    "target_primary": first_present(summary, spec["target"], spec.get("target_alt")),
                    "reference_primary": first_present(
                        summary, spec["reference"], spec.get("reference_alt")
                    ),
                    "target_delta_primary": first_present(summary, spec["target_delta"]),
                    "reference_delta_primary": first_present(summary, spec["reference_delta"]),
                    "constraint_status_for_plot": constraint_status(manifest)
                    if kind == "safe"
                    else "",
                    "manifest_path": str(manifest_path.resolve()),
                    "summary_path": str(summary_path) if summary_path else "",
                }
            )

    out_rows = []
    for experiment in sorted(base_rows):
        out_rows.append(base_rows[experiment])
        out_rows.extend(row for row in rows if row["experiment"] == experiment)

    writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(out_rows)


if __name__ == "__main__":
    main()
