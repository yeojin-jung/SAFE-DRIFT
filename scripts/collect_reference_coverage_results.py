#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REFERENCE_METRICS = {
    "bbq": {
        "score": "extra_eval_bbq_bbq_accuracy",
        "base": "base_extra_eval_bbq_bbq_accuracy",
        "kl": "extra_eval_bbq_heldout_kl",
    },
    "truthfulqa": {
        "score": "extra_eval_truthfulqa_truthfulqa_token_f1",
        "base": "base_extra_eval_truthfulqa_truthfulqa_token_f1",
        "kl": "extra_eval_truthfulqa_heldout_kl",
    },
    "mmlu_non_medical": {
        "score": "extra_eval_mmlu_non_medical_mmlu_accuracy",
        "base": "base_extra_eval_mmlu_non_medical_mmlu_accuracy",
        "kl": "extra_eval_mmlu_non_medical_heldout_kl",
    },
}

UNSEEN_METRICS = {
    "ifeval": {
        "score": "extra_eval_ifeval_ifeval_proxy_instruction_accuracy",
        "base": "base_extra_eval_ifeval_ifeval_proxy_instruction_accuracy",
    }
}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def loss_from_base(summary: dict[str, Any], score_key: str, base_key: str) -> float | None:
    score = as_float(summary.get(score_key))
    base = as_float(summary.get(base_key))
    if score is None or base is None:
        return None
    return max(0.0, base - score)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def method_name(summary: dict[str, Any], selection: dict[str, Any]) -> str:
    method = str(summary.get("selection_method") or selection.get("selector") or "")
    if method.startswith("less"):
        return "LESS"
    if method.startswith("safe"):
        composition = selection.get("reference_composition_name") or summary.get("reference_composition_name")
        return f"SAFE-({composition})" if composition else "SAFE"
    return method or "unknown"


def load_row(summary_path: Path) -> dict[str, Any] | None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = summary.get("selection") if isinstance(summary.get("selection"), dict) else {}
    if summary.get("completed_steps") is None:
        return None

    target_score = as_float(summary.get("target_medqa_accuracy", summary.get("medqa_accuracy")))
    target_base = as_float(summary.get("base_target_medqa_accuracy"))
    target_gain = None if target_score is None or target_base is None else target_score - target_base

    domains = selection.get("reference_composition_domains", summary.get("reference_composition_domains") or [])
    domains = [str(domain) for domain in domains]
    row: dict[str, Any] = {
        "summary_path": str(summary_path.resolve()),
        "seed": summary.get("seed"),
        "method": method_name(summary, selection),
        "reference_composition": selection.get("reference_composition_name", summary.get("reference_composition_name")),
        "included_references": ",".join(domains),
        "safe_beta": selection.get("safe_cost_beta"),
        "safe_rho": summary.get("safe_rho", selection.get("safe_cost_c")),
        "safe_epsilon": summary.get("safe_epsilon", selection.get("safe_epsilon")),
        "target_medqa_accuracy": target_score,
        "base_target_medqa_accuracy": target_base,
        "medqa_gain": target_gain,
    }

    reference_losses: list[float] = []
    reference_kls: list[float] = []
    for name, keys in REFERENCE_METRICS.items():
        score = as_float(summary.get(keys["score"]))
        base = as_float(summary.get(keys["base"]))
        loss = loss_from_base(summary, keys["score"], keys["base"])
        kl = as_float(summary.get(keys["kl"]))
        row[f"{name}_included"] = name in domains
        row[f"{name}_score"] = score
        row[f"{name}_base_score"] = base
        row[f"{name}_loss"] = loss
        row[f"{name}_heldout_kl"] = kl
        if loss is not None:
            reference_losses.append(loss)
        if kl is not None:
            reference_kls.append(kl)

    for name, keys in UNSEEN_METRICS.items():
        row[f"{name}_score"] = as_float(summary.get(keys["score"]))
        row[f"{name}_base_score"] = as_float(summary.get(keys["base"]))
        row[f"{name}_loss"] = loss_from_base(summary, keys["score"], keys["base"])

    row["mean_reference_loss"] = mean(reference_losses)
    row["worst_reference_loss"] = max(reference_losses) if reference_losses else None
    row["mean_reference_kl"] = mean(reference_kls)
    row["worst_reference_kl"] = max(reference_kls) if reference_kls else None
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect the Setting 2 reference-coverage table.")
    parser.add_argument("roots", nargs="+", help="Run roots searched recursively for summary.json.")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for root in args.roots:
        for summary_path in sorted(Path(root).rglob("summary.json")):
            try:
                row = load_row(summary_path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if row is not None:
                rows.append(row)
    write_csv(Path(args.output_csv), rows)
    print(json.dumps({"rows": len(rows), "output_csv": str(Path(args.output_csv).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
