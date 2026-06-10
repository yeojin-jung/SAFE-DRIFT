#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.entanglement_analysis import analyze_entanglement, headline_metrics, selected_indices_from_records
from utils.data_construction import load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze reference/task entanglement for a selector feature cache.")
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--candidate-file", required=True, type=Path)
    parser.add_argument("--subset-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preconditioner-cache", default=None, type=Path)
    parser.add_argument("--reference-rank", default=None, type=int)
    parser.add_argument("--target-sample-size", default=128, type=int)
    parser.add_argument("--selected-csv-max-rows", default=5000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def load_projected_preconditioner(path: Path | None, feature_cache: Path) -> torch.Tensor | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if not isinstance(payload, dict):
        return None
    projected = payload.get("projected_preconditioners")
    if isinstance(projected, dict):
        value = projected.get(feature_cache.stem)
        if isinstance(value, torch.Tensor):
            return value
    value = payload.get("projected_preconditioner")
    return value if isinstance(value, torch.Tensor) else None


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = torch.load(args.feature_cache, map_location="cpu")
    candidates = load_records(args.candidate_file)
    selected = load_records(args.subset_file)
    selected_indices = selected_indices_from_records(candidates, selected)
    summary = analyze_entanglement(
        candidate_features=payload["candidate_features"],
        target_feature=payload["target_feature"],
        target_features=payload.get("target_features"),
        reference_fisher=payload["reference_fisher"],
        selector_preconditioner=load_projected_preconditioner(args.preconditioner_cache, args.feature_cache),
        selected_indices=selected_indices,
        output_dir=args.output_dir,
        metadata=payload.get("metadata") or {},
        reference_rank=args.reference_rank,
        target_sample_size=args.target_sample_size,
        seed=args.seed,
        selected_csv_max_rows=args.selected_csv_max_rows,
        label=args.label or args.subset_file.stem,
    )
    print(json.dumps(headline_metrics(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
