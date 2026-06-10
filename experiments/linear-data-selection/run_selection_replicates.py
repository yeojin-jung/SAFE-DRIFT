#!/usr/local/bin/python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_selection_sims.experiment_modules.selection_replicates import (
    run_selection_replicates_experiment,
    run_selection_replicates_orthogonal_target_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated versions of experiment 2 and aggregate the tradeoff metrics.")
    parser.add_argument("--output-root", default="outputs", help="Directory where figures and CSV files are written.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of independent experiment-2 replicates.")
    parser.add_argument(
        "--target-variant",
        choices=["default", "orthogonal"],
        default="default",
        help="Which target-direction variant to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = (
        run_selection_replicates_orthogonal_target_experiment
        if args.target_variant == "orthogonal"
        else run_selection_replicates_experiment
    )
    manifest = runner(output_root=Path(args.output_root), seed=args.seed, num_runs=args.num_runs)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
