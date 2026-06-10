#!/usr/local/bin/python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_selection_sims.experiment_modules import (
    DimRedSelectorConfig,
    ReferenceTargetSpectrumGridConfig,
    run_reference_target_spectrum_grid,
)


def _parse_score_rank_pairs(value: str | None, default: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if value is None:
        return default
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":")
        pairs.append((int(left), int(right)))
    return tuple(pairs)


def _parse_string_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a repeated grid sweep over reference and target covariance spectra.")
    parser.add_argument("--output-root", default="outputs", help="Directory where figures and CSV files are written.")
    parser.add_argument("--seed", type=int, default=11, help="Base random seed.")
    parser.add_argument("--d", type=int, default=200, help="Ambient dimension.")
    parser.add_argument("--r", type=int, default=40, help="True reference rank.")
    parser.add_argument("--n-candidates", type=int, default=800, help="Number of candidate gradients.")
    parser.add_argument("--n-target", type=int, default=32, help="Number of target gradients used to estimate g_T.")
    parser.add_argument("--n-reference", type=int, default=300, help="Number of reference samples used to estimate F_R.")
    parser.add_argument("--k-select", type=int, default=40, help="Subset budget.")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds / repeats per grid cell.")
    parser.add_argument(
        "--score-rank-pairs",
        default="5:5,10:10,20:20,30:30,40:40,60:60",
        help="Comma-separated K_R:K_T pairs.",
    )
    parser.add_argument("--spike-high", type=float, default=25.0, help="Largest spike in the reference spectrum.")
    parser.add_argument("--spike-low", type=float, default=2.0, help="Smallest spike among retained reference directions.")
    parser.add_argument("--tail-eig", type=float, default=0.05, help="Flat tail eigenvalue outside the reference rank.")
    parser.add_argument("--target-cov-scale", type=float, default=1.0, help="Global scale factor applied to the target covariance.")
    parser.add_argument("--target-r", type=int, default=None, help="Optional target rank. Defaults to the reference rank.")
    parser.add_argument("--target-spike-high", type=float, default=None, help="Optional largest spike in the target spectrum.")
    parser.add_argument("--target-spike-low", type=float, default=None, help="Optional smallest retained spike in the target spectrum.")
    parser.add_argument("--target-tail-eig", type=float, default=None, help="Optional flat tail eigenvalue for the target spectrum.")
    parser.add_argument(
        "--reference-decays",
        default="geometric,linear,powerlaw",
        help="Comma-separated reference decay families to compare.",
    )
    parser.add_argument(
        "--target-decays",
        default="geometric,linear,powerlaw",
        help="Comma-separated target decay families to compare.",
    )
    parser.add_argument(
        "--reference-powerlaw-exponents",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated reference powerlaw exponents for the powerlaw-vs-powerlaw grid.",
    )
    parser.add_argument(
        "--target-powerlaw-exponents",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated target powerlaw exponents for the powerlaw-vs-powerlaw grid.",
    )
    parser.add_argument(
        "--comparison-reference-powerlaw-exponent",
        type=float,
        default=1.0,
        help="Reference powerlaw exponent used in the decay-family 3x3 grid.",
    )
    parser.add_argument(
        "--comparison-target-powerlaw-exponent",
        type=float,
        default=1.0,
        help="Target powerlaw exponent used in the decay-family 3x3 grid.",
    )
    parser.add_argument("--epsilon", type=float, default=1.0, help="SAFE norm radius.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed alpha used when rho is not set.")
    parser.add_argument("--rho", type=float, default=None, help="Optional SAFE reference-drift budget.")
    parser.add_argument(
        "--regimes",
        default="reference_aligned,residual_cheap,adversarial_omitted",
        help="Comma-separated regime names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = DimRedSelectorConfig()
    dimred_config = DimRedSelectorConfig(
        d=int(args.d),
        r=int(args.r),
        n_candidates=int(args.n_candidates),
        n_target=int(args.n_target),
        n_reference=int(args.n_reference),
        k_select=int(args.k_select),
        seeds=int(args.seeds),
        base_seed=int(args.seed),
        score_rank_pairs=_parse_score_rank_pairs(args.score_rank_pairs, default_config.score_rank_pairs),
        spike_high=float(args.spike_high),
        spike_low=float(args.spike_low),
        tail_eig=float(args.tail_eig),
        spectrum_decay="geometric",
        powerlaw_exponent=float(args.comparison_reference_powerlaw_exponent),
        target_cov_scale=float(args.target_cov_scale),
        target_r=None if args.target_r is None else int(args.target_r),
        target_spike_high=None if args.target_spike_high is None else float(args.target_spike_high),
        target_spike_low=None if args.target_spike_low is None else float(args.target_spike_low),
        target_tail_eig=None if args.target_tail_eig is None else float(args.target_tail_eig),
        target_spectrum_decay="geometric",
        target_powerlaw_exponent=float(args.comparison_target_powerlaw_exponent),
        epsilon=float(args.epsilon),
        alpha=float(args.alpha),
        rho=None if args.rho is None else float(args.rho),
        regimes=_parse_string_list(args.regimes),
    )
    grid_config = ReferenceTargetSpectrumGridConfig(
        dimred_config=dimred_config,
        reference_decays=_parse_string_list(args.reference_decays),
        target_decays=_parse_string_list(args.target_decays),
        reference_powerlaw_exponents=_parse_float_list(args.reference_powerlaw_exponents),
        target_powerlaw_exponents=_parse_float_list(args.target_powerlaw_exponents),
        comparison_reference_powerlaw_exponent=float(args.comparison_reference_powerlaw_exponent),
        comparison_target_powerlaw_exponent=float(args.comparison_target_powerlaw_exponent),
    )
    manifest = run_reference_target_spectrum_grid(
        output_root=Path(args.output_root),
        seed=int(args.seed),
        config=grid_config,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
