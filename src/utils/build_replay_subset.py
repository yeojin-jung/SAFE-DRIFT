#!/usr/bin/env python3
"""
Build a replay-based training subset: k% of the budget comes from an
already-computed LESS subset (target-relevant, top-ranked first) and the
remaining (100 - k)% is sampled uniformly at random from a reference/replay
pool. This is the standard replay recipe for mitigating catastrophic
forgetting/off-target drift: mix task-relevant selected data with samples
from previously-seen or off-target data rather than training on the
selected subset alone.

Typical usage (setting 1: CodeAlpaca candidates, StereoSet reference fetched live,
same source run_selector_sft_sweep_code.py uses via --reference-hf-stereoset):
  python src/utils/build_replay_subset.py \
      --less-subset-file outputs/.../subsets/<...>_less_low_rank_adam_pct5.0_budget1000_seed42.jsonl \
      --reference-hf-stereoset \
      --output-file outputs/.../subsets/replay_k80_budget1000_seed42.jsonl \
      --k 80 --seed 42

Settings 2/3 (prepared local reference files) use --reference-file instead, e.g.:
  python src/utils/build_replay_subset.py \
      --less-subset-file <...>_less_..._pct5.0_budget<N>_seed42.jsonl \
      --reference-file data/setting_2_medical/prepared/reference_prompts.jsonl \
      --output-file <...>/replay_k80_budget<N>_seed42.jsonl \
      --k 80 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from baseline_selectors.replay_selector import select_replay
from utils.data_construction import load_records, load_stereoset_records, split_records_by_proportions, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix k%% of a LESS-selected subset with (100-k)%% random reference replay examples."
    )
    parser.add_argument(
        "--less-subset-file",
        required=True,
        type=Path,
        help="JSONL subset already produced by LESS (e.g. via ensure_less_subset), ranked highest-score first.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        type=Path,
        help="JSONL replay pool to sample uniformly at random from (e.g. a setting's prepared reference_prompts.jsonl). "
        "Mutually exclusive with --reference-hf-stereoset.",
    )
    parser.add_argument(
        "--reference-hf-stereoset",
        action="store_true",
        help="Fetch the replay pool from HF StereoSet instead of a local file (setting 1's reference source). "
        "Mirrors run_selector_sft_sweep_code.py's --reference-hf-stereoset handling, including the 50/50 "
        "reference/bias_eval split, so the replay pool here matches the reference half actually used for "
        "selection rather than leaking the held-out bias_eval half.",
    )
    parser.add_argument("--reference-hf-subset", default="intrasentence")
    parser.add_argument("--reference-hf-split", default="validation")
    parser.add_argument("--reference-hf-label", default="stereotype,anti-stereotype")
    parser.add_argument("--reference-hf-format", default="instruction", choices=["instruction", "messages"])
    parser.add_argument("--reference-split-proportions", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--reference-split-seed", type=int, default=42)
    parser.add_argument("--reference-split-group-key", default="bias_type")
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument(
        "--k",
        type=float,
        default=80.0,
        help="Percentage of the final subset drawn from --less-subset-file (top-ranked entries); "
        "the remaining (100-k) percent is sampled at random from --reference-file. Default: 80.",
    )
    parser.add_argument(
        "--total-budget",
        type=int,
        default=None,
        help="Total number of examples in the output subset. Defaults to the size of --less-subset-file, "
        "i.e. keep the overall budget fixed and replace (100-k)%% of it with replay examples.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reference sampling and shuffling.")
    parser.add_argument(
        "--with-replacement",
        action="store_true",
        help="Allow sampling the same reference example more than once if the replay budget "
        "exceeds the reference pool size.",
    )
    parser.add_argument(
        "--dedupe-key",
        default="id",
        help="Record field used to drop reference replay examples that duplicate a selected LESS example "
        "(e.g. reference and candidate pools overlap). Set to '' to disable.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep LESS examples first followed by replay examples instead of shuffling them together.",
    )
    return parser.parse_args()


def tag_source(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    tagged = []
    for record in records:
        record = dict(record)
        record["replay_source"] = source
        tagged.append(record)
    return tagged


def dedupe_reference_records(
    less_records: list[dict[str, Any]],
    reference_records: list[dict[str, Any]],
    dedupe_key: str,
) -> list[dict[str, Any]]:
    if not dedupe_key:
        return reference_records
    less_keys = {record[dedupe_key] for record in less_records if dedupe_key in record}
    if not less_keys:
        return reference_records
    return [record for record in reference_records if record.get(dedupe_key) not in less_keys]


def load_reference_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    if bool(args.reference_file) == bool(args.reference_hf_stereoset):
        raise ValueError("Pass exactly one of --reference-file or --reference-hf-stereoset.")

    if args.reference_file:
        return load_records(args.reference_file)

    all_reference_records = load_stereoset_records(
        subset=args.reference_hf_subset,
        split=args.reference_hf_split,
        label=args.reference_hf_label,
        output_format=args.reference_hf_format,
    )
    reference_splits = split_records_by_proportions(
        all_reference_records,
        args.reference_split_proportions,
        seed=args.reference_split_seed,
        shuffle=True,
        group_key=args.reference_split_group_key,
    )
    # First split is "reference" (matches names=["reference", "bias_eval"] in
    # run_selector_sft_sweep_code.py's split_lookup) -- the held-out bias_eval
    # half is intentionally excluded from the replay pool.
    return reference_splits[0]


def main() -> None:
    args = parse_args()

    less_records = load_records(args.less_subset_file)
    reference_pool = load_reference_pool(args)
    reference_pool = dedupe_reference_records(less_records, reference_pool, args.dedupe_key)

    total_budget = args.total_budget if args.total_budget is not None else len(less_records)

    less_indices, reference_indices = select_replay(
        num_less_candidates=len(less_records),
        num_reference_candidates=len(reference_pool),
        subset_budget=total_budget,
        k=args.k,
        seed=args.seed,
        with_replacement=args.with_replacement,
    )

    less_selected = tag_source([less_records[i] for i in less_indices], "less")
    replay_selected = tag_source([reference_pool[i] for i in reference_indices], "replay")

    combined = less_selected + replay_selected
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(combined)

    write_jsonl(combined, args.output_file)

    metadata = {
        "less_subset_file": str(args.less_subset_file.resolve()),
        "reference_file": str(args.reference_file.resolve()) if args.reference_file else None,
        "reference_hf_stereoset": bool(args.reference_hf_stereoset),
        "reference_hf_subset": args.reference_hf_subset if args.reference_hf_stereoset else None,
        "reference_hf_split": args.reference_hf_split if args.reference_hf_stereoset else None,
        "reference_hf_label": args.reference_hf_label if args.reference_hf_stereoset else None,
        "k": args.k,
        "total_budget": total_budget,
        "less_count": len(less_selected),
        "replay_count": len(replay_selected),
        "seed": args.seed,
        "with_replacement": args.with_replacement,
        "dedupe_key": args.dedupe_key or None,
        "shuffled": not args.no_shuffle,
    }
    metadata_path = args.output_file.with_suffix(args.output_file.suffix + ".selection.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[replay] wrote {len(combined)} examples "
        f"({len(less_selected)} less + {len(replay_selected)} replay) to {args.output_file}"
    )


if __name__ == "__main__":
    main()
