#!/usr/bin/env python3
"""
Build a replay-based training subset: k% of the budget comes from an
already-computed LESS subset (target-relevant, top-ranked first) and the
remaining (100 - k)% is sampled uniformly at random from a reference/replay
pool. This is the standard replay recipe for mitigating catastrophic
forgetting/off-target drift: mix task-relevant selected data with samples
from previously-seen or off-target data rather than training on the
selected subset alone.

For --reference-hf-stereoset, StereoSet is split at the *triplet* level into
the same two halves run_selector_sft_sweep_code.py uses: "reference" (the
half whose Fisher/gradient geometry defines drift during LESS/SAFE selection)
and "bias_eval" (held out for the post-training StereoSet SS/LMS/ICAT score).
The replay pool is drawn only from the "reference" half; the "bias_eval" half
is written out separately (--stereoset-eval-output-file) so it stays
guaranteed-disjoint from anything used in training and is ready to pass to
train_lora_sft.py's --bias-eval-data-path later.

Typical usage (setting 1: CodeAlpaca candidates, StereoSet reference fetched live,
same source run_selector_sft_sweep_code.py uses via --reference-hf-stereoset):
  python src/utils/build_replay_subset.py \
      --less-subset-file outputs/.../subsets/<...>_less_low_rank_adam_pct5.0_budget1000_seed42.jsonl \
      --reference-hf-stereoset \
      --output-file outputs/.../subsets/replay_k80_budget1000_seed42.jsonl \
      --stereoset-eval-output-file outputs/.../subsets/replay_k80_stereoset_eval.json \
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
from utils.data_construction import (
    load_records,
    load_stereoset_triplet_records,
    split_records_by_proportions,
    write_jsonl,
)


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
        "Splits StereoSet at the triplet level into the same 'reference'/'bias_eval' halves "
        "run_selector_sft_sweep_code.py uses; the replay pool is drawn only from the 'reference' "
        "(drift-calculation) half.",
    )
    parser.add_argument("--reference-hf-subset", default="intrasentence")
    parser.add_argument("--reference-hf-split", default="validation")
    parser.add_argument(
        "--reference-hf-label",
        default="stereotype,anti-stereotype",
        help="Comma-separated StereoSet labels (or 'all') expanded into replay-pool records per reference triplet.",
    )
    parser.add_argument("--reference-split-proportions", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--reference-split-seed", type=int, default=42)
    parser.add_argument("--reference-split-group-key", default="bias_type")
    parser.add_argument(
        "--stereoset-eval-output-file",
        default=None,
        type=Path,
        help="Only with --reference-hf-stereoset: write the held-out 'bias_eval' StereoSet triplet half here "
        "(same JSON-list-of-triplets format train_lora_sft.py's --bias-eval-data-path expects), guaranteed "
        "disjoint from the replay pool used in training.",
    )
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument(
        "--k",
        type=float,
        default=80.0,
        help="Percentage of the final subset drawn from --less-subset-file (top-ranked entries); "
        "the remaining (100-k) percent is sampled at random from the reference pool. Default: 80.",
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


def parse_stereoset_labels(label: str) -> list[str]:
    if label == "all":
        return ["stereotype", "anti-stereotype", "unrelated"]
    labels = [value.strip() for value in str(label).split(",") if value.strip()]
    if not labels:
        raise ValueError("--reference-hf-label must be 'all' or a non-empty comma-separated list of StereoSet labels.")
    return labels


def triplets_to_labeled_records(triplets: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    """Expand StereoSet triplets into instruction/output rows, one per requested label.

    Mirrors utils.data_construction.load_stereoset_records's record shape, but
    operates on triplets already split into a specific half, so the expansion
    never crosses the reference/bias_eval boundary.
    """
    records: list[dict[str, Any]] = []
    for triplet in triplets:
        instruction = str(triplet.get("context") or "").strip() or "Complete the sentence."
        outputs = {
            "stereotype": triplet.get("stereotype", ""),
            "anti-stereotype": triplet.get("anti_stereotype", triplet.get("anti-stereotype", "")),
            "unrelated": triplet.get("unrelated", ""),
        }
        for label in labels:
            records.append(
                {
                    "id": triplet.get("id"),
                    "bias_type": triplet.get("bias_type"),
                    "target": triplet.get("target"),
                    "subset": triplet.get("subset"),
                    "label": label,
                    "instruction": instruction,
                    "output": outputs[label],
                }
            )
    return records


def load_reference_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """Returns (replay_pool_records, bias_eval_triplets_or_None)."""
    if bool(args.reference_file) == bool(args.reference_hf_stereoset):
        raise ValueError("Pass exactly one of --reference-file or --reference-hf-stereoset.")

    if args.reference_file:
        if args.stereoset_eval_output_file is not None:
            raise ValueError("--stereoset-eval-output-file is only supported with --reference-hf-stereoset.")
        return load_records(args.reference_file), None

    all_triplets = load_stereoset_triplet_records(
        subset=args.reference_hf_subset,
        split=args.reference_hf_split,
    )
    # Single triplet-level split so "reference" (drift-calculation half, used
    # below for the replay pool) and "bias_eval" (held out for the StereoSet
    # score) are guaranteed disjoint at the triplet level -- unlike deriving
    # them from two independent splits over differently-shaped record lists.
    reference_triplets, bias_eval_triplets = split_records_by_proportions(
        all_triplets,
        args.reference_split_proportions,
        seed=args.reference_split_seed,
        shuffle=True,
        group_key=args.reference_split_group_key,
    )
    replay_pool = triplets_to_labeled_records(reference_triplets, parse_stereoset_labels(args.reference_hf_label))
    return replay_pool, bias_eval_triplets


def main() -> None:
    args = parse_args()

    less_records = load_records(args.less_subset_file)
    reference_pool, bias_eval_triplets = load_reference_pool(args)
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

    stereoset_eval_output_file = None
    if args.stereoset_eval_output_file is not None:
        if bias_eval_triplets is None:
            raise ValueError("--stereoset-eval-output-file requires --reference-hf-stereoset.")
        stereoset_eval_output_file = args.stereoset_eval_output_file
        stereoset_eval_output_file.parent.mkdir(parents=True, exist_ok=True)
        stereoset_eval_output_file.write_text(
            json.dumps(bias_eval_triplets, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    metadata = {
        "less_subset_file": str(args.less_subset_file.resolve()),
        "reference_file": str(args.reference_file.resolve()) if args.reference_file else None,
        "reference_hf_stereoset": bool(args.reference_hf_stereoset),
        "reference_hf_subset": args.reference_hf_subset if args.reference_hf_stereoset else None,
        "reference_hf_split": args.reference_hf_split if args.reference_hf_stereoset else None,
        "reference_hf_label": args.reference_hf_label if args.reference_hf_stereoset else None,
        "reference_split_proportions": args.reference_split_proportions if args.reference_hf_stereoset else None,
        "reference_split_seed": args.reference_split_seed if args.reference_hf_stereoset else None,
        "reference_split_group_key": args.reference_split_group_key if args.reference_hf_stereoset else None,
        "k": args.k,
        "total_budget": total_budget,
        "less_count": len(less_selected),
        "replay_count": len(replay_selected),
        "seed": args.seed,
        "with_replacement": args.with_replacement,
        "dedupe_key": args.dedupe_key or None,
        "shuffled": not args.no_shuffle,
        "stereoset_eval_output_file": str(stereoset_eval_output_file.resolve()) if stereoset_eval_output_file else None,
        "stereoset_eval_triplet_count": len(bias_eval_triplets) if bias_eval_triplets is not None else None,
    }
    metadata_path = args.output_file.with_suffix(args.output_file.suffix + ".selection.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[replay] wrote {len(combined)} examples "
        f"({len(less_selected)} less + {len(replay_selected)} replay) to {args.output_file}"
    )
    if stereoset_eval_output_file is not None:
        print(
            f"[replay] wrote {len(bias_eval_triplets)} held-out StereoSet triplets "
            f"(disjoint from replay pool) to {stereoset_eval_output_file}"
        )


if __name__ == "__main__":
    main()
