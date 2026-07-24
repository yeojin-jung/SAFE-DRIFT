#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def flag_span(command: list[str], flag: str) -> tuple[int, int] | None:
    try:
        start = command.index(flag)
    except ValueError:
        return None
    end = start + 1
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    return start, end


def get_value(command: list[str], flag: str, default: str | None = None) -> str | None:
    span = flag_span(command, flag)
    if span is None:
        return default
    start, end = span
    return command[start + 1] if end > start + 1 else default


def set_flag(command: list[str], flag: str, *values: str) -> None:
    span = flag_span(command, flag)
    if span is not None:
        del command[span[0] : span[1]]
    command.extend([flag, *values])


def add_bool(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def safe_slug(value: Any) -> str:
    return (
        "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in str(value)
        ).strip("_")
        or "item"
    )


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def command_weight(entry: dict[str, Any]) -> float:
    command = entry["command"]
    weight = 1.0
    if "--skip-base-eval" not in command:
        weight += 1.0
    if float(get_value(command, "--kl-regularization-lambda", "0") or 0.0) > 0.0:
        weight += 0.45
    if get_value(command, "--safe-solver") == "relaxed":
        weight += 0.8
    if get_value(command, "--safe-training-constraint-mode", "none") != "none":
        weight += 0.1
    return weight


def cache_mode(entry: dict[str, Any]) -> str | None:
    if entry.get("kind") != "safe":
        return None
    command = entry["command"]
    adam_mode = get_value(command, "--adam-selection-mode", "post_projection")
    return str(adam_mode)


def cache_prep_entries(
    entries: list[dict[str, Any]],
    *,
    dispatch_dir: Path,
) -> list[dict[str, Any]]:
    representatives: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        mode = cache_mode(entry)
        if mode is None:
            continue
        key = (str(entry.get("repeat_tag")), mode)
        representatives.setdefault(key, entry)

    prep_entries = []
    for (repeat_tag, mode), source in sorted(representatives.items()):
        prep = copy.deepcopy(source)
        command = prep["command"]
        prep_output = dispatch_dir / "cache_prep_outputs" / repeat_tag / safe_slug(mode)
        set_flag(command, "--output-dir", str(prep_output))
        set_flag(command, "--selection-output-dir", str(prep_output / "subsets"))
        set_flag(command, "--training-output-dir", str(prep_output / "training_runs"))
        set_flag(command, "--train-cache-dir", str(prep_output / "train_cache"))
        add_bool(command, "--skip-base-eval")
        add_bool(command, "--prepare-shared-selector-cache-only")
        prep["name"] = f"cache_prep/{repeat_tag}/{mode}"
        prep["kind"] = "cache_prep"
        prep["done_file"] = str(prep_output / "selector_sft_sweep_manifest.json")
        prep_entries.append(prep)
    return prep_entries


def prepare_run_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base_eval_seen: set[str] = set()
    prepared = []
    for source in entries:
        entry = copy.deepcopy(source)
        repeat_tag = str(entry.get("repeat_tag"))
        if repeat_tag in base_eval_seen:
            add_bool(entry["command"], "--skip-base-eval")
        else:
            base_eval_seen.add(repeat_tag)
        output_dir = Path(get_value(entry["command"], "--output-dir") or "")
        entry["done_file"] = str(output_dir / "selector_sft_sweep_manifest.json")
        prepared.append(entry)
    return prepared


def balanced_shards(
    entries: list[dict[str, Any]],
    *,
    num_workers: int,
) -> tuple[list[list[dict[str, Any]]], list[float]]:
    shards = [[] for _ in range(num_workers)]
    loads = [0.0] * num_workers
    for entry in sorted(entries, key=command_weight, reverse=True):
        worker = min(range(num_workers), key=lambda index: loads[index])
        shards[worker].append(entry)
        loads[worker] += command_weight(entry)
    return shards, loads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced cache-prep and run shards from a pipeline manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = prepare_run_entries(list(manifest["run_commands"]))
    prep_entries = cache_prep_entries(entries, dispatch_dir=args.out_dir.resolve())
    prep_shards, prep_loads = balanced_shards(
        prep_entries,
        num_workers=args.num_workers,
    )
    run_shards, run_loads = balanced_shards(entries, num_workers=args.num_workers)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for worker in range(args.num_workers):
        write_jsonl(args.out_dir / f"worker_{worker}_cache_prep.jsonl", prep_shards[worker])
        write_jsonl(args.out_dir / f"worker_{worker}_runs.jsonl", run_shards[worker])

    summary = {
        "manifest": str(args.manifest.resolve()),
        "num_workers": args.num_workers,
        "cache_prep_count": len(prep_entries),
        "run_count": len(entries),
        "workers": [
            {
                "worker": worker,
                "cache_prep_count": len(prep_shards[worker]),
                "cache_prep_weight": prep_loads[worker],
                "run_count": len(run_shards[worker]),
                "run_weight": run_loads[worker],
            }
            for worker in range(args.num_workers)
        ],
    }
    (args.out_dir / "shard_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
