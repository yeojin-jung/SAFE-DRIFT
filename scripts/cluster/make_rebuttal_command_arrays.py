#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shlex
from pathlib import Path
from typing import Any, Iterable


VALUE_FLAGS = {
    "--candidate-file",
    "--candidate-validation-file",
    "--candidate-test-file",
    "--target-file",
    "--validation-file",
    "--eval-file",
    "--ood-eval-file",
    "--reference-file",
    "--reference-validation-file",
    "--reference-test-file",
    "--reference-hf-subset",
    "--reference-hf-split",
    "--reference-hf-label",
    "--reference-hf-format",
    "--reference-bias-eval-output-file",
    "--bias-eval-file",
    "--seed",
    "--model-name",
    "--train-model-name",
    "--output-dir",
    "--selection-output-dir",
    "--training-output-dir",
    "--feature-cache-dir",
    "--train-cache-dir",
    "--max-seq-len",
    "--lora-r",
    "--lora-alpha",
    "--lora-dropout",
    "--less-similarity",
    "--selector-feature-method",
    "--selector-preconditioner",
    "--adam-beta2",
    "--adam-eps",
    "--adam-warmup-steps",
    "--preconditioner-split-seed",
    "--selector-projection-dim",
    "--selector-projection-seed",
    "--selector-projection-chunk-size",
    "--selector-projection-cache-dtype",
    "--safe-alpha",
    "--safe-learning-rate",
    "--safe-geometry",
    "--safe-solver",
    "--safe-shortlist-size",
    "--safe-cost-c",
    "--safe-epsilon",
    "--reference-fisher-max-examples",
    "--low-rank-reference-rank",
    "--low-rank-builder-alpha",
    "--low-rank-delta",
    "--low-rank-max-reference-rank",
    "--low-rank-reference-shrinkage",
    "--low-rank-task-rank",
    "--low-rank-common-rank",
    "--low-rank-delta-task",
    "--low-rank-max-task-rank",
    "--low-rank-target-max-examples",
    "--low-rank-candidate-max-examples",
    "--low-rank-candidate-projection-chunk-rows",
    "--train-max-seq-length",
    "--train-torch-dtype",
    "--selection-torch-dtype",
    "--train-lora-r",
    "--train-lora-alpha",
    "--train-lora-dropout",
    "--per-device-train-batch-size",
    "--per-device-eval-batch-size",
    "--gradient-accumulation-steps",
    "--learning-rate",
    "--weight-decay",
    "--warmup-ratio",
    "--lr-scheduler-type",
    "--max-grad-norm",
    "--logging-steps",
    "--eval-steps",
    "--save-steps",
    "--evaluator",
    "--ood-evaluator",
    "--reference-evaluator",
    "--eval-max-examples",
    "--generation-max-new-tokens",
    "--humaneval-num-samples",
    "--humaneval-temperature",
    "--humaneval-top-p",
    "--benchmark-max-examples",
    "--benchmark-mmlu-subset",
    "--benchmark-mmlu-split",
    "--benchmark-gsm8k-subset",
    "--benchmark-gsm8k-split",
    "--validation-gradient-max-examples",
    "--entanglement-output-dir",
    "--entanglement-selector-top-k",
    "--entanglement-target-sample-size",
    "--entanglement-selected-csv-max-rows",
    "--bias-eval-domain",
    "--bias-eval-layers",
    "--bias-eval-alpha",
    "--bias-eval-geometry",
    "--bias-eval-direction-split",
    "--bias-eval-batch-size",
    "--bias-eval-max-length",
    "--bias-eval-max-examples",
    "--num-train-epochs",
    "--max-steps",
    "--target-split-seed",
    "--reference-split-seed",
    "--reference-split-group-key",
}

MULTI_FLAGS = {
    "--target-split-proportions",
    "--reference-split-proportions",
    "--selectors",
    "--subset-percentages",
    "--lora-target-modules",
    "--train-lora-target-modules",
    "--humaneval-pass-at-ks",
    "--benchmark-evals",
}

BOOL_FLAGS = {
    "--reference-hf-stereoset",
    "--dry-run",
    "--overwrite-subsets",
    "--overwrite-selection-cache",
    "--skip-base-eval",
    "--prepare-shared-selector-cache-only",
    "--skip-training",
    "--skip-final-evaluation",
    "--save-selected-artifacts",
    "--no-save-selected-artifacts",
    "--run-entanglement-analysis",
    "--no-run-entanglement-analysis",
    "--safe-average-by-budget",
    "--no-safe-average-by-budget",
    "--low-rank-auto-task-rank",
    "--save-all-target-features",
    "--gradient-checkpointing",
    "--add-bos-token",
    "--trust-remote-code",
    "--low-cpu-mem-usage",
    "--compute-validation-gradient",
    "--compute-reference-fisher",
    "--compute-entanglement-metrics",
}


def safe_slug(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text).strip("_") or "value"


def flag_span(command: list[str], flag: str) -> tuple[int, int] | None:
    try:
        start = command.index(flag)
    except ValueError:
        return None
    if flag in BOOL_FLAGS:
        return start, start + 1
    end = start + 1
    while end < len(command) and not command[end].startswith("--"):
        end += 1
    return start, end


def get_values(command: list[str], flag: str) -> list[str]:
    span = flag_span(command, flag)
    if span is None:
        return []
    start, end = span
    return command[start + 1 : end]


def get_value(command: list[str], flag: str, default: str | None = None) -> str | None:
    values = get_values(command, flag)
    return values[0] if values else default


def remove_flag(command: list[str], flag: str) -> None:
    while True:
        span = flag_span(command, flag)
        if span is None:
            return
        del command[span[0] : span[1]]


def set_flag(command: list[str], flag: str, values: str | Iterable[str]) -> None:
    remove_flag(command, flag)
    value_list = [values] if isinstance(values, str) else [str(value) for value in values]
    command.extend([flag, *[str(value) for value in value_list]])


def add_bool(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def set_bool(command: list[str], positive_flag: str, enabled: bool) -> None:
    negative_flag = "--no-" + positive_flag.removeprefix("--")
    remove_flag(command, positive_flag)
    remove_flag(command, negative_flag)
    command.append(positive_flag if enabled else negative_flag)


def command_map(command: list[str]) -> dict[str, tuple[str, ...] | bool]:
    parsed: dict[str, tuple[str, ...] | bool] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        if item in BOOL_FLAGS:
            parsed[item] = True
            index += 1
            continue
        end = index + 1
        while end < len(command) and not command[end].startswith("--"):
            end += 1
        parsed[item] = tuple(command[index + 1 : end])
        index = end
    return parsed


def relative_output_parts(entry: dict[str, Any]) -> list[str]:
    name_parts = str(entry["name"]).split("/")
    if len(name_parts) >= 5:
        return name_parts[:5]
    return name_parts


def shared_feature_cache_path(
    command: list[str],
    entry: dict[str, Any],
    *,
    project_dir: Path,
    shared_cache_root: Path,
) -> tuple[Path, Path]:
    prefix = relative_output_parts(entry)
    method = safe_slug(get_value(command, "--selector-feature-method", "low_rank"))
    preconditioner = safe_slug(get_value(command, "--selector-preconditioner", "adam"))
    target_seed = safe_slug(get_value(command, "--target-split-seed", "42"))
    reference_seed = safe_slug(get_value(command, "--reference-split-seed", "42"))
    projection_seed = safe_slug(get_value(command, "--selector-projection-seed", "13"))
    shared_tag = f"{method}_{preconditioner}_target{target_seed}_ref{reference_seed}_proj{projection_seed}"
    root = (project_dir / shared_cache_root / Path(*prefix) / shared_tag).resolve()
    return root / "selector_feature_cache"


def transform_common(
    command: list[str],
    entry: dict[str, Any],
    *,
    project_dir: Path,
    shared_cache_root: Path,
    save_steps: int,
    run_entanglement_analysis: bool,
) -> Path:
    feature_cache = shared_feature_cache_path(
        command,
        entry,
        project_dir=project_dir,
        shared_cache_root=shared_cache_root,
    )
    set_flag(command, "--feature-cache-dir", str(feature_cache))
    set_flag(command, "--save-steps", str(save_steps))
    set_bool(command, "--run-entanglement-analysis", run_entanglement_analysis)
    return feature_cache


def split_baseline_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    command = list(entry["command"])
    selectors = get_values(command, "--selectors")
    if len(selectors) <= 1:
        return [copy.deepcopy(entry)]
    split_entries: list[dict[str, Any]] = []
    for index, selector in enumerate(selectors):
        split_entry = copy.deepcopy(entry)
        split_command = list(command)
        set_flag(split_command, "--selectors", selector)
        output_dir = Path(get_value(split_command, "--output-dir") or "outputs/baseline")
        selector_output_dir = output_dir / safe_slug(selector)
        set_flag(split_command, "--output-dir", str(selector_output_dir))
        set_flag(split_command, "--selection-output-dir", str(selector_output_dir / "subsets"))
        set_flag(split_command, "--training-output-dir", str(selector_output_dir / "training_runs"))
        set_flag(split_command, "--train-cache-dir", str(selector_output_dir / "train_cache"))
        if index > 0:
            add_bool(split_command, "--skip-base-eval")
        split_entry["name"] = f"{entry['name']}/{safe_slug(selector)}"
        split_entry["selector"] = selector
        split_entry["command"] = split_command
        split_entries.append(split_entry)
    return split_entries


def build_cache_prep_item(
    entry: dict[str, Any],
    command: list[str],
    *,
    project_dir: Path,
    shared_cache_root: Path,
    save_steps: int,
) -> dict[str, Any]:
    prep_entry = copy.deepcopy(entry)
    prep_command = list(command)
    feature_cache = transform_common(
        prep_command,
        prep_entry,
        project_dir=project_dir,
        shared_cache_root=shared_cache_root,
        save_steps=save_steps,
        run_entanglement_analysis=True,
    )
    prep_output = feature_cache.resolve().parent / "cache_prep_output"
    set_flag(prep_command, "--output-dir", str(prep_output))
    set_flag(prep_command, "--selection-output-dir", str(prep_output / "subsets"))
    set_flag(prep_command, "--training-output-dir", str(prep_output / "training_runs"))
    add_bool(prep_command, "--skip-base-eval")
    add_bool(prep_command, "--prepare-shared-selector-cache-only")
    prep_entry["name"] = f"cache_prep/{entry['name']}"
    prep_entry["kind"] = "cache_prep"
    prep_entry["command"] = prep_command
    prep_entry["done_file"] = str((prep_output / "selector_sft_sweep_manifest.json").resolve())
    prep_entry["feature_cache_dir"] = str(feature_cache.resolve())
    return prep_entry


def is_same_cache_prep(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_map = command_map(left["command"])
    right_map = command_map(right["command"])
    ignored = {
        "--output-dir",
        "--selection-output-dir",
        "--training-output-dir",
        "--train-cache-dir",
        "--safe-alpha",
        "--safe-cost-c",
        "--safe-epsilon",
        "--safe-geometry",
        "--safe-solver",
        "--safe-learning-rate",
        "--save-steps",
        "--selectors",
    }
    for key in set(left_map) | set(right_map):
        if key in ignored:
            continue
        if left_map.get(key) != right_map.get(key):
            return False
    return True


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_shell_preview(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, item in enumerate(items):
        lines.append(f"# {index}: {item['name']}")
        lines.append(shlex.join(item["command"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Slurm JSONL command arrays for setting 2/3 rebuttal sweeps.")
    parser.add_argument("--manifest", action="append", required=True, help="Pipeline manifest JSON. Repeat for each setting.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--project-dir", default="/net/projects2/mercury/cdonnat/SAFE-DRIFT")
    parser.add_argument("--shared-cache-root", default="outputs/rebuttal_shared_cache")
    parser.add_argument("--save-steps", type=int, default=25)
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    shared_cache_root = Path(args.shared_cache_root)

    all_prepare: list[dict[str, Any]] = []
    all_cache_prep: list[dict[str, Any]] = []
    all_runs: list[dict[str, Any]] = []

    for manifest_path_raw in args.manifest:
        manifest_path = Path(manifest_path_raw)
        manifest = load_manifest(manifest_path)
        experiment_id = str(manifest.get("experiment", {}).get("id") or manifest_path.stem)
        setting_prepare: list[dict[str, Any]] = []
        setting_cache_prep: list[dict[str, Any]] = []
        setting_runs: list[dict[str, Any]] = []

        for index, command in enumerate(manifest.get("prepare_commands", [])):
            item = {
                "name": f"{experiment_id}/prepare/{index}",
                "kind": "prepare",
                "command": command,
            }
            setting_prepare.append(item)
            all_prepare.append(item)

        expanded_runs: list[dict[str, Any]] = []
        for entry in manifest.get("run_commands", []):
            if entry.get("kind") == "baseline":
                expanded_runs.extend(split_baseline_entry(entry))
            else:
                expanded_runs.append(copy.deepcopy(entry))

        for entry in expanded_runs:
            command = list(entry["command"])
            transform_common(
                command,
                entry,
                project_dir=project_dir,
                shared_cache_root=shared_cache_root,
                save_steps=args.save_steps,
                run_entanglement_analysis=False,
            )
            if entry.get("kind") == "safe":
                add_bool(command, "--skip-base-eval")
            run_output = Path(get_value(command, "--output-dir") or "")
            entry["command"] = command
            entry["done_file"] = str((run_output / "selector_sft_sweep_manifest.json").resolve())
            setting_runs.append(entry)
            all_runs.append(entry)

            prep_item = build_cache_prep_item(
                entry,
                command,
                project_dir=project_dir,
                shared_cache_root=shared_cache_root,
                save_steps=args.save_steps,
            )
            if not any(is_same_cache_prep(prep_item, existing) for existing in setting_cache_prep):
                setting_cache_prep.append(prep_item)
                all_cache_prep.append(prep_item)

        prefix = safe_slug(experiment_id)
        write_jsonl(out_dir / f"{prefix}_prepare_commands.jsonl", setting_prepare)
        write_jsonl(out_dir / f"{prefix}_cache_prep_commands.jsonl", setting_cache_prep)
        write_jsonl(out_dir / f"{prefix}_run_commands.jsonl", setting_runs)
        write_shell_preview(out_dir / f"{prefix}_run_commands.sh", setting_runs)
        print(
            f"[cluster-manifest] {experiment_id}: "
            f"prepare={len(setting_prepare)} cache_prep={len(setting_cache_prep)} runs={len(setting_runs)}"
        )

    write_jsonl(out_dir / "all_prepare_commands.jsonl", all_prepare)
    write_jsonl(out_dir / "all_cache_prep_commands.jsonl", all_cache_prep)
    write_jsonl(out_dir / "all_run_commands.jsonl", all_runs)
    print(
        "[cluster-manifest] all: "
        f"prepare={len(all_prepare)} cache_prep={len(all_cache_prep)} runs={len(all_runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
