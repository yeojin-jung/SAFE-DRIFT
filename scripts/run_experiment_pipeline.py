#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run_selector_sft_sweep_code.py"


def safe_slug(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text).strip("_") or "value"


def parse_simple_yaml_value(raw: str) -> Any:
    value = raw.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        data: dict[str, Any] = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid config line {line_no} in {path}: {line!r}")
            key, value = stripped.split(":", 1)
            data[key.strip()] = parse_simple_yaml_value(value)
        return data
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return dict(loaded)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def add_flag(command: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{name.replace('_', '-')}"
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, list):
        if not value:
            return
        command.append(flag)
        command.extend(str(item) for item in value)
        return
    command.extend([flag, str(value)])


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def rank_overrides(rank_spec: dict[str, Any]) -> dict[str, Any]:
    mode = rank_spec.get("mode", "fixed")
    if mode == "auto":
        return {
            "low_rank_reference_rank": "auto",
            "low_rank_task_rank": "auto",
            "low_rank_common_rank": "auto",
            "low_rank_auto_task_rank": True,
        }
    return {
        "low_rank_reference_rank": rank_spec["K_R"],
        "low_rank_task_rank": rank_spec["K_T"],
        "low_rank_common_rank": rank_spec.get("K", int(rank_spec["K_R"]) + int(rank_spec["K_T"])),
        "low_rank_auto_task_rank": False,
    }


def build_sweep_command(
    config: dict[str, Any],
    *,
    run_kind: str,
    model_variant: dict[str, Any],
    rank_spec: dict[str, Any],
    subset_percentage: float,
    rho: float | None,
) -> tuple[str, list[str]]:
    experiment = config["experiment"]
    data = config["data"]
    selection = config.get("selection", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    splits = config.get("splits", {})
    output_root = repo_path(config.get("output_root", "outputs"))

    rank_name = str(rank_spec.get("name") or ("auto" if rank_spec.get("mode") == "auto" else f"KR{rank_spec['K_R']}_KT{rank_spec['K_T']}"))
    model_name = str(model_variant["name"])
    budget_name = f"pct{safe_slug(f'{float(subset_percentage):g}')}"
    if run_kind == "safe":
        run_name = f"{experiment['id']}/{model_name}/{rank_name}/{budget_name}/safe_rho{safe_slug(f'{rho:g}')}"
    else:
        run_name = f"{experiment['id']}/{model_name}/{rank_name}/{budget_name}/baselines"
    run_output = output_root / run_name

    command = [sys.executable, str(SWEEP_SCRIPT)]
    base_args: dict[str, Any] = {
        "candidate_file": data["candidate_file"],
        "target_file": data["target_file"],
        "ood_eval_file": data.get("ood_eval_file"),
        "reference_file": data.get("reference_file"),
        "reference_hf_stereoset": data.get("reference_hf_stereoset", False),
        "reference_hf_subset": data.get("reference_hf_subset"),
        "reference_hf_split": data.get("reference_hf_split"),
        "reference_hf_label": data.get("reference_hf_label"),
        "reference_hf_format": data.get("reference_hf_format"),
        "reference_bias_eval_output_file": str(run_output / "reference_bias_eval_triplets.json")
        if data.get("reference_hf_stereoset", False)
        else None,
        "target_split_proportions": splits.get("target", [0.34, 0.33, 0.33]),
        "reference_split_proportions": splits.get("reference", [0.5, 0.5]),
        "reference_split_group_key": splits.get("reference_group_key"),
        "subset_percentages": [subset_percentage],
        "selectors": config.get("baseline_selectors", ["full", "random", "dsir", "less", "prismatic"])
        if run_kind == "baseline"
        else ["safe"],
        "seed": config.get("seed", 42),
        "model_name": model_variant["gradient_model"],
        "train_model_name": model_variant.get("finetune_model", model_variant["gradient_model"]),
        "output_dir": str(run_output),
        "selection_output_dir": str(run_output / "subsets"),
        "training_output_dir": str(run_output / "training_runs"),
        "feature_cache_dir": str(run_output / "selector_feature_cache"),
        "train_cache_dir": str(run_output / "train_cache"),
    }
    base_args.update(selection)
    base_args.update(training)
    base_args.update(evaluation)
    base_args.update(rank_overrides(rank_spec))
    if run_kind == "safe":
        base_args.update({"safe_alpha": "auto", "safe_cost_c": rho, "safe_epsilon": config["safe"]["epsilon"]})
    for key, value in base_args.items():
        add_flag(command, key, value)
    return run_name, command


def build_prepare_commands(config: dict[str, Any]) -> list[list[str]]:
    commands = []
    for item in as_list(config.get("prepare")):
        if not item:
            continue
        raw = item.get("command") if isinstance(item, dict) else item
        command = [str(part) for part in raw]
        if command and command[0] == "python":
            command[0] = sys.executable
        commands.append(command)
    return commands


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []
    model_variants = [variant for variant in config.get("model_variants", []) if variant.get("enabled", True)]
    for model_variant in model_variants:
        for rank_spec in config["rank_grid"]:
            for subset_percentage in config["subset_percentages"]:
                if config.get("run_baselines", True):
                    name, command = build_sweep_command(
                        config,
                        run_kind="baseline",
                        model_variant=model_variant,
                        rank_spec=rank_spec,
                        subset_percentage=float(subset_percentage),
                        rho=None,
                    )
                    commands.append({"name": name, "kind": "baseline", "command": command})
                for rho in config["safe"]["rho_values"]:
                    name, command = build_sweep_command(
                        config,
                        run_kind="safe",
                        model_variant=model_variant,
                        rank_spec=rank_spec,
                        subset_percentage=float(subset_percentage),
                        rho=float(rho),
                    )
                    commands.append({"name": name, "kind": "safe", "rho": float(rho), "command": command})
    return {
        "experiment": config["experiment"],
        "prepare_commands": build_prepare_commands(config),
        "run_commands": commands,
    }


def run_commands(commands: list[list[str]], *, dry_run: bool, print_commands: bool) -> int:
    for command in commands:
        if print_commands or not dry_run:
            print(" ".join(command))
        if dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SAFE-DRIFT experiment setting from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--defaults", default=str(REPO_ROOT / "configs" / "pipeline_defaults.yaml"))
    parser.add_argument("--prepare", action="store_true", help="Run configured dataset preparation before sweeps.")
    parser.add_argument("--execute", action="store_true", help="Actually launch commands. Default writes/prints manifest only.")
    parser.add_argument("--print-commands", action="store_true", help="Print expanded commands without executing them.")
    parser.add_argument("--manifest-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    defaults = load_yaml(repo_path(args.defaults)) if args.defaults and repo_path(args.defaults).exists() else {}
    config = deep_merge(defaults, load_yaml(repo_path(args.config)))
    manifest = build_manifest(config)
    manifest_path = Path(args.manifest_out) if args.manifest_out else repo_path(config.get("output_root", "outputs")) / config["experiment"]["id"] / "pipeline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[pipeline] wrote manifest: {manifest_path}")

    if args.prepare:
        rc = run_commands(manifest["prepare_commands"], dry_run=not args.execute, print_commands=args.print_commands)
        if rc != 0:
            return rc
    run_command_list = [entry["command"] for entry in manifest["run_commands"]]
    return run_commands(run_command_list, dry_run=not args.execute, print_commands=args.print_commands)


if __name__ == "__main__":
    raise SystemExit(main())
