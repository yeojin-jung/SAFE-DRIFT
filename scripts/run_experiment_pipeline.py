#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
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


def format_repeat_templates(value: Any, repeat: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**repeat) if "{" in value else value
    if isinstance(value, list):
        return [format_repeat_templates(item, repeat) for item in value]
    if isinstance(value, dict):
        return {key: format_repeat_templates(item, repeat) for key, item in value.items()}
    return value


def repeat_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_specs = config.get("repeats")
    if raw_specs is not None:
        specs: list[dict[str, Any]] = []
        for index, raw_spec in enumerate(as_list(raw_specs)):
            if not isinstance(raw_spec, dict):
                raise ValueError("repeats entries must be mappings.")
            spec = dict(raw_spec)
            seed = int(spec.get("repeat_seed", spec.get("seed", config.get("seed", 42))))
            spec.setdefault("repeat_index", index)
            spec.setdefault("repeat_seed", seed)
            spec.setdefault("repeat_tag", f"seed{safe_slug(seed)}")
            specs.append(spec)
        if not specs:
            raise ValueError("repeats must contain at least one entry.")
        return specs

    raw_seeds = config.get("repeat_seeds", config.get("seeds"))
    if raw_seeds is None:
        raw_seeds = [config.get("seed", 42)]
    seeds = [int(seed) for seed in as_list(raw_seeds)]
    if not seeds:
        raise ValueError("repeat_seeds must contain at least one seed.")
    specs: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        tag = f"seed{safe_slug(seed)}"
        specs.append(
            {
                "repeat_index": index,
                "repeat_seed": seed,
                "repeat_tag": tag,
            }
        )
    return specs


def config_for_repeat(config: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    formatted = format_repeat_templates(copy.deepcopy(config), repeat)
    seed = int(repeat["repeat_seed"])
    formatted["_repeat_index"] = int(repeat["repeat_index"])
    formatted["_repeat_seed"] = seed
    formatted["_repeat_tag"] = str(repeat["repeat_tag"])
    formatted["seed"] = int(formatted.get("seed", seed))

    splits = dict(formatted.get("splits", {}))
    splits.setdefault("target_seed", seed)
    splits.setdefault("reference_seed", seed)
    formatted["splits"] = splits

    selection = dict(formatted.get("selection", {}))
    selection.setdefault("preconditioner_split_seed", seed)
    selection.setdefault("selector_projection_seed", seed)
    formatted["selection"] = selection
    return formatted


def rank_overrides(rank_spec: dict[str, Any]) -> dict[str, Any]:
    mode = rank_spec.get("mode", "fixed")
    if mode == "auto":
        return {
            "low_rank_reference_rank": "auto",
            "low_rank_task_rank": "auto",
            "low_rank_common_rank": "auto",
            "low_rank_auto_task_rank": True,
        }
    overrides = {
        "low_rank_reference_rank": rank_spec["K_R"],
        "low_rank_task_rank": rank_spec["K_T"],
        "low_rank_auto_task_rank": False,
    }
    if "K" in rank_spec:
        overrides["low_rank_common_rank"] = rank_spec["K"]
    return overrides


def build_sweep_command(
    config: dict[str, Any],
    *,
    run_kind: str,
    model_variant: dict[str, Any],
    rank_spec: dict[str, Any],
    subset_percentage: float,
    rho: float | None,
    beta: float | None = None,
    gamma: float | None = None,
    baseline_variant: dict[str, Any] | None = None,
    safe_variant: dict[str, Any] | None = None,
    safe_epsilon: float | None = None,
    safe_epsilon_multiplier: float | None = None,
    reference_composition: dict[str, Any] | None = None,
    include_safe_epsilon_in_name: bool = False,
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
    experiment_prefix = str(experiment["id"])
    repeat_tag = config.get("_repeat_tag")
    if repeat_tag is not None:
        experiment_prefix = f"{experiment_prefix}/{repeat_tag}"
    shared_cache_root = config.get("shared_cache_root")
    if shared_cache_root is None:
        shared_scope = None
    else:
        shared_experiment_prefix = str(
            config.get("shared_cache_namespace") or experiment["id"]
        )
        if repeat_tag is not None:
            shared_experiment_prefix = f"{shared_experiment_prefix}/{repeat_tag}"
        shared_scope = (
            repo_path(shared_cache_root)
            / shared_experiment_prefix
            / model_name
            / rank_name
            / budget_name
        )
    if run_kind == "safe":
        variant_name = str((safe_variant or {}).get("name") or "safe")
        composition_name = (
            None
            if reference_composition is None
            else str(reference_composition.get("name") or "reference")
        )
        budget_name_for_run = (
            f"gamma{safe_slug(f'{float(gamma):g}')}"
            if gamma is not None
            else (
                f"beta{safe_slug(f'{float(beta):g}')}"
                if beta is not None
                else f"rho{safe_slug(f'{float(rho):g}')}"
            )
        )
        safe_name = f"{variant_name}_{budget_name_for_run}"
        if composition_name:
            safe_name = f"{variant_name}_ref{safe_slug(composition_name)}_{budget_name_for_run}"
        if include_safe_epsilon_in_name:
            if safe_epsilon_multiplier is not None:
                safe_name += (
                    f"_epsm{safe_slug(f'{float(safe_epsilon_multiplier):g}')}"
                )
            else:
                safe_name += f"_eps{safe_slug(f'{float(safe_epsilon):g}')}"
        run_name = f"{experiment_prefix}/{model_name}/{rank_name}/{budget_name}/{safe_name}"
    else:
        baseline_name = str((baseline_variant or {}).get("name") or "baselines")
        run_name = (
            f"{experiment_prefix}/{model_name}/{rank_name}/{budget_name}/"
            f"baseline_{safe_slug(baseline_name)}"
        )
    run_output = output_root / run_name

    command = [sys.executable, str(SWEEP_SCRIPT)]
    target_split_seed = splits.get("target_seed", config.get("target_split_seed", config.get("seed", 42)))
    reference_split_seed = splits.get("reference_seed", config.get("reference_split_seed", config.get("seed", 42)))
    base_args: dict[str, Any] = {
        "candidate_file": data["candidate_file"],
        "candidate_validation_file": data.get("candidate_validation_file"),
        "candidate_test_file": data.get("candidate_test_file"),
        "target_file": data["target_file"],
        "validation_file": data.get("validation_file"),
        "eval_file": data.get("eval_file"),
        "ood_eval_file": data.get("ood_eval_file"),
        "reference_file": data.get("reference_file"),
        "reference_validation_file": data.get("reference_validation_file"),
        "reference_test_file": data.get("reference_test_file"),
        "reference_hf_stereoset": data.get("reference_hf_stereoset", False),
        "reference_hf_subset": data.get("reference_hf_subset"),
        "reference_hf_split": data.get("reference_hf_split"),
        "reference_hf_label": data.get("reference_hf_label"),
        "reference_hf_format": data.get("reference_hf_format"),
        "reference_bias_eval_output_file": str(run_output / "reference_bias_eval_triplets.json")
        if data.get("reference_hf_stereoset", False)
        else None,
        "target_split_proportions": splits.get("target", [0.34, 0.33, 0.33]),
        "target_split_seed": target_split_seed,
        "reference_split_proportions": splits.get("reference", [0.5, 0.5]),
        "reference_split_seed": reference_split_seed,
        "reference_split_group_key": splits.get("reference_group_key"),
        "subset_percentages": [subset_percentage],
        "selectors": (
            as_list(
                (baseline_variant or {}).get(
                    "selectors",
                    (baseline_variant or {}).get(
                        "selector",
                        config.get("baseline_selectors", ["full", "random", "dsir", "less", "prismatic"]),
                    ),
                )
            )
            if run_kind == "baseline"
            else ["safe"]
        ),
        "seed": config.get("seed", 42),
        "model_name": model_variant["gradient_model"],
        "train_model_name": model_variant.get("finetune_model", model_variant["gradient_model"]),
        "output_dir": str(run_output),
        "selection_output_dir": str((shared_scope or run_output) / "subsets"),
        "training_output_dir": str(run_output / "training_runs"),
        "feature_cache_dir": str((shared_scope or run_output) / "selector_feature_cache"),
        "train_cache_dir": str((shared_scope or run_output) / "train_cache"),
    }
    base_args.update(selection)
    if run_kind == "safe" and safe_variant:
        base_args.update(
            {
                key: value
                for key, value in safe_variant.items()
                if key
                not in {
                    "name",
                    "rho_values",
                    "beta_values",
                    "rho_beta_values",
                    "gamma_values",
                    "epsilon",
                    "epsilon_values",
                    "epsilon_multiplier",
                    "epsilon_multipliers",
                    "reference_compositions",
                }
            }
        )
    if run_kind == "safe" and reference_composition is not None:
        reference_file = reference_composition.get("reference_file", reference_composition.get("file"))
        if reference_file is not None:
            base_args["reference_file"] = reference_file
        base_args["reference_composition_name"] = reference_composition.get("name")
        domains = reference_composition.get("domains", reference_composition.get("datasets"))
        if domains is not None:
            base_args["reference_composition_domains"] = domains
    base_args.update(training)
    base_args.update(evaluation)
    if run_kind == "baseline" and baseline_variant:
        base_args.update(dict(baseline_variant.get("training", {})))
        base_args.update(
            {
                key: value
                for key, value in baseline_variant.items()
                if key not in {"name", "selector", "selectors", "training"}
            }
        )
    base_args.update(rank_overrides(rank_spec))
    if run_kind == "safe":
        base_args["safe_alpha"] = "auto"
        if safe_epsilon_multiplier is not None:
            base_args["safe_epsilon_multiplier"] = safe_epsilon_multiplier
        else:
            epsilon = (
                config["safe"].get("epsilon")
                if safe_epsilon is None
                else safe_epsilon
            )
            base_args["safe_epsilon"] = epsilon
        if gamma is not None:
            base_args["safe_cost_gamma"] = gamma
        elif beta is None:
            base_args["safe_cost_c"] = rho
        else:
            base_args["safe_cost_beta"] = beta
    for key, value in base_args.items():
        add_flag(command, key, value)
    return run_name, command


def safe_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_variants = config.get("safe", {}).get("variants")
    if raw_variants is None:
        return [{"name": "safe"}]
    variants: list[dict[str, Any]] = []
    for item in as_list(raw_variants):
        if isinstance(item, dict):
            variants.append(dict(item))
        else:
            variants.append({"name": str(item)})
    if not variants:
        raise ValueError("safe.variants must not be empty when provided.")
    return variants


def baseline_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_variants = config.get("baseline_variants")
    if raw_variants is None:
        return [
            {
                "name": "baselines",
                "selectors": config.get(
                    "baseline_selectors",
                    ["full", "random", "dsir", "less", "prismatic"],
                ),
            }
        ]
    variants: list[dict[str, Any]] = []
    for item in as_list(raw_variants):
        if not isinstance(item, dict):
            raise ValueError("baseline_variants entries must be mappings.")
        variant = dict(item)
        if "selector" not in variant and "selectors" not in variant:
            raise ValueError("Each baseline_variants entry requires selector or selectors.")
        variant.setdefault(
            "name",
            safe_slug(variant.get("selector") or "_".join(variant.get("selectors", []))),
        )
        variants.append(variant)
    if not variants:
        raise ValueError("baseline_variants must not be empty when provided.")
    return variants


def safe_epsilon_specs(
    config: dict[str, Any],
    variant: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    safe_config = config.get("safe", {})
    variant = variant or {}
    raw_values = variant.get("epsilon_values", safe_config.get("epsilon_values"))
    raw_multipliers = variant.get(
        "epsilon_multipliers",
        safe_config.get("epsilon_multipliers"),
    )
    scalar_multiplier = variant.get(
        "epsilon_multiplier",
        safe_config.get("epsilon_multiplier"),
    )
    if raw_multipliers is None and scalar_multiplier is not None:
        raw_multipliers = [scalar_multiplier]
    if raw_values is None:
        scalar_epsilon = variant.get("epsilon", safe_config.get("epsilon"))
        if scalar_epsilon is not None:
            raw_values = [scalar_epsilon]
    if raw_values is not None and raw_multipliers is not None:
        raise ValueError(
            "Use epsilon/epsilon_values or epsilon_multiplier/epsilon_multipliers, "
            "not both."
        )
    if raw_multipliers is not None:
        values = [
            float(value)
            for value in as_list(raw_multipliers)
            if value is not None
        ]
        if not values:
            raise ValueError("epsilon_multipliers must not be empty.")
        return [
            {"epsilon_multiplier": value}
            for value in dict.fromkeys(values)
        ]
    values = [float(value) for value in as_list(raw_values) if value is not None]
    if not values:
        raise ValueError(
            "SAFE runs require epsilon values or epsilon multipliers in the config."
        )
    return [{"epsilon": value} for value in dict.fromkeys(values)]


def safe_rho_values(config: dict[str, Any], variant: dict[str, Any] | None = None) -> list[float]:
    safe_config = config.get("safe", {})
    variant = variant or {}
    raw_values = variant.get("rho_values", safe_config.get("rho_values"))
    return [float(value) for value in as_list(raw_values)]


def safe_beta_values(config: dict[str, Any], variant: dict[str, Any] | None = None) -> list[float]:
    safe_config = config.get("safe", {})
    variant = variant or {}
    raw_values = variant.get(
        "beta_values",
        variant.get("rho_beta_values", safe_config.get("beta_values", safe_config.get("rho_beta_values"))),
    )
    if raw_values is None:
        return []
    return [float(value) for value in as_list(raw_values)]


def safe_gamma_values(
    config: dict[str, Any],
    variant: dict[str, Any] | None = None,
) -> list[float]:
    safe_config = config.get("safe", {})
    variant = variant or {}
    raw_values = variant.get("gamma_values", safe_config.get("gamma_values"))
    if raw_values is None:
        return []
    return [float(value) for value in as_list(raw_values)]


def safe_budget_specs(config: dict[str, Any], variant: dict[str, Any] | None = None) -> list[dict[str, float]]:
    specs = [{"rho": rho} for rho in safe_rho_values(config, variant)]
    specs.extend({"beta": beta} for beta in safe_beta_values(config, variant))
    specs.extend({"gamma": gamma} for gamma in safe_gamma_values(config, variant))
    if not specs:
        raise ValueError(
            "SAFE runs require rho_values, beta_values, gamma_values, or "
            "per-variant equivalents."
        )
    return specs


def reference_compositions(
    config: dict[str, Any],
    variant: dict[str, Any] | None = None,
) -> list[dict[str, Any] | None]:
    variant = variant or {}
    raw_values = variant.get("reference_compositions", config.get("reference_compositions"))
    if raw_values is None:
        return [None]
    compositions: list[dict[str, Any] | None] = []
    for item in as_list(raw_values):
        if item is None:
            compositions.append(None)
        elif isinstance(item, dict):
            spec = dict(item)
            spec.setdefault("name", spec.get("label", spec.get("file", "reference")))
            compositions.append(spec)
        else:
            compositions.append({"name": str(item)})
    if not compositions:
        raise ValueError("reference_compositions must not be empty when provided.")
    return compositions


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
    prepare_commands: list[list[str]] = []
    repeats = repeat_specs(config)
    for repeat in repeats:
        repeat_config = config_for_repeat(config, repeat)
        shared_cache_experiment_id = repeat_config.get(
            "shared_cache_experiment_id"
        )
        for prepare_command in build_prepare_commands(repeat_config):
            if prepare_command not in prepare_commands:
                prepare_commands.append(prepare_command)
        model_variants = [variant for variant in repeat_config.get("model_variants", []) if variant.get("enabled", True)]
        variants = safe_variants(repeat_config)
        baseline_specs = baseline_variants(repeat_config)
        for model_variant in model_variants:
            for rank_spec in repeat_config["rank_grid"]:
                for subset_percentage in repeat_config["subset_percentages"]:
                    if repeat_config.get("run_baselines", True):
                        for baseline_spec in baseline_specs:
                            name, command = build_sweep_command(
                                repeat_config,
                                run_kind="baseline",
                                model_variant=model_variant,
                                rank_spec=rank_spec,
                                subset_percentage=float(subset_percentage),
                                rho=None,
                                baseline_variant=baseline_spec,
                            )
                            entry = {
                                "name": name,
                                "kind": "baseline",
                                "baseline_variant": str(
                                    baseline_spec.get("name", "baseline")
                                ),
                                "repeat_index": repeat["repeat_index"],
                                "repeat_seed": repeat["repeat_seed"],
                                "repeat_tag": repeat["repeat_tag"],
                                "command": command,
                            }
                            if shared_cache_experiment_id is not None:
                                entry["shared_cache_experiment_id"] = str(
                                    shared_cache_experiment_id
                                )
                            commands.append(entry)
                    for variant in variants:
                        epsilon_specs = safe_epsilon_specs(repeat_config, variant)
                        include_epsilon_in_name = (
                            len(epsilon_specs) > 1
                            or "epsilon_values" in variant
                            or "epsilon_values" in repeat_config.get("safe", {})
                            or "epsilon_multipliers" in variant
                            or "epsilon_multipliers" in repeat_config.get("safe", {})
                        )
                        for composition in reference_compositions(repeat_config, variant):
                            for epsilon_spec in epsilon_specs:
                                epsilon = epsilon_spec.get("epsilon")
                                epsilon_multiplier = epsilon_spec.get(
                                    "epsilon_multiplier"
                                )
                                for budget_spec in safe_budget_specs(repeat_config, variant):
                                    rho = budget_spec.get("rho")
                                    beta = budget_spec.get("beta")
                                    gamma = budget_spec.get("gamma")
                                    name, command = build_sweep_command(
                                        repeat_config,
                                        run_kind="safe",
                                        model_variant=model_variant,
                                        rank_spec=rank_spec,
                                        subset_percentage=float(subset_percentage),
                                        rho=None if rho is None else float(rho),
                                        beta=None if beta is None else float(beta),
                                        gamma=None if gamma is None else float(gamma),
                                        safe_variant=variant,
                                        safe_epsilon=(
                                            None
                                            if epsilon is None
                                            else float(epsilon)
                                        ),
                                        safe_epsilon_multiplier=(
                                            None
                                            if epsilon_multiplier is None
                                            else float(epsilon_multiplier)
                                        ),
                                        reference_composition=composition,
                                        include_safe_epsilon_in_name=include_epsilon_in_name,
                                    )
                                    entry = {
                                        "name": name,
                                        "kind": "safe",
                                        "variant": str(variant.get("name", "safe")),
                                        "rho": None if rho is None else float(rho),
                                        "beta": None if beta is None else float(beta),
                                        "gamma": (
                                            None
                                            if gamma is None
                                            else float(gamma)
                                        ),
                                        "epsilon": (
                                            None
                                            if epsilon is None
                                            else float(epsilon)
                                        ),
                                        "epsilon_multiplier": (
                                            None
                                            if epsilon_multiplier is None
                                            else float(epsilon_multiplier)
                                        ),
                                        "repeat_index": repeat["repeat_index"],
                                        "repeat_seed": repeat["repeat_seed"],
                                        "repeat_tag": repeat["repeat_tag"],
                                        "command": command,
                                    }
                                    if shared_cache_experiment_id is not None:
                                        entry["shared_cache_experiment_id"] = str(
                                            shared_cache_experiment_id
                                        )
                                    if composition is not None:
                                        entry["reference_composition"] = {
                                            "name": composition.get("name"),
                                            "domains": composition.get("domains", composition.get("datasets")),
                                            "reference_file": composition.get("reference_file", composition.get("file")),
                                        }
                                    commands.append(entry)
    return {
        "experiment": config["experiment"],
        "repeats": repeats,
        "prepare_commands": prepare_commands,
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
