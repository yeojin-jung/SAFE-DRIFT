from __future__ import annotations

import argparse
import ast
import csv
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from baseline_selectors.dsir import select_dsir
from baseline_selectors.less_selector import select_less
from baseline_selectors.prismatic_selector import select_prismatic
from baseline_selectors.random_selector import select_random
from evaluation import get_evaluator, humaneval as humaneval_evaluator
from evaluation.bias_disentangle import evaluate_base_model as evaluate_base_bias_metrics
from evaluation.entanglement_analysis import (
    analyze_entanglement,
    headline_metrics as entanglement_headline_metrics,
    selected_indices_from_records,
)
from utils.extract_gradients import (
    compute_projected_feature,
    compute_projected_reference_fisher,
    compute_projected_target_gradient,
    load_model_with_lora,
    project_preconditioner_count_sketch,
)
from utils.adam_selection import planned_learning_rate_sum
from utils.low_rank_fisher_builder import compute_low_rank_safe_inputs_from_examples
from utils.preconditioner import build_adam_preconditioner, estimate_adam_second_moment
from utils.safe_budget_calibration import (
    coupled_reference_budget,
    random_subset_update_norm_calibration,
)
from safe_drift.safe_drift_selector import select_safe_subset_from_gradients
from utils.data_construction import (
    load_records,
    load_stereoset_records,
    load_stereoset_triplet_records,
    split_records_by_proportions,
    to_instruction_output,
)
from utils.train_lora_sft import build_base_model_and_tokenizer

DEFAULT_TRAIN_SCRIPT = SRC_DIR / "utils" / "train_lora_sft.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "selector_sft"
DEFAULT_MODEL_NAME = "allenai/OLMo-2-1124-7B"
DEFAULT_CANDIDATE_FILE = REPO_ROOT / "data" / "setting_1_code" / "codealpaca.jsonl"
DEFAULT_TARGET_FILE = REPO_ROOT / "data" / "setting_1_code" / "humaneval.jsonl"
DEFAULT_OOD_EVAL_FILE = REPO_ROOT / "data" / "setting_1_code" / "humaneval.jsonl"
DEFAULT_REFERENCE_FILE = REPO_ROOT / "data" / "references" / "stereoset_reference.jsonl"


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_") or "value"


def parse_extra_eval_spec(spec: str) -> dict[str, str]:
    try:
        name, rest = str(spec).split("=", 1)
        evaluator, path = rest.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "Extra eval specs must use NAME=EVALUATOR:/path/to/file.jsonl syntax; "
            f"got {spec!r}."
        ) from exc
    name = safe_slug(name)
    evaluator = evaluator.strip()
    path = path.strip()
    if not name or not evaluator or not path:
        raise ValueError(f"Invalid extra eval spec: {spec!r}")
    return {"name": name, "evaluator": evaluator, "file": path}


def parse_extra_eval_specs(specs: list[str] | tuple[str, ...] | None) -> list[dict[str, str]]:
    return [parse_extra_eval_spec(spec) for spec in (specs or [])]


def parse_auto_float(value: str) -> float | None:
    lowered = str(value).strip().lower()
    if lowered in {"auto", "none", "derived"}:
        return None
    return float(value)


def parse_auto_int(value: str) -> int | None:
    lowered = str(value).strip().lower()
    if lowered in {"auto", "none", "derived"}:
        return None
    return int(value)


def file_fingerprint(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    stem = safe_slug(resolved.stem)[:24]
    digest = hashlib.sha1(f"{resolved}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_instruction_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    write_jsonl([to_instruction_output(record) for record in records], path)


def count_optional_records(path: str | Path | None) -> int | None:
    if path is None:
        return None
    return len(load_records(path))


def selection_count_from_percentage(num_candidates: int, subset_percentage: float) -> int:
    if not (0 < subset_percentage <= 100):
        raise ValueError("subset_percentage must be in (0, 100].")
    return min(num_candidates, max(1, math.ceil(num_candidates * subset_percentage / 100.0)))


def split_lookup(
    records: list[dict[str, Any]],
    *,
    proportions: list[float],
    seed: int,
    names: list[str],
    group_key: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if len(proportions) != len(names):
        raise ValueError(f"Expected {len(names)} proportions for split names {names}, got {len(proportions)}.")
    return {
        name: split
        for name, split in zip(
            names,
            split_records_by_proportions(records, proportions, seed=seed, shuffle=True, group_key=group_key),
            strict=True,
        )
    }


def prepare_candidate_pools(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Prepare candidate data and optional Adam warmup examples.

    The default selector pool is stable across Adam/SGD and SAFE
    hyperparameters. Legacy reproduction mode restores the original behavior:
    remove the deterministic warmup holdout, then subsample the selection pool.
    """
    legacy_disjoint = bool(args.legacy_disjoint_preconditioner_pool)
    split_info: dict[str, Any] = {
        "preconditioner_data_disjoint": legacy_disjoint,
        "preconditioner_candidate_count": 0,
        "selection_candidate_count": len(candidates),
        "preconditioner_candidate_file": None,
        "selection_candidate_file": str(Path(args.candidate_file).resolve()),
        "preconditioner_split_seed": None,
        "legacy_disjoint_preconditioner_pool": legacy_disjoint,
        "selection_candidate_max_examples": args.selection_candidate_max_examples,
        "selection_candidate_subsample_seed": args.selection_candidate_subsample_seed,
    }
    if (
        args.selector_preconditioner != "adam"
        and not legacy_disjoint
        and args.selection_candidate_max_examples is None
    ):
        setattr(args, "_preconditioner_candidates", candidates)
        return candidates, candidates, split_info

    if args.selector_preconditioner == "adam" or legacy_disjoint:
        if len(candidates) < 2:
            raise ValueError("Need at least 2 candidates to sample preconditioner warmup data.")
        holdout_size = max(1, math.ceil(len(candidates) * float(args.adam_warmup_steps)))
        holdout_size = min(holdout_size, len(candidates) - 1)
        indices = list(range(len(candidates)))
        random.Random(int(args.preconditioner_split_seed)).shuffle(indices)
        preconditioner_ids = set(indices[:holdout_size])
        preconditioner_candidates = [record for idx, record in enumerate(candidates) if idx in preconditioner_ids]
        selection_candidates = (
            [record for idx, record in enumerate(candidates) if idx not in preconditioner_ids]
            if legacy_disjoint
            else list(candidates)
        )
    else:
        preconditioner_candidates = list(candidates)
        selection_candidates = list(candidates)

    if args.selection_candidate_max_examples is not None:
        max_examples = int(args.selection_candidate_max_examples)
        if max_examples <= 0:
            raise ValueError("--selection-candidate-max-examples must be positive.")
        if len(selection_candidates) > max_examples:
            subsample_seed = (
                int(args.selection_candidate_subsample_seed)
                if args.selection_candidate_subsample_seed is not None
                else int(args.seed)
            )
            indices = list(range(len(selection_candidates)))
            random.Random(subsample_seed).shuffle(indices)
            selection_candidates = [selection_candidates[index] for index in indices[:max_examples]]

    pool_dir = Path(args.feature_cache_dir).resolve().parent / "candidate_pools"
    preconditioner_file = pool_dir / "preconditioner_warmup_candidates.jsonl"
    materialized_selection = legacy_disjoint or args.selection_candidate_max_examples is not None
    selection_file = pool_dir / (
        "selection_candidates.jsonl" if materialized_selection else "selection_candidates_full_pool.jsonl"
    )
    if args.overwrite_selection_cache or not preconditioner_file.exists():
        write_jsonl(preconditioner_candidates, preconditioner_file)
    if args.overwrite_selection_cache or not selection_file.exists():
        write_jsonl(selection_candidates, selection_file)

    setattr(args, "_preconditioner_candidates", preconditioner_candidates)
    setattr(args, "_preconditioner_candidate_file", str(preconditioner_file))
    setattr(
        args,
        "_selection_candidate_file",
        str(selection_file) if materialized_selection else str(Path(args.candidate_file).resolve()),
    )
    setattr(args, "_selection_candidate_inspection_file", str(selection_file))

    split_info.update(
        {
            "preconditioner_candidate_count": len(preconditioner_candidates),
            "selection_candidate_count": len(selection_candidates),
            "preconditioner_candidate_file": str(preconditioner_file.resolve()),
            "selection_candidate_file": str(selection_file.resolve()),
            "preconditioner_split_seed": int(args.preconditioner_split_seed),
            "preconditioner_warmup_excluded_from_selection": legacy_disjoint,
        }
    )
    return preconditioner_candidates, selection_candidates, split_info


def print_preconditioner_split_summary(split_info: dict[str, Any]) -> None:
    if not split_info.get("preconditioner_data_disjoint", False):
        if split_info.get("preconditioner_candidate_file"):
            print(
                "[candidate split] Adam warmup uses a deterministic candidate subset; "
                "selection keeps the full candidate pool "
                f"(warmup={split_info.get('preconditioner_candidate_count')}, "
                f"selection={split_info.get('selection_candidate_count')}, "
                f"seed={split_info.get('preconditioner_split_seed')})."
            )
            print(f"[candidate split] preconditioner file: {split_info.get('preconditioner_candidate_file')}")
            print(f"[candidate split] selection file:     {split_info.get('selection_candidate_file')}")
            return
        print(
            "[candidate split] preconditioner and selection share the same candidate pool "
            f"(selection={split_info.get('selection_candidate_count')})."
        )
        return

    print(
        "[candidate split] disjoint candidate pools: "
        f"preconditioner={split_info.get('preconditioner_candidate_count')} "
        f"selection={split_info.get('selection_candidate_count')} "
        f"(seed={split_info.get('preconditioner_split_seed')})."
    )
    print(f"[candidate split] preconditioner file: {split_info.get('preconditioner_candidate_file')}")
    print(f"[candidate split] selection file:     {split_info.get('selection_candidate_file')}")


def print_named_split_summary(label: str, split_map: dict[str, list[dict[str, Any]]]) -> None:
    summary = ", ".join(f"{name}={len(records)}" for name, records in split_map.items())
    print(f"[{label} split] {summary}")


def print_shared_feature_summary(info: dict[str, Any]) -> None:
    print(
        "[selector features] "
        f"method={info.get('selector_feature_method')} "
        f"cache_hit={info.get('selector_feature_cache_hit')} "
        f"runtime_cache_hit={info.get('selector_feature_runtime_cache_hit', False)}"
    )
    print(
        "[selector features] "
        f"candidate_shape={info.get('candidate_features_shape')} "
        f"target_shape={info.get('target_feature_shape')} "
        f"reference_fisher_shape={info.get('reference_fisher_shape')}"
    )
    print(f"[selector features] cache path: {info.get('selector_feature_cache_path')}")


def print_low_rank_plan(
    *,
    total_candidates: int,
    basis_candidates: int,
    projection_candidates: int,
    target_examples: int,
    reference_examples: int,
    reference_cap: int | None,
    target_cap: int | None,
    candidate_cap: int | None,
    projection_chunk_rows: int,
) -> None:
    print(
        "[low-rank] "
        f"reference={reference_examples}"
        + (f" (cap={reference_cap})" if reference_cap is not None else "")
        + f", target={target_examples}"
        + (f" (cap={target_cap})" if target_cap is not None else "")
        + f", candidates={total_candidates}"
        + (f" (basis cap={candidate_cap})" if candidate_cap is not None else "")
    )
    print(
        "[low-rank] "
        f"basis candidates={basis_candidates}, projection candidates={projection_candidates}, "
        f"projection chunk rows={projection_chunk_rows}"
    )


def print_low_rank_result(info: dict[str, Any]) -> None:
    print(
        "[low-rank] "
        f"resolved K_R={info.get('low_rank_resolved_reference_rank')} "
        f"K_T={info.get('low_rank_resolved_task_rank')} "
        f"K={info.get('low_rank_resolved_rank')}"
    )
    print(
        "[low-rank] "
        f"basis candidates={info.get('low_rank_basis_candidate_count')} "
        f"projection candidates={info.get('low_rank_projection_candidate_count')} "
        f"wallclock={info.get('low_rank_wallclock_seconds'):.2f}s"
    )


def selection_backbone_tag(args: argparse.Namespace) -> str:
    parts = [
        safe_slug(args.model_name),
        f"r{args.lora_r}",
        f"a{args.lora_alpha}",
        f"d{args.lora_dropout}",
        f"seq{args.max_seq_len}",
    ]
    return "_".join(parts)


def candidate_cache_tag(args: argparse.Namespace) -> str:
    candidate_file = getattr(args, "_selection_candidate_file", args.candidate_file)
    return "_".join([file_fingerprint(candidate_file), selection_backbone_tag(args)])


def preconditioner_data_cache_tag(args: argparse.Namespace) -> str:
    candidate_file = getattr(args, "_preconditioner_candidate_file", args.candidate_file)
    return "_".join([file_fingerprint(candidate_file), selection_backbone_tag(args)])


def target_cache_tag(args: argparse.Namespace) -> str:
    return "_".join(
        [
            file_fingerprint(args.target_file),
            f"tsplit{'-'.join(str(value) for value in args.target_split_proportions)}",
            f"tsplitseed{args.target_split_seed}",
            selection_backbone_tag(args),
        ]
    )


def reference_cache_prefix(args: argparse.Namespace) -> str:
    if args.reference_file is not None:
        reference_tag = file_fingerprint(args.reference_file)
    elif args.reference_hf_stereoset:
        reference_tag = "_".join(
            [
                "stereoset",
                safe_slug(args.reference_hf_subset),
                safe_slug(args.reference_hf_split),
                safe_slug(args.reference_hf_label),
            ]
        )
    else:
        raise ValueError("Either reference_file or reference_hf_stereoset is required to build a reference Fisher cache.")
    return "_".join(
        [
            reference_tag,
            f"refsplit{'-'.join(str(value) for value in args.reference_split_proportions)}",
            f"refsplitseed{args.reference_split_seed}",
            selection_backbone_tag(args),
        ]
    )


def optional_rank_tag(value: int | float | None, auto_label: str = "auto") -> str:
    return auto_label if value is None else safe_slug(f"{value:g}" if isinstance(value, float) else str(value))


@dataclass
class SharedSelectorFeatures:
    candidate_features: torch.Tensor
    target_feature: torch.Tensor
    target_features: torch.Tensor | None
    reference_fisher: torch.Tensor | None
    basis: torch.Tensor | None
    selector_preconditioner: torch.Tensor | None
    info: dict[str, Any]


def selector_feature_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference_file is not None or args.reference_hf_stereoset:
        reference_tag = reference_cache_prefix(args)
    else:
        reference_tag = "none"

    descriptor = {
        "candidate": candidate_cache_tag(args),
        "target": target_cache_tag(args),
        "reference": reference_tag,
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "selector_feature_method": args.selector_feature_method,
        "reference_max_examples": int(args.reference_fisher_max_examples),
        "K_R": args.low_rank_reference_rank,
        "delta": float(args.low_rank_delta),
        "max_reference_rank": args.low_rank_max_reference_rank,
        "reference_eigenvalue_shrinkage": args.low_rank_reference_shrinkage,
        "K_T": args.low_rank_task_rank,
        "K_common": args.low_rank_common_rank,
        "auto_task_rank": args.low_rank_auto_task_rank,
        "delta_task": args.low_rank_delta_task,
        "max_task_rank": args.low_rank_max_task_rank,
        "low_rank_builder_alpha": args.low_rank_builder_alpha,
        "safe_alpha": args.safe_alpha,
        "target_max_examples": args.low_rank_target_max_examples,
        "candidate_max_examples": args.low_rank_candidate_max_examples,
        "candidate_subset_seed": args.seed,
        "adam_selection_mode": args.adam_selection_mode,
    }
    if adam_preconditions_before_projection(args):
        descriptor.update(
            {
                "adam_preconditioner_candidate": preconditioner_data_cache_tag(args),
                "adam_beta2": args.adam_beta2,
                "adam_eps": args.adam_eps,
                "adam_warmup_steps": args.adam_warmup_steps,
            }
        )
    if args.selector_feature_method == "random_sketch":
        descriptor.update(
            {
                "projection_dim": args.selector_projection_dim,
                "projection_seed": args.selector_projection_seed,
                "projection_chunk_size": args.selector_projection_chunk_size,
                "projection_cache_dtype": args.selector_projection_cache_dtype,
            }
        )
    return descriptor


def selector_feature_cache_path(args: argparse.Namespace) -> Path:
    descriptor = selector_feature_descriptor(args)
    digest = hashlib.sha1(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    parts = [
        file_fingerprint(getattr(args, "_selection_candidate_file", args.candidate_file)),
        safe_slug(args.selector_feature_method),
    ]
    if args.selector_feature_method == "random_sketch":
        parts.extend(
            [
                f"proj{args.selector_projection_dim}",
                f"projseed{args.selector_projection_seed}",
            ]
        )
    else:
        parts.extend(
            [
                f"KR{optional_rank_tag(args.low_rank_reference_rank)}",
                f"delta{optional_rank_tag(float(args.low_rank_delta))}",
                f"KT{optional_rank_tag(args.low_rank_task_rank, auto_label='none')}",
                f"K{optional_rank_tag(args.low_rank_common_rank)}",
                f"builderalpha{optional_rank_tag(float(args.low_rank_builder_alpha))}",
            ]
        )
        if adam_preconditions_before_projection(args):
            parts.append("adam_precondition_before_projection")
    parts.append(f"refmax{args.reference_fisher_max_examples}")
    parts.extend(
        [
            digest,
            "selector_features",
        ]
    )
    return Path(args.feature_cache_dir) / f"{'_'.join(parts)}.pt"


def write_shared_feature_artifacts(
    args: argparse.Namespace,
    cache_path: Path,
    candidate_records: list[dict[str, Any]],
    candidate_features: torch.Tensor,
    target_feature: torch.Tensor,
    target_features: torch.Tensor | None,
    reference_fisher: torch.Tensor | None,
    basis: torch.Tensor | None,
    metadata: dict[str, Any],
    reference_evals_full: torch.Tensor | None = None,
    task_evals_full: torch.Tensor | None = None,
) -> None:
    artifact_root = Path(args.feature_cache_dir).resolve().parent
    features_dir = artifact_root / "features"
    fisher_dir = artifact_root / "fisher"
    spectra_dir = artifact_root / "spectra"
    metadata_dir = artifact_root / "metadata"
    features_dir.mkdir(parents=True, exist_ok=True)
    fisher_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    stem = cache_path.stem
    torch.save(candidate_features.cpu(), features_dir / f"{stem}_candidate_features.pt")
    torch.save(candidate_features.cpu(), features_dir / f"{stem}_projected_candidate_gradients.pt")
    torch.save(target_feature.cpu(), features_dir / f"{stem}_target_feature.pt")
    if target_features is not None:
        torch.save(target_features.cpu(), features_dir / f"{stem}_target_features.pt")
    if reference_fisher is not None:
        torch.save(reference_fisher.cpu(), fisher_dir / f"{stem}_reference_fisher.pt")
    if reference_evals_full is not None:
        torch.save(reference_evals_full.cpu(), spectra_dir / f"{stem}_reference_evals_full.pt")
    if task_evals_full is not None:
        torch.save(task_evals_full.cpu(), spectra_dir / f"{stem}_task_evals_full.pt")
    row_records = []
    for candidate_row, record in enumerate(candidate_records):
        row_records.append(
            {
                "candidate_row": candidate_row,
                "id": record.get("id"),
                "source": record.get("source"),
                "task": record.get("task"),
                "text_hash": record.get("text_hash"),
            }
        )
    row_map_path = metadata_dir / f"{stem}_candidate_gradient_rows.jsonl"
    write_jsonl(row_records, row_map_path)
    ordered_ids = "\n".join(str(row.get("id") or row["candidate_row"]) for row in row_records)
    artifact_metadata = {
        **metadata,
        "gradient_representation": "low_rank_projected_coordinates"
        if args.selector_feature_method == "low_rank"
        else "count_sketch_projected_coordinates",
        "gradient_representation_is_preconditioned_update": bool(
            adam_preconditions_before_projection(args)
        ),
        "projected_candidate_gradients_file": str(
            (features_dir / f"{stem}_projected_candidate_gradients.pt").resolve()
        ),
        "candidate_gradient_row_map_file": str(row_map_path.resolve()),
        "candidate_gradient_row_map_sha256": hashlib.sha256(ordered_ids.encode("utf-8")).hexdigest(),
        "projection_basis_embedded_in_cache": bool(basis is not None),
        "projection_basis_cache_file": str(cache_path.resolve()) if basis is not None else None,
    }
    (metadata_dir / f"{stem}_metadata.json").write_text(
        json.dumps(artifact_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def resolved_adam_warmup_steps(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> int:
    warmup_data = getattr(args, "_preconditioner_candidates", candidates)
    if getattr(args, "separate_preconditioner_data", True):
        return len(warmup_data)
    return max(1, math.ceil(len(candidates) * float(args.adam_warmup_steps)))


def adam_preconditions_before_projection(args: argparse.Namespace) -> bool:
    return (
        args.selector_preconditioner == "adam"
        and args.adam_selection_mode == "precondition_before_projection"
    )


def selector_preconditioner_tag(args: argparse.Namespace) -> str:
    if adam_preconditions_before_projection(args):
        return "adam_precondition_before_projection"
    return str(args.selector_preconditioner)


def selector_learning_rate(
    args: argparse.Namespace,
    subset_budget: int,
) -> tuple[float, dict[str, Any]]:
    if not adam_preconditions_before_projection(args):
        return float(args.safe_learning_rate), {
            "selector_learning_rate": float(args.safe_learning_rate),
            "selector_learning_rate_source": "safe_learning_rate",
        }
    if args.adam_selection_effective_learning_rate is not None:
        value = float(args.adam_selection_effective_learning_rate)
        if value <= 0.0:
            raise ValueError("--adam-selection-effective-learning-rate must be positive.")
        return value, {
            "selector_learning_rate": value,
            "selector_learning_rate_source": "explicit_adam_selection_effective_learning_rate",
        }
    value, metadata = planned_learning_rate_sum(args, subset_budget)
    if value <= 0.0:
        raise ValueError(
            "The planned learning-rate schedule has zero total scale. "
            "Set --adam-selection-effective-learning-rate explicitly."
        )
    return value, {
        **metadata,
        "selector_learning_rate": value,
        "selector_learning_rate_source": "planned_training_schedule_sum",
    }


def selector_preconditioner_cache_path(args: argparse.Namespace, warmup_steps: int) -> Path:
    descriptor = {
        "candidate": preconditioner_data_cache_tag(args),
        "selector_preconditioner": args.selector_preconditioner,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "adam_warmup_steps": args.adam_warmup_steps,
        "adam_warmup_steps_resolved": warmup_steps,
    }
    digest = hashlib.sha1(json.dumps(descriptor, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    parts = [
        file_fingerprint(getattr(args, "_preconditioner_candidate_file", args.candidate_file)),
        safe_slug(args.selector_preconditioner),
        f"warmup{warmup_steps}",
        f"beta2{safe_slug(str(args.adam_beta2))}",
        f"eps{safe_slug(str(args.adam_eps))}",
        digest,
        "preconditioner",
    ]
    return Path(args.feature_cache_dir) / "preconditioners" / f"{'_'.join(parts)}.pt"


@contextmanager
def exclusive_cache_lock(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def torch_save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def projected_dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported projected feature dtype: {dtype_name}")
    return mapping[key]


def build_selector_preconditioner(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if args.selector_preconditioner == "sgd":
        return None, {
            "selector_preconditioner": "sgd",
            "preconditioner_wallclock_seconds": 0.0,
            "adam_warmup_steps_resolved": None,
        }
    if args.selector_preconditioner != "adam":
        raise ValueError(f"Unsupported selector preconditioner: {args.selector_preconditioner}")

    warmup_data = getattr(args, "_preconditioner_candidates", candidates)
    if not warmup_data:
        raise ValueError("Adam preconditioner warmup data is empty.")
    warmup_steps = resolved_adam_warmup_steps(args, candidates)
    cache_path = selector_preconditioner_cache_path(args, warmup_steps)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_cache_lock(cache_path):
        if cache_path.exists() and not args.overwrite_selection_cache:
            payload = torch.load(cache_path, map_location="cpu")
            preconditioner = payload["preconditioner"].cpu()
            metadata = payload.get("metadata", {})
            return preconditioner, {
                **metadata,
                "selector_preconditioner": "adam",
                "preconditioner_cache_hit": True,
                "preconditioner_cache_path": str(cache_path),
                "preconditioner_wallclock_seconds": 0.0,
                "adam_warmup_steps_resolved": warmup_steps,
            }

        start = time.perf_counter()
        v_bar = estimate_adam_second_moment(
            model=model,
            tokenizer=tokenizer,
            warmup_data=warmup_data,
            beta2=args.adam_beta2,
            n_steps=warmup_steps,
            device=device,
            max_seq_len=args.max_seq_len,
            use_lora=True,
        )
        preconditioner = build_adam_preconditioner(v_bar, eps=args.adam_eps, eta=1.0).cpu()
        elapsed = time.perf_counter() - start
        metadata = {
            "selector_preconditioner": "adam",
            "preconditioner_cache_hit": False,
            "preconditioner_cache_path": str(cache_path),
            "preconditioner_wallclock_seconds": elapsed,
            "adam_warmup_steps_resolved": warmup_steps,
            "adam_warmup_pool_size": len(warmup_data),
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            "adam_warmup_steps": args.adam_warmup_steps,
        }
        torch_save_atomic(
            {
                "preconditioner": preconditioner.cpu(),
                "v_bar": v_bar.cpu(),
                "metadata": metadata,
            },
            cache_path,
        )
    return preconditioner, metadata


def projected_preconditioner_cache_key(info: dict[str, Any]) -> str | None:
    selector_feature_cache_path = info.get("selector_feature_cache_path")
    if not selector_feature_cache_path:
        return None
    return Path(selector_feature_cache_path).stem


def maybe_load_projected_preconditioner(info: dict[str, Any]) -> torch.Tensor | None:
    path_raw = info.get("preconditioner_cache_path")
    if not path_raw:
        return None
    path = Path(path_raw)
    if not path.exists():
        return None
    with exclusive_cache_lock(path):
        payload = torch.load(path, map_location="cpu")
    key = projected_preconditioner_cache_key(info)
    projected_dict = payload.get("projected_preconditioners")
    if isinstance(projected_dict, dict) and key is not None:
        projected = projected_dict.get(key)
        if projected is not None:
            return projected.cpu()
    projected = payload.get("projected_preconditioner")
    return None if projected is None else projected.cpu()


def save_projected_preconditioner(
    *,
    selector_feature_method: str,
    selector_feature_cache_path: Path,
    preconditioner_info: dict[str, Any],
    full_preconditioner: torch.Tensor | None,
    basis: torch.Tensor | None,
    projection_dim: int | None = None,
    projection_seed: int | None = None,
) -> torch.Tensor | None:
    if full_preconditioner is None:
        return None
    path_raw = preconditioner_info.get("preconditioner_cache_path")
    if not path_raw:
        return None
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = full_preconditioner.detach().to(dtype=torch.float32, device="cpu")
    if selector_feature_method == "low_rank":
        if basis is None:
            raise ValueError("low_rank projected preconditioner requires a basis.")
        U_K = basis.detach().to(dtype=torch.float32, device="cpu")
        if p.ndim != 1 or p.numel() != U_K.shape[0]:
            raise ValueError("full_preconditioner must be a 1D tensor matching the low-rank basis height.")
        projected = U_K.T @ (p.unsqueeze(1) * U_K)
        projected = 0.5 * (projected + projected.T)
        projected_basis = "low_rank_common_basis"
    elif selector_feature_method == "random_sketch":
        if projection_dim is None or projection_seed is None:
            raise ValueError("random_sketch projected preconditioner requires projection_dim and projection_seed.")
        projected = project_preconditioner_count_sketch(
            p,
            output_dim=int(projection_dim),
            seed=int(projection_seed),
            dtype=torch.float32,
        )
        projected_basis = "count_sketch"
    else:
        raise ValueError(f"Unsupported selector_feature_method: {selector_feature_method}")

    with exclusive_cache_lock(path):
        payload = torch.load(path, map_location="cpu") if path.exists() else {"metadata": dict(preconditioner_info)}
        metadata = payload.get("metadata", {})
        key = selector_feature_cache_path.stem
        projected_metadata = payload.get("projected_preconditioner_metadata")
        if not isinstance(projected_metadata, dict):
            projected_metadata = {}
        projected_store = payload.get("projected_preconditioners")
        if not isinstance(projected_store, dict):
            projected_store = {}
        metadata.update(
            {
                **preconditioner_info,
            }
        )
        payload["metadata"] = metadata
        projected_store[key] = projected.cpu()
        payload["projected_preconditioners"] = projected_store
        projected_metadata[key] = {
            "selector_feature_method": selector_feature_method,
            "selector_feature_cache_path": str(selector_feature_cache_path),
            "projected_preconditioner_shape": tuple(projected.shape),
            "projected_preconditioner_basis": projected_basis,
        }
        payload["projected_preconditioner_metadata"] = projected_metadata
        payload["projected_preconditioner"] = projected.cpu()
        torch_save_atomic(payload, path)
    return projected.cpu()


def compute_random_sketch_selector_features(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    output_dtype = projected_dtype_from_name(args.selector_projection_cache_dtype)
    start = time.perf_counter()
    candidate_rows = []
    for example in tqdm(candidates, desc="Computing shared random-sketch candidate features"):
        feature = compute_projected_feature(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=args.max_seq_len,
            output_dim=args.selector_projection_dim,
            projection_seed=args.selector_projection_seed,
            use_lora=True,
            chunk_size=args.selector_projection_chunk_size,
            output_dtype=output_dtype,
        )
        candidate_rows.append(feature.cpu())
    candidate_features = torch.stack(candidate_rows, dim=0)
    candidate_seconds = time.perf_counter() - start

    start = time.perf_counter()
    target_feature = compute_projected_target_gradient(
        model=model,
        tokenizer=tokenizer,
        target_exemplars=targets,
        device=device,
        max_seq_len=args.max_seq_len,
        output_dim=args.selector_projection_dim,
        projection_seed=args.selector_projection_seed,
        use_lora=True,
        chunk_size=args.selector_projection_chunk_size,
        output_dtype=torch.float32,
        normalize=False,
    ).cpu()
    target_seconds = time.perf_counter() - start

    reference_fisher = None
    fisher_seconds = 0.0
    if references:
        reference_subset = references[: max(1, min(len(references), int(args.reference_fisher_max_examples)))]
        start = time.perf_counter()
        reference_fisher = compute_projected_reference_fisher(
            model=model,
            tokenizer=tokenizer,
            reference_exemplars=reference_subset,
            device=device,
            max_seq_len=args.max_seq_len,
            output_dim=args.selector_projection_dim,
            projection_seed=args.selector_projection_seed,
            use_lora=True,
            chunk_size=args.selector_projection_chunk_size,
            output_dtype=torch.float32,
        ).cpu()
        fisher_seconds = time.perf_counter() - start

    return candidate_features, target_feature, reference_fisher, {
        "selector_feature_method": "random_sketch",
        "selector_feature_dim": int(candidate_features.shape[1]),
        "projected_dim": args.selector_projection_dim,
        "projected_seed": args.selector_projection_seed,
        "projected_cache_dtype": args.selector_projection_cache_dtype,
        "candidate_feature_wallclock_seconds": candidate_seconds,
        "target_feature_wallclock_seconds": target_seconds,
        "reference_fisher_wallclock_seconds": fisher_seconds,
        "reference_fisher_available": reference_fisher is not None,
        "low_rank_features_are_preconditioned": False,
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "reference_fisher_weighted": any(
            record.get("reference_fisher_weight") is not None for record in references
        ),
    }, None, None, None


def compute_low_rank_selector_features(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    model,
    tokenizer,
    device: torch.device,
    update_preconditioner: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, dict[str, Any], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not references:
        raise ValueError("--selector-feature-method low_rank requires reference samples.")
    if args.low_rank_task_rank is not None and int(args.low_rank_task_rank) <= 0:
        raise ValueError("--low-rank-task-rank must be positive when provided.")
    if args.low_rank_max_task_rank is not None and int(args.low_rank_max_task_rank) <= 0:
        raise ValueError("--low-rank-max-task-rank must be positive when provided.")
    if args.low_rank_reference_rank is not None and int(args.low_rank_reference_rank) <= 0:
        raise ValueError("--low-rank-reference-rank must be positive when provided.")
    if args.low_rank_common_rank is not None and int(args.low_rank_common_rank) <= 0:
        raise ValueError("--low-rank-common-rank must be positive when provided.")
    if args.low_rank_candidate_max_examples is not None and int(args.low_rank_candidate_max_examples) <= 0:
        raise ValueError("--low-rank-candidate-max-examples must be positive when provided.")

    low_rank_processed_candidate_count = (
        len(candidates)
        if args.low_rank_candidate_max_examples is None
        else min(len(candidates), int(args.low_rank_candidate_max_examples))
    )

    print_low_rank_plan(
        total_candidates=len(candidates),
        basis_candidates=low_rank_processed_candidate_count,
        projection_candidates=len(candidates),
        target_examples=len(targets),
        reference_examples=len(references),
        reference_cap=args.reference_fisher_max_examples,
        target_cap=args.low_rank_target_max_examples,
        candidate_cap=args.low_rank_candidate_max_examples,
        projection_chunk_rows=args.low_rank_candidate_projection_chunk_rows,
    )

    start = time.perf_counter()
    safe_inputs = compute_low_rank_safe_inputs_from_examples(
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        targets=targets,
        references=references,
        alpha=float(args.low_rank_builder_alpha),
        device=device,
        max_seq_len=args.max_seq_len,
        K_R=args.low_rank_reference_rank,
        delta=args.low_rank_delta,
        max_reference_rank=args.low_rank_max_reference_rank,
        reference_eigenvalue_shrinkage=args.low_rank_reference_shrinkage,
        K_T=args.low_rank_task_rank,
        K_common=args.low_rank_common_rank,
        auto_task_rank=args.low_rank_auto_task_rank,
        delta_task=args.low_rank_delta_task,
        max_task_rank=args.low_rank_max_task_rank,
        use_lora=True,
        max_reference_examples=args.reference_fisher_max_examples,
        max_target_examples=args.low_rank_target_max_examples,
        max_candidate_examples=args.low_rank_candidate_max_examples,
        projection_examples=candidates,
        show_progress=True,
        artifact_path=None,
        include_full_gradients=False,
        include_basis=False,
        save_target_features=bool(getattr(args, "save_all_target_features", False)),
        candidate_projection_chunk_rows=int(args.low_rank_candidate_projection_chunk_rows),
        update_preconditioner=update_preconditioner,
    )
    candidate_features = safe_inputs.candidate_features.cpu()
    target_feature = safe_inputs.target_feature.cpu()
    target_features = None if safe_inputs.target_features is None else safe_inputs.target_features.cpu()
    reference_fisher = safe_inputs.reference_fisher.cpu()
    feature_info = {
        "selector_feature_method": "low_rank",
        "selector_feature_dim": int(safe_inputs.common_basis.K),
        "low_rank_wallclock_seconds": time.perf_counter() - start,
        "low_rank_builder_alpha": float(args.low_rank_builder_alpha),
        "low_rank_reference_rank_requested": args.low_rank_reference_rank,
        "low_rank_task_rank_requested": args.low_rank_task_rank,
        "low_rank_common_rank_requested": args.low_rank_common_rank,
        "low_rank_auto_task_rank": bool(args.low_rank_auto_task_rank),
        "low_rank_delta_task": float(args.low_rank_delta_task),
        "low_rank_max_task_rank": args.low_rank_max_task_rank,
        "low_rank_delta": float(args.low_rank_delta),
        "low_rank_max_reference_rank": args.low_rank_max_reference_rank,
        "low_rank_reference_shrinkage": args.low_rank_reference_shrinkage,
        "low_rank_candidate_max_examples": args.low_rank_candidate_max_examples,
        "low_rank_basis_candidate_count": low_rank_processed_candidate_count,
        "low_rank_projection_candidate_count": len(candidates),
        "low_rank_resolved_rank": safe_inputs.common_basis.K,
        "low_rank_resolved_reference_rank": safe_inputs.reference_low_rank.K_R,
        "low_rank_reference_shrinkage_gamma": safe_inputs.reference_low_rank.shrinkage_gamma,
        "low_rank_reference_shrinkage_valid_count": None
        if safe_inputs.reference_low_rank.shrinkage_valid_mask is None
        else int(safe_inputs.reference_low_rank.shrinkage_valid_mask.sum().item()),
        "low_rank_resolved_task_rank": (
            None if safe_inputs.task_candidate_low_rank is None else safe_inputs.task_candidate_low_rank.K_T
        ),
        "reference_fisher_available": True,
        "low_rank_features_are_preconditioned": bool(update_preconditioner is not None),
        "low_rank_task_basis_uses_preconditioned_updates": bool(update_preconditioner is not None),
        "save_all_target_features": bool(getattr(args, "save_all_target_features", False)),
        "target_features_shape": None if target_features is None else tuple(target_features.shape),
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "reference_fisher_weighted": any(
            record.get("reference_fisher_weight") is not None for record in references
        ),
    }
    basis = safe_inputs.common_basis.U_K.cpu()
    reference_evals_full = safe_inputs.reference_low_rank.evals_full.cpu()
    task_evals_full = None if safe_inputs.task_candidate_low_rank is None else safe_inputs.task_candidate_low_rank.evals_full.cpu()
    print_low_rank_result(feature_info)
    del safe_inputs
    gc.collect()
    return (
        candidate_features,
        target_feature,
        target_features,
        reference_fisher,
        feature_info,
        basis,
        reference_evals_full,
        task_evals_full,
    )


def compute_or_load_shared_selector_features(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> SharedSelectorFeatures:
    if adam_preconditions_before_projection(args) and args.selector_feature_method != "low_rank":
        raise ValueError(
            "--adam-selection-mode precondition_before_projection currently requires "
            "--selector-feature-method low_rank."
        )
    cache_path = selector_feature_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[selector features] requested cache path: {cache_path}")
    if cache_path.exists() and not args.overwrite_selection_cache:
        payload = torch.load(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        info = {
            **metadata,
            "selector_preconditioner": args.selector_preconditioner,
            "selector_feature_cache_hit": True,
            "selector_feature_cache_path": str(cache_path),
            "selector_feature_wallclock_seconds": 0.0,
        }
        selector_preconditioner = None
        if args.selector_preconditioner == "adam" and not adam_preconditions_before_projection(args):
            basis = payload.get("basis")
            warmup_steps = resolved_adam_warmup_steps(args, candidates)
            preconditioner_cache = selector_preconditioner_cache_path(args, warmup_steps)
            device = torch.device("cpu")
            model = tokenizer = None
            if args.overwrite_selection_cache or not preconditioner_cache.exists():
                device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
                model, tokenizer = load_model_for_selection(args, device=device)
            preconditioner, preconditioner_info = build_selector_preconditioner(
                args,
                candidates,
                model,
                tokenizer,
                device,
            )
            selector_preconditioner = save_projected_preconditioner(
                selector_feature_method=args.selector_feature_method,
                selector_feature_cache_path=cache_path,
                preconditioner_info=preconditioner_info,
                full_preconditioner=preconditioner,
                basis=basis,
                projection_dim=args.selector_projection_dim if args.selector_feature_method == "random_sketch" else None,
                projection_seed=args.selector_projection_seed if args.selector_feature_method == "random_sketch" else None,
            )
            info.update(preconditioner_info)
            if model is not None:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return SharedSelectorFeatures(
            candidate_features=payload["candidate_features"],
            target_feature=payload["target_feature"],
            target_features=payload.get("target_features"),
            reference_fisher=payload.get("reference_fisher"),
            basis=payload.get("basis"),
            selector_preconditioner=selector_preconditioner,
            info=info,
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer = load_model_for_selection(args, device=device)
    preconditioner, preconditioner_info = build_selector_preconditioner(args, candidates, model, tokenizer, device)

    start = time.perf_counter()
    if args.selector_feature_method == "random_sketch":
        target_features = None
        candidate_features, target_feature, reference_fisher, feature_info, basis, reference_evals_full, task_evals_full = compute_random_sketch_selector_features(
            args=args,
            candidates=candidates,
            targets=targets,
            references=references,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    elif args.selector_feature_method == "low_rank":
        candidate_features, target_feature, target_features, reference_fisher, feature_info, basis, reference_evals_full, task_evals_full = compute_low_rank_selector_features(
            args=args,
            candidates=candidates,
            targets=targets,
            references=references,
            model=model,
            tokenizer=tokenizer,
            device=device,
            update_preconditioner=preconditioner if adam_preconditions_before_projection(args) else None,
        )
    else:
        raise ValueError(f"Unsupported selector feature method: {args.selector_feature_method}")

    projected_preconditioner = None
    if not adam_preconditions_before_projection(args):
        projected_preconditioner = save_projected_preconditioner(
            selector_feature_method=args.selector_feature_method,
            selector_feature_cache_path=cache_path,
            preconditioner_info=preconditioner_info,
            full_preconditioner=preconditioner,
            basis=basis,
            projection_dim=args.selector_projection_dim if args.selector_feature_method == "random_sketch" else None,
            projection_seed=args.selector_projection_seed if args.selector_feature_method == "random_sketch" else None,
        )

    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    reference_gradient_count = min(len(references), int(args.reference_fisher_max_examples))
    target_gradient_count = min(
        len(targets),
        len(targets)
        if args.low_rank_target_max_examples is None
        else int(args.low_rank_target_max_examples),
    )
    candidate_gradient_count = len(candidates)
    preconditioner_gradient_count = (
        resolved_adam_warmup_steps(args, candidates)
        if args.selector_preconditioner == "adam"
        else 0
    )
    estimated_selection_flops = (
        6.0
        * float(model_parameter_count)
        * float(args.max_seq_len)
        * float(
            reference_gradient_count
            + target_gradient_count
            + candidate_gradient_count
            + preconditioner_gradient_count
        )
    )
    feature_metadata = {
        **selector_feature_descriptor(args),
        **feature_info,
        "selector_feature_cache_path": str(cache_path),
        "candidate_features_shape": tuple(candidate_features.shape),
        "target_feature_shape": tuple(target_feature.shape),
        "target_features_shape": None if target_features is None else tuple(target_features.shape),
        "reference_fisher_shape": None if reference_fisher is None else tuple(reference_fisher.shape),
        "estimated_selection_flops": estimated_selection_flops,
        "estimated_selection_flops_method": (
            "6x_parameter_token_upper_bound_over_reference_target_candidate_and_adam_warmup_gradients"
        ),
        "selection_model_parameter_count": int(model_parameter_count),
        "selection_gradient_example_count": int(
            reference_gradient_count
            + target_gradient_count
            + candidate_gradient_count
            + preconditioner_gradient_count
        ),
    }
    metadata = {
        **feature_metadata,
        **preconditioner_info,
        "selector_preconditioner": args.selector_preconditioner,
        "selector_feature_cache_hit": False,
        "selector_feature_wallclock_seconds": time.perf_counter() - start,
    }
    print(
        "[selector features] saving cache with "
        f"candidate_shape={tuple(candidate_features.shape)} "
        f"target_shape={tuple(target_feature.shape)} "
        f"reference_fisher_shape={None if reference_fisher is None else tuple(reference_fisher.shape)}"
    )
    torch.save(
        {
            "candidate_features": candidate_features.cpu(),
            "target_feature": target_feature.cpu(),
            "target_features": None if target_features is None else target_features.cpu(),
            "reference_fisher": None if reference_fisher is None else reference_fisher.cpu(),
            "basis": None if basis is None else basis.cpu(),
            "metadata": metadata,
        },
        cache_path,
    )
    write_shared_feature_artifacts(
        args=args,
        cache_path=cache_path,
        candidate_records=candidates,
        candidate_features=candidate_features,
        target_feature=target_feature,
        target_features=target_features,
        reference_fisher=reference_fisher,
        basis=basis,
        metadata=metadata,
        reference_evals_full=reference_evals_full,
        task_evals_full=task_evals_full,
    )
    # Training runs in a subprocess and loads its own model. Release the
    # selection model first so both copies never occupy the same GPU.
    del model
    del preconditioner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[selector features] released selection model before training")
    return SharedSelectorFeatures(
        candidate_features=candidate_features.cpu(),
        target_feature=target_feature.cpu(),
        target_features=None if target_features is None else target_features.cpu(),
        reference_fisher=None if reference_fisher is None else reference_fisher.cpu(),
        basis=None if basis is None else basis.cpu(),
        selector_preconditioner=projected_preconditioner,
        info=metadata,
    )


def get_shared_selector_features(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> SharedSelectorFeatures:
    cache_key = str(selector_feature_cache_path(args))
    cached = getattr(args, "_shared_selector_features", None)
    cached_key = getattr(args, "_shared_selector_features_key", None)
    if cached is not None and cached_key == cache_key:
        features = cached
        info = {
            **features.info,
            "selector_feature_runtime_cache_hit": True,
            "selector_feature_cache_path": cache_key,
        }
        print_shared_feature_summary(info)
        entanglement_info = maybe_run_entanglement_analysis(args, features)
        if entanglement_info is not None:
            info["entanglement_analysis"] = entanglement_info
        return SharedSelectorFeatures(
            candidate_features=features.candidate_features,
            target_feature=features.target_feature,
            target_features=features.target_features,
            reference_fisher=features.reference_fisher,
            basis=features.basis,
            selector_preconditioner=features.selector_preconditioner,
            info=info,
        )

    features = compute_or_load_shared_selector_features(args, candidates, targets, references)
    print_shared_feature_summary(features.info)
    entanglement_info = maybe_run_entanglement_analysis(args, features)
    if entanglement_info is not None:
        features.info["entanglement_analysis"] = entanglement_info
    setattr(args, "_shared_selector_features", features)
    setattr(args, "_shared_selector_features_key", cache_key)
    return features


def entanglement_analysis_output_dir(args: argparse.Namespace) -> Path:
    if args.entanglement_output_dir:
        return Path(args.entanglement_output_dir)
    return Path(args.feature_cache_dir).resolve().parent / "entanglement_analysis"


def maybe_run_entanglement_analysis(
    args: argparse.Namespace,
    features: SharedSelectorFeatures,
) -> dict[str, Any] | None:
    if not args.run_entanglement_analysis:
        return None
    if args.selector_feature_method != "low_rank":
        return {
            "status": "skipped",
            "reason": "requires low_rank selector features",
        }

    cache_path_raw = features.info.get("selector_feature_cache_path")
    if not cache_path_raw:
        return {
            "status": "skipped",
            "reason": "selector feature cache path unavailable",
        }
    cache_path = Path(cache_path_raw).resolve()
    if not cache_path.exists():
        return {
            "status": "skipped",
            "reason": "selector feature cache missing",
            "selector_feature_cache_path": str(cache_path),
        }

    completed: dict[str, dict[str, Any]] = getattr(args, "_entanglement_analysis_results", {})
    cache_key = str(cache_path)
    if cache_key in completed:
        return completed[cache_key]

    from evaluation.entanglement_analysis import analyze_one, discover_feature_files

    cache_root = Path(args.feature_cache_dir).resolve().parent
    out_root = entanglement_analysis_output_dir(args).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        feature_files = discover_feature_files(cache_root)
    except FileNotFoundError:
        feature_files = {}
    tag = next((candidate_tag for candidate_tag, path in feature_files.items() if path.resolve() == cache_path), cache_path.stem)

    print(f"[entanglement] analyzing selector cache: {cache_path}")
    summary = analyze_one(tag, cache_path, out_root, selector_top_k=args.entanglement_selector_top_k)
    summary_path = out_root / "entanglement_summary.json"

    if summary is None:
        result = {
            "status": "skipped",
            "reason": "analysis returned no summary",
            "selector_feature_cache_path": str(cache_path),
            "output_dir": str((out_root / tag).resolve()),
            "summary_file": None,
            "tag": tag,
        }
    else:
        all_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        all_summary[tag] = summary
        summary_path.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        result = {
            "status": "completed",
            "selector_feature_cache_path": str(cache_path),
            "output_dir": str((out_root / tag).resolve()),
            "summary_file": str(summary_path.resolve()),
            "tag": tag,
        }
        print(f"[entanglement] wrote summary: {summary_path}")

    completed[cache_key] = result
    setattr(args, "_entanglement_analysis_results", completed)
    return result


def collect_entanglement_analysis_results(args: argparse.Namespace) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = getattr(args, "_entanglement_analysis_results", {})
    return list(completed.values())


def selector_update_preconditioner(args: argparse.Namespace, features: SharedSelectorFeatures) -> torch.Tensor | None:
    if args.selector_preconditioner != "adam":
        return None
    if adam_preconditions_before_projection(args):
        return None
    if features.selector_preconditioner is None:
        raise ValueError(
            "selector_preconditioner=adam requires a projected preconditioner. "
            "Rebuild the selector feature cache with --overwrite-selection-cache."
        )
    return features.selector_preconditioner


def subset_output_path(args: argparse.Namespace, selector_name: str, subset_budget: int) -> Path:
    budget_tag = f"pct{float(args.subset_percentage):g}_budget{subset_budget}"
    filename = (
        f"{file_fingerprint(getattr(args, '_selection_candidate_file', args.candidate_file))}_"
        f"{safe_slug(selector_name)}_{safe_slug(args.model_name)}_"
        f"{budget_tag}_seed{args.seed}.jsonl"
    )
    return Path(args.selection_output_dir) / filename


def selection_metadata_path(subset_path: Path) -> Path:
    return subset_path.with_suffix(subset_path.suffix + ".selection.json")


def load_model_for_selection(args: argparse.Namespace, device: torch.device):
    return load_model_with_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device=device,
        torch_dtype=args.selection_torch_dtype or args.train_torch_dtype,
    )


def build_subset_records(candidates: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [candidates[int(index)] for index in indices]


def build_weighted_safe_records(
    *,
    candidates: list[dict[str, Any]],
    indices: list[int],
    weights: list[float],
    subset_budget: int,
    average_by_budget: bool,
    solver: str,
) -> list[dict[str, Any]]:
    if len(indices) != len(weights):
        raise ValueError("SAFE selection indices and weights must have the same length.")
    support_size = len(indices)
    if support_size == 0:
        if not candidates:
            raise ValueError("Cannot materialize a no-op SAFE run from an empty candidate pool.")
        sentinel = dict(candidates[0])
        sentinel["safe_selection_weight"] = 0.0
        sentinel["safe_training_weight"] = 0.0
        sentinel["safe_selection_solver"] = str(solver)
        sentinel["safe_noop_sentinel"] = True
        return [sentinel]

    # DataLoader training averages over the materialized support. Rescale each
    # loss so that the average gradient matches the selector's fixed 1 / S
    # dictionary normalization (or its unnormalized sum variant).
    training_scale = (
        float(support_size) / float(subset_budget)
        if average_by_budget
        else float(support_size)
    )
    records: list[dict[str, Any]] = []
    for index, weight in zip(indices, weights, strict=True):
        record = dict(candidates[int(index)])
        record["safe_selection_weight"] = float(weight)
        record["safe_training_weight"] = float(weight) * training_scale
        record["safe_selection_solver"] = str(solver)
        records.append(record)
    return records


def write_tensor_if_present(tensor: torch.Tensor | None, path: Path) -> str | None:
    if tensor is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    return str(path.resolve())


def constraint_status(
    *,
    cost: float | None,
    budget: float | None,
    relative_tolerance: float = 1.0e-5,
    absolute_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    if cost is None or budget is None:
        return {
            "cost": cost,
            "budget": budget,
            "satisfied": None,
            "active": None,
            "gap": None,
            "relative_gap": None,
        }
    cost = float(cost)
    budget = float(budget)
    gap = budget - cost
    tolerance = max(float(absolute_tolerance), abs(budget) * float(relative_tolerance))
    return {
        "cost": cost,
        "budget": budget,
        "satisfied": bool(cost <= budget + tolerance),
        "active": bool(abs(gap) <= tolerance),
        "gap": float(gap),
        "relative_gap": float(gap / max(abs(budget), absolute_tolerance)),
    }


def safe_alpha_case(alpha_status: str | None) -> str:
    if alpha_status == "feasible":
        return "feasible_rho_and_epsilon_boundaries"
    if alpha_status == "too_strict":
        return "rho_budget_too_strict_alpha_min_reference_budget_violated"
    if alpha_status == "inactive":
        return "rho_budget_inactive_alpha_max_reference_budget_slack"
    if alpha_status:
        return f"alpha_status_{safe_slug(alpha_status)}"
    return "fixed_alpha_or_unreported"


def safe_constraint_report(
    *,
    result,
    rho: float | None,
    epsilon: float | None,
) -> dict[str, Any]:
    norm_budget = None if epsilon is None else 0.5 * float(epsilon) * float(epsilon)
    reference_budget = None if rho is None else float(rho)
    safe_update = result.safe_update
    alpha_status = safe_update.get("alpha_status")
    alpha_status_text = None if alpha_status is None else str(alpha_status)

    continuous_reference = constraint_status(
        cost=safe_update.get("reference_cost"),
        budget=reference_budget,
    )
    continuous_norm = constraint_status(
        cost=safe_update.get("norm_cost"),
        budget=norm_budget,
    )
    selected_reference = constraint_status(
        cost=result.reference_cost,
        budget=reference_budget,
    )
    selected_norm = constraint_status(
        cost=result.norm_cost,
        budget=norm_budget,
    )

    return {
        "case": safe_alpha_case(alpha_status_text),
        "alpha_status": alpha_status_text,
        "rho": reference_budget,
        "epsilon": None if epsilon is None else float(epsilon),
        "norm_budget": norm_budget,
        "continuous_update": {
            "reference": continuous_reference,
            "epsilon_norm": continuous_norm,
            "predicted_gain": safe_update.get("predicted_gain"),
            "lambda": safe_update.get("lambda"),
            "alpha": safe_update.get("alpha"),
            "beta": safe_update.get("beta"),
        },
        "selected_subset": {
            "reference": selected_reference,
            "epsilon_norm": selected_norm,
            "predicted_target_gain": result.predicted_target_gain,
        },
        "alpha_boundary": {
            "target_ratio": safe_update.get("alpha_target_ratio"),
            "achieved_ratio": safe_update.get("alpha_achieved_ratio"),
            "phi_min": safe_update.get("alpha_phi_min"),
            "phi_max": safe_update.get("alpha_phi_max"),
            "iterations": safe_update.get("alpha_iterations"),
        },
    }


def unconstrained_target_reference_cost(
    target_feature: torch.Tensor,
    reference_fisher: torch.Tensor,
    epsilon: float,
) -> float:
    target = target_feature.detach().float()
    fisher = reference_fisher.detach().float()
    norm_sq = float(torch.dot(target, target).item())
    if norm_sq <= 0.0:
        raise ValueError("Cannot derive beta-normalized SAFE rho from a zero target gradient.")
    fisher_quadratic = float(torch.dot(target, fisher * target).item())
    return 0.5 * float(epsilon) * float(epsilon) * fisher_quadratic / norm_sq


def resolve_safe_epsilon(
    args: argparse.Namespace,
    features: SharedSelectorFeatures,
    *,
    subset_budget: int,
    learning_rate: float,
) -> tuple[float | None, dict[str, Any]]:
    if args.safe_epsilon is not None and args.safe_epsilon_multiplier is not None:
        raise ValueError("Use either --safe-epsilon or --safe-epsilon-multiplier, not both.")
    if args.safe_epsilon_multiplier is None:
        return args.safe_epsilon, {
            "safe_configured_epsilon": args.safe_epsilon,
            "safe_epsilon_multiplier": None,
            "safe_epsilon_calibration_base": None,
        }

    calibration_seed = (
        int(args.safe_epsilon_calibration_seed)
        if args.safe_epsilon_calibration_seed is not None
        else int(args.seed)
    )
    epsilon0, calibration_info = random_subset_update_norm_calibration(
        features.candidate_features,
        subset_budget=subset_budget,
        learning_rate=learning_rate,
        preconditioner=selector_update_preconditioner(args, features),
        sample_count=int(args.safe_epsilon_calibration_samples),
        seed=calibration_seed,
    )
    multiplier = float(args.safe_epsilon_multiplier)
    if multiplier <= 0.0:
        raise ValueError("--safe-epsilon-multiplier must be positive.")
    resolved_epsilon = multiplier * epsilon0
    return resolved_epsilon, {
        "safe_configured_epsilon": None,
        "safe_epsilon_multiplier": multiplier,
        "safe_epsilon_calibration_base": epsilon0,
        **calibration_info,
    }


def resolve_safe_reference_budget(
    args: argparse.Namespace,
    features: SharedSelectorFeatures,
    *,
    epsilon: float | None,
) -> tuple[float | None, dict[str, Any]]:
    configured = [
        args.safe_cost_c is not None,
        args.safe_cost_beta is not None,
        args.safe_cost_gamma is not None,
    ]
    if sum(configured) > 1:
        raise ValueError(
            "Use exactly one of --safe-cost-c, --safe-cost-beta, or "
            "--safe-cost-gamma."
        )
    if args.safe_cost_beta is None and args.safe_cost_gamma is None:
        return args.safe_cost_c, {
            "safe_cost_beta": None,
            "safe_cost_gamma": None,
            "safe_unconstrained_target_reference_cost": None,
            "safe_effective_cost_c": args.safe_cost_c,
        }
    if epsilon is None:
        raise ValueError("Gamma/beta-normalized SAFE rho requires an epsilon budget.")
    if features.reference_fisher is None:
        raise ValueError("Gamma/beta-normalized SAFE rho requires a reference Fisher.")
    gamma = (
        float(args.safe_cost_gamma)
        if args.safe_cost_gamma is not None
        else float(args.safe_cost_beta)
    )
    effective_cost_c, coupled_info = coupled_reference_budget(
        target_gradient=features.target_feature,
        reference_fisher=features.reference_fisher,
        epsilon=float(epsilon),
        gamma=gamma,
        preconditioner=selector_update_preconditioner(args, features),
    )
    return effective_cost_c, {
        "safe_cost_beta": (
            None if args.safe_cost_beta is None else float(args.safe_cost_beta)
        ),
        "safe_unconstrained_target_reference_cost": (
            0.5 * coupled_info["safe_reference_curvature_q0"] * float(epsilon) ** 2
        ),
        **coupled_info,
    }


def ensure_random_subset(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    output_path = subset_output_path(args, "random", subset_budget)
    if output_path.exists() and not args.overwrite_subsets:
        return output_path, {"selector": "random", "cache_hit": True, "selection_wallclock_seconds": 0.0}

    start = time.perf_counter()
    indices = select_random(len(candidates), subset_budget=subset_budget, seed=args.seed)
    write_jsonl(build_subset_records(candidates, indices), output_path)
    return output_path, {
        "selector": "random",
        "cache_hit": False,
        "selection_wallclock_seconds": time.perf_counter() - start,
    }


def ensure_dsir_subset(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    output_path = subset_output_path(args, "dsir", subset_budget)
    if output_path.exists() and not args.overwrite_subsets:
        return output_path, {"selector": "dsir", "cache_hit": True, "selection_wallclock_seconds": 0.0}

    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as temp_dir:
        candidate_path = Path(temp_dir) / "candidate.jsonl"
        target_path = Path(temp_dir) / "target.jsonl"
        write_instruction_jsonl(candidates, candidate_path)
        write_instruction_jsonl(targets, target_path)
        indices = select_dsir(
            pool_jsonl_path=str(candidate_path),
            target_jsonl_path=str(target_path),
            k=subset_budget,
            num_buckets=args.num_buckets,
            q_sample_ratio=args.q_sample_ratio,
            q_seed=args.seed,
        )
    write_jsonl(build_subset_records(candidates, [int(index) for index in indices]), output_path)
    return output_path, {
        "selector": "dsir",
        "cache_hit": False,
        "selection_wallclock_seconds": time.perf_counter() - start,
    }


def ensure_less_subset(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    selector_name = f"less_{args.selector_feature_method}_{selector_preconditioner_tag(args)}"
    output_path = subset_output_path(args, selector_name, subset_budget)
    if output_path.exists() and not args.overwrite_subsets:
        return output_path, {"selector": selector_name, "cache_hit": True, "selection_wallclock_seconds": 0.0}

    features = get_shared_selector_features(args, candidates, targets, references)
    start = time.perf_counter()
    indices = select_less(
        candidate_gradients=features.candidate_features.float(),
        target_gradient=features.target_feature.float(),
        subset_budget=subset_budget,
        preconditioner=selector_update_preconditioner(args, features),
        similarity=args.less_similarity,
    )
    wallclock = time.perf_counter() - start
    write_jsonl(build_subset_records(candidates, indices), output_path)
    return output_path, {
        "selector": selector_name,
        "cache_hit": False,
        "selection_wallclock_seconds": wallclock,
        "less_similarity": args.less_similarity,
        **features.info,
    }


def ensure_prismatic_subset(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    selector_name = f"prismatic_{args.selector_feature_method}_{selector_preconditioner_tag(args)}"
    output_path = subset_output_path(args, selector_name, subset_budget)
    if output_path.exists() and not args.overwrite_subsets:
        return output_path, {"selector": selector_name, "cache_hit": True, "selection_wallclock_seconds": 0.0}

    features = get_shared_selector_features(args, candidates, targets, references)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.perf_counter()
    indices = select_prismatic(
        candidate_gradients=features.candidate_features.to(device).float(),
        subset_budget=subset_budget,
        cluster_ratio=args.cluster_ratio,
        sparsity=args.sparsity,
        num_iters=args.num_iters,
        method=args.method,
        seed=args.seed,
    )
    wallclock = time.perf_counter() - start
    write_jsonl(build_subset_records(candidates, [int(index) for index in indices]), output_path)
    return output_path, {
        "selector": selector_name,
        "cache_hit": False,
        "selection_wallclock_seconds": wallclock,
        **features.info,
    }


def ensure_safe_subset(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    effective_learning_rate, learning_rate_info = selector_learning_rate(args, subset_budget)
    selector_tag = (
        f"safe_{args.selector_feature_method}_{selector_preconditioner_tag(args)}_"
        f"{args.safe_geometry}_{args.safe_solver}_"
        f"eta{safe_slug(f'{effective_learning_rate:.8g}')}"
    )
    if args.safe_geometry == "reference":
        selector_tag = f"{selector_tag}_metricFplusAlpha"
    if args.safe_solver == "constrained_greedy":
        selector_tag = (
            f"{selector_tag}_swaps{int(args.safe_greedy_swap_passes)}_"
            f"feastol{safe_slug(f'{args.safe_feasibility_tolerance:g}')}"
        )
    elif args.safe_solver == "relaxed":
        selector_tag = (
            f"{selector_tag}_{safe_slug(args.safe_relaxed_cvx_solver.lower())}_"
            f"wthresh{safe_slug(f'{args.safe_relaxed_weight_threshold:g}')}"
        )
    if args.safe_alpha is None:
        if (
            args.safe_epsilon is None
            and args.safe_epsilon_multiplier is None
        ) or (
            args.safe_cost_c is None
            and args.safe_cost_beta is None
            and args.safe_cost_gamma is None
        ):
            raise ValueError(
                "--safe-alpha auto requires an epsilon or epsilon multiplier and "
                "one of --safe-cost-c, --safe-cost-beta, or --safe-cost-gamma."
            )
        if sum(
            value is not None
            for value in (
                args.safe_cost_c,
                args.safe_cost_beta,
                args.safe_cost_gamma,
            )
        ) > 1:
            raise ValueError(
                "Use exactly one of --safe-cost-c, --safe-cost-beta, or "
                "--safe-cost-gamma."
            )
        if args.safe_cost_gamma is not None:
            cost_tag = f"gamma{safe_slug(f'{args.safe_cost_gamma:g}')}"
        elif args.safe_cost_beta is not None:
            cost_tag = f"beta{safe_slug(f'{args.safe_cost_beta:g}')}"
        else:
            cost_tag = f"cost{safe_slug(f'{args.safe_cost_c:g}')}"
        epsilon_tag = (
            (
                f"epsm{safe_slug(f'{args.safe_epsilon_multiplier:g}')}_"
                f"caln{int(args.safe_epsilon_calibration_samples)}_"
                f"calseed{int(args.safe_epsilon_calibration_seed if args.safe_epsilon_calibration_seed is not None else args.seed)}"
            )
            if args.safe_epsilon_multiplier is not None
            else f"eps{safe_slug(f'{args.safe_epsilon:g}')}"
        )
        selector_tag = (
            f"{selector_tag}_{cost_tag}_"
            f"{epsilon_tag}"
        )
    output_path = subset_output_path(args, selector_tag, subset_budget)
    metadata_path = selection_metadata_path(output_path)
    output_exists = output_path.exists()
    if output_exists and not args.overwrite_subsets and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["cache_hit"] = True
        metadata["selection_wallclock_seconds"] = 0.0
        return output_path, metadata
    if not references:
        raise ValueError("safe selector requires reference samples from --reference-file or --reference-hf-stereoset.")

    features = get_shared_selector_features(args, candidates, targets, references)
    if features.reference_fisher is None:
        raise ValueError("safe selector requires reference_fisher in the shared selector feature cache.")
    resolved_epsilon, epsilon_info = resolve_safe_epsilon(
        args,
        features,
        subset_budget=subset_budget,
        learning_rate=effective_learning_rate,
    )
    effective_cost_c, budget_info = resolve_safe_reference_budget(
        args,
        features,
        epsilon=resolved_epsilon,
    )

    start = time.perf_counter()
    result = select_safe_subset_from_gradients(
        candidate_gradients=features.candidate_features.float(),
        candidates=candidates,
        target_gradient=features.target_feature.float(),
        fisher=features.reference_fisher.float(),
        subset_budget=subset_budget,
        alpha=args.safe_alpha,
        learning_rate=effective_learning_rate,
        cost_c=effective_cost_c,
        epsilon=resolved_epsilon,
        geometry=args.safe_geometry,
        solver=args.safe_solver,
        shortlist_size=args.safe_shortlist_size,
        average_by_budget=args.safe_average_by_budget,
        preconditioner=selector_update_preconditioner(args, features),
        feasibility_tolerance=args.safe_feasibility_tolerance,
        objective_tolerance=args.safe_objective_tolerance,
        greedy_swap_passes=args.safe_greedy_swap_passes,
        relaxed_cvx_solver=args.safe_relaxed_cvx_solver,
        relaxed_weight_threshold=args.safe_relaxed_weight_threshold,
    )
    wallclock = time.perf_counter() - start
    optimization_trace_path = output_path.with_suffix(output_path.suffix + ".optimization_trace.csv")
    if args.overwrite_subsets or not output_exists:
        weighted_records = build_weighted_safe_records(
            candidates=candidates,
            indices=result.selected_indices,
            weights=result.selection_weights,
            subset_budget=subset_budget,
            average_by_budget=result.average_by_budget,
            solver=result.solver,
        )
        write_jsonl(weighted_records, output_path)
        write_csv(result.optimization_trace, optimization_trace_path)
    constraint_report = safe_constraint_report(
        result=result,
        rho=effective_cost_c,
        epsilon=resolved_epsilon,
    )
    selection_info = {
        "selector": selector_tag,
        "cache_hit": bool(output_exists and not args.overwrite_subsets),
        "selection_wallclock_seconds": wallclock,
        "safe_geometry": result.geometry,
        "safe_solver": result.solver,
        "safe_average_by_budget": result.average_by_budget,
        "safe_shortlist_size": result.shortlist_size,
        "safe_selected_count": len(result.selected_indices),
        "safe_is_noop": not result.selected_indices,
        "safe_selection_weight_sum": float(sum(result.selection_weights)),
        "safe_selection_weight_min": (
            float(min(result.selection_weights)) if result.selection_weights else None
        ),
        "safe_selection_weight_max": (
            float(max(result.selection_weights)) if result.selection_weights else None
        ),
        "safe_solver_status": result.solver_status,
        "safe_optimization_trace": result.optimization_trace,
        "safe_optimization_trace_file": str(optimization_trace_path.resolve()),
        "safe_objective_value": result.objective_value,
        "safe_reference_cost": result.reference_cost,
        "safe_norm_cost": result.norm_cost,
        "safe_predicted_target_gain": result.predicted_target_gain,
        "safe_alpha": result.alpha,
        "safe_cost_c": effective_cost_c,
        "safe_configured_cost_c": args.safe_cost_c,
        "safe_epsilon": resolved_epsilon,
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "adam_selection_mode": args.adam_selection_mode,
        **learning_rate_info,
        "safe_target_update_reference_cost": result.safe_update.get("reference_cost"),
        "safe_target_update_norm_cost": result.safe_update.get("norm_cost"),
        "safe_alpha_status": result.safe_update.get("alpha_status"),
        "safe_alpha_target_ratio": result.safe_update.get("alpha_target_ratio"),
        "safe_alpha_achieved_ratio": result.safe_update.get("alpha_achieved_ratio"),
        "safe_alpha_phi_min": result.safe_update.get("alpha_phi_min"),
        "safe_alpha_phi_max": result.safe_update.get("alpha_phi_max"),
        "safe_alpha_iterations": result.safe_update.get("alpha_iterations"),
        "safe_constraint_case": constraint_report["case"],
        "safe_continuous_reference_budget_satisfied": constraint_report["continuous_update"]["reference"]["satisfied"],
        "safe_continuous_reference_budget_active": constraint_report["continuous_update"]["reference"]["active"],
        "safe_continuous_reference_budget_gap": constraint_report["continuous_update"]["reference"]["gap"],
        "safe_continuous_epsilon_budget_satisfied": constraint_report["continuous_update"]["epsilon_norm"]["satisfied"],
        "safe_continuous_epsilon_budget_active": constraint_report["continuous_update"]["epsilon_norm"]["active"],
        "safe_continuous_epsilon_budget_gap": constraint_report["continuous_update"]["epsilon_norm"]["gap"],
        "safe_selected_reference_budget_satisfied": constraint_report["selected_subset"]["reference"]["satisfied"],
        "safe_selected_reference_budget_active": constraint_report["selected_subset"]["reference"]["active"],
        "safe_selected_reference_budget_gap": constraint_report["selected_subset"]["reference"]["gap"],
        "safe_selected_epsilon_budget_satisfied": constraint_report["selected_subset"]["epsilon_norm"]["satisfied"],
        "safe_selected_epsilon_budget_active": constraint_report["selected_subset"]["epsilon_norm"]["active"],
        "safe_selected_epsilon_budget_gap": constraint_report["selected_subset"]["epsilon_norm"]["gap"],
        "safe_constraint_report": constraint_report,
        **epsilon_info,
        **budget_info,
        **features.info,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(selection_info, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path, selection_info


def save_selected_artifacts_for_run(
    *,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    train_file: str | Path,
    run_output_dir: Path,
    selector: str,
    method_tag: str,
    selection_info: dict[str, Any],
) -> dict[str, Any] | None:
    if not args.save_selected_artifacts:
        return None
    if args.dry_run:
        return {"enabled": True, "skipped": True, "reason": "dry_run"}

    if selector == "full":
        selected_indices = list(range(len(candidates)))
        selected_records = list(candidates)
    elif selection_info.get("safe_is_noop"):
        selected_records = []
        selected_indices = []
    else:
        selected_records = load_records(train_file)
        selected_indices = selected_indices_from_records(candidates, selected_records)

    artifact_dir = run_output_dir / "selection_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    selected_candidates_path = artifact_dir / "selected_candidates.jsonl"
    write_jsonl(selected_records, selected_candidates_path)

    selected_indices_path = artifact_dir / "selected_candidate_indices.json"
    selected_indices_path.write_text(
        json.dumps(
            {
                "selector": selector,
                "method_tag": method_tag,
                "train_file": str(Path(train_file).resolve()),
                "candidate_file": str(Path(args.candidate_file).resolve()),
                "indices": [int(index) for index in selected_indices],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    selected_row_records = [
        {
            "selected_rank": selected_rank,
            "candidate_row": int(candidate_index),
            "id": candidates[int(candidate_index)].get("id"),
            "source": candidates[int(candidate_index)].get("source"),
            "task": candidates[int(candidate_index)].get("task"),
            "text_hash": candidates[int(candidate_index)].get("text_hash"),
            "safe_selection_weight": selected_records[selected_rank - 1].get("safe_selection_weight"),
            "safe_training_weight": selected_records[selected_rank - 1].get("safe_training_weight"),
        }
        for selected_rank, candidate_index in enumerate(selected_indices, start=1)
    ]
    selected_rows_path = artifact_dir / "selected_candidate_gradient_rows.jsonl"
    write_jsonl(selected_row_records, selected_rows_path)
    selected_row_ids = "\n".join(
        str(row.get("id") or row["candidate_row"]) for row in selected_row_records
    )

    info: dict[str, Any] = {
        "enabled": True,
        "skipped": False,
        "artifact_dir": str(artifact_dir.resolve()),
        "selected_candidates_file": str(selected_candidates_path.resolve()),
        "selected_candidate_indices_file": str(selected_indices_path.resolve()),
        "selected_candidate_gradient_row_map_file": str(selected_rows_path.resolve()),
        "selected_candidate_gradient_row_map_sha256": hashlib.sha256(
            selected_row_ids.encode("utf-8")
        ).hexdigest(),
        "selected_count": len(selected_indices),
        "candidate_count": len(candidates),
        "feature_cache_path": selection_info.get("selector_feature_cache_path"),
        "selector_feature_method": args.selector_feature_method,
        "selector_preconditioner": args.selector_preconditioner,
    }

    if references:
        features = get_shared_selector_features(args, candidates, targets, references)
        selected_tensor = torch.tensor(selected_indices, dtype=torch.long)
        selected_features = features.candidate_features[selected_tensor].detach().cpu()
        selected_features_path = artifact_dir / "selected_candidate_features.pt"
        torch.save(selected_features, selected_features_path)
        selected_gradients_path = artifact_dir / "selected_candidate_gradients.pt"
        torch.save(selected_features, selected_gradients_path)

        info.update(
            {
                "selected_candidate_features_file": str(selected_features_path.resolve()),
                "selected_candidate_gradients_file": str(selected_gradients_path.resolve()),
                "selected_candidate_features_shape": tuple(selected_features.shape),
                "gradient_representation": (
                    "low_rank_preconditioned_update_coordinates"
                    if adam_preconditions_before_projection(args)
                    else "low_rank_projected_coordinates"
                    if args.selector_feature_method == "low_rank"
                    else "count_sketch_projected_coordinates"
                ),
                "target_feature_file": write_tensor_if_present(features.target_feature, artifact_dir / "target_feature.pt"),
                "target_features_file": write_tensor_if_present(features.target_features, artifact_dir / "target_features.pt"),
                "reference_fisher_file": write_tensor_if_present(features.reference_fisher, artifact_dir / "reference_fisher.pt"),
                "selector_preconditioner_file": write_tensor_if_present(
                    features.selector_preconditioner,
                    artifact_dir / "selector_preconditioner.pt",
                ),
                "feature_cache_path": features.info.get("selector_feature_cache_path"),
                "projection_basis_embedded_in_feature_cache": bool(features.basis is not None),
                "selector_feature_dim": features.info.get("selector_feature_dim"),
                "reference_fisher_shape": features.info.get("reference_fisher_shape"),
            }
        )
        metadata_path = artifact_dir / "selector_feature_metadata.json"
        metadata_path.write_text(json.dumps(features.info, indent=2, ensure_ascii=False), encoding="utf-8")
        info["selector_feature_metadata_file"] = str(metadata_path.resolve())
    else:
        info.update(
            {
                "selected_candidate_features_file": None,
                "selected_candidate_gradients_file": None,
                "reason": "no reference records available to build selector feature cache",
            }
        )
    return info


def compute_entanglement_for_run(
    *,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    train_file: str | Path,
    run_output_dir: Path,
    selector: str,
    method_tag: str,
) -> dict[str, Any] | None:
    if not args.compute_entanglement_metrics:
        return None
    if args.dry_run:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "dry_run",
        }
    if args.selector_feature_method != "low_rank":
        return {
            "enabled": False,
            "skipped": True,
            "reason": "entanglement metrics require selector_feature_method=low_rank",
        }
    if not references:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "entanglement metrics require reference records",
        }

    features = get_shared_selector_features(args, candidates, targets, references)
    if features.reference_fisher is None:
        return {
            "enabled": False,
            "skipped": True,
            "reason": "shared selector features do not include reference_fisher",
        }

    if selector == "full":
        selected_indices = list(range(len(candidates)))
    else:
        selected_records = load_records(train_file)
        selected_indices = selected_indices_from_records(candidates, selected_records)

    out_dir = run_output_dir / "entanglement"
    summary = analyze_entanglement(
        candidate_features=features.candidate_features,
        target_feature=features.target_feature,
        target_features=features.target_features,
        reference_fisher=features.reference_fisher,
        selector_preconditioner=features.selector_preconditioner,
        selected_indices=selected_indices,
        output_dir=out_dir,
        metadata=features.info,
        reference_rank=features.info.get("low_rank_resolved_reference_rank"),
        target_sample_size=args.entanglement_target_sample_size,
        seed=args.seed,
        selected_csv_max_rows=args.entanglement_selected_csv_max_rows,
        label=method_tag,
    )
    info = {
        "enabled": True,
        "skipped": False,
        **entanglement_headline_metrics(summary),
    }
    return info


def merge_entanglement_into_training_summary(run_output_dir: Path, entanglement_info: dict[str, Any] | None) -> None:
    if not entanglement_info:
        return
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    summary["entanglement"] = entanglement_info
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_selection_artifacts_into_training_summary(
    run_output_dir: Path,
    selection_artifacts: dict[str, Any] | None,
) -> None:
    if not selection_artifacts:
        return
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    summary["selection_artifacts"] = selection_artifacts
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_selection_info_into_training_summary(
    run_output_dir: Path,
    selection_info: dict[str, Any],
) -> None:
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    summary["selection"] = selection_info
    selection_flops = selection_info.get("estimated_selection_flops")
    if selection_flops is not None:
        summary["estimated_selection_flops"] = float(selection_flops)
        training_flops = summary.get("estimated_total_flops")
        if training_flops is not None:
            summary["estimated_total_flops_including_selection"] = (
                float(training_flops) + float(selection_flops)
            )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_update_trajectory_plot(run_output_dir: Path) -> dict[str, Any]:
    plot_script = REPO_ROOT / "scripts" / "plot_update_constraint_trajectory.py"
    mpl_config_dir = run_output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    completed = subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--run-dir",
            str(run_output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        return {
            "generated": False,
            "return_code": int(completed.returncode),
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result = {"stdout": completed.stdout.strip()}
    return {"generated": True, **result}


def merge_update_trajectory_plot_into_summary(
    run_output_dir: Path,
    plot_info: dict[str, Any] | None,
) -> None:
    if not plot_info:
        return
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["update_constraint_trajectory_plot"] = plot_info
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_base_model_eval_into_training_summary(
    run_output_dir: Path,
    base_model_eval: dict[str, Any] | None,
) -> None:
    if not base_model_eval:
        return
    summary_path = run_output_dir / "summary.json"
    if not summary_path.exists():
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    summary["base_model_eval"] = base_model_eval
    summary.update(
        {
            key: value
            for key, value in base_model_eval.items()
            if key.startswith("base_") and isinstance(value, (int, float, str, bool))
        }
    )
    for key, value in list(summary.items()):
        if key.startswith("reference_"):
            prefix = "reference_"
            base_prefix = "base_reference_"
            delta_prefix = "reference_delta_"
        elif key.startswith("ood_"):
            prefix = "ood_"
            base_prefix = "base_ood_"
            delta_prefix = "ood_delta_"
        elif key.startswith("target_"):
            prefix = "target_"
            base_prefix = "base_target_"
            delta_prefix = "target_delta_"
        elif key.startswith("extra_eval_"):
            prefix = "extra_eval_"
            base_prefix = "base_extra_eval_"
            delta_prefix = "extra_eval_delta_"
        else:
            continue
        metric_name = key.removeprefix(prefix)
        base_key = f"{base_prefix}{metric_name}"
        if (
            base_key in summary
            and isinstance(value, (int, float))
            and isinstance(summary[base_key], (int, float))
        ):
            summary[f"{delta_prefix}{metric_name}"] = float(value) - float(summary[base_key])
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_train_command(
    args: argparse.Namespace,
    train_file: str | Path,
    run_output_dir: Path,
    *,
    selector: str,
    selection_info: dict[str, Any],
) -> list[str]:
    train_script = Path(args.train_script)
    train_lora_target_modules = args.train_lora_target_modules or args.lora_target_modules
    train_model_name = args.train_model_name or args.model_name
    validation_file = args.validation_file or args.target_file
    eval_file = args.eval_file or args.target_file
    command = []
    if args.launcher == "accelerate":
        command.extend(["accelerate", "launch", "--num_processes", str(args.num_processes)])
    else:
        command.append(sys.executable)

    command.extend(
        [
            str(train_script),
            "--model-name-or-path",
            train_model_name,
            "--train-file",
            str(train_file),
            "--validation-file",
            str(validation_file),
            "--eval-file",
            str(eval_file),
            "--ood-eval-file",
            str(args.ood_eval_file),
            "--output-dir",
            str(run_output_dir),
            "--cache-dir",
            str(args.train_cache_dir),
            "--max-seq-length",
            str(args.train_max_seq_length),
            "--torch-dtype",
            args.train_torch_dtype,
            "--lora-r",
            str(args.train_lora_r),
            "--lora-alpha",
            str(args.train_lora_alpha),
            "--lora-dropout",
            str(args.train_lora_dropout),
            "--lora-target-modules",
            *list(train_lora_target_modules),
            "--per-device-train-batch-size",
            str(args.per_device_train_batch_size),
            "--per-device-eval-batch-size",
            str(args.per_device_eval_batch_size),
            "--gradient-accumulation-steps",
            str(args.gradient_accumulation_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--warmup-ratio",
            str(args.warmup_ratio),
            "--lr-scheduler-type",
            args.lr_scheduler_type,
            "--seed",
            str(args.seed),
            "--logging-steps",
            str(args.logging_steps),
            "--eval-steps",
            str(args.eval_steps),
            "--evaluator",
            args.evaluator,
            "--generation-max-new-tokens",
            str(args.generation_max_new_tokens),
            "--humaneval-num-samples",
            str(args.humaneval_num_samples),
            "--humaneval-pass-at-ks",
            *(str(value) for value in args.humaneval_pass_at_ks),
            "--humaneval-temperature",
            str(args.humaneval_temperature),
            "--humaneval-top-p",
            str(args.humaneval_top_p),
            "--max-grad-norm",
            str(args.max_grad_norm),
            "--kl-regularization-lambda",
            str(args.kl_regularization_lambda),
            "--kl-reference-max-examples",
            str(args.kl_reference_max_examples),
            "--trajectory-eval-steps",
            str(args.trajectory_eval_steps),
            "--trajectory-reference-max-examples",
            str(args.trajectory_reference_max_examples),
            "--trajectory-target-max-examples",
            str(args.trajectory_target_max_examples),
            "--trajectory-generation-eval-steps",
            str(args.trajectory_generation_eval_steps),
            "--trajectory-generation-max-examples",
            str(args.trajectory_generation_max_examples),
            "--trajectory-target-evaluator",
            str(args.trajectory_target_evaluator or args.evaluator),
            "--trajectory-reference-evaluator",
            str(args.trajectory_reference_evaluator),
            "--trajectory-generation-max-new-tokens",
            str(args.trajectory_generation_max_new_tokens),
            "--selection-method",
            str(selection_info.get("selector") or selector),
            "--selection-preconditioner",
            str(args.selector_preconditioner),
            "--safe-training-constraint-mode",
            str(args.safe_training_constraint_mode),
        ]
    )
    safe_rho_for_training = selection_info.get("safe_effective_cost_c", selection_info.get("safe_cost_c", args.safe_cost_c))
    if safe_rho_for_training is not None:
        command.extend(["--safe-rho", str(safe_rho_for_training)])
    safe_epsilon_for_training = selection_info.get("safe_epsilon", args.safe_epsilon)
    if safe_epsilon_for_training is not None:
        command.extend(["--safe-epsilon", str(safe_epsilon_for_training)])
    if args.safe_training_epsilon_scale_by_steps:
        command.append("--safe-training-epsilon-scale-by-steps")
    if selection_info.get("safe_reference_cost") is not None:
        command.extend(
            [
                "--selected-predicted-fisher-cost",
                str(selection_info["safe_reference_cost"]),
            ]
        )
    if selection_info.get("safe_norm_cost") is not None:
        command.extend(
            [
                "--selected-predicted-norm-cost",
                str(selection_info["safe_norm_cost"]),
            ]
        )
    if args.ood_generation_max_new_tokens is not None:
        command.extend(["--ood-generation-max-new-tokens", str(args.ood_generation_max_new_tokens)])
    if args.reference_generation_max_new_tokens is not None:
        command.extend(["--reference-generation-max-new-tokens", str(args.reference_generation_max_new_tokens)])
    if args.validation_file is None:
        command.extend(
            [
                "--validation-split-proportions",
                *(str(value) for value in args.target_split_proportions),
                "--validation-split-seed",
                str(args.target_split_seed),
                "--validation-split-role",
                "val",
            ]
        )
    if args.eval_file is None:
        command.extend(
            [
                "--eval-split-proportions",
                *(str(value) for value in args.target_split_proportions),
                "--eval-split-seed",
                str(args.target_split_seed),
                "--eval-split-role",
                "final_test",
            ]
        )
    if args.ood_evaluator is not None:
        command.extend(["--ood-evaluator", args.ood_evaluator])
    if args.reference_evaluator is not None:
        command.extend(["--reference-evaluator", args.reference_evaluator])
    if args.candidate_validation_file is not None:
        command.extend(["--candidate-validation-file", str(args.candidate_validation_file)])
    if args.candidate_test_file is not None:
        command.extend(["--candidate-test-file", str(args.candidate_test_file)])
    if args.reference_validation_file is not None:
        command.extend(["--reference-validation-file", str(args.reference_validation_file)])
    if args.reference_test_file is not None:
        command.extend(["--reference-test-file", str(args.reference_test_file)])
    if args.reference_composition_name is not None:
        command.extend(["--reference-composition-name", str(args.reference_composition_name)])
    if args.reference_composition_domains:
        command.extend(["--reference-composition-domains", *list(args.reference_composition_domains)])
    if args.trajectory_instruction_file is not None:
        command.extend(
            [
                "--trajectory-instruction-file",
                str(args.trajectory_instruction_file),
                "--trajectory-instruction-evaluator",
                str(args.trajectory_instruction_evaluator),
            ]
        )
    if args.benchmark_evals:
        command.extend(["--benchmark-evals", *list(args.benchmark_evals)])
    if args.extra_evals:
        command.extend(["--extra-evals", *list(args.extra_evals)])
    if args.eval_max_examples is not None:
        command.extend(["--eval-max-examples", str(args.eval_max_examples)])
    if args.benchmark_max_examples is not None:
        command.extend(["--benchmark-max-examples", str(args.benchmark_max_examples)])
    command.extend(
        [
            "--benchmark-mmlu-subset",
            args.benchmark_mmlu_subset,
            "--benchmark-mmlu-split",
            args.benchmark_mmlu_split,
            "--benchmark-gsm8k-subset",
            args.benchmark_gsm8k_subset,
            "--benchmark-gsm8k-split",
            args.benchmark_gsm8k_split,
        ]
    )
    bias_eval_data_path = args.bias_eval_file or args.reference_bias_eval_output_file
    if bias_eval_data_path is not None:
        command.extend(
            [
                "--bias-eval-data-path",
                str(bias_eval_data_path),
                "--bias-eval-domain",
                args.bias_eval_domain,
                "--bias-eval-layers",
                args.bias_eval_layers,
                "--bias-eval-alpha",
                str(args.bias_eval_alpha),
                "--bias-eval-geometry",
                args.bias_eval_geometry,
                "--bias-eval-direction-split",
                str(args.bias_eval_direction_split),
                "--bias-eval-batch-size",
                str(args.bias_eval_batch_size),
                "--bias-eval-max-length",
                str(args.bias_eval_max_length),
            ]
        )
        if args.bias_eval_max_examples is not None:
            command.extend(["--bias-eval-max-examples", str(args.bias_eval_max_examples)])

    if args.gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if args.skip_final_evaluation:
        command.append("--skip-final-evaluation")
    if args.add_bos_token:
        command.append("--add-bos-token")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.low_cpu_mem_usage:
        command.append("--low-cpu-mem-usage")
    if args.compute_validation_gradient:
        command.extend(["--compute-validation-gradient", "--validation-gradient-max-examples", str(args.validation_gradient_max_examples)])
    if args.reference_file is not None:
        command.extend(["--reference-file", str(args.reference_file)])
    elif args.reference_hf_stereoset:
        command.extend(
            [
                "--reference-hf-stereoset",
                "--reference-hf-subset",
                args.reference_hf_subset,
                "--reference-hf-split",
                args.reference_hf_split,
                "--reference-hf-label",
                args.reference_hf_label,
                "--reference-hf-format",
                args.reference_hf_format,
                "--reference-split-proportions",
                *(str(value) for value in args.reference_split_proportions),
                "--reference-split-seed",
                str(args.reference_split_seed),
                "--reference-split-group-key",
                str(args.reference_split_group_key),
                "--reference-split-role",
                "reference",
            ]
        )
    if args.compute_reference_fisher and (args.reference_file is not None or args.reference_hf_stereoset):
        command.extend(["--compute-reference-fisher", "--reference-fisher-max-examples", str(args.reference_fisher_max_examples)])

    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    else:
        command.extend(["--num-train-epochs", str(args.num_train_epochs)])

    if args.save_steps and args.save_steps > 0:
        command.extend(["--save-steps", str(args.save_steps)])
    return command


def run_training_command(command: list[str], dry_run: bool) -> tuple[float, int]:
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return 0.0, 0
    print(f"[training] launching command: {' '.join(command)}")
    start = time.perf_counter()
    completed = subprocess.run(command, check=False)
    return time.perf_counter() - start, int(completed.returncode)


def completed_training_run(run_output_dir: Path) -> bool:
    summary_path = run_output_dir / "summary.json"
    final_adapter = run_output_dir / "final_adapter"
    if not summary_path.exists() or not final_adapter.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(summary.get("completed_steps") or 0) > 0


def copy_file_unless_same(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copytree_unless_same(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def evaluate_base_model_once(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    bias_eval_data_path: str | None,
) -> dict[str, Any] | None:
    base_eval_path = output_dir / "base_model_eval.json"
    base_eval_output_dir = output_dir / "base_model_eval"
    target_evaluator = args.evaluator
    ood_evaluator = args.ood_evaluator or args.evaluator
    reference_evaluator = args.reference_evaluator or "none"
    instruction_evaluator = args.trajectory_instruction_evaluator
    extra_eval_specs = parse_extra_eval_specs(args.extra_evals)
    target_eval_cache_tag = "target_none"
    if args.eval_file and target_evaluator not in {"none", "bias_disentangle"}:
        target_eval_cache_tag = f"target_{safe_slug(target_evaluator)}_{file_fingerprint(args.eval_file)}"
    ood_eval_cache_tag = "ood_none"
    if args.ood_eval_file and ood_evaluator not in {"none", "bias_disentangle"}:
        ood_eval_cache_tag = f"ood_{safe_slug(ood_evaluator)}_{file_fingerprint(args.ood_eval_file)}"
    reference_eval_cache_tag = "reference_none"
    if args.reference_test_file and reference_evaluator not in {"none", "loss", "bias_disentangle"}:
        reference_eval_cache_tag = (
            f"reference_{safe_slug(reference_evaluator)}_{file_fingerprint(args.reference_test_file)}"
        )
    instruction_eval_cache_tag = "instruction_none"
    if args.trajectory_instruction_file and instruction_evaluator not in {"none", "bias_disentangle"}:
        instruction_eval_cache_tag = (
            f"instruction_{safe_slug(instruction_evaluator)}_"
            f"{file_fingerprint(args.trajectory_instruction_file)}"
        )
    extra_eval_cache_tag = "extra_none"
    if extra_eval_specs:
        extra_eval_cache_tag = "extra_" + hashlib.sha1(
            json.dumps(
                [
                    {
                        **spec,
                        "fingerprint": file_fingerprint(spec["file"]),
                    }
                    for spec in extra_eval_specs
                ],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
    bias_eval_cache_tag = "bias_none"
    if bias_eval_data_path is not None:
        bias_eval_cache_tag = f"bias_{file_fingerprint(bias_eval_data_path)}"
    shared_cache_stem = "_".join(
        [
            "base_model_eval",
            safe_slug(args.model_name),
            safe_slug(args.train_torch_dtype),
            target_eval_cache_tag,
            ood_eval_cache_tag,
            reference_eval_cache_tag,
            instruction_eval_cache_tag,
            extra_eval_cache_tag,
            bias_eval_cache_tag,
        ]
    )
    shared_base_eval_path = Path(args.feature_cache_dir).resolve().parent / f"{shared_cache_stem}.json"
    shared_base_eval_output_dir = Path(args.feature_cache_dir).resolve().parent / shared_cache_stem
    if args.dry_run:
        payload = {
            "output_file": str(base_eval_path.resolve()),
            "shared_output_file": str(shared_base_eval_path.resolve()),
            "output_dir": str(base_eval_output_dir.resolve()),
            "target_eval_file": None if args.eval_file is None else str(Path(args.eval_file).resolve()),
            "target_evaluator": target_evaluator,
            "ood_eval_file": None if args.ood_eval_file is None else str(Path(args.ood_eval_file).resolve()),
            "ood_evaluator": ood_evaluator,
            "reference_eval_file": None
            if args.reference_test_file is None
            else str(Path(args.reference_test_file).resolve()),
            "reference_evaluator": reference_evaluator,
            "instruction_eval_file": None
            if args.trajectory_instruction_file is None
            else str(Path(args.trajectory_instruction_file).resolve()),
            "instruction_evaluator": instruction_evaluator,
            "extra_evals": extra_eval_specs,
            "bias_eval_data_path": None if bias_eval_data_path is None else str(Path(bias_eval_data_path).resolve()),
            "eval_max_examples": args.eval_max_examples,
            "bias_eval_max_examples": args.bias_eval_max_examples,
            "skipped": "dry_run",
        }
        write_json(payload, base_eval_path)
        return payload

    target_records = (
        load_records(args.eval_file)
        if args.eval_file and target_evaluator not in {"none", "bias_disentangle"}
        else []
    )
    ood_records = (
        load_records(args.ood_eval_file)
        if args.ood_eval_file and ood_evaluator not in {"none", "bias_disentangle"}
        else []
    )
    reference_records = (
        load_records(args.reference_test_file)
        if args.reference_test_file
        and reference_evaluator not in {"none", "loss", "bias_disentangle"}
        else []
    )
    instruction_records = (
        load_records(args.trajectory_instruction_file)
        if args.trajectory_instruction_file
        and instruction_evaluator not in {"none", "bias_disentangle"}
        else []
    )
    extra_eval_records = [
        {
            **spec,
            "records": load_records(spec["file"])
            if spec["evaluator"] not in {"none", "bias_disentangle", "loss"}
            else [],
        }
        for spec in extra_eval_specs
    ]
    if not any(
        (
            target_records,
            ood_records,
            reference_records,
            instruction_records,
            any(item["records"] for item in extra_eval_records),
        )
    ) and bias_eval_data_path is None:
        return None
    if shared_base_eval_path.exists():
        payload = json.loads(shared_base_eval_path.read_text(encoding="utf-8"))
        copy_file_unless_same(shared_base_eval_path, base_eval_path)
        if shared_base_eval_output_dir.exists():
            copytree_unless_same(shared_base_eval_output_dir, base_eval_output_dir)
        return payload

    payload: dict[str, Any] = {
        "model_name": args.model_name,
        "target_eval_file": None if args.eval_file is None else str(Path(args.eval_file).resolve()),
        "target_evaluator": target_evaluator,
        "ood_eval_file": None if args.ood_eval_file is None else str(Path(args.ood_eval_file).resolve()),
        "ood_evaluator": ood_evaluator,
        "reference_eval_file": None
        if args.reference_test_file is None
        else str(Path(args.reference_test_file).resolve()),
        "reference_evaluator": reference_evaluator,
        "instruction_eval_file": None
        if args.trajectory_instruction_file is None
        else str(Path(args.trajectory_instruction_file).resolve()),
        "instruction_evaluator": instruction_evaluator,
        "extra_evals": extra_eval_specs,
        "bias_eval_data_path": None if bias_eval_data_path is None else str(Path(bias_eval_data_path).resolve()),
        "eval_max_examples": args.eval_max_examples,
        "bias_eval_max_examples": args.bias_eval_max_examples,
    }

    model = None
    tokenizer = None
    if any((target_records, ood_records, reference_records, instruction_records, any(item["records"] for item in extra_eval_records))):
        base_eval_args = argparse.Namespace(
            model_name_or_path=args.model_name,
            use_slow_tokenizer=False,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=args.train_torch_dtype,
        )
        model, tokenizer = build_base_model_and_tokenizer(base_eval_args)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
        model.eval()
        if target_records:
            evaluator = get_evaluator(target_evaluator)
            target_metrics = evaluator(
                model=model,
                tokenizer=tokenizer,
                records=target_records,
                device=device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
            )
            payload.update({f"base_target_{key}": value for key, value in target_metrics.items()})
        if ood_records:
            if ood_evaluator == "humaneval":
                ood_metrics = humaneval_evaluator.evaluate_records(
                    model=model,
                    tokenizer=tokenizer,
                    records=ood_records,
                    device=device,
                    max_examples=args.eval_max_examples,
                    max_new_tokens=args.ood_generation_max_new_tokens or args.generation_max_new_tokens,
                    add_bos_token=args.add_bos_token,
                    num_samples=args.humaneval_num_samples,
                    pass_at_ks=tuple(args.humaneval_pass_at_ks),
                    temperature=args.humaneval_temperature,
                    top_p=args.humaneval_top_p,
                )
            else:
                evaluator = get_evaluator(ood_evaluator)
                ood_metrics = evaluator(
                    model=model,
                    tokenizer=tokenizer,
                    records=ood_records,
                    device=device,
                    max_examples=args.eval_max_examples,
                    max_new_tokens=args.ood_generation_max_new_tokens or args.generation_max_new_tokens,
                    add_bos_token=args.add_bos_token,
                )
            payload.update({f"base_ood_{key}": value for key, value in ood_metrics.items()})
        if reference_records:
            evaluator = get_evaluator(reference_evaluator)
            reference_metrics = evaluator(
                model=model,
                tokenizer=tokenizer,
                records=reference_records,
                device=device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
            )
            payload.update({f"base_reference_{key}": value for key, value in reference_metrics.items()})
        if instruction_records:
            evaluator = get_evaluator(instruction_evaluator)
            instruction_metrics = evaluator(
                model=model,
                tokenizer=tokenizer,
                records=instruction_records,
                device=device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
            )
            payload.update({f"base_instruction_{key}": value for key, value in instruction_metrics.items()})
        for item in extra_eval_records:
            if not item["records"]:
                continue
            evaluator = get_evaluator(item["evaluator"])
            metrics = evaluator(
                model=model,
                tokenizer=tokenizer,
                records=item["records"],
                device=device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
            )
            payload.update(
                {
                    f"base_extra_eval_{item['name']}_{key}": value
                    for key, value in metrics.items()
                }
            )

    if model is not None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if bias_eval_data_path is not None:
        base_bias_output_dir = shared_base_eval_output_dir / "bias_disentangle"
        bias_metrics = run_base_bias_eval_suite(
            base_model_name_or_path=args.model_name,
            data_path=bias_eval_data_path,
            requested_domains=args.bias_eval_domain,
            layers=args.bias_eval_layers,
            alpha=args.bias_eval_alpha,
            geometry=args.bias_eval_geometry,
            direction_split=args.bias_eval_direction_split,
            seed=args.seed,
            batch_size=args.bias_eval_batch_size,
            max_length=args.bias_eval_max_length,
            max_examples=args.bias_eval_max_examples,
            torch_dtype=args.train_torch_dtype,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            trust_remote_code=args.trust_remote_code,
            output_dir=base_bias_output_dir,
        )
        payload.update(bias_metrics)

    write_json(payload, shared_base_eval_path)
    copy_file_unless_same(shared_base_eval_path, base_eval_path)
    if shared_base_eval_output_dir.exists():
        copytree_unless_same(shared_base_eval_output_dir, base_eval_output_dir)
    return payload


def run_base_bias_eval_suite(
    *,
    base_model_name_or_path: str,
    data_path: str | Path,
    requested_domains: str,
    layers: str,
    alpha: float,
    geometry: str,
    direction_split: float,
    seed: int,
    batch_size: int,
    max_length: int,
    max_examples: int | None,
    torch_dtype: str,
    device: str,
    trust_remote_code: bool,
    output_dir: Path,
) -> dict[str, float]:
    from utils.train_lora_sft import resolve_bias_eval_domains

    domains, include_all = resolve_bias_eval_domains(data_path, requested_domains)
    if not domains:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    results_by_domain: dict[str, dict[str, float]] = {}
    merged_metrics: dict[str, float] = {}
    for domain in domains:
        domain_metrics = evaluate_base_bias_metrics(
            base_model_name_or_path=base_model_name_or_path,
            data_path=data_path,
            domain=domain,
            layers=layers,
            alpha=alpha,
            geometry=geometry,
            direction_split=direction_split,
            seed=seed,
            batch_size=batch_size,
            max_length=max_length,
            max_examples=max_examples,
            torch_dtype=torch_dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            output_dir=output_dir / domain,
        )
        results_by_domain[domain] = domain_metrics
        for key, value in domain_metrics.items():
            suffix = key[len("bias_disentangle_") :] if key.startswith("bias_disentangle_") else key
            merged_metrics[f"bias_disentangle_{domain}_{suffix}"] = float(value)

    if include_all:
        shared_keys = set.intersection(*(set(metrics.keys()) for metrics in results_by_domain.values()))
        for key in sorted(shared_keys):
            suffix = key[len("bias_disentangle_") :] if key.startswith("bias_disentangle_") else key
            merged_metrics[f"bias_disentangle_all_{suffix}"] = float(
                sum(results_by_domain[domain][key] for domain in domains) / len(domains)
            )

    merged_metrics["bias_disentangle_domain_count"] = float(len(domains))
    return merged_metrics


def ensure_selection_subset(
    selector: str,
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    references: list[dict[str, Any]],
    subset_budget: int,
) -> tuple[Path, dict[str, Any]]:
    if selector == "random":
        return ensure_random_subset(args, candidates, subset_budget)
    if selector == "dsir":
        return ensure_dsir_subset(args, candidates, targets, subset_budget)
    if selector == "less":
        return ensure_less_subset(args, candidates, targets, references, subset_budget)
    if selector == "prismatic":
        return ensure_prismatic_subset(args, candidates, targets, references, subset_budget)
    if selector == "safe":
        if not references:
            raise ValueError("safe selector requires reference samples from --reference-file or --reference-hf-stereoset.")
        return ensure_safe_subset(args, candidates, targets, references, subset_budget)
    raise ValueError(f"Unsupported selector: {selector}")


def _parse_simple_yaml_value(raw: str) -> Any:
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


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a flat YAML config and translate keys to argparse dest names."""
    try:
        import yaml  # type: ignore
    except ImportError:
        config: dict[str, Any] = {}
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.split("#", 1)[0].strip()
            if not stripped:
                continue
            if ":" not in stripped:
                raise ValueError(f"Invalid config line {line_no} in {path}: {line!r}")
            key, value = stripped.split(":", 1)
            config[key.strip()] = _parse_simple_yaml_value(value)
    else:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected a YAML mapping in {path}.")
        config = dict(loaded)

    normalized: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            raise ValueError(
                f"{path} uses nested key {key!r}; this runner expects flat YAML keys matching CLI arguments."
            )
        normalized[str(key).replace("-", "_")] = value
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selector subset generation and LoRA SFT sweeps.")
    parser.add_argument("--config", default=None, help="Optional flat YAML config file with argument defaults.")
    parser.add_argument(
        "--defaults-config",
        default=str(REPO_ROOT / "configs" / "defaults.yaml"),
        help="Optional flat YAML defaults loaded before --config.",
    )
    parser.add_argument("--candidate-file", default=str(DEFAULT_CANDIDATE_FILE))
    parser.add_argument("--candidate-validation-file", default=None)
    parser.add_argument("--candidate-test-file", default=None)
    parser.add_argument("--target-file", default=str(DEFAULT_TARGET_FILE))
    parser.add_argument(
        "--validation-file",
        default=None,
        help="Optional pre-split validation/metrics file passed directly to SFT training.",
    )
    parser.add_argument(
        "--eval-file",
        default=None,
        help="Optional pre-split final evaluation file passed directly to SFT training.",
    )
    parser.add_argument("--ood-eval-file", default=str(DEFAULT_OOD_EVAL_FILE))
    parser.add_argument("--target-split-proportions", nargs=3, type=float, default=[0.34, 0.33, 0.33])
    parser.add_argument("--target-split-seed", type=int, default=42)
    parser.add_argument("--reference-file", default=None)
    parser.add_argument("--reference-validation-file", default=None)
    parser.add_argument("--reference-test-file", default=None)
    parser.add_argument("--reference-composition-name", default=None)
    parser.add_argument("--reference-composition-domains", nargs="*", default=[])
    parser.add_argument("--reference-hf-stereoset", action="store_true")
    parser.add_argument("--reference-hf-subset", default="intrasentence")
    parser.add_argument("--reference-hf-split", default="validation")
    parser.add_argument("--reference-hf-label", default="stereotype,anti-stereotype")
    parser.add_argument("--reference-hf-format", default="instruction", choices=["instruction", "messages"])
    parser.add_argument("--reference-split-proportions", nargs=2, type=float, default=[0.5, 0.5])
    parser.add_argument("--reference-split-seed", type=int, default=42)
    parser.add_argument("--reference-split-group-key", default="bias_type")
    parser.add_argument(
        "--reference-bias-eval-output-file",
        default=None,
        help=(
            "Optional JSON file for the held-out StereoSet triplet records reserved for final bias evaluation. "
            "Only supported when --reference-hf-stereoset is used; this split is disjoint from the reference "
            "records used to build the selector Fisher."
        ),
    )
    parser.add_argument("--bias-eval-file", default=None)
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=["full", "random", "dsir", "less", "prismatic", "safe"],
        default=["full", "random", "dsir", "less", "prismatic", "safe"],
    )
    parser.add_argument("--subset-percentages", nargs="+", type=float, default=[5.0, 15.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-subsets", action="store_true")
    parser.add_argument("--overwrite-selection-cache", action="store_true")
    parser.add_argument("--skip-base-eval", action="store_true")
    parser.add_argument("--prepare-shared-selector-cache-only", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument(
        "--resume-completed-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip training outputs that already contain a completed summary and final adapter.",
    )
    parser.add_argument("--save-selected-artifacts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--plot-update-trajectory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot selector-prefix and optimizer-step SAFE constraint diagnostics.",
    )
    parser.add_argument(
        "--run-entanglement-analysis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run entanglement analysis for low-rank selector feature caches.",
    )
    parser.add_argument(
        "--entanglement-output-dir",
        default=None,
        help="Optional entanglement analysis output directory. Defaults to <feature-cache-dir parent>/entanglement_analysis.",
    )
    parser.add_argument(
        "--entanglement-selector-top-k",
        type=int,
        default=200,
        help="Number of top-influence candidates to highlight in entanglement analysis plots.",
    )

    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--selection-output-dir", default=str(DEFAULT_OUTPUT_DIR / "subsets"))
    parser.add_argument("--training-output-dir", default=str(DEFAULT_OUTPUT_DIR / "training_runs"))
    parser.add_argument("--feature-cache-dir", default=str(DEFAULT_OUTPUT_DIR / "selector_feature_cache"))
    parser.add_argument("--train-cache-dir", default=str(DEFAULT_OUTPUT_DIR / "train_cache"))

    parser.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT))
    parser.add_argument("--launcher", choices=["python", "accelerate"], default="python")
    parser.add_argument("--num-processes", type=int, default=1)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--train-model-name",
        default=None,
        help="Optional fine-tuning model. Defaults to --model-name; set this for gradient-model != FT-model transfer runs.",
    )
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    parser.add_argument("--num-buckets", type=int, default=10000)
    parser.add_argument("--q-sample-ratio", type=float, default=0.3)
    parser.add_argument("--cluster-ratio", type=float, default=0.1)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--method", choices=["hard", "soft"], default="soft")
    parser.add_argument("--selector-feature-method", choices=["random_sketch", "low_rank"], default="low_rank")
    parser.add_argument("--selector-preconditioner", choices=["sgd", "adam"], default="adam")
    parser.add_argument(
        "--adam-selection-mode",
        choices=["post_projection", "precondition_before_projection"],
        default="post_projection",
        help=(
            "Adam selector approximation. 'post_projection' preserves the legacy "
            "P_K B_K path; 'precondition_before_projection' builds candidate features "
            "and the residual task basis from full-coordinate frozen-Adam updates."
        ),
    )
    parser.add_argument(
        "--adam-selection-effective-learning-rate",
        type=float,
        default=None,
        help=(
            "Optional cumulative learning-rate scale for "
            "precondition_before_projection. By default it is the sum of the planned "
            "training learning-rate schedule."
        ),
    )
    parser.add_argument("--less-similarity", choices=["dot", "cosine"], default="cosine")
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-warmup-steps", type=float, default=0.1)
    parser.add_argument("--selection-candidate-max-examples", type=int, default=None)
    parser.add_argument("--selection-candidate-subsample-seed", type=int, default=None)
    parser.add_argument("--legacy-disjoint-preconditioner-pool", action="store_true")
    parser.add_argument("--preconditioner-split-seed", type=int, default=42)
    parser.add_argument("--selector-projection-dim", type=int, default=8192)
    parser.add_argument("--selector-projection-seed", type=int, default=13)
    parser.add_argument("--selector-projection-chunk-size", type=int, default=1_048_576)
    parser.add_argument("--selector-projection-cache-dtype", choices=["float16", "bfloat16", "float32"], default="float16")

    parser.add_argument("--safe-alpha", type=parse_auto_float, default=1e-3)
    parser.add_argument("--safe-learning-rate", type=float, default=1.0)
    parser.add_argument("--safe-geometry", choices=["euclidean", "reference"], default="reference")
    parser.add_argument(
        "--safe-solver",
        choices=["rank", "greedy_marginal", "constrained_greedy", "relaxed"],
        default="constrained_greedy",
    )
    parser.add_argument("--safe-shortlist-size", type=int, default=None)
    parser.add_argument("--safe-average-by-budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-feasibility-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--safe-objective-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--safe-greedy-swap-passes", type=int, default=1)
    parser.add_argument("--safe-relaxed-cvx-solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument("--safe-relaxed-weight-threshold", type=float, default=1.0e-8)
    parser.add_argument("--safe-cost-c", type=float, default=None)
    parser.add_argument(
        "--safe-cost-beta",
        type=float,
        default=None,
        help="Set rho to beta times the Fisher cost of the norm-epsilon unconstrained target step.",
    )
    parser.add_argument(
        "--safe-cost-gamma",
        type=float,
        default=None,
        help="Set rho = 0.5 * gamma * q0 * epsilon^2 for a normalized target direction.",
    )
    parser.add_argument("--safe-epsilon", type=float, default=None)
    parser.add_argument("--safe-epsilon-multiplier", type=float, default=None)
    parser.add_argument("--safe-epsilon-calibration-samples", type=int, default=64)
    parser.add_argument("--safe-epsilon-calibration-seed", type=int, default=None)
    parser.add_argument(
        "--safe-training-constraint-mode",
        choices=["none", "cumulative_line_search", "equal_allocation"],
        default="none",
        help=(
            "Optional projected optimizer updates: cumulative_line_search enforces the "
            "global Fisher/norm budgets from the base checkpoint; equal_allocation uses "
            "the conservative per-step rho/T^2 and epsilon/T allocation."
        ),
    )
    parser.add_argument(
        "--safe-training-epsilon-scale-by-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use selector epsilon times the resolved optimizer-step horizon as the "
            "training trajectory radius."
        ),
    )
    parser.add_argument("--reference-fisher-max-examples", type=int, default=1024)
    parser.add_argument("--low-rank-reference-rank", type=parse_auto_int, default=256)
    parser.add_argument("--low-rank-builder-alpha", type=float, default=1e-3)
    parser.add_argument("--low-rank-delta", type=float, default=0.01)
    parser.add_argument("--low-rank-max-reference-rank", type=int, default=None)
    parser.add_argument("--low-rank-reference-shrinkage", choices=["none", "stein"], default="none")
    parser.add_argument("--low-rank-task-rank", type=parse_auto_int, default=64)
    parser.add_argument("--low-rank-common-rank", type=parse_auto_int, default=None)
    parser.add_argument("--low-rank-auto-task-rank", action="store_true")
    parser.add_argument("--low-rank-delta-task", type=float, default=0.01)
    parser.add_argument("--low-rank-max-task-rank", type=int, default=None)
    parser.add_argument("--low-rank-target-max-examples", type=int, default=None)
    parser.add_argument("--low-rank-candidate-max-examples", type=int, default=None)
    parser.add_argument("--low-rank-candidate-projection-chunk-rows", type=int, default=256)
    parser.add_argument("--save-all-target-features", action="store_true")

    parser.add_argument("--train-max-seq-length", type=int, default=1024)
    parser.add_argument("--train-torch-dtype", default="float32")
    parser.add_argument("--selection-torch-dtype", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--train-lora-r", type=int, default=32)
    parser.add_argument("--train-lora-alpha", type=int, default=32)
    parser.add_argument("--train-lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-lora-target-modules", nargs="+", default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--add-bos-token", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--kl-regularization-lambda", type=float, default=0.0)
    parser.add_argument("--kl-reference-max-examples", type=int, default=128)
    parser.add_argument("--trajectory-eval-steps", type=int, default=0)
    parser.add_argument("--trajectory-reference-max-examples", type=int, default=64)
    parser.add_argument("--trajectory-target-max-examples", type=int, default=64)
    parser.add_argument("--trajectory-generation-eval-steps", type=int, default=0)
    parser.add_argument("--trajectory-generation-max-examples", type=int, default=16)
    parser.add_argument("--trajectory-generation-max-new-tokens", type=int, default=32)
    parser.add_argument("--trajectory-target-evaluator", default=None)
    parser.add_argument("--trajectory-reference-evaluator", default="medical_reference")
    parser.add_argument("--trajectory-instruction-file", default=None)
    parser.add_argument("--trajectory-instruction-evaluator", default="ifeval_proxy")
    parser.add_argument("--evaluator", default="humaneval")
    parser.add_argument("--ood-evaluator", default=None)
    parser.add_argument("--reference-evaluator", default=None)
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=256)
    parser.add_argument("--ood-generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--reference-generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--humaneval-num-samples", type=int, default=1)
    parser.add_argument("--humaneval-pass-at-ks", nargs="+", type=int, default=[1])
    parser.add_argument("--humaneval-temperature", type=float, default=0.8)
    parser.add_argument("--humaneval-top-p", type=float, default=0.95)
    parser.add_argument("--benchmark-evals", nargs="*", choices=["mmlu", "gsm8k"], default=[])
    parser.add_argument(
        "--extra-evals",
        nargs="*",
        default=[],
        help="Named held-out evals as NAME=EVALUATOR:/path/to/file.jsonl.",
    )
    parser.add_argument("--benchmark-max-examples", type=int, default=None)
    parser.add_argument("--benchmark-mmlu-subset", default="all")
    parser.add_argument("--benchmark-mmlu-split", default="test")
    parser.add_argument("--benchmark-gsm8k-subset", default="main")
    parser.add_argument("--benchmark-gsm8k-split", default="test")
    parser.add_argument("--compute-validation-gradient", action="store_true")
    parser.add_argument("--validation-gradient-max-examples", type=int, default=128)
    parser.add_argument("--compute-reference-fisher", action="store_true")
    parser.add_argument("--compute-entanglement-metrics", action="store_true")
    parser.add_argument("--entanglement-target-sample-size", type=int, default=128)
    parser.add_argument("--entanglement-selected-csv-max-rows", type=int, default=5000)
    parser.add_argument("--bias-eval-domain", default="all")
    parser.add_argument("--bias-eval-layers", default="-1")
    parser.add_argument("--bias-eval-alpha", type=float, default=0.5)
    parser.add_argument("--bias-eval-geometry", choices=["sand-e", "sand-w"], default="sand-e")
    parser.add_argument("--bias-eval-direction-split", type=float, default=0.5)
    parser.add_argument("--bias-eval-batch-size", type=int, default=8)
    parser.add_argument("--bias-eval-max-length", type=int, default=512)
    parser.add_argument("--bias-eval-max-examples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default=None)
    config_probe.add_argument("--defaults-config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    probe_args, _ = config_probe.parse_known_args()
    defaults_path = Path(probe_args.defaults_config)
    if defaults_path.exists():
        parser.set_defaults(**load_yaml_config(defaults_path))
    if probe_args.config:
        parser.set_defaults(**load_yaml_config(Path(probe_args.config)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(
        value is not None
        for value in (args.safe_cost_c, args.safe_cost_beta, args.safe_cost_gamma)
    ) > 1:
        raise ValueError(
            "Use exactly one of --safe-cost-c, --safe-cost-beta, or "
            "--safe-cost-gamma."
        )
    if args.safe_epsilon is not None and args.safe_epsilon_multiplier is not None:
        raise ValueError("Use either --safe-epsilon or --safe-epsilon-multiplier, not both.")
    if args.safe_epsilon_calibration_samples <= 0:
        raise ValueError("--safe-epsilon-calibration-samples must be positive.")
    if args.safe_training_epsilon_scale_by_steps:
        if args.safe_training_constraint_mode == "none":
            raise ValueError(
                "--safe-training-epsilon-scale-by-steps requires a SAFE training "
                "constraint mode."
            )
        if args.safe_epsilon is None and args.safe_epsilon_multiplier is None:
            raise ValueError(
                "--safe-training-epsilon-scale-by-steps requires --safe-epsilon "
                "or --safe-epsilon-multiplier."
            )
    if not args.compute_validation_gradient:
        args.compute_validation_gradient = True
    if (args.reference_file is not None or args.reference_hf_stereoset) and not args.compute_reference_fisher:
        args.compute_reference_fisher = True
    output_dir = Path(args.output_dir)
    selection_output_dir = Path(args.selection_output_dir)
    training_output_dir = Path(args.training_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_output_dir.mkdir(parents=True, exist_ok=True)
    training_output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.feature_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.train_cache_dir).mkdir(parents=True, exist_ok=True)

    raw_candidates = load_records(args.candidate_file)
    preconditioner_candidates, candidates, candidate_split_info = prepare_candidate_pools(args, raw_candidates)
    print(
        f"[dataset] candidate file: {Path(args.candidate_file).resolve()} "
        f"(raw={len(raw_candidates)}, selection={len(candidates)})."
    )
    print_preconditioner_split_summary(candidate_split_info)
    all_target_records = load_records(args.target_file)
    target_splits = split_lookup(
        all_target_records,
        proportions=args.target_split_proportions,
        seed=args.target_split_seed,
        names=["selector_target", "val", "final_test"],
    )
    targets = target_splits["selector_target"]
    print(f"[dataset] target file: {Path(args.target_file).resolve()} (raw={len(all_target_records)}).")
    print_named_split_summary("target", target_splits)
    if args.reference_file:
        all_reference_records = load_records(args.reference_file)
        reference_source = "file"
    elif args.reference_hf_stereoset:
        all_reference_records = load_stereoset_records(
            subset=args.reference_hf_subset,
            split=args.reference_hf_split,
            label=args.reference_hf_label,
            output_format=args.reference_hf_format,
        )
        reference_source = "hf_stereoset"
    else:
        all_reference_records = []
        reference_source = "none"
    if all_reference_records:
        reference_splits = split_lookup(
            all_reference_records,
            proportions=args.reference_split_proportions,
            seed=args.reference_split_seed,
            names=["reference", "bias_eval"],
            group_key=args.reference_split_group_key,
        )
        references = reference_splits["reference"]
        print(f"[dataset] reference source: {reference_source} (raw={len(all_reference_records)}).")
        print_named_split_summary("reference", reference_splits)
    else:
        references = []
        print("[dataset] no reference records configured.")

    reference_bias_eval_output_file = None
    reference_bias_eval_split_size = None
    if args.reference_bias_eval_output_file is not None:
        if not args.reference_hf_stereoset:
            raise ValueError("--reference-bias-eval-output-file is only supported with --reference-hf-stereoset.")
        all_reference_triplets = load_stereoset_triplet_records(
            subset=args.reference_hf_subset,
            split=args.reference_hf_split,
        )
        reference_triplet_splits = split_lookup(
            all_reference_triplets,
            proportions=args.reference_split_proportions,
            seed=args.reference_split_seed,
            names=["reference", "bias_eval"],
            group_key=args.reference_split_group_key,
        )
        bias_eval_triplets = reference_triplet_splits["bias_eval"]
        reference_bias_eval_output_file = str(Path(args.reference_bias_eval_output_file).resolve())
        reference_bias_eval_split_size = len(bias_eval_triplets)
        write_json(bias_eval_triplets, Path(reference_bias_eval_output_file))

    base_bias_eval_data_path = args.bias_eval_file or reference_bias_eval_output_file
    base_model_eval = None
    if not args.skip_base_eval and not args.prepare_shared_selector_cache_only:
        base_model_eval = evaluate_base_model_once(
            args,
            output_dir=output_dir,
            bias_eval_data_path=base_bias_eval_data_path,
        )

    manifest: dict[str, Any] = {
        "candidate_file": str(Path(args.candidate_file).resolve()),
        "raw_candidate_count": len(raw_candidates),
        "candidate_pool": candidate_split_info,
        "candidate_validation_file": None if args.candidate_validation_file is None else str(Path(args.candidate_validation_file).resolve()),
        "candidate_validation_count": count_optional_records(args.candidate_validation_file),
        "candidate_test_file": None if args.candidate_test_file is None else str(Path(args.candidate_test_file).resolve()),
        "candidate_test_count": count_optional_records(args.candidate_test_file),
        "target_file": str(Path(args.target_file).resolve()),
        "validation_file": None if args.validation_file is None else str(Path(args.validation_file).resolve()),
        "eval_file": None if args.eval_file is None else str(Path(args.eval_file).resolve()),
        "ood_eval_file": None if args.ood_eval_file is None else str(Path(args.ood_eval_file).resolve()),
        "target_split_proportions": args.target_split_proportions,
        "target_split_seed": args.target_split_seed,
        "target_split_sizes": {name: len(split) for name, split in target_splits.items()},
        "reference_file": None if args.reference_file is None else str(Path(args.reference_file).resolve()),
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "reference_validation_file": None if args.reference_validation_file is None else str(Path(args.reference_validation_file).resolve()),
        "reference_validation_count": count_optional_records(args.reference_validation_file),
        "reference_test_file": None if args.reference_test_file is None else str(Path(args.reference_test_file).resolve()),
        "reference_test_count": count_optional_records(args.reference_test_file),
        "bias_eval_file": None if args.bias_eval_file is None else str(Path(args.bias_eval_file).resolve()),
        "reference_source": reference_source,
        "reference_hf_stereoset": bool(args.reference_hf_stereoset),
        "reference_hf_subset": args.reference_hf_subset if args.reference_hf_stereoset else None,
        "reference_hf_split": args.reference_hf_split if args.reference_hf_stereoset else None,
        "reference_hf_label": args.reference_hf_label if args.reference_hf_stereoset else None,
        "reference_split_proportions": args.reference_split_proportions if all_reference_records else None,
        "reference_split_seed": args.reference_split_seed if all_reference_records else None,
        "reference_split_group_key": args.reference_split_group_key if all_reference_records else None,
        "reference_split_sizes": None if not all_reference_records else {name: len(split) for name, split in reference_splits.items()},
        "reference_bias_eval_output_file": reference_bias_eval_output_file,
        "reference_bias_eval_split_size": reference_bias_eval_split_size,
        "base_model_eval_file": None if base_model_eval is None else str((output_dir / "base_model_eval.json").resolve()),
        "seed": args.seed,
        "model_name": args.model_name,
        "train_model_name": args.train_model_name or args.model_name,
        "subset_percentages": args.subset_percentages,
        "selectors": args.selectors,
        "selector_feature_method": args.selector_feature_method,
        "selector_preconditioner": args.selector_preconditioner,
        "adam_selection_mode": args.adam_selection_mode,
        "adam_selection_effective_learning_rate": args.adam_selection_effective_learning_rate,
        "safe_cost_c": args.safe_cost_c,
        "safe_cost_beta": args.safe_cost_beta,
        "safe_cost_gamma": args.safe_cost_gamma,
        "safe_epsilon": args.safe_epsilon,
        "safe_epsilon_multiplier": args.safe_epsilon_multiplier,
        "safe_epsilon_calibration_samples": args.safe_epsilon_calibration_samples,
        "safe_epsilon_calibration_seed": args.safe_epsilon_calibration_seed,
        "safe_training_constraint_mode": args.safe_training_constraint_mode,
        "safe_training_epsilon_scale_by_steps": bool(
            args.safe_training_epsilon_scale_by_steps
        ),
        "preconditioner_split_seed": args.preconditioner_split_seed,
        "less_similarity": args.less_similarity,
        "feature_cache_dir": str(Path(args.feature_cache_dir).resolve()),
        "entanglement_analysis": {
            "enabled": bool(args.run_entanglement_analysis),
            "output_dir": str(entanglement_analysis_output_dir(args).resolve()),
            "summary_file": str((entanglement_analysis_output_dir(args).resolve() / "entanglement_summary.json")),
            "selector_top_k": int(args.entanglement_selector_top_k),
        },
        "selector_projection_dim": args.selector_projection_dim,
        "selector_projection_seed": args.selector_projection_seed,
        "selector_projection_chunk_size": args.selector_projection_chunk_size,
        "selector_projection_cache_dtype": args.selector_projection_cache_dtype,
        "low_rank_reference_rank": args.low_rank_reference_rank,
        "low_rank_builder_alpha": args.low_rank_builder_alpha,
        "low_rank_delta": args.low_rank_delta,
        "low_rank_max_reference_rank": args.low_rank_max_reference_rank,
        "low_rank_reference_shrinkage": args.low_rank_reference_shrinkage,
        "low_rank_task_rank": args.low_rank_task_rank,
        "low_rank_common_rank": args.low_rank_common_rank,
        "low_rank_auto_task_rank": args.low_rank_auto_task_rank,
        "low_rank_delta_task": args.low_rank_delta_task,
        "low_rank_max_task_rank": args.low_rank_max_task_rank,
        "low_rank_target_max_examples": args.low_rank_target_max_examples,
        "save_selected_artifacts": bool(args.save_selected_artifacts),
        "compute_entanglement_metrics": bool(args.compute_entanglement_metrics),
        "entanglement_target_sample_size": args.entanglement_target_sample_size,
        "entanglement_selected_csv_max_rows": args.entanglement_selected_csv_max_rows,
        "base_model_eval": base_model_eval,
        "extra_evals": parse_extra_eval_specs(args.extra_evals),
        "runs": [],
    }

    if args.prepare_shared_selector_cache_only:
        get_shared_selector_features(args, candidates, targets, references)
        manifest["prepared_shared_selector_cache_only"] = True
        manifest["entanglement_analysis"]["results"] = collect_entanglement_analysis_results(args)
        manifest_path = output_dir / "selector_sft_sweep_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return

    first_subset_percentage = float(args.subset_percentages[0])

    for subset_percentage in args.subset_percentages:
        args.subset_percentage = subset_percentage
        subset_budget = selection_count_from_percentage(len(candidates), subset_percentage)
        for selector in args.selectors:
            if selector == "full":
                if float(subset_percentage) != first_subset_percentage:
                    continue
                train_file = Path(getattr(args, "_selection_candidate_file", args.candidate_file))
                selection_info = {
                    "selector": "full",
                    "subset_budget": len(candidates),
                    "subset_percentage": 100.0,
                    "selection_wallclock_seconds": 0.0,
                    "cache_hit": True,
                    "preconditioner_disjoint": bool(candidate_split_info["preconditioner_data_disjoint"]),
                }
                method_tag = "full"
            else:
                print(
                    f"[selection] selector={selector} subset_percentage={subset_percentage:g} "
                    f"subset_budget={subset_budget}"
                )
                train_file, selection_info = ensure_selection_subset(
                    selector=selector,
                    args=args,
                    candidates=candidates,
                    targets=targets,
                    references=references,
                    subset_budget=subset_budget,
                )
                selection_info["subset_budget"] = subset_budget
                selection_info["subset_percentage"] = subset_percentage
                method_tag = str(selection_info["selector"])

            if args.safe_training_constraint_mode != "none":
                method_tag = (
                    f"{method_tag}_trainconstraint_"
                    f"{safe_slug(args.safe_training_constraint_mode)}"
                )
                if args.safe_training_epsilon_scale_by_steps:
                    method_tag = f"{method_tag}_eps_times_steps"
            if selector == "full":
                run_output_dir = training_output_dir / f"{method_tag}_seed{args.seed}"
            else:
                run_output_dir = training_output_dir / f"{method_tag}_pct{subset_percentage:g}_seed{args.seed}"
            print(
                f"[selection] selector={selector} method_tag={method_tag} "
                f"train_file={train_file} output_dir={run_output_dir}"
            )
            if args.resume_completed_runs and completed_training_run(run_output_dir):
                command = build_train_command(
                    args,
                    train_file=train_file,
                    run_output_dir=run_output_dir,
                    selector=selector,
                    selection_info=selection_info,
                )
                print(f"[training] reusing completed run: {run_output_dir}")
                manifest["runs"].append(
                    {
                        "selector": selector,
                        "method_tag": method_tag,
                        "subset_percentage": subset_percentage,
                        "subset_budget": subset_budget if selector != "full" else len(candidates),
                        "train_file": str(train_file),
                        "run_output_dir": str(run_output_dir),
                        "selection": selection_info,
                        "train_command": command,
                        "train_wallclock_seconds": 0.0,
                        "train_return_code": 0,
                        "reused_completed_run": True,
                    }
                )
                continue
            selection_artifacts = save_selected_artifacts_for_run(
                args=args,
                candidates=candidates,
                targets=targets,
                references=references,
                train_file=train_file,
                run_output_dir=run_output_dir,
                selector=selector,
                method_tag=method_tag,
                selection_info=selection_info,
            )
            entanglement_info = compute_entanglement_for_run(
                args=args,
                candidates=candidates,
                targets=targets,
                references=references,
                train_file=train_file,
                run_output_dir=run_output_dir,
                selector=selector,
                method_tag=method_tag,
            )
            command = build_train_command(
                args,
                train_file=train_file,
                run_output_dir=run_output_dir,
                selector=selector,
                selection_info=selection_info,
            )
            train_seconds = 0.0
            return_code = 0
            if not args.skip_training:
                train_seconds, return_code = run_training_command(command, dry_run=args.dry_run)
            merge_selection_info_into_training_summary(run_output_dir, selection_info)
            merge_base_model_eval_into_training_summary(run_output_dir, base_model_eval)
            merge_selection_artifacts_into_training_summary(run_output_dir, selection_artifacts)
            merge_entanglement_into_training_summary(run_output_dir, entanglement_info)
            update_trajectory_plot = None
            if (
                selector == "safe"
                and args.plot_update_trajectory
                and not args.dry_run
                and (run_output_dir / "summary.json").exists()
            ):
                update_trajectory_plot = generate_update_trajectory_plot(run_output_dir)
                merge_update_trajectory_plot_into_summary(run_output_dir, update_trajectory_plot)

            manifest["runs"].append(
                {
                    "selector": selector,
                    "method_tag": method_tag,
                    "subset_percentage": subset_percentage,
                    "subset_budget": subset_budget if selector != "full" else len(candidates),
                    "train_file": str(train_file),
                    "run_output_dir": str(run_output_dir),
                    "selection": selection_info,
                    "selection_artifacts": selection_artifacts,
                    "entanglement": entanglement_info,
                    "update_constraint_trajectory_plot": update_trajectory_plot,
                    "train_command": command,
                    "train_wallclock_seconds": train_seconds,
                    "train_return_code": return_code,
                }
            )

    manifest_path = output_dir / "selector_sft_sweep_manifest.json"
    manifest["entanglement_analysis"]["results"] = collect_entanglement_analysis_results(args)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    failed_runs = [run for run in manifest.get("runs", []) if int(run.get("train_return_code", 0)) != 0]
    if failed_runs:
        failed_tags = ", ".join(str(run.get("method_tag", "unknown")) for run in failed_runs)
        print(f"[training] {len(failed_runs)} run(s) failed: {failed_tags}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
