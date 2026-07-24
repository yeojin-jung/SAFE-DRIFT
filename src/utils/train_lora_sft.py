from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent.parent

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation import get_evaluator
from evaluation import humaneval as humaneval_evaluator
from utils.data_construction import (
    coerce_record,
    dataset_fingerprint,
    file_fingerprint,
    load_gsm8k_records,
    load_mmlu_records,
    load_records,
    load_stereoset_records,
    split_records_by_proportions,
)


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TimingResult:
    tensor: torch.Tensor
    wallclock_seconds: float
    cache_hit: bool
    cache_path: str


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


def concat_messages(messages: list[dict[str, str]], tokenizer, add_bos_token: bool = False) -> str:
    chunks: list[str] = []
    for message in messages:
        role = message["role"]
        content = str(message["content"]).strip()
        if role == "system":
            chunks.append(f"<|system|>\n{content}\n")
        elif role == "user":
            chunks.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            chunks.append(f"<|assistant|>\n{content}{tokenizer.eos_token}\n")
        else:
            raise ValueError(f"Invalid role: {role}")
    text = "".join(chunks).strip()
    if add_bos_token and tokenizer.bos_token is not None:
        text = tokenizer.bos_token + text
    return text


def encode_with_messages_format(
    example: dict[str, Any],
    tokenizer,
    max_seq_length: int,
    add_bos_token: bool = False,
) -> dict[str, torch.Tensor]:
    messages = example["messages"]
    if not messages:
        raise ValueError("messages field is empty.")

    example_text = concat_messages(messages, tokenizer=tokenizer, add_bos_token=add_bos_token)
    tokenized_example = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    for message_idx, message in enumerate(messages):
        if message["role"] == "assistant":
            continue

        if message_idx == 0:
            message_start_idx = 0
        else:
            message_start_idx = tokenizer(
                concat_messages(messages[:message_idx], tokenizer=tokenizer, add_bos_token=add_bos_token),
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).input_ids.shape[1]

        if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
            messages_so_far = (
                concat_messages(messages[: message_idx + 1], tokenizer=tokenizer, add_bos_token=add_bos_token)
                + "<|assistant|>\n"
            )
        else:
            messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer=tokenizer, add_bos_token=add_bos_token)

        message_end_idx = tokenizer(
            messages_so_far,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
        ).input_ids.shape[1]
        labels[:, message_start_idx:message_end_idx] = -100
        if message_end_idx >= max_seq_length:
            break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def has_supervised_targets(encoded_example: dict[str, torch.Tensor] | torch.Tensor) -> bool:
    labels = encoded_example["labels"] if isinstance(encoded_example, dict) else encoded_example
    return bool(labels.ne(-100).any().item())


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        example_weights = [instance.get("example_weight") for instance in instances]
        if any(weight is not None for weight in example_weights):
            if not all(weight is not None for weight in example_weights):
                raise ValueError("A training batch cannot mix weighted and unweighted examples.")
            batch["example_weights"] = torch.stack(
                [weight.float().reshape(()) for weight in example_weights if weight is not None]
            )
        return batch


def tokenize_records(records: list[dict[str, Any]], tokenizer, max_seq_length: int, add_bos_token: bool) -> list[dict[str, torch.Tensor]]:
    tokenized_records: list[dict[str, torch.Tensor]] = []
    for record in records:
        encoded = encode_with_messages_format(
            record,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            add_bos_token=add_bos_token,
        )
        if has_supervised_targets(encoded):
            if "safe_training_weight" in record:
                weight = float(record["safe_training_weight"])
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"safe_training_weight must be finite and non-negative, got {weight}.")
                encoded["example_weight"] = torch.tensor(weight, dtype=torch.float32)
            tokenized_records.append(encoded)
    return tokenized_records


def weighted_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    example_weights: torch.Tensor,
) -> torch.Tensor:
    """Apply example weights while preserving the ordinary token-mean loss at weight one."""
    if example_weights.ndim != 1 or example_weights.shape[0] != labels.shape[0]:
        raise ValueError("example_weights must have shape [batch_size].")
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(-100)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)
    weighted_losses = token_losses * valid * example_weights.to(token_losses.dtype).unsqueeze(1)
    return weighted_losses.sum() / valid.sum().clamp_min(1)


def constraint_audit(
    *,
    cost: float,
    budget: float | None,
    prefix: str,
    relative_tolerance: float = 1.0e-6,
    absolute_tolerance: float = 1.0e-12,
) -> dict[str, float | bool | None]:
    if budget is None:
        return {
            f"{prefix}_budget": None,
            f"{prefix}_budget_ratio": None,
            f"{prefix}_budget_gap": None,
            f"{prefix}_budget_satisfied": None,
        }
    budget = float(budget)
    tolerance = max(float(absolute_tolerance), abs(budget) * float(relative_tolerance))
    return {
        f"{prefix}_budget": budget,
        f"{prefix}_budget_ratio": float(cost) / max(abs(budget), absolute_tolerance),
        f"{prefix}_budget_gap": budget - float(cost),
        f"{prefix}_budget_satisfied": bool(float(cost) <= budget + tolerance),
    }


def largest_scale_for_quadratic(
    *,
    quadratic: float,
    linear: float,
    constant: float,
    tolerance: float = 1.0e-12,
) -> float:
    """Largest s in [0, 1] with quadratic * s^2 + linear * s + constant <= 0."""
    quadratic = max(0.0, float(quadratic))
    linear = float(linear)
    constant = float(constant)
    tolerance = float(tolerance)
    if quadratic + linear + constant <= tolerance:
        return 1.0
    if quadratic <= tolerance:
        if abs(linear) <= tolerance:
            return 1.0 if constant <= tolerance else 0.0
        boundary = (tolerance - constant) / linear
        if linear > 0.0:
            return max(0.0, min(1.0, boundary))
        return 1.0 if boundary <= 1.0 else 0.0

    shifted_constant = constant - tolerance
    discriminant = linear * linear - 4.0 * quadratic * shifted_constant
    if discriminant < 0.0:
        return 0.0
    root_radius = math.sqrt(max(0.0, discriminant))
    lower_root = (-linear - root_radius) / (2.0 * quadratic)
    upper_root = (-linear + root_radius) / (2.0 * quadratic)
    feasible_lower = max(0.0, lower_root)
    feasible_upper = min(1.0, upper_root)
    if feasible_lower > feasible_upper:
        return 0.0
    return max(0.0, feasible_upper)


def cumulative_safe_scale(
    *,
    displacement: torch.Tensor,
    proposal: torch.Tensor,
    fisher: torch.Tensor | None,
    rho: float | None,
    epsilon: float | None,
    tolerance: float = 1.0e-12,
) -> tuple[float, dict[str, float | str | bool | None]]:
    """Line-search an optimizer proposal in the cumulative safe set anchored at theta0."""
    displacement = displacement.detach().float()
    proposal = proposal.detach().float()
    proposed_displacement = displacement + proposal
    scales = {"unit_interval": 1.0}

    if rho is not None and fisher is None:
        raise ValueError("A Fisher tensor is required when rho is constrained.")

    proposed_reference_cost = None
    current_reference_cost = None
    if fisher is not None:
        fisher = fisher.detach().to(device=proposal.device, dtype=torch.float32)
        fisher_proposal = fisher * proposal
        current_reference_cost = float(
            0.5 * torch.dot(displacement, fisher * displacement).item()
        )
        proposed_reference_cost = float(
            0.5 * torch.dot(proposed_displacement, fisher * proposed_displacement).item()
        )
    if rho is not None:
        scales["reference"] = largest_scale_for_quadratic(
            quadratic=float(0.5 * torch.dot(proposal, fisher_proposal).item()),
            linear=float(torch.dot(displacement, fisher_proposal).item()),
            constant=current_reference_cost - float(rho),
            tolerance=tolerance,
        )

    current_norm_cost = float(0.5 * torch.dot(displacement, displacement).item())
    proposed_norm_cost = float(
        0.5 * torch.dot(proposed_displacement, proposed_displacement).item()
    )
    if epsilon is not None:
        scales["norm"] = largest_scale_for_quadratic(
            quadratic=float(0.5 * torch.dot(proposal, proposal).item()),
            linear=float(torch.dot(displacement, proposal).item()),
            constant=current_norm_cost - 0.5 * float(epsilon) ** 2,
            tolerance=tolerance,
        )

    active_constraint = min(scales, key=scales.get)
    scale = min(scales.values())
    if scale < 1.0:
        scale *= 1.0 - 1.0e-7
    accepted_displacement = displacement + scale * proposal
    accepted_reference_cost = None
    if fisher is not None:
        accepted_reference_cost = float(
            0.5 * torch.dot(accepted_displacement, fisher * accepted_displacement).item()
        )
    accepted_norm_cost = float(
        0.5 * torch.dot(accepted_displacement, accepted_displacement).item()
    )
    rejected_proposal = (1.0 - scale) * proposal
    projection_fisher_error = None
    if fisher is not None:
        projection_fisher_error = float(
            0.5 * torch.dot(rejected_proposal, fisher * rejected_proposal).item()
        )
    diagnostics: dict[str, float | str | bool | None] = {
        "trajectory_projection_scale": float(scale),
        "trajectory_projection_active": bool(scale < 1.0 - 1.0e-8),
        "trajectory_projection_active_constraint": (
            active_constraint if scale < 1.0 - 1.0e-8 else "none"
        ),
        "trajectory_proposal_norm": float(proposal.norm().item()),
        "trajectory_projection_l2_error": float(rejected_proposal.norm().item()),
        "trajectory_projection_fisher_error": projection_fisher_error,
        "current_cumulative_reference_cost_before_update": current_reference_cost,
        "proposed_cumulative_reference_cost": proposed_reference_cost,
        "accepted_cumulative_reference_cost": accepted_reference_cost,
        "current_cumulative_norm_cost_before_update": current_norm_cost,
        "proposed_cumulative_norm_cost": proposed_norm_cost,
        "accepted_cumulative_norm_cost": accepted_norm_cost,
    }
    return float(scale), diagnostics


def safe_training_update_scale(
    *,
    mode: str,
    displacement: torch.Tensor,
    proposal: torch.Tensor,
    fisher: torch.Tensor | None,
    rho: float | None,
    epsilon: float | None,
    max_steps: int,
    tolerance: float = 1.0e-12,
) -> tuple[float, dict[str, float | str | bool | None]]:
    if mode not in {"cumulative_line_search", "equal_allocation"}:
        raise ValueError(f"Unsupported SAFE training constraint mode: {mode!r}")

    global_scale, diagnostics = cumulative_safe_scale(
        displacement=displacement,
        proposal=proposal,
        fisher=fisher,
        rho=rho,
        epsilon=epsilon,
        tolerance=tolerance,
    )
    diagnostics["safe_training_constraint_mode"] = mode
    if mode == "cumulative_line_search":
        return global_scale, diagnostics

    horizon = max(1, int(max_steps))
    step_rho = None if rho is None else float(rho) / float(horizon * horizon)
    step_epsilon = None if epsilon is None else float(epsilon) / float(horizon)
    step_scale, step_diagnostics = cumulative_safe_scale(
        displacement=torch.zeros_like(displacement),
        proposal=proposal,
        fisher=fisher,
        rho=step_rho,
        epsilon=step_epsilon,
        tolerance=tolerance,
    )
    scale = min(global_scale, step_scale)
    accepted_displacement = displacement + scale * proposal
    accepted_update = scale * proposal
    rejected_proposal = (1.0 - scale) * proposal
    active_constraint = "global_" + str(
        diagnostics["trajectory_projection_active_constraint"]
    )
    if step_scale <= global_scale and step_scale < 1.0 - 1.0e-8:
        active_constraint = "equal_allocation_" + str(
            step_diagnostics["trajectory_projection_active_constraint"]
        )

    diagnostics.update(
        {
            "trajectory_projection_scale": float(scale),
            "trajectory_projection_active": bool(scale < 1.0 - 1.0e-8),
            "trajectory_projection_active_constraint": (
                active_constraint if scale < 1.0 - 1.0e-8 else "none"
            ),
            "trajectory_projection_l2_error": float(rejected_proposal.norm().item()),
            "accepted_cumulative_norm_cost": float(
                0.5 * torch.dot(accepted_displacement, accepted_displacement).item()
            ),
            "equal_allocation_horizon": horizon,
            "equal_allocation_step_rho": step_rho,
            "equal_allocation_step_epsilon": step_epsilon,
            "equal_allocation_proposal_scale": float(step_scale),
            "equal_allocation_accepted_update_norm_cost": float(
                0.5 * torch.dot(accepted_update, accepted_update).item()
            ),
        }
    )
    if fisher is not None:
        fisher = fisher.detach().to(device=proposal.device, dtype=torch.float32)
        diagnostics.update(
            {
                "accepted_cumulative_reference_cost": float(
                    0.5
                    * torch.dot(
                        accepted_displacement,
                        fisher * accepted_displacement,
                    ).item()
                ),
                "trajectory_projection_fisher_error": float(
                    0.5
                    * torch.dot(
                        rejected_proposal,
                        fisher * rejected_proposal,
                    ).item()
                ),
                "equal_allocation_accepted_update_reference_cost": float(
                    0.5 * torch.dot(accepted_update, fisher * accepted_update).item()
                ),
            }
        )
    return float(scale), diagnostics


def assign_trainable_parameters_from_flat(model, flat_parameters: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, parameter in get_trainable_named_parameters(model):
            next_offset = offset + parameter.numel()
            if next_offset > flat_parameters.numel():
                raise ValueError("Flat parameter vector is shorter than the trainable parameter set.")
            parameter.copy_(
                flat_parameters[offset:next_offset]
                .view_as(parameter)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            offset = next_offset
    if offset != flat_parameters.numel():
        raise ValueError("Flat parameter vector is longer than the trainable parameter set.")


def get_tokenized_cache_path(
    cache_dir: Path,
    dataset_path: str | Path,
    tokenizer_name: str,
    max_seq_length: int,
    add_bos_token: bool,
    dataset_cache_tag: str | None = None,
) -> Path:
    fingerprint = file_fingerprint(dataset_path) if dataset_cache_tag is None else safe_slug(dataset_cache_tag)
    tokenizer_base = safe_slug(Path(str(tokenizer_name)).name or str(tokenizer_name).split("/")[-1])[:24]
    tokenizer_hash = hashlib.sha1(str(tokenizer_name).encode("utf-8")).hexdigest()[:8]
    tokenizer_tag = f"{tokenizer_base}_{tokenizer_hash}"
    suffix = f"{fingerprint}_{tokenizer_tag}_seq{max_seq_length}_bos{int(add_bos_token)}.pt"
    return cache_dir / "tokenized" / suffix


def prepare_tokenized_dataset(
    *,
    accelerator: Accelerator,
    records: list[dict[str, Any]],
    dataset_path: str | Path,
    tokenizer,
    max_seq_length: int,
    add_bos_token: bool,
    cache_dir: Path,
    overwrite_cache: bool,
    dataset_cache_tag: str | None = None,
) -> list[dict[str, torch.Tensor]]:
    cache_path = get_tokenized_cache_path(
        cache_dir=cache_dir,
        dataset_path=dataset_path,
        tokenizer_name=tokenizer.name_or_path,
        max_seq_length=max_seq_length,
        add_bos_token=add_bos_token,
        dataset_cache_tag=dataset_cache_tag,
    )
    if accelerator.is_main_process and (overwrite_cache or not cache_path.exists()):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tokenized = tokenize_records(records, tokenizer, max_seq_length=max_seq_length, add_bos_token=add_bos_token)
        torch.save(tokenized, cache_path)
    accelerator.wait_for_everyone()
    return torch.load(cache_path, map_location="cpu")


def apply_named_split(
    records: list[dict[str, Any]],
    *,
    proportions: list[float] | None,
    seed: int,
    role: str | None,
    names: list[str],
    group_key: str | None = None,
) -> list[dict[str, Any]]:
    if not proportions or role is None:
        return records
    if len(proportions) != len(names):
        raise ValueError(f"Expected {len(names)} proportions for split names {names}, got {len(proportions)}.")
    if role not in names:
        raise ValueError(f"Unknown split role {role!r}. Expected one of {names}.")
    split_lookup = {
        name: split
        for name, split in zip(
            names,
            split_records_by_proportions(records, proportions, seed=seed, shuffle=True, group_key=group_key),
            strict=True,
        )
    }
    return split_lookup[role]


def resolve_bias_eval_domains(data_path: str | Path, requested: str) -> tuple[list[str], bool]:
    from evaluation.bias_disentangle import load_examples

    examples = load_examples(str(data_path))
    available_domains = sorted({str(example.bias_type).strip().lower() for example in examples if str(example.bias_type).strip()})
    requested_tokens = [token.strip().lower() for token in str(requested).split(",") if token.strip()]
    if not requested_tokens:
        requested_tokens = ["gender"]

    include_all = "all" in requested_tokens
    explicit_domains = [token for token in requested_tokens if token != "all"]
    if include_all:
        domains = list(available_domains)
        for domain in explicit_domains:
            if domain not in available_domains:
                raise ValueError(
                    f"Unknown bias evaluation domain {domain!r}. Available domains: {available_domains}"
                )
        return domains, True

    ordered_domains: list[str] = []
    for domain in explicit_domains:
        if domain not in available_domains:
            raise ValueError(
                f"Unknown bias evaluation domain {domain!r}. Available domains: {available_domains}"
            )
        if domain not in ordered_domains:
            ordered_domains.append(domain)
    return ordered_domains, include_all


def run_bias_eval_suite(
    *,
    base_model_name_or_path: str,
    adapter_path: str | Path,
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
    from evaluation.bias_disentangle import evaluate_saved_adapter

    domains, include_all = resolve_bias_eval_domains(data_path, requested_domains)
    if not domains:
        return {}

    results_by_domain: dict[str, dict[str, float]] = {}
    merged_metrics: dict[str, float] = {}
    for domain in domains:
        domain_metrics = evaluate_saved_adapter(
            base_model_name_or_path=base_model_name_or_path,
            adapter_path=adapter_path,
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


def get_dtype(dtype_name: str) -> torch.dtype:
    lowered = dtype_name.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[lowered]


def configure_attention_backend() -> str | None:
    disable_cudnn_sdp = os.environ.get("SAFEDRIFT_DISABLE_CUDNN_SDP", "1").strip().lower()
    if disable_cudnn_sdp not in {"0", "false", "no"} and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)

    implementation = os.environ.get("SAFEDRIFT_ATTENTION_IMPLEMENTATION", "eager").strip()
    if implementation.lower() in {"", "auto", "default", "none"}:
        return None
    return implementation


def build_base_model_and_tokenizer(args: argparse.Namespace):
    attn_implementation = configure_attention_backend()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": get_dtype(args.torch_dtype),
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    required_vocab_size = max(tokenizer.get_vocab().values()) + 1
    if model.get_input_embeddings().weight.shape[0] < required_vocab_size:
        model.resize_token_embeddings(required_vocab_size)
    model.config.use_cache = False
    return model, tokenizer


def attach_lora_adapter(model, args: argparse.Namespace):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=list(args.lora_target_modules),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        # PEFT + gradient checkpointing needs input activations to require grad,
        # otherwise the checkpointed graph can be detached and loss.backward()
        # fails with "does not require grad".
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            input_embeddings = model.get_input_embeddings()

            def _make_inputs_require_grad(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    output.requires_grad_(True)
                return output

            input_embeddings.register_forward_hook(_make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    return model


def get_trainable_named_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found in LoRA model.")
    return params


def flatten_trainable_parameters(model) -> torch.Tensor:
    return torch.cat([param.detach().float().cpu().view(-1) for _, param in get_trainable_named_parameters(model)])


def trainable_parameter_delta_metrics(model, initial_parameters: torch.Tensor) -> dict[str, float | int]:
    current_parameters = flatten_trainable_parameters(model)
    delta = current_parameters - initial_parameters
    numel = int(delta.numel())
    norm = float(delta.norm().item()) if numel else 0.0
    max_abs = float(delta.abs().max().item()) if numel else 0.0
    return {
        "final_trainable_parameter_count": numel,
        "final_trainable_parameter_delta_l2_norm": norm,
        "final_trainable_parameter_delta_l2_norm_per_sqrt_param": norm / math.sqrt(numel) if numel else 0.0,
        "final_trainable_parameter_delta_max_abs": max_abs,
    }


def active_lora_adapter_names(module) -> list[str]:
    active = getattr(module, "active_adapters", None)
    if callable(active):
        active = active()
    if active is None:
        active = getattr(module, "active_adapter", None)
        if callable(active):
            active = active()
    if isinstance(active, str):
        names = [active]
    elif active is None:
        names = []
    else:
        try:
            names = list(active)
        except TypeError:
            names = []
    if not names and hasattr(module, "lora_A"):
        names = list(getattr(module, "lora_A").keys())
    return [str(name) for name in names]


def lora_delta_weight(module, adapter_name: str) -> torch.Tensor:
    if hasattr(module, "get_delta_weight"):
        return module.get_delta_weight(adapter_name)
    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
        raise ValueError("module does not expose LoRA delta weights")
    lora_a = getattr(module, "lora_A")[adapter_name].weight
    lora_b = getattr(module, "lora_B")[adapter_name].weight
    scaling = getattr(module, "scaling", {}).get(adapter_name, 1.0)
    delta = (lora_b @ lora_a) * float(scaling)
    if bool(getattr(module, "fan_in_fan_out", False)):
        delta = delta.T
    return delta


@torch.no_grad()
def lora_model_delta_metrics(model) -> dict[str, float | int | list[dict[str, str]]]:
    total_sq_norm = 0.0
    total_numel = 0
    max_abs = 0.0
    module_count = 0
    adapter_delta_count = 0
    skipped: list[dict[str, str]] = []

    for module_name, module in model.named_modules():
        has_lora_delta = hasattr(module, "get_delta_weight") or (hasattr(module, "lora_A") and hasattr(module, "lora_B"))
        if not has_lora_delta:
            continue
        module_had_delta = False
        for adapter_name in active_lora_adapter_names(module):
            try:
                delta = lora_delta_weight(module, adapter_name).detach().float()
            except Exception as exc:
                skipped.append({"module": module_name, "adapter": adapter_name, "error": str(exc)})
                continue
            if delta.numel() == 0:
                continue
            total_sq_norm += float(delta.square().sum().item())
            max_abs = max(max_abs, float(delta.abs().max().item()))
            total_numel += int(delta.numel())
            adapter_delta_count += 1
            module_had_delta = True
        if module_had_delta:
            module_count += 1

    norm = math.sqrt(total_sq_norm)
    return {
        "final_model_weight_delta_l2_norm": norm,
        "final_model_weight_delta_l2_norm_per_sqrt_param": norm / math.sqrt(total_numel) if total_numel else 0.0,
        "final_model_weight_delta_max_abs": max_abs,
        "final_model_weight_delta_parameter_count": total_numel,
        "final_model_weight_delta_module_count": module_count,
        "final_model_weight_delta_adapter_count": adapter_delta_count,
        "final_model_weight_delta_skipped_module_count": len(skipped),
        "final_model_weight_delta_skipped_modules": skipped[:20],
    }


def flatten_trainable_gradients(model) -> torch.Tensor:
    grads = []
    for _, param in get_trainable_named_parameters(model):
        if param.grad is None:
            grads.append(torch.zeros_like(param, dtype=torch.float32, device=param.device).view(-1).cpu())
        else:
            grads.append(param.grad.detach().float().view(-1).cpu())
    return torch.cat(grads)


def metric_cache_key(
    *,
    data_path: str | Path | None,
    model_name_or_path: str,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    add_bos_token: bool,
    seed: int,
    max_examples: int | None,
    dataset_name: str | None = None,
    dataset_subset: str | None = None,
    dataset_split: str | None = None,
    dataset_label: str | None = None,
    split_tag: str | None = None,
) -> str:
    if data_path is not None:
        data_tag = file_fingerprint(data_path)
    else:
        data_tag = dataset_fingerprint(
            dataset_name or "unknown",
            dataset_subset or "unknown",
            dataset_split or "unknown",
            dataset_label or "unknown",
        )
    payload = {
        "data": data_tag,
        "model": model_name_or_path,
        "seq": max_seq_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "add_bos_token": add_bos_token,
        "seed": seed,
        "max_examples": max_examples,
        "split_tag": split_tag,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def seeded_record_subset(
    records: list[dict[str, Any]],
    *,
    max_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_examples is None or max_examples <= 0 or len(records) <= max_examples:
        return list(records)
    indices = list(range(len(records)))
    random.Random(int(seed)).shuffle(indices)
    return [records[index] for index in indices[: int(max_examples)]]


def compute_average_gradient(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_seq_length: int,
    add_bos_token: bool,
    max_examples: int | None,
) -> torch.Tensor:
    subset = records if max_examples is None else records[: max(1, min(len(records), int(max_examples)))]
    model.eval()
    gradient_sum = None
    used_examples = 0
    for record in subset:
        batch = encode_with_messages_format(
            record,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            add_bos_token=add_bos_token,
        )
        if not has_supervised_targets(batch):
            continue
        batch = move_batch_to_device({key: value.unsqueeze(0) for key, value in batch.items()}, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        gradient = flatten_trainable_gradients(model)
        if gradient_sum is None:
            gradient_sum = gradient
        else:
            gradient_sum += gradient
        model.zero_grad(set_to_none=True)
        used_examples += 1
    if gradient_sum is None or used_examples == 0:
        raise ValueError("Cannot compute a validation gradient on an empty record list.")
    return gradient_sum / float(used_examples)


def compute_diagonal_fisher(
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_seq_length: int,
    add_bos_token: bool,
    max_examples: int | None,
) -> torch.Tensor:
    subset = records if max_examples is None else records[: max(1, min(len(records), int(max_examples)))]
    model.eval()
    fisher_sum = None
    used_examples = 0
    for record in subset:
        batch = encode_with_messages_format(
            record,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            add_bos_token=add_bos_token,
        )
        if not has_supervised_targets(batch):
            continue
        batch = move_batch_to_device({key: value.unsqueeze(0) for key, value in batch.items()}, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        outputs.loss.backward()
        gradient = flatten_trainable_gradients(model)
        squared = gradient.square()
        if fisher_sum is None:
            fisher_sum = squared
        else:
            fisher_sum += squared
        model.zero_grad(set_to_none=True)
        used_examples += 1
    if fisher_sum is None or used_examples == 0:
        raise ValueError("Cannot compute reference Fisher on an empty record list.")
    return fisher_sum / float(used_examples)


def get_or_compute_metric_tensor(
    *,
    accelerator: Accelerator,
    cache_path: Path,
    overwrite_cache: bool,
    compute_fn,
) -> TimingResult:
    wallclock_seconds = 0.0
    cache_hit = cache_path.exists() and not overwrite_cache
    if accelerator.is_main_process and not cache_hit:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        tensor = compute_fn().cpu()
        wallclock_seconds = time.perf_counter() - start
        torch.save(tensor, cache_path)
    accelerator.wait_for_everyone()
    tensor = torch.load(cache_path, map_location="cpu")
    return TimingResult(
        tensor=tensor,
        wallclock_seconds=wallclock_seconds,
        cache_hit=cache_hit,
        cache_path=str(cache_path),
    )


def estimate_batch_training_flops(model, batch: dict[str, torch.Tensor]) -> float:
    """Estimate forward+backward FLOPs using the Transformers convention."""
    input_batch = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    floating_point_ops = getattr(model, "floating_point_ops", None)
    if callable(floating_point_ops):
        try:
            return float(floating_point_ops(input_batch))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        token_count = int(attention_mask.detach().sum().item())
    else:
        token_count = int(batch["input_ids"].numel())
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    return float(6 * token_count * parameter_count)


@contextmanager
def adapters_disabled(model):
    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        with disable_adapter():
            yield
        return
    yield


def shifted_supervised_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch.get("labels")
    if labels is not None:
        mask = labels[:, 1:].ne(-100)
        if mask.any():
            return mask
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return attention_mask[:, 1:].bool()
    return torch.ones_like(batch["input_ids"][:, 1:], dtype=torch.bool)


def token_kl_from_logits(
    *,
    adapted_logits: torch.Tensor,
    base_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    adapted_log_probs = F.log_softmax(adapted_logits[:, :-1, :].float(), dim=-1)
    base_log_probs = F.log_softmax(base_logits[:, :-1, :].float(), dim=-1)
    token_kl = F.kl_div(
        adapted_log_probs,
        base_log_probs.exp(),
        reduction="none",
        log_target=False,
    ).sum(dim=-1)
    mask = token_mask.to(device=token_kl.device, dtype=token_kl.dtype)
    return (token_kl * mask).sum(), mask.sum()


def compute_reference_kl_regularization_loss(
    model,
    batch: dict[str, torch.Tensor],
    accelerator: Accelerator,
) -> torch.Tensor:
    unwrapped = accelerator.unwrap_model(model)
    was_training = bool(model.training)
    model.eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "token_type_ids"}
    }
    with torch.no_grad():
        with adapters_disabled(unwrapped):
            base_logits = unwrapped(**model_inputs).logits.detach()
    adapted_logits = model(**model_inputs).logits
    kl_sum, token_count = token_kl_from_logits(
        adapted_logits=adapted_logits,
        base_logits=base_logits,
        token_mask=shifted_supervised_mask(batch),
    )
    if was_training:
        model.train()
    return kl_sum / token_count.clamp_min(1.0)


@torch.no_grad()
def evaluate_realized_heldout_kl(
    model,
    dataloader,
    accelerator: Accelerator,
) -> tuple[float, float]:
    unwrapped = accelerator.unwrap_model(model)
    was_training = bool(model.training)
    model.eval()
    local_kl_sum = 0.0
    local_token_count = 0.0
    estimated_forward_flops = 0.0
    for batch in dataloader:
        model_inputs = {
            key: value
            for key, value in batch.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with adapters_disabled(unwrapped):
            base_logits = unwrapped(**model_inputs).logits
        adapted_logits = model(**model_inputs).logits
        kl_sum, token_count = token_kl_from_logits(
            adapted_logits=adapted_logits,
            base_logits=base_logits,
            token_mask=shifted_supervised_mask(batch),
        )
        local_kl_sum += float(kl_sum.item())
        local_token_count += float(token_count.item())
        estimated_forward_flops += (2.0 / 3.0) * estimate_batch_training_flops(unwrapped, batch)
    totals = torch.tensor(
        [local_kl_sum, local_token_count, estimated_forward_flops],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered = accelerator.gather_for_metrics(totals).reshape(-1, 3).sum(dim=0)
    if was_training:
        model.train()
    token_count = max(1.0, float(gathered[1].item()))
    return float(gathered[0].item()) / token_count, float(gathered[2].item())


@torch.no_grad()
def evaluate_validation_loss_with_flops(model, dataloader, accelerator: Accelerator) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    estimated_forward_flops = 0.0
    unwrapped = accelerator.unwrap_model(model)
    for batch in dataloader:
        if not batch["labels"].ne(-100).any():
            continue
        outputs = model(**batch)
        loss = outputs.loss.detach()
        reduced = accelerator.gather_for_metrics(loss.unsqueeze(0))
        total_loss += float(reduced.mean().item())
        total_batches += 1
        estimated_forward_flops += estimate_batch_training_flops(unwrapped, batch) / 3.0
    gathered_flops = accelerator.gather_for_metrics(
        torch.tensor([estimated_forward_flops], dtype=torch.float64, device=accelerator.device)
    ).sum()
    model.train()
    if total_batches == 0:
        return 0.0, float(gathered_flops.item())
    return total_loss / float(total_batches), float(gathered_flops.item())


def evaluate_validation_loss(model, dataloader, accelerator: Accelerator) -> float:
    loss, _ = evaluate_validation_loss_with_flops(model, dataloader, accelerator)
    return loss


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = 0.5 * ((position + 1) + end)
        for ordered_index in order[position:end]:
            ranks[ordered_index] = average_rank
        position = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float | None:
    pairs = [
        (float(left_value), float(right_value))
        for left_value, right_value in zip(left, right)
        if math.isfinite(float(left_value)) and math.isfinite(float(right_value))
    ]
    if len(pairs) < 3:
        return None
    left_ranks = average_ranks([pair[0] for pair in pairs])
    right_ranks = average_ranks([pair[1] for pair in pairs])
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((rank - left_mean) ** 2 for rank in left_ranks))
    right_scale = math.sqrt(sum((rank - right_mean) ** 2 for rank in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    return numerator / (left_scale * right_scale)


def trajectory_correlation_summary(points: list[dict[str, Any]]) -> dict[str, float | int | None]:
    fisher = [float(point["predicted_fisher_cost"]) for point in points]
    kl = [float(point["realized_heldout_kl"]) for point in points]
    degradation = [float(point["target_loss_degradation"]) for point in points]
    performance = [float(point["target_performance_proxy"]) for point in points]
    summary: dict[str, float | int | None] = {
        "trajectory_point_count": len(points),
        "spearman_predicted_fisher_vs_realized_kl": spearman_correlation(fisher, kl),
        "spearman_realized_kl_vs_target_degradation": spearman_correlation(kl, degradation),
        "spearman_realized_kl_vs_target_performance": spearman_correlation(kl, performance),
        "spearman_predicted_fisher_vs_target_degradation": spearman_correlation(fisher, degradation),
    }
    optional_pairs = {
        "spearman_realized_kl_vs_medqa_accuracy": "target_medqa_accuracy",
        "spearman_realized_kl_vs_reference_degradation": "reference_performance_degradation",
        "spearman_realized_kl_vs_instruction_degradation": "instruction_performance_degradation",
    }
    for output_key, point_key in optional_pairs.items():
        metric_points = [point for point in points if point_key in point]
        if metric_points:
            summary[output_key] = spearman_correlation(
                [float(point["realized_heldout_kl"]) for point in metric_points],
                [float(point[point_key]) for point in metric_points],
            )
    return summary


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_named_benchmark_records(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    benchmark_records: dict[str, list[dict[str, Any]]] = {}
    for benchmark_name in args.benchmark_evals:
        if benchmark_name == "mmlu":
            benchmark_records["mmlu"] = load_mmlu_records(
                subset=args.benchmark_mmlu_subset,
                split=args.benchmark_mmlu_split,
            )
        elif benchmark_name == "gsm8k":
            benchmark_records["gsm8k"] = load_gsm8k_records(
                subset=args.benchmark_gsm8k_subset,
                split=args.benchmark_gsm8k_split,
            )
        else:
            raise ValueError(f"Unsupported benchmark evaluator: {benchmark_name}")
    return benchmark_records


def evaluate_named_record_sets(
    *,
    model,
    tokenizer,
    device: torch.device,
    record_sets: dict[str, list[dict[str, Any]]],
    max_examples: int | None,
    max_new_tokens: int,
    add_bos_token: bool,
    prefix: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for evaluator_name, records in record_sets.items():
        evaluator = get_evaluator(evaluator_name)
        evaluator_metrics = evaluator(
            model,
            tokenizer,
            records,
            device=device,
            max_examples=max_examples,
            max_new_tokens=max_new_tokens,
            add_bos_token=add_bos_token,
        )
        metrics.update({f"{prefix}{key}": value for key, value in evaluator_metrics.items()})
    return metrics


def load_extra_eval_record_sets(args: argparse.Namespace) -> list[dict[str, Any]]:
    record_sets: list[dict[str, Any]] = []
    for spec in parse_extra_eval_specs(args.extra_evals):
        if spec["evaluator"] in {"none", "loss", "bias_disentangle"}:
            records: list[dict[str, Any]] = []
        else:
            records = load_records(spec["file"])
        record_sets.append({**spec, "records": records})
    return record_sets


def evaluate_extra_record_sets(
    *,
    model,
    tokenizer,
    device: torch.device,
    record_sets: list[dict[str, Any]],
    max_examples: int | None,
    max_new_tokens: int,
    add_bos_token: bool,
    prefix: str,
    humaneval_num_samples: int,
    humaneval_pass_at_ks: tuple[int, ...],
    humaneval_temperature: float,
    humaneval_top_p: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for item in record_sets:
        records = item.get("records") or []
        if not records:
            continue
        evaluator_metrics = evaluate_records_with_config(
            evaluator_name=str(item["evaluator"]),
            model=model,
            tokenizer=tokenizer,
            records=records,
            device=device,
            max_examples=max_examples,
            max_new_tokens=max_new_tokens,
            add_bos_token=add_bos_token,
            humaneval_num_samples=humaneval_num_samples,
            humaneval_pass_at_ks=humaneval_pass_at_ks,
            humaneval_temperature=humaneval_temperature,
            humaneval_top_p=humaneval_top_p,
        )
        metrics.update(
            {
                f"{prefix}{item['name']}_{key}": value
                for key, value in evaluator_metrics.items()
            }
        )
    return metrics


def evaluate_records_with_config(
    *,
    evaluator_name: str,
    model,
    tokenizer,
    records: list[dict[str, Any]],
    device: torch.device,
    max_examples: int | None,
    max_new_tokens: int,
    add_bos_token: bool,
    humaneval_num_samples: int,
    humaneval_pass_at_ks: tuple[int, ...],
    humaneval_temperature: float,
    humaneval_top_p: float,
) -> dict[str, float]:
    if evaluator_name == "humaneval":
        return humaneval_evaluator.evaluate_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            device=device,
            max_examples=max_examples,
            max_new_tokens=max_new_tokens,
            add_bos_token=add_bos_token,
            num_samples=humaneval_num_samples,
            pass_at_ks=humaneval_pass_at_ks,
            temperature=humaneval_temperature,
            top_p=humaneval_top_p,
        )
    evaluator = get_evaluator(evaluator_name)
    return evaluator(
        model,
        tokenizer,
        records,
        device=device,
        max_examples=max_examples,
        max_new_tokens=max_new_tokens,
        add_bos_token=add_bos_token,
    )


REFERENCE_NUMERIC_METRICS = (
    "SS",
    "LMS",
    "ICAT",
    "mean_bias_margin",
    "mean_lm_margin",
)


def _extract_bias_eval_aliases(metrics: dict[str, float]) -> dict[str, float]:
    aliases: dict[str, float] = {}
    preferred_layer = "layerneg1"

    for metric in REFERENCE_NUMERIC_METRICS:
        ft_key = f"bias_disentangle_all_{preferred_layer}_{metric}_ft"
        base_key = f"bias_disentangle_all_{preferred_layer}_{metric}_base"
        delta_key = f"bias_disentangle_all_{preferred_layer}_delta_{metric}"

        if ft_key not in metrics:
            ft_key = next((key for key in metrics if key.endswith(f"_{metric}_ft")), "")
        if base_key not in metrics:
            base_key = next((key for key in metrics if key.endswith(f"_{metric}_base")), "")
        if delta_key not in metrics:
            delta_key = next((key for key in metrics if key.endswith(f"_delta_{metric}")), "")

        if ft_key:
            aliases[f"reference_{metric}"] = float(metrics[ft_key])
        if base_key:
            aliases[f"reference_base_{metric}"] = float(metrics[base_key])
        if delta_key:
            aliases[f"reference_delta_{metric}"] = float(metrics[delta_key])
    return aliases


def run_required_final_evaluations(
    *,
    model,
    tokenizer,
    args: argparse.Namespace,
    accelerator: Accelerator,
    eval_records: list[dict[str, Any]],
    ood_eval_records: list[dict[str, Any]],
    reference_validation_records: list[dict[str, Any]],
    reference_eval_records: list[dict[str, Any]],
    extra_eval_record_sets: list[dict[str, Any]],
    final_dir: Path,
    output_dir: Path,
) -> dict[str, float]:
    summary_updates: dict[str, float] = {}
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.eval()

    if args.evaluator not in {"none", "bias_disentangle"} and eval_records:
        target_metrics = evaluate_records_with_config(
            evaluator_name=args.evaluator,
            model=unwrapped,
            tokenizer=tokenizer,
            records=eval_records,
            device=accelerator.device,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(target_metrics)
        summary_updates.update(prefixed_metrics("target_", target_metrics))

    ood_evaluator = args.ood_evaluator or args.evaluator
    if ood_evaluator not in {"none", "bias_disentangle"} and ood_eval_records:
        ood_metrics = evaluate_records_with_config(
            evaluator_name=ood_evaluator,
            model=unwrapped,
            tokenizer=tokenizer,
            records=ood_eval_records,
            device=accelerator.device,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.ood_generation_max_new_tokens or args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(prefixed_metrics("ood_", ood_metrics))

    if args.reference_evaluator not in {"none", "loss", "bias_disentangle"} and reference_validation_records:
        reference_validation_metrics = evaluate_records_with_config(
            evaluator_name=args.reference_evaluator,
            model=unwrapped,
            tokenizer=tokenizer,
            records=reference_validation_records,
            device=accelerator.device,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(prefixed_metrics("reference_validation_", reference_validation_metrics))

    if args.reference_evaluator not in {"none", "loss", "bias_disentangle"} and reference_eval_records:
        reference_metrics = evaluate_records_with_config(
            evaluator_name=args.reference_evaluator,
            model=unwrapped,
            tokenizer=tokenizer,
            records=reference_eval_records,
            device=accelerator.device,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(prefixed_metrics("reference_", reference_metrics))

    if extra_eval_record_sets:
        extra_metrics = evaluate_extra_record_sets(
            model=unwrapped,
            tokenizer=tokenizer,
            device=accelerator.device,
            record_sets=extra_eval_record_sets,
            max_examples=args.eval_max_examples,
            max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            prefix="extra_eval_",
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(extra_metrics)

    if args.bias_eval_data_path:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        bias_metrics = run_bias_eval_suite(
            base_model_name_or_path=args.model_name_or_path,
            adapter_path=final_dir,
            data_path=args.bias_eval_data_path,
            requested_domains=args.bias_eval_domain,
            layers=args.bias_eval_layers,
            alpha=args.bias_eval_alpha,
            geometry=args.bias_eval_geometry,
            direction_split=args.bias_eval_direction_split,
            seed=args.seed,
            batch_size=args.bias_eval_batch_size,
            max_length=args.bias_eval_max_length,
            max_examples=args.bias_eval_max_examples,
            torch_dtype=args.torch_dtype,
            device=str(accelerator.device),
            trust_remote_code=args.trust_remote_code,
            output_dir=output_dir / "bias_disentangle",
        )
        summary_updates.update(bias_metrics)
        summary_updates.update(_extract_bias_eval_aliases(bias_metrics))
    else:
        unwrapped.train()
    return summary_updates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General LoRA SFT runner with pluggable evaluation heads.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--candidate-validation-file", default=None)
    parser.add_argument("--candidate-test-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--validation-split-proportions", nargs=3, type=float, default=None)
    parser.add_argument("--validation-split-seed", type=int, default=42)
    parser.add_argument("--validation-split-role", choices=["selector_target", "val", "final_test"], default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--eval-split-proportions", nargs=3, type=float, default=None)
    parser.add_argument("--eval-split-seed", type=int, default=42)
    parser.add_argument("--eval-split-role", choices=["selector_target", "val", "final_test"], default=None)
    parser.add_argument("--ood-eval-file", default=None)
    parser.add_argument("--reference-file", default=None)
    parser.add_argument("--reference-validation-file", default=None)
    parser.add_argument("--reference-test-file", default=None)
    parser.add_argument("--reference-composition-name", default=None)
    parser.add_argument("--reference-composition-domains", nargs="*", default=[])
    parser.add_argument("--reference-hf-stereoset", action="store_true")
    parser.add_argument("--reference-hf-subset", default="intrasentence")
    parser.add_argument("--reference-hf-split", default="validation")
    parser.add_argument("--reference-hf-label", default="stereotype")
    parser.add_argument("--reference-hf-format", default="instruction", choices=["instruction", "messages"])
    parser.add_argument("--reference-split-proportions", nargs=2, type=float, default=None)
    parser.add_argument("--reference-split-seed", type=int, default=42)
    parser.add_argument("--reference-split-role", choices=["reference", "bias_eval"], default=None)
    parser.add_argument("--reference-split-group-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--overwrite-cache", action="store_true")

    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--add-bos-token", action="store_true")
    parser.add_argument("--use-slow-tokenizer", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--torch-dtype", default="bf16")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=list(DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--skip-final-evaluation", action="store_true")
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
    parser.add_argument("--selection-method", default=None)
    parser.add_argument("--selection-preconditioner", default=None)
    parser.add_argument("--safe-rho", type=float, default=None)
    parser.add_argument("--safe-epsilon", type=float, default=None)
    parser.add_argument(
        "--safe-training-constraint-mode",
        choices=["none", "cumulative_line_search", "equal_allocation"],
        default="none",
        help=(
            "Optional hard control for realized optimizer updates. cumulative_line_search "
            "keeps every displacement from theta0 inside the global SAFE set; equal_allocation "
            "uses per-step epsilon/T and rho/T^2 budgets."
        ),
    )
    parser.add_argument(
        "--safe-training-epsilon-scale-by-steps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use safe_epsilon * T as the realized-training norm radius, where T is "
            "the resolved optimizer-step horizon. The selector still uses safe_epsilon."
        ),
    )
    parser.add_argument("--selected-predicted-fisher-cost", type=float, default=None)
    parser.add_argument("--selected-predicted-norm-cost", type=float, default=None)

    parser.add_argument(
        "--evaluator",
        default="none",
        choices=[
            "none",
            "medqa",
            "mmlu",
            "gsm8k",
            "esconv",
            "humaneval",
            "ifeval",
            "ifeval_proxy",
            "medical_reference",
            "mcq_ood",
            "bias_disentangle",
        ],
    )
    parser.add_argument(
        "--ood-evaluator",
        default=None,
        choices=[
            "none",
            "medqa",
            "mmlu",
            "gsm8k",
            "esconv",
            "humaneval",
            "ifeval",
            "ifeval_proxy",
            "medical_reference",
            "mcq_ood",
            "bias_disentangle",
        ],
    )
    parser.add_argument(
        "--reference-evaluator",
        default="none",
        choices=[
            "none",
            "loss",
            "medqa",
            "mmlu",
            "gsm8k",
            "esconv",
            "humaneval",
            "ifeval",
            "ifeval_proxy",
            "medical_reference",
            "bias_disentangle",
        ],
    )
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--ood-generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--reference-generation-max-new-tokens", type=int, default=None)
    parser.add_argument("--humaneval-num-samples", type=int, default=1)
    parser.add_argument("--humaneval-pass-at-ks", nargs="+", type=int, default=[1])
    parser.add_argument("--humaneval-temperature", type=float, default=0.8)
    parser.add_argument("--humaneval-top-p", type=float, default=0.95)
    parser.add_argument("--benchmark-evals", nargs="+", choices=["mmlu", "gsm8k"], default=[])
    parser.add_argument(
        "--extra-evals",
        nargs="*",
        default=[],
        help="Named held-out evals as NAME=EVALUATOR:/path/to/file.jsonl.",
    )
    parser.add_argument("--evaluate-base-model", action="store_true")
    parser.add_argument("--evaluate-base-ood", action="store_true")
    parser.add_argument("--benchmark-max-examples", type=int, default=None)
    parser.add_argument("--benchmark-mmlu-subset", default="all")
    parser.add_argument("--benchmark-mmlu-split", default="test")
    parser.add_argument("--benchmark-gsm8k-subset", default="main")
    parser.add_argument("--benchmark-gsm8k-split", default="test")
    parser.add_argument("--bias-eval-data-path", default=None)
    parser.add_argument("--bias-eval-domain", default="gender")
    parser.add_argument("--bias-eval-layers", default="-1")
    parser.add_argument("--bias-eval-alpha", type=float, default=0.5)
    parser.add_argument("--bias-eval-geometry", choices=["sand-e", "sand-w"], default="sand-e")
    parser.add_argument("--bias-eval-direction-split", type=float, default=0.5)
    parser.add_argument("--bias-eval-batch-size", type=int, default=8)
    parser.add_argument("--bias-eval-max-length", type=int, default=512)
    parser.add_argument("--bias-eval-max-examples", type=int, default=None)

    parser.add_argument("--compute-validation-gradient", action="store_true")
    parser.add_argument("--validation-gradient-max-examples", type=int, default=128)
    parser.add_argument("--compute-reference-fisher", action="store_true")
    parser.add_argument("--reference-fisher-max-examples", type=int, default=1024)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kl_regularization_lambda < 0:
        raise ValueError("--kl-regularization-lambda must be non-negative.")
    if args.trajectory_eval_steps < 0:
        raise ValueError("--trajectory-eval-steps must be non-negative.")
    if args.trajectory_generation_eval_steps < 0:
        raise ValueError("--trajectory-generation-eval-steps must be non-negative.")
    if args.safe_training_constraint_mode != "none":
        if args.safe_rho is None and args.safe_epsilon is None:
            raise ValueError(
                "--safe-training-constraint-mode requires --safe-rho and/or --safe-epsilon."
            )
        if args.safe_rho is not None and not args.compute_reference_fisher:
            raise ValueError(
                "Fisher trajectory control requires --compute-reference-fisher."
            )
    if args.safe_training_epsilon_scale_by_steps:
        if args.safe_training_constraint_mode == "none":
            raise ValueError(
                "--safe-training-epsilon-scale-by-steps requires a SAFE training "
                "constraint mode."
            )
        if args.safe_epsilon is None:
            raise ValueError(
                "--safe-training-epsilon-scale-by-steps requires --safe-epsilon."
            )
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else output_dir / "cache"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "run_config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() and int(os.environ.get("RANK", "0")) == 0:
        metrics_path.unlink()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    set_seed(args.seed)
    if args.evaluator == "bias_disentangle" and not args.bias_eval_data_path:
        raise ValueError("--bias-eval-data-path is required when --evaluator bias_disentangle is used.")

    if accelerator.is_main_process:
        save_json(config_path, vars(args))

    train_records = load_records(args.train_file)
    safe_training_weights = [
        float(record["safe_training_weight"])
        for record in train_records
        if "safe_training_weight" in record
    ]
    if safe_training_weights and len(safe_training_weights) != len(train_records):
        raise ValueError("SAFE training files must provide safe_training_weight for every record or none.")
    noop_training = bool(safe_training_weights) and max(safe_training_weights) == 0.0
    candidate_validation_records = load_records(args.candidate_validation_file) if args.candidate_validation_file else []
    candidate_test_records = load_records(args.candidate_test_file) if args.candidate_test_file else []
    validation_records = load_records(args.validation_file) if args.validation_file else []
    validation_records = apply_named_split(
        validation_records,
        proportions=args.validation_split_proportions,
        seed=args.validation_split_seed,
        role=args.validation_split_role,
        names=["selector_target", "val", "final_test"],
    )
    if args.eval_file:
        eval_records = load_records(args.eval_file)
    else:
        eval_records = list(validation_records)
    eval_records = apply_named_split(
        eval_records,
        proportions=args.eval_split_proportions,
        seed=args.eval_split_seed,
        role=args.eval_split_role,
        names=["selector_target", "val", "final_test"],
    )
    ood_eval_records = load_records(args.ood_eval_file) if args.ood_eval_file else []
    if args.reference_file:
        all_reference_records = load_records(args.reference_file)
        reference_source = "file"
    elif args.reference_hf_stereoset:
        all_reference_records = [
            coerce_record(record)
            for record in load_stereoset_records(
                subset=args.reference_hf_subset,
                split=args.reference_hf_split,
                label=args.reference_hf_label,
                output_format=args.reference_hf_format,
            )
        ]
        reference_source = "hf"
    else:
        all_reference_records = []
        reference_source = "none"
    reference_records = apply_named_split(
        all_reference_records,
        proportions=args.reference_split_proportions,
        seed=args.reference_split_seed,
        role=args.reference_split_role,
        names=["reference", "bias_eval"],
        group_key=args.reference_split_group_key,
    )
    reference_validation_records = load_records(args.reference_validation_file) if args.reference_validation_file else []
    reference_eval_records = load_records(args.reference_test_file) if args.reference_test_file else []
    if not reference_eval_records and all_reference_records and args.reference_split_proportions:
        reference_eval_records = apply_named_split(
            all_reference_records,
            proportions=args.reference_split_proportions,
            seed=args.reference_split_seed,
            role="bias_eval",
            names=["reference", "bias_eval"],
            group_key=args.reference_split_group_key,
        )
    kl_regularization_records = seeded_record_subset(
        reference_records,
        max_examples=args.kl_reference_max_examples,
        seed=args.seed + 104729,
    )
    trajectory_reference_source = reference_eval_records or reference_validation_records
    trajectory_reference_records = seeded_record_subset(
        trajectory_reference_source,
        max_examples=args.trajectory_reference_max_examples,
        seed=args.seed + 130363,
    )
    trajectory_target_records = seeded_record_subset(
        validation_records,
        max_examples=args.trajectory_target_max_examples,
        seed=args.seed + 155921,
    )
    trajectory_generation_target_records = seeded_record_subset(
        validation_records,
        max_examples=args.trajectory_generation_max_examples,
        seed=args.seed + 179953,
    )
    trajectory_generation_reference_records = seeded_record_subset(
        trajectory_reference_source,
        max_examples=args.trajectory_generation_max_examples,
        seed=args.seed + 196613,
    )
    trajectory_instruction_source = (
        load_records(args.trajectory_instruction_file)
        if args.trajectory_instruction_file
        else []
    )
    trajectory_generation_instruction_records = seeded_record_subset(
        trajectory_instruction_source,
        max_examples=args.trajectory_generation_max_examples,
        seed=args.seed + 214748,
    )
    if args.kl_regularization_lambda > 0 and not kl_regularization_records:
        raise ValueError("KL regularization requires reference fitting records.")
    if args.trajectory_eval_steps > 0 and not trajectory_reference_records:
        raise ValueError("Trajectory diagnostics require held-out reference records.")
    if args.trajectory_eval_steps > 0 and not trajectory_target_records:
        raise ValueError("Trajectory diagnostics require target validation records.")
    if args.trajectory_generation_eval_steps > 0 and not trajectory_generation_target_records:
        raise ValueError("Trajectory generation diagnostics require target validation records.")
    validation_split_tag = None
    if args.validation_split_proportions and args.validation_split_role:
        validation_split_tag = (
            f"{file_fingerprint(args.validation_file)}_"
            f"{safe_slug(args.validation_split_role)}_"
            f"{'-'.join(str(value) for value in args.validation_split_proportions)}_"
            f"seed{args.validation_split_seed}"
        )
    reference_split_tag = None
    if args.reference_split_proportions and args.reference_split_role:
        source_name = args.reference_file if args.reference_file else f"stereoset_{args.reference_hf_subset}_{args.reference_hf_split}_{args.reference_hf_label}"
        reference_split_tag = (
            f"{safe_slug(str(source_name))}_"
            f"{safe_slug(args.reference_split_role)}_"
            f"{'-'.join(str(value) for value in args.reference_split_proportions)}_"
            f"seed{args.reference_split_seed}"
        )

    benchmark_records = load_named_benchmark_records(args) if args.benchmark_evals else {}
    extra_eval_record_sets = load_extra_eval_record_sets(args)

    base_model, tokenizer = build_base_model_and_tokenizer(args)
    if args.evaluate_base_model:
        base_model.to(accelerator.device)
        base_model.eval()
    base_eval_metrics: dict[str, float] = {}
    if accelerator.is_main_process and args.evaluate_base_model:
        if benchmark_records:
            base_eval_metrics.update(
                evaluate_named_record_sets(
                    model=base_model,
                    tokenizer=tokenizer,
                    device=accelerator.device,
                    record_sets=benchmark_records,
                    max_examples=args.benchmark_max_examples,
                    max_new_tokens=args.generation_max_new_tokens,
                    add_bos_token=args.add_bos_token,
                    prefix="base_",
                )
            )
            append_jsonl(metrics_path, {"type": "benchmark_base", **base_eval_metrics})
        if args.evaluator not in {"none", "bias_disentangle"} and eval_records:
            target_base_metrics = evaluate_records_with_config(
                evaluator_name=args.evaluator,
                model=base_model,
                tokenizer=tokenizer,
                records=eval_records,
                device=accelerator.device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
                humaneval_num_samples=args.humaneval_num_samples,
                humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                humaneval_temperature=args.humaneval_temperature,
                humaneval_top_p=args.humaneval_top_p,
            )
            prefixed_target_base = {f"base_target_{key}": value for key, value in target_base_metrics.items()}
            base_eval_metrics.update(prefixed_target_base)
            append_jsonl(metrics_path, {"type": "base_target_evaluation", **prefixed_target_base})
        ood_evaluator = args.ood_evaluator or args.evaluator
        if args.evaluate_base_ood and ood_evaluator not in {"none", "bias_disentangle"} and ood_eval_records:
            ood_metrics = evaluate_records_with_config(
                evaluator_name=ood_evaluator,
                model=base_model,
                tokenizer=tokenizer,
                records=ood_eval_records,
                device=accelerator.device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.ood_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
                humaneval_num_samples=args.humaneval_num_samples,
                humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                humaneval_temperature=args.humaneval_temperature,
                humaneval_top_p=args.humaneval_top_p,
            )
            prefixed_ood_base = {f"base_ood_{key}": value for key, value in ood_metrics.items()}
            base_eval_metrics.update(prefixed_ood_base)
            append_jsonl(metrics_path, {"type": "base_ood_evaluation", **prefixed_ood_base})
        if args.reference_evaluator not in {"none", "loss", "bias_disentangle"} and reference_eval_records:
            reference_base_metrics = evaluate_records_with_config(
                evaluator_name=args.reference_evaluator,
                model=base_model,
                tokenizer=tokenizer,
                records=reference_eval_records,
                device=accelerator.device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
                humaneval_num_samples=args.humaneval_num_samples,
                humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                humaneval_temperature=args.humaneval_temperature,
                humaneval_top_p=args.humaneval_top_p,
            )
            prefixed_reference_base = {f"base_reference_{key}": value for key, value in reference_base_metrics.items()}
            base_eval_metrics.update(prefixed_reference_base)
            append_jsonl(metrics_path, {"type": "reference_base_evaluation", **prefixed_reference_base})
        if args.reference_evaluator not in {"none", "loss", "bias_disentangle"} and reference_validation_records:
            reference_validation_base_metrics = evaluate_records_with_config(
                evaluator_name=args.reference_evaluator,
                model=base_model,
                tokenizer=tokenizer,
                records=reference_validation_records,
                device=accelerator.device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
                humaneval_num_samples=args.humaneval_num_samples,
                humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                humaneval_temperature=args.humaneval_temperature,
                humaneval_top_p=args.humaneval_top_p,
            )
            prefixed_reference_validation_base = {
                f"base_reference_validation_{key}": value
                for key, value in reference_validation_base_metrics.items()
            }
            base_eval_metrics.update(prefixed_reference_validation_base)
            append_jsonl(
                metrics_path,
                {"type": "reference_validation_base_evaluation", **prefixed_reference_validation_base},
            )
        if extra_eval_record_sets:
            extra_base_metrics = evaluate_extra_record_sets(
                model=base_model,
                tokenizer=tokenizer,
                device=accelerator.device,
                record_sets=extra_eval_record_sets,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.reference_generation_max_new_tokens or args.generation_max_new_tokens,
                add_bos_token=args.add_bos_token,
                prefix="base_extra_eval_",
                humaneval_num_samples=args.humaneval_num_samples,
                humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                humaneval_temperature=args.humaneval_temperature,
                humaneval_top_p=args.humaneval_top_p,
            )
            base_eval_metrics.update(extra_base_metrics)
            append_jsonl(metrics_path, {"type": "extra_eval_base", **extra_base_metrics})
    accelerator.wait_for_everyone()
    model = attach_lora_adapter(base_model, args)
    # Validation-gradient and reference-Fisher probes run before `accelerator.prepare`,
    # so we need the model on the target device already.
    model.to(accelerator.device)
    tokenized_train = prepare_tokenized_dataset(
        accelerator=accelerator,
        records=train_records,
        dataset_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        add_bos_token=args.add_bos_token,
        cache_dir=cache_dir,
        overwrite_cache=args.overwrite_cache,
    )
    if not tokenized_train:
        raise ValueError("No train examples contain supervised assistant tokens after truncation.")
    tokenized_candidate_validation = []
    if candidate_validation_records:
        tokenized_candidate_validation = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=candidate_validation_records,
            dataset_path=args.candidate_validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag="candidate_validation_loss",
        )
    tokenized_candidate_test = []
    if candidate_test_records:
        tokenized_candidate_test = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=candidate_test_records,
            dataset_path=args.candidate_test_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag="candidate_test_loss",
        )
    tokenized_validation = []
    if validation_records:
        tokenized_validation = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=validation_records,
            dataset_path=args.validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=validation_split_tag,
        )
    tokenized_eval = []
    if eval_records:
        tokenized_eval = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=eval_records,
            dataset_path=args.eval_file or args.validation_file or args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=(
                f"{file_fingerprint(args.eval_file or args.validation_file or args.train_file)}_"
                f"{safe_slug(args.eval_split_role or 'eval')}_"
                f"{'-'.join(str(value) for value in args.eval_split_proportions or [])}_"
                f"seed{args.eval_split_seed}"
            ),
        )
    tokenized_reference_validation = []
    if args.reference_evaluator == "loss" and reference_validation_records:
        tokenized_reference_validation = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=reference_validation_records,
            dataset_path=args.reference_validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag="reference_validation_loss",
        )
    tokenized_reference_eval = []
    if args.reference_evaluator == "loss" and reference_eval_records:
        tokenized_reference_eval = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=reference_eval_records,
            dataset_path=args.reference_test_file or args.reference_file or f"stereoset_{args.reference_hf_subset}_{args.reference_hf_split}",
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=(reference_split_tag or "reference_eval") + "_loss",
        )
    tokenized_extra_eval: dict[str, list[dict[str, torch.Tensor]]] = {}
    for item in extra_eval_record_sets:
        records = item.get("records") or []
        if not records:
            continue
        tokenized = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=records,
            dataset_path=item["file"],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=f"extra_eval_{item['name']}_kl",
        )
        if tokenized:
            tokenized_extra_eval[str(item["name"])] = tokenized
    tokenized_kl_reference = []
    if args.kl_regularization_lambda > 0:
        tokenized_kl_reference = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=kl_regularization_records,
            dataset_path=args.reference_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=f"kl_reference_seed{args.seed}_n{len(kl_regularization_records)}",
        )
    tokenized_trajectory_reference = []
    tokenized_trajectory_target = []
    if args.trajectory_eval_steps > 0:
        tokenized_trajectory_reference = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=trajectory_reference_records,
            dataset_path=args.reference_test_file or args.reference_validation_file or args.reference_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=f"trajectory_reference_seed{args.seed}_n{len(trajectory_reference_records)}",
        )
        tokenized_trajectory_target = prepare_tokenized_dataset(
            accelerator=accelerator,
            records=trajectory_target_records,
            dataset_path=args.validation_file or args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos_token=args.add_bos_token,
            cache_dir=cache_dir,
            overwrite_cache=args.overwrite_cache,
            dataset_cache_tag=f"trajectory_target_seed{args.seed}_n{len(trajectory_target_records)}",
        )

    validation_gradient_result = None
    if args.compute_validation_gradient and validation_records:
        cache_key = metric_cache_key(
            data_path=args.validation_file,
            model_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            add_bos_token=args.add_bos_token,
            seed=args.seed,
            max_examples=args.validation_gradient_max_examples,
            split_tag=validation_split_tag,
        )
        validation_gradient_result = get_or_compute_metric_tensor(
            accelerator=accelerator,
            cache_path=cache_dir / "validation_gradients" / f"{cache_key}.pt",
            overwrite_cache=args.overwrite_cache,
            compute_fn=lambda: compute_average_gradient(
                model=model,
                tokenizer=tokenizer,
                records=validation_records,
                device=accelerator.device,
                max_seq_length=args.max_seq_length,
                add_bos_token=args.add_bos_token,
                max_examples=args.validation_gradient_max_examples,
            ),
        )

    reference_fisher_result = None
    if args.compute_reference_fisher and reference_records:
        cache_key = metric_cache_key(
            data_path=args.reference_file if reference_source == "file" else None,
            model_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            add_bos_token=args.add_bos_token,
            seed=args.seed,
            max_examples=args.reference_fisher_max_examples,
            dataset_name="stereoset" if reference_source == "hf" else None,
            dataset_subset=args.reference_hf_subset if reference_source == "hf" else None,
            dataset_split=args.reference_hf_split if reference_source == "hf" else None,
            dataset_label=args.reference_hf_label if reference_source == "hf" else None,
            split_tag=reference_split_tag,
        )
        reference_fisher_result = get_or_compute_metric_tensor(
            accelerator=accelerator,
            cache_path=cache_dir / "reference_fisher" / f"{cache_key}.pt",
            overwrite_cache=args.overwrite_cache,
            compute_fn=lambda: compute_diagonal_fisher(
                model=model,
                tokenizer=tokenizer,
                records=reference_records,
                device=accelerator.device,
                max_seq_length=args.max_seq_length,
                add_bos_token=args.add_bos_token,
                max_examples=args.reference_fisher_max_examples,
            ),
        )

    collator = DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader = DataLoader(
        tokenized_train,
        shuffle=True,
        collate_fn=collator,
        batch_size=args.per_device_train_batch_size,
    )
    candidate_validation_dataloader = None
    if tokenized_candidate_validation:
        candidate_validation_dataloader = DataLoader(
            tokenized_candidate_validation,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    candidate_test_dataloader = None
    if tokenized_candidate_test:
        candidate_test_dataloader = DataLoader(
            tokenized_candidate_test,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    validation_dataloader = None
    if tokenized_validation:
        validation_dataloader = DataLoader(
            tokenized_validation,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    eval_dataloader = None
    if tokenized_eval:
        eval_dataloader = DataLoader(
            tokenized_eval,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    reference_validation_dataloader = None
    if tokenized_reference_validation:
        reference_validation_dataloader = DataLoader(
            tokenized_reference_validation,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    reference_eval_dataloader = None
    if tokenized_reference_eval:
        reference_eval_dataloader = DataLoader(
            tokenized_reference_eval,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    extra_eval_dataloaders: dict[str, Any] = {}
    for name, tokenized in tokenized_extra_eval.items():
        extra_eval_dataloaders[name] = DataLoader(
            tokenized,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    kl_reference_dataloader = None
    if tokenized_kl_reference:
        kl_reference_dataloader = DataLoader(
            tokenized_kl_reference,
            shuffle=True,
            collate_fn=collator,
            batch_size=args.per_device_train_batch_size,
        )
    trajectory_reference_dataloader = None
    if tokenized_trajectory_reference:
        trajectory_reference_dataloader = DataLoader(
            tokenized_trajectory_reference,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )
    trajectory_target_dataloader = None
    if tokenized_trajectory_target:
        trajectory_target_dataloader = DataLoader(
            tokenized_trajectory_target,
            shuffle=False,
            collate_fn=collator,
            batch_size=args.per_device_eval_batch_size,
        )

    optimizer = torch.optim.AdamW(
        [param for _, param in get_trainable_named_parameters(model)],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    if noop_training:
        max_steps = 0
    elif args.max_steps is None:
        max_steps = max(1, math.ceil(args.num_train_epochs * num_update_steps_per_epoch))
    else:
        max_steps = int(args.max_steps)
    num_warmup_steps = int(math.ceil(max_steps * float(args.warmup_ratio)))
    safe_training_epsilon = args.safe_epsilon
    if (
        safe_training_epsilon is not None
        and args.safe_training_epsilon_scale_by_steps
    ):
        safe_training_epsilon = float(safe_training_epsilon) * float(max_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max(1, max_steps),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    if candidate_validation_dataloader is not None:
        candidate_validation_dataloader = accelerator.prepare(candidate_validation_dataloader)
    if candidate_test_dataloader is not None:
        candidate_test_dataloader = accelerator.prepare(candidate_test_dataloader)
    if validation_dataloader is not None:
        validation_dataloader = accelerator.prepare(validation_dataloader)
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    if reference_validation_dataloader is not None:
        reference_validation_dataloader = accelerator.prepare(reference_validation_dataloader)
    if reference_eval_dataloader is not None:
        reference_eval_dataloader = accelerator.prepare(reference_eval_dataloader)
    for name, dataloader in list(extra_eval_dataloaders.items()):
        extra_eval_dataloaders[name] = accelerator.prepare(dataloader)
    if kl_reference_dataloader is not None:
        kl_reference_dataloader = accelerator.prepare(kl_reference_dataloader)
    if trajectory_reference_dataloader is not None:
        trajectory_reference_dataloader = accelerator.prepare(trajectory_reference_dataloader)
    if trajectory_target_dataloader is not None:
        trajectory_target_dataloader = accelerator.prepare(trajectory_target_dataloader)

    theta0 = flatten_trainable_parameters(accelerator.unwrap_model(model))
    previous_theta = theta0.clone()
    completed_steps = 0
    train_start_time = time.perf_counter()
    warmup_end_time = None
    evaluator = None if args.evaluator == "bias_disentangle" else get_evaluator(args.evaluator)
    summary: dict[str, Any] = {
        "train_examples": len(train_records),
        "effective_train_examples": len(tokenized_train),
        "weighted_training": bool(safe_training_weights),
        "noop_training": noop_training,
        "safe_training_weight_sum": (
            float(sum(safe_training_weights)) if safe_training_weights else None
        ),
        "safe_training_weight_min": (
            float(min(safe_training_weights)) if safe_training_weights else None
        ),
        "safe_training_weight_max": (
            float(max(safe_training_weights)) if safe_training_weights else None
        ),
        "candidate_validation_examples": len(candidate_validation_records),
        "candidate_test_examples": len(candidate_test_records),
        "effective_candidate_validation_examples": len(tokenized_candidate_validation),
        "effective_candidate_test_examples": len(tokenized_candidate_test),
        "validation_examples": len(validation_records),
        "evaluation_examples": len(eval_records),
        "ood_evaluation_examples": len(ood_eval_records),
        "effective_validation_examples": len(tokenized_validation),
        "reference_examples": len(reference_records),
        "reference_composition_name": args.reference_composition_name,
        "reference_composition_domains": list(args.reference_composition_domains or []),
        "reference_validation_examples": len(reference_validation_records),
        "reference_eval_examples": len(reference_eval_records),
        "reference_test_examples": len(reference_eval_records),
        "extra_eval_examples": {
            str(item["name"]): len(item.get("records") or [])
            for item in extra_eval_record_sets
        },
        "effective_extra_eval_examples": {
            name: len(tokenized)
            for name, tokenized in tokenized_extra_eval.items()
        },
        "extra_evals": [
            {key: value for key, value in item.items() if key != "records"}
            for item in extra_eval_record_sets
        ],
        "effective_reference_validation_examples": len(tokenized_reference_validation),
        "effective_reference_eval_examples": len(tokenized_reference_eval),
        "effective_reference_test_examples": len(tokenized_reference_eval),
        "kl_regularization_lambda": float(args.kl_regularization_lambda),
        "kl_reference_examples": len(kl_regularization_records),
        "trajectory_eval_steps": int(args.trajectory_eval_steps),
        "trajectory_reference_examples": len(trajectory_reference_records),
        "trajectory_target_examples": len(trajectory_target_records),
        "trajectory_generation_eval_steps": int(args.trajectory_generation_eval_steps),
        "trajectory_generation_target_examples": len(trajectory_generation_target_records),
        "trajectory_generation_reference_examples": len(trajectory_generation_reference_records),
        "trajectory_generation_instruction_examples": len(trajectory_generation_instruction_records),
        "trajectory_target_evaluator": args.trajectory_target_evaluator or args.evaluator,
        "trajectory_reference_evaluator": args.trajectory_reference_evaluator,
        "trajectory_instruction_evaluator": args.trajectory_instruction_evaluator,
        "selection_method": args.selection_method,
        "selection_preconditioner": args.selection_preconditioner,
        "safe_rho": args.safe_rho,
        "safe_epsilon": args.safe_epsilon,
        "safe_training_epsilon": safe_training_epsilon,
        "safe_training_epsilon_scale_by_steps": bool(
            args.safe_training_epsilon_scale_by_steps
        ),
        "safe_training_epsilon_scale_horizon": (
            max_steps if args.safe_training_epsilon_scale_by_steps else None
        ),
        "safe_training_constraint_mode": args.safe_training_constraint_mode,
        "selected_predicted_fisher_cost": args.selected_predicted_fisher_cost,
        "selected_predicted_norm_cost": args.selected_predicted_norm_cost,
        "seed": args.seed,
        "num_warmup_steps": num_warmup_steps,
        "max_steps": max_steps,
        "validation_gradient": None,
        "reference_fisher": None,
    }
    if validation_gradient_result is not None:
        summary["validation_gradient"] = {
            "cache_hit": validation_gradient_result.cache_hit,
            "wallclock_seconds": validation_gradient_result.wallclock_seconds,
            "cache_path": validation_gradient_result.cache_path,
        }
    if reference_fisher_result is not None:
        summary["reference_fisher"] = {
            "cache_hit": reference_fisher_result.cache_hit,
            "wallclock_seconds": reference_fisher_result.wallclock_seconds,
            "cache_path": reference_fisher_result.cache_path,
        }
    summary.update(base_eval_metrics)

    if args.trajectory_eval_steps > 0 and reference_fisher_result is None:
        raise ValueError("Trajectory diagnostics require --compute-reference-fisher.")

    estimated_sft_flops = 0.0
    estimated_kl_regularization_flops = 0.0
    estimated_trajectory_diagnostic_flops = 0.0
    estimated_trajectory_generation_flops = 0.0
    trajectory_points: list[dict[str, Any]] = []
    initial_trajectory_target_loss: float | None = None
    initial_trajectory_generation_metrics: dict[str, float] = {}
    kl_reference_iterator = iter(kl_reference_dataloader) if kl_reference_dataloader is not None else None

    def collect_trajectory_point(step: int, epoch: int) -> None:
        nonlocal estimated_trajectory_diagnostic_flops
        nonlocal estimated_trajectory_generation_flops
        nonlocal initial_trajectory_target_loss
        if trajectory_reference_dataloader is None or trajectory_target_dataloader is None:
            return
        current_theta = flatten_trainable_parameters(accelerator.unwrap_model(model))
        cumulative_delta = current_theta - theta0
        fisher = reference_fisher_result.tensor.float()
        predicted_fisher_cost = float(
            (0.5 * torch.dot(cumulative_delta.float(), fisher * cumulative_delta.float())).item()
        )
        cumulative_norm_cost = float(0.5 * torch.dot(cumulative_delta.float(), cumulative_delta.float()).item())
        target_loss, target_flops = evaluate_validation_loss_with_flops(
            model,
            trajectory_target_dataloader,
            accelerator,
        )
        realized_kl, reference_flops = evaluate_realized_heldout_kl(
            model,
            trajectory_reference_dataloader,
            accelerator,
        )
        estimated_trajectory_diagnostic_flops += target_flops + reference_flops
        if initial_trajectory_target_loss is None:
            initial_trajectory_target_loss = float(target_loss)
        degradation = float(target_loss) - float(initial_trajectory_target_loss)
        point = {
            "type": "trajectory_diagnostics",
            "step": int(step),
            "epoch": int(epoch),
            "predicted_fisher_cost": predicted_fisher_cost,
            "trained_lora_fisher_cost": predicted_fisher_cost,
            "cumulative_reference_cost": predicted_fisher_cost,
            "cumulative_norm_cost": cumulative_norm_cost,
            "realized_heldout_kl": float(realized_kl),
            "target_validation_loss": float(target_loss),
            "target_loss_degradation": degradation,
            "target_performance_proxy": -float(target_loss),
            "target_performance_delta": -degradation,
            "cumulative_delta_theta_norm": float(cumulative_delta.norm().item()),
            "safe_rho": args.safe_rho,
            "safe_epsilon": args.safe_epsilon,
            "safe_training_epsilon": safe_training_epsilon,
            "selection_method": args.selection_method,
            "selection_preconditioner": args.selection_preconditioner,
            "selected_predicted_fisher_cost": args.selected_predicted_fisher_cost,
            "selected_predicted_norm_cost": args.selected_predicted_norm_cost,
            **constraint_audit(
                cost=predicted_fisher_cost,
                budget=args.safe_rho,
                prefix="cumulative_reference",
            ),
            **constraint_audit(
                cost=cumulative_norm_cost,
                budget=(
                    None
                    if safe_training_epsilon is None
                    else 0.5 * float(safe_training_epsilon) ** 2
                ),
                prefix="cumulative_norm",
            ),
            "estimated_sft_flops": estimated_sft_flops,
            "estimated_kl_regularization_flops": estimated_kl_regularization_flops,
            "estimated_trajectory_diagnostic_flops": estimated_trajectory_diagnostic_flops,
            "estimated_trajectory_generation_flops": estimated_trajectory_generation_flops,
            "estimated_total_flops": (
                estimated_sft_flops
                + estimated_kl_regularization_flops
                + estimated_trajectory_diagnostic_flops
                + estimated_trajectory_generation_flops
            ),
        }
        should_run_generation = (
            args.trajectory_generation_eval_steps > 0
            and (
                step == 0
                or step % args.trajectory_generation_eval_steps == 0
                or step == max_steps
            )
        )
        if should_run_generation and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            was_training = bool(unwrapped.training)
            generation_groups = [
                (
                    "target",
                    args.trajectory_target_evaluator or args.evaluator,
                    trajectory_generation_target_records,
                ),
                (
                    "reference",
                    args.trajectory_reference_evaluator,
                    trajectory_generation_reference_records,
                ),
                (
                    "instruction",
                    args.trajectory_instruction_evaluator,
                    trajectory_generation_instruction_records,
                ),
            ]
            for group_name, evaluator_name, records in generation_groups:
                if not records or evaluator_name in {None, "none", "loss", "bias_disentangle"}:
                    continue
                group_metrics = evaluate_records_with_config(
                    evaluator_name=evaluator_name,
                    model=unwrapped,
                    tokenizer=tokenizer,
                    records=records,
                    device=accelerator.device,
                    max_examples=args.trajectory_generation_max_examples,
                    max_new_tokens=args.trajectory_generation_max_new_tokens,
                    add_bos_token=args.add_bos_token,
                    humaneval_num_samples=args.humaneval_num_samples,
                    humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                    humaneval_temperature=args.humaneval_temperature,
                    humaneval_top_p=args.humaneval_top_p,
                )
                parameter_count = sum(
                    parameter.numel() for parameter in unwrapped.parameters()
                )
                estimated_trajectory_generation_flops += (
                    2.0
                    * float(parameter_count)
                    * float(args.max_seq_length + args.trajectory_generation_max_new_tokens)
                    * float(min(len(records), args.trajectory_generation_max_examples))
                )
                for metric_name, metric_value in group_metrics.items():
                    key = f"{group_name}_{metric_name}"
                    value = float(metric_value)
                    point[key] = value
                    initial_trajectory_generation_metrics.setdefault(key, value)
                    if not metric_name.endswith(("_count", "_instruction_count")):
                        point[f"{key}_delta"] = value - initial_trajectory_generation_metrics[key]
            if "target_medqa_accuracy" in point:
                point["target_performance"] = point["target_medqa_accuracy"]
                point["target_performance_delta_actual"] = point["target_medqa_accuracy_delta"]
            if "reference_medical_reference_primary_score" in point:
                point["reference_performance_degradation"] = -point[
                    "reference_medical_reference_primary_score_delta"
                ]
            if "instruction_ifeval_proxy_instruction_accuracy" in point:
                point["instruction_performance_degradation"] = -point[
                    "instruction_ifeval_proxy_instruction_accuracy_delta"
                ]
            if was_training:
                unwrapped.train()
            point["estimated_trajectory_generation_flops"] = estimated_trajectory_generation_flops
            point["estimated_total_flops"] = (
                estimated_sft_flops
                + estimated_kl_regularization_flops
                + estimated_trajectory_diagnostic_flops
                + estimated_trajectory_generation_flops
            )
        trajectory_points.append(point)
        if accelerator.is_main_process:
            append_jsonl(metrics_path, point)

    if args.trajectory_eval_steps > 0:
        collect_trajectory_point(step=0, epoch=0)
    train_start_time = time.perf_counter()
    sum_incremental_reference_cost = 0.0
    sum_incremental_norm_cost = 0.0
    all_incremental_reference_constraints_satisfied = True
    all_incremental_norm_constraints_satisfied = True
    trajectory_projection_steps = 0
    trajectory_projection_active_steps = 0
    trajectory_projection_scale_sum = 0.0
    trajectory_projection_min_scale = 1.0
    model.train()
    for epoch in range(math.ceil(args.num_train_epochs) if args.max_steps is None else 10**9):
        if completed_steps >= max_steps:
            break

        for batch in train_dataloader:
            if completed_steps >= max_steps:
                break

            unwrapped_for_flops = accelerator.unwrap_model(model)
            estimated_sft_flops += estimate_batch_training_flops(unwrapped_for_flops, batch)
            with accelerator.accumulate(model):
                example_weights = batch.pop("example_weights", None)
                outputs = model(**batch)
                sft_loss = (
                    outputs.loss
                    if example_weights is None
                    else weighted_causal_lm_loss(outputs.logits, batch["labels"], example_weights)
                )
                kl_regularization_loss = None
                if kl_reference_iterator is not None:
                    try:
                        kl_batch = next(kl_reference_iterator)
                    except StopIteration:
                        kl_reference_iterator = iter(kl_reference_dataloader)
                        kl_batch = next(kl_reference_iterator)
                    kl_regularization_loss = compute_reference_kl_regularization_loss(
                        model,
                        kl_batch,
                        accelerator,
                    )
                    estimated_kl_regularization_flops += (
                        (4.0 / 3.0)
                        * estimate_batch_training_flops(unwrapped_for_flops, kl_batch)
                    )
                loss = sft_loss
                if kl_regularization_loss is not None:
                    loss = loss + float(args.kl_regularization_lambda) * kl_regularization_loss
                accelerator.backward(loss)

                grad_vector = None
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    grad_vector = flatten_trainable_gradients(unwrapped)
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                projected_current_theta: torch.Tensor | None = None
                projection_metrics: dict[str, Any] = {}
                if (
                    accelerator.sync_gradients
                    and args.safe_training_constraint_mode != "none"
                ):
                    unwrapped = accelerator.unwrap_model(model)
                    proposed_theta = flatten_trainable_parameters(unwrapped)
                    proposal = proposed_theta - previous_theta
                    displacement_before_update = previous_theta - theta0
                    projection_fisher = (
                        None
                        if reference_fisher_result is None
                        else reference_fisher_result.tensor.float()
                    )
                    projection_scale, projection_metrics = safe_training_update_scale(
                        mode=args.safe_training_constraint_mode,
                        displacement=displacement_before_update,
                        proposal=proposal,
                        fisher=projection_fisher,
                        rho=args.safe_rho,
                        epsilon=safe_training_epsilon,
                        max_steps=max_steps,
                    )
                    accepted_theta = previous_theta + projection_scale * proposal
                    if projection_scale < 1.0:
                        assign_trainable_parameters_from_flat(unwrapped, accepted_theta)

                    projected_current_theta = flatten_trainable_parameters(unwrapped)
                    accepted_displacement = projected_current_theta - theta0
                    realized_projection_error = proposed_theta - projected_current_theta
                    accepted_norm_cost = float(
                        0.5
                        * torch.dot(
                            accepted_displacement.float(),
                            accepted_displacement.float(),
                        ).item()
                    )
                    projection_metrics.update(
                        {
                            "trajectory_projection_l2_error": float(
                                realized_projection_error.norm().item()
                            ),
                            "accepted_cumulative_norm_cost": accepted_norm_cost,
                        }
                    )
                    accepted_norm_audit = constraint_audit(
                        cost=accepted_norm_cost,
                        budget=(
                            None
                            if safe_training_epsilon is None
                            else 0.5 * float(safe_training_epsilon) ** 2
                        ),
                        prefix="accepted_cumulative_norm",
                    )
                    projection_metrics.update(accepted_norm_audit)
                    proposed_norm_cost = projection_metrics.get(
                        "proposed_cumulative_norm_cost"
                    )
                    if proposed_norm_cost is not None:
                        projection_metrics.update(
                            constraint_audit(
                                cost=float(proposed_norm_cost),
                                budget=(
                                    None
                                    if safe_training_epsilon is None
                                    else 0.5 * float(safe_training_epsilon) ** 2
                                ),
                                prefix="proposed_cumulative_norm",
                            )
                        )

                    accepted_reference_audit: dict[str, Any] = {}
                    if projection_fisher is not None:
                        accepted_reference_cost = float(
                            0.5
                            * torch.dot(
                                accepted_displacement.float(),
                                projection_fisher * accepted_displacement.float(),
                            ).item()
                        )
                        projection_metrics.update(
                            {
                                "accepted_cumulative_reference_cost": accepted_reference_cost,
                                "trajectory_projection_fisher_error": float(
                                    0.5
                                    * torch.dot(
                                        realized_projection_error.float(),
                                        projection_fisher
                                        * realized_projection_error.float(),
                                    ).item()
                                ),
                            }
                        )
                        accepted_reference_audit = constraint_audit(
                            cost=accepted_reference_cost,
                            budget=args.safe_rho,
                            prefix="accepted_cumulative_reference",
                        )
                        projection_metrics.update(accepted_reference_audit)
                        proposed_reference_cost = projection_metrics.get(
                            "proposed_cumulative_reference_cost"
                        )
                        if proposed_reference_cost is not None:
                            projection_metrics.update(
                                constraint_audit(
                                    cost=float(proposed_reference_cost),
                                    budget=args.safe_rho,
                                    prefix="proposed_cumulative_reference",
                                )
                            )

                    if (
                        accepted_norm_audit[
                            "accepted_cumulative_norm_budget_satisfied"
                        ]
                        is False
                        or accepted_reference_audit.get(
                            "accepted_cumulative_reference_budget_satisfied"
                        )
                        is False
                    ):
                        raise RuntimeError(
                            "Projected optimizer update did not satisfy the requested "
                            "cumulative SAFE constraints."
                        )

                    trajectory_projection_steps += 1
                    trajectory_projection_scale_sum += projection_scale
                    trajectory_projection_min_scale = min(
                        trajectory_projection_min_scale,
                        projection_scale,
                    )
                    if projection_scale < 1.0 - 1.0e-8:
                        trajectory_projection_active_steps += 1
                lr_scheduler.step()
                optimizer.zero_grad()

                if not accelerator.sync_gradients:
                    continue

                accelerator.wait_for_everyone()
                completed_steps += 1
                unwrapped = accelerator.unwrap_model(model)
                current_theta = (
                    projected_current_theta
                    if projected_current_theta is not None
                    else flatten_trainable_parameters(unwrapped)
                )
                delta_theta = current_theta - previous_theta
                cumulative_delta = current_theta - theta0
                previous_theta = current_theta

                metrics = {
                    "step": completed_steps,
                    "epoch": epoch,
                    "train_loss": float(accelerator.gather_for_metrics(loss.detach().unsqueeze(0)).mean().item()),
                    "sft_loss": float(
                        accelerator.gather_for_metrics(sft_loss.detach().unsqueeze(0)).mean().item()
                    ),
                    "kl_regularization_loss": (
                        None
                        if kl_regularization_loss is None
                        else float(
                            accelerator.gather_for_metrics(
                                kl_regularization_loss.detach().unsqueeze(0)
                            ).mean().item()
                        )
                    ),
                    "kl_regularization_lambda": float(args.kl_regularization_lambda),
                    "safe_training_constraint_mode": args.safe_training_constraint_mode,
                    "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_vector.norm().item()) if grad_vector is not None else None,
                    "delta_theta_norm": float(delta_theta.norm().item()),
                    "cumulative_delta_theta_norm": float(cumulative_delta.norm().item()),
                    "step_wallclock_seconds": time.perf_counter() - train_start_time,
                    "estimated_sft_flops": estimated_sft_flops,
                    "estimated_kl_regularization_flops": estimated_kl_regularization_flops,
                    "estimated_trajectory_diagnostic_flops": estimated_trajectory_diagnostic_flops,
                    "estimated_trajectory_generation_flops": estimated_trajectory_generation_flops,
                    "estimated_total_flops": (
                        estimated_sft_flops
                        + estimated_kl_regularization_flops
                        + estimated_trajectory_diagnostic_flops
                        + estimated_trajectory_generation_flops
                    ),
                }
                metrics.update(projection_metrics)
                incremental_norm_cost = float(0.5 * torch.dot(delta_theta.float(), delta_theta.float()).item())
                cumulative_norm_cost = float(
                    0.5 * torch.dot(cumulative_delta.float(), cumulative_delta.float()).item()
                )
                sum_incremental_norm_cost += incremental_norm_cost
                norm_budget = (
                    None
                    if safe_training_epsilon is None
                    else 0.5 * float(safe_training_epsilon) ** 2
                )
                incremental_norm_audit = constraint_audit(
                    cost=incremental_norm_cost,
                    budget=norm_budget,
                    prefix="incremental_norm",
                )
                cumulative_norm_audit = constraint_audit(
                    cost=cumulative_norm_cost,
                    budget=norm_budget,
                    prefix="cumulative_norm",
                )
                if incremental_norm_audit["incremental_norm_budget_satisfied"] is False:
                    all_incremental_norm_constraints_satisfied = False
                metrics.update(
                    {
                        "incremental_norm_cost": incremental_norm_cost,
                        "cumulative_norm_cost": cumulative_norm_cost,
                        "sum_incremental_norm_cost": sum_incremental_norm_cost,
                        "norm_cost_cross_term_total": cumulative_norm_cost - sum_incremental_norm_cost,
                        "all_incremental_norm_constraints_satisfied_so_far": (
                            all_incremental_norm_constraints_satisfied
                        ),
                        **incremental_norm_audit,
                        **cumulative_norm_audit,
                    }
                )
                if args.selected_predicted_norm_cost is not None:
                    selected_norm_cost = float(args.selected_predicted_norm_cost)
                    metrics["incremental_selected_norm_cost_calibration_abs_error"] = abs(
                        incremental_norm_cost - selected_norm_cost
                    )
                    metrics["cumulative_selected_norm_cost_calibration_abs_error"] = abs(
                        cumulative_norm_cost - selected_norm_cost
                    )

                if reference_fisher_result is not None:
                    fisher = reference_fisher_result.tensor.float()
                    incremental_reference_cost = float(
                        (0.5 * torch.dot(delta_theta.float(), fisher * delta_theta.float())).item()
                    )
                    cumulative_reference_cost = float(
                        (0.5 * torch.dot(cumulative_delta.float(), fisher * cumulative_delta.float())).item()
                    )
                    previous_cumulative_delta = cumulative_delta.float() - delta_theta.float()
                    reference_cross_term = float(
                        torch.dot(previous_cumulative_delta, fisher * delta_theta.float()).item()
                    )
                    sum_incremental_reference_cost += incremental_reference_cost
                    incremental_reference_audit = constraint_audit(
                        cost=incremental_reference_cost,
                        budget=args.safe_rho,
                        prefix="incremental_reference",
                    )
                    cumulative_reference_audit = constraint_audit(
                        cost=cumulative_reference_cost,
                        budget=args.safe_rho,
                        prefix="cumulative_reference",
                    )
                    if incremental_reference_audit["incremental_reference_budget_satisfied"] is False:
                        all_incremental_reference_constraints_satisfied = False
                    metrics.update(
                        {
                            "delta_theta_reference_cost": incremental_reference_cost,
                            "incremental_reference_cost": incremental_reference_cost,
                            "cumulative_reference_cost": cumulative_reference_cost,
                            "reference_cost_cross_term": reference_cross_term,
                            "sum_incremental_reference_cost": sum_incremental_reference_cost,
                            "reference_cost_cross_term_total": (
                                cumulative_reference_cost - sum_incremental_reference_cost
                            ),
                            "all_incremental_reference_constraints_satisfied_so_far": (
                                all_incremental_reference_constraints_satisfied
                            ),
                            **incremental_reference_audit,
                            **cumulative_reference_audit,
                        }
                    )
                    if args.selected_predicted_fisher_cost is not None:
                        selected_fisher_cost = float(args.selected_predicted_fisher_cost)
                        metrics["incremental_selected_fisher_cost_calibration_abs_error"] = abs(
                            incremental_reference_cost - selected_fisher_cost
                        )
                        metrics["cumulative_selected_fisher_cost_calibration_abs_error"] = abs(
                            cumulative_reference_cost - selected_fisher_cost
                        )
                if validation_gradient_result is not None:
                    validation_grad = validation_gradient_result.tensor.float()
                    step_dot = torch.dot(validation_grad, delta_theta.float())
                    cumulative_dot = torch.dot(validation_grad, cumulative_delta.float())
                    metrics["validation_gradient_dot_delta"] = float(step_dot.item())
                    metrics["validation_gain"] = float((-step_dot).item())
                    metrics["cumulative_validation_gain"] = float((-cumulative_dot).item())

                if warmup_end_time is None and completed_steps >= num_warmup_steps:
                    warmup_end_time = time.perf_counter()
                    metrics["warmup_wallclock_seconds"] = warmup_end_time - train_start_time

                if accelerator.is_main_process:
                    append_jsonl(metrics_path, metrics)

                should_log_trajectory = (
                    args.trajectory_eval_steps > 0
                    and (
                        completed_steps % args.trajectory_eval_steps == 0
                        or completed_steps == max_steps
                    )
                )
                if should_log_trajectory:
                    accelerator.wait_for_everyone()
                    collect_trajectory_point(step=completed_steps, epoch=epoch)
                    accelerator.wait_for_everyone()

                should_log_eval = (
                    validation_dataloader is not None
                    and args.eval_steps > 0
                    and (completed_steps % args.eval_steps == 0 or completed_steps == max_steps)
                )
                if should_log_eval:
                    accelerator.wait_for_everyone()
                    eval_metrics: dict[str, float] = {}
                    if validation_dataloader is not None:
                        eval_metrics["validation_loss"] = evaluate_validation_loss(model, validation_dataloader, accelerator)
                    if args.evaluator not in {"none", "bias_disentangle"} and validation_records and accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.eval()
                        step_eval_metrics = evaluate_records_with_config(
                            evaluator_name=args.evaluator,
                            model=unwrapped,
                            tokenizer=tokenizer,
                            records=validation_records,
                            device=accelerator.device,
                            max_examples=args.eval_max_examples,
                            max_new_tokens=args.generation_max_new_tokens,
                            add_bos_token=args.add_bos_token,
                            humaneval_num_samples=args.humaneval_num_samples,
                            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
                            humaneval_temperature=args.humaneval_temperature,
                            humaneval_top_p=args.humaneval_top_p,
                        )
                        if "medqa_accuracy" in step_eval_metrics:
                            step_eval_metrics["val_medqa_accuracy"] = step_eval_metrics.pop("medqa_accuracy")
                        if "medqa_count" in step_eval_metrics:
                            step_eval_metrics["val_medqa_count"] = step_eval_metrics.pop("medqa_count")
                        eval_metrics.update(step_eval_metrics)
                        unwrapped.train()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and eval_metrics:
                        eval_payload = {"step": completed_steps, "type": "evaluation", **eval_metrics}
                        append_jsonl(metrics_path, eval_payload)

                if args.save_steps and args.save_steps > 0 and completed_steps % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = output_dir / f"checkpoint-step-{completed_steps}"
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)

                if completed_steps >= max_steps:
                    break

    accelerator.wait_for_everyone()
    total_train_seconds = time.perf_counter() - train_start_time
    final_validation_loss = None
    if validation_dataloader is not None:
        final_validation_loss = evaluate_validation_loss(model, validation_dataloader, accelerator)
    final_test_loss = None
    if eval_dataloader is not None:
        final_test_loss = evaluate_validation_loss(model, eval_dataloader, accelerator)
    candidate_validation_loss = None
    if candidate_validation_dataloader is not None:
        candidate_validation_loss = evaluate_validation_loss(model, candidate_validation_dataloader, accelerator)
    candidate_test_loss = None
    if candidate_test_dataloader is not None:
        candidate_test_loss = evaluate_validation_loss(model, candidate_test_dataloader, accelerator)
    reference_validation_loss = None
    if reference_validation_dataloader is not None:
        reference_validation_loss = evaluate_validation_loss(model, reference_validation_dataloader, accelerator)
    reference_eval_loss = None
    if reference_eval_dataloader is not None:
        reference_eval_loss = evaluate_validation_loss(model, reference_eval_dataloader, accelerator)
    extra_eval_kl_metrics: dict[str, float] = {}
    estimated_extra_eval_kl_flops = 0.0
    for name, dataloader in extra_eval_dataloaders.items():
        realized_kl, kl_flops = evaluate_realized_heldout_kl(model, dataloader, accelerator)
        extra_eval_kl_metrics[f"extra_eval_{name}_heldout_kl"] = float(realized_kl)
        extra_eval_kl_metrics[f"extra_eval_{name}_heldout_kl_forward_flops"] = float(kl_flops)
        estimated_extra_eval_kl_flops += float(kl_flops)
    if accelerator.is_main_process:
        final_dir = output_dir / "final_adapter"
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        summary.update(
            {
                "train_wallclock_seconds": total_train_seconds,
                "warmup_wallclock_seconds": None if warmup_end_time is None else warmup_end_time - train_start_time,
                "output_dir": str(output_dir),
                "final_adapter_dir": str(final_dir),
                "completed_steps": completed_steps,
                "estimated_sft_flops": estimated_sft_flops,
                "estimated_kl_regularization_flops": estimated_kl_regularization_flops,
                "estimated_trajectory_diagnostic_flops": estimated_trajectory_diagnostic_flops,
                "estimated_trajectory_generation_flops": estimated_trajectory_generation_flops,
                "estimated_extra_eval_kl_flops": estimated_extra_eval_kl_flops,
                "estimated_total_flops": (
                    estimated_sft_flops
                    + estimated_kl_regularization_flops
                    + estimated_trajectory_diagnostic_flops
                    + estimated_trajectory_generation_flops
                    + estimated_extra_eval_kl_flops
                ),
                "flops_estimation_method": (
                    "transformers_floating_point_ops_with_6x_token_parameter_fallback;"
                    "generation_uses_2x_parameter_token_upper_bound"
                ),
                "trajectory_correlations": (
                    trajectory_correlation_summary(trajectory_points)
                    if trajectory_points
                    else None
                ),
                "trajectory_points": trajectory_points,
                "all_incremental_reference_constraints_satisfied": (
                    all_incremental_reference_constraints_satisfied
                ),
                "all_incremental_norm_constraints_satisfied": (
                    all_incremental_norm_constraints_satisfied
                ),
                "sum_incremental_reference_cost": sum_incremental_reference_cost,
                "sum_incremental_norm_cost": sum_incremental_norm_cost,
                "trajectory_projection_steps": trajectory_projection_steps,
                "trajectory_projection_active_steps": trajectory_projection_active_steps,
                "trajectory_projection_active_fraction": (
                    float(trajectory_projection_active_steps)
                    / float(trajectory_projection_steps)
                    if trajectory_projection_steps
                    else 0.0
                ),
                "trajectory_projection_mean_scale": (
                    trajectory_projection_scale_sum / float(trajectory_projection_steps)
                    if trajectory_projection_steps
                    else 1.0
                ),
                "trajectory_projection_min_scale": trajectory_projection_min_scale,
            }
        )
        if final_validation_loss is not None:
            summary["final_validation_loss"] = final_validation_loss
        if final_test_loss is not None:
            summary["final_test_loss"] = final_test_loss
        if candidate_validation_loss is not None:
            summary["candidate_validation_loss"] = candidate_validation_loss
        if candidate_test_loss is not None:
            summary["candidate_test_loss"] = candidate_test_loss
        if reference_validation_loss is not None:
            summary["reference_validation_loss"] = reference_validation_loss
        if reference_eval_loss is not None:
            summary["reference_test_loss"] = reference_eval_loss
            summary["reference_eval_loss"] = reference_eval_loss
        summary.update(extra_eval_kl_metrics)
        unwrapped_model = accelerator.unwrap_model(model)
        parameter_change_metrics = {
            **trainable_parameter_delta_metrics(unwrapped_model, theta0),
            **lora_model_delta_metrics(unwrapped_model),
        }
        if reference_fisher_result is not None:
            final_theta = flatten_trainable_parameters(unwrapped_model)
            final_delta = final_theta - theta0
            fisher = reference_fisher_result.tensor.float()
            final_fisher_cost = float(
                (0.5 * torch.dot(final_delta.float(), fisher * final_delta.float())).item()
            )
            final_norm_cost = float(0.5 * torch.dot(final_delta.float(), final_delta.float()).item())
            parameter_change_metrics.update(
                {
                    "trained_lora_fisher_cost": final_fisher_cost,
                    "trained_lora_norm_cost": final_norm_cost,
                    **constraint_audit(
                        cost=final_fisher_cost,
                        budget=args.safe_rho,
                        prefix="final_cumulative_reference",
                    ),
                    **constraint_audit(
                        cost=final_norm_cost,
                        budget=(
                            None
                            if safe_training_epsilon is None
                            else 0.5 * float(safe_training_epsilon) ** 2
                        ),
                        prefix="final_cumulative_norm",
                    ),
                }
            )
        summary.update(parameter_change_metrics)
        append_jsonl(metrics_path, {"type": "final_parameter_change", **parameter_change_metrics})
        if args.skip_final_evaluation:
            summary["final_evaluation_skipped"] = True
            save_json(summary_path, summary)
            return
        summary.update(
            run_required_final_evaluations(
                model=model,
                tokenizer=tokenizer,
                args=args,
                accelerator=accelerator,
                eval_records=eval_records,
                ood_eval_records=ood_eval_records,
                reference_validation_records=reference_validation_records,
                reference_eval_records=reference_eval_records,
                extra_eval_record_sets=extra_eval_record_sets,
                final_dir=final_dir,
                output_dir=output_dir,
            )
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
            if base_key in summary and isinstance(value, (int, float)) and isinstance(summary[base_key], (int, float)):
                summary[f"{delta_prefix}{metric_name}"] = float(value) - float(summary[base_key])
        save_json(summary_path, summary)


if __name__ == "__main__":
    main()
