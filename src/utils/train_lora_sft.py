from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
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
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


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
            tokenized_records.append(encoded)
    return tokenized_records


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


@torch.no_grad()
def evaluate_validation_loss(model, dataloader, accelerator: Accelerator) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        if not batch["labels"].ne(-100).any():
            continue
        outputs = model(**batch)
        loss = outputs.loss.detach()
        reduced = accelerator.gather_for_metrics(loss.unsqueeze(0))
        total_loss += float(reduced.mean().item())
        total_batches += 1
    model.train()
    if total_batches == 0:
        return 0.0
    return total_loss / float(total_batches)


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
            max_new_tokens=args.generation_max_new_tokens,
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
            max_new_tokens=args.generation_max_new_tokens,
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
            max_new_tokens=args.generation_max_new_tokens,
            add_bos_token=args.add_bos_token,
            humaneval_num_samples=args.humaneval_num_samples,
            humaneval_pass_at_ks=tuple(args.humaneval_pass_at_ks),
            humaneval_temperature=args.humaneval_temperature,
            humaneval_top_p=args.humaneval_top_p,
        )
        summary_updates.update(prefixed_metrics("reference_", reference_metrics))

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

    parser.add_argument(
        "--evaluator",
        default="none",
        choices=["none", "medqa", "mmlu", "gsm8k", "esconv", "humaneval", "ifeval", "bias_disentangle"],
    )
    parser.add_argument(
        "--ood-evaluator",
        default=None,
        choices=["none", "medqa", "mmlu", "gsm8k", "esconv", "humaneval", "ifeval", "bias_disentangle"],
    )
    parser.add_argument(
        "--reference-evaluator",
        default="none",
        choices=["none", "loss", "medqa", "mmlu", "gsm8k", "esconv", "humaneval", "ifeval", "bias_disentangle"],
    )
    parser.add_argument("--eval-max-examples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument("--humaneval-num-samples", type=int, default=1)
    parser.add_argument("--humaneval-pass-at-ks", nargs="+", type=int, default=[1])
    parser.add_argument("--humaneval-temperature", type=float, default=0.8)
    parser.add_argument("--humaneval-top-p", type=float, default=0.95)
    parser.add_argument("--benchmark-evals", nargs="+", choices=["mmlu", "gsm8k"], default=[])
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
        ood_evaluator = args.ood_evaluator or args.evaluator
        if args.evaluate_base_ood and ood_evaluator not in {"none", "bias_disentangle"} and ood_eval_records:
            ood_metrics = evaluate_records_with_config(
                evaluator_name=ood_evaluator,
                model=base_model,
                tokenizer=tokenizer,
                records=ood_eval_records,
                device=accelerator.device,
                max_examples=args.eval_max_examples,
                max_new_tokens=args.generation_max_new_tokens,
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
                max_new_tokens=args.generation_max_new_tokens,
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
                max_new_tokens=args.generation_max_new_tokens,
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

    optimizer = torch.optim.AdamW(
        [param for _, param in get_trainable_named_parameters(model)],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / args.gradient_accumulation_steps))
    if args.max_steps is None:
        max_steps = max(1, math.ceil(args.num_train_epochs * num_update_steps_per_epoch))
    else:
        max_steps = int(args.max_steps)
    num_warmup_steps = int(math.ceil(max_steps * float(args.warmup_ratio)))

    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_steps,
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

    theta0 = flatten_trainable_parameters(accelerator.unwrap_model(model))
    previous_theta = theta0.clone()
    completed_steps = 0
    train_start_time = time.perf_counter()
    warmup_end_time = None
    evaluator = None if args.evaluator == "bias_disentangle" else get_evaluator(args.evaluator)
    summary: dict[str, Any] = {
        "train_examples": len(train_records),
        "effective_train_examples": len(tokenized_train),
        "candidate_validation_examples": len(candidate_validation_records),
        "candidate_test_examples": len(candidate_test_records),
        "effective_candidate_validation_examples": len(tokenized_candidate_validation),
        "effective_candidate_test_examples": len(tokenized_candidate_test),
        "validation_examples": len(validation_records),
        "evaluation_examples": len(eval_records),
        "ood_evaluation_examples": len(ood_eval_records),
        "effective_validation_examples": len(tokenized_validation),
        "reference_examples": len(reference_records),
        "reference_validation_examples": len(reference_validation_records),
        "reference_eval_examples": len(reference_eval_records),
        "reference_test_examples": len(reference_eval_records),
        "effective_reference_validation_examples": len(tokenized_reference_validation),
        "effective_reference_eval_examples": len(tokenized_reference_eval),
        "effective_reference_test_examples": len(tokenized_reference_eval),
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

    model.train()
    for epoch in range(math.ceil(args.num_train_epochs) if args.max_steps is None else 10**9):
        if completed_steps >= max_steps:
            break

        for batch in train_dataloader:
            if completed_steps >= max_steps:
                break

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                grad_vector = None
                if accelerator.sync_gradients:
                    unwrapped = accelerator.unwrap_model(model)
                    grad_vector = flatten_trainable_gradients(unwrapped)
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if not accelerator.sync_gradients:
                    continue

                accelerator.wait_for_everyone()
                completed_steps += 1
                unwrapped = accelerator.unwrap_model(model)
                current_theta = flatten_trainable_parameters(unwrapped)
                delta_theta = current_theta - previous_theta
                cumulative_delta = current_theta - theta0
                previous_theta = current_theta

                metrics = {
                    "step": completed_steps,
                    "epoch": epoch,
                    "train_loss": float(accelerator.gather_for_metrics(loss.detach().unsqueeze(0)).mean().item()),
                    "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_vector.norm().item()) if grad_vector is not None else None,
                    "delta_theta_norm": float(delta_theta.norm().item()),
                    "cumulative_delta_theta_norm": float(cumulative_delta.norm().item()),
                    "step_wallclock_seconds": time.perf_counter() - train_start_time,
                }

                if reference_fisher_result is not None:
                    fisher = reference_fisher_result.tensor.float()
                    metrics["delta_theta_reference_cost"] = float(
                        (0.5 * torch.dot(delta_theta.float(), fisher * delta_theta.float())).item()
                    )
                    metrics["cumulative_reference_cost"] = float(
                        (0.5 * torch.dot(cumulative_delta.float(), fisher * cumulative_delta.float())).item()
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
            else:
                continue
            metric_name = key.removeprefix(prefix)
            base_key = f"{base_prefix}{metric_name}"
            if base_key in summary and isinstance(value, (int, float)) and isinstance(summary[base_key], (int, float)):
                summary[f"{delta_prefix}{metric_name}"] = float(value) - float(summary[base_key])
        save_json(summary_path, summary)


if __name__ == "__main__":
    main()
