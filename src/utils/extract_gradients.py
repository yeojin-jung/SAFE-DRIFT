from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_dtype(value: str | torch.dtype | None) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    lowered = "float32" if value is None else str(value).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported model dtype: {value}")
    return mapping[lowered]


def load_model_with_lora(model_name, lora_r, lora_alpha, lora_dropout,
                         lora_target_modules, device, torch_dtype=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_model_dtype(torch_dtype),
    )
    required_vocab_size = max(tokenizer.get_vocab().values()) + 1
    if base_model.get_input_embeddings().weight.shape[0] < required_vocab_size:
        base_model.resize_token_embeddings(required_vocab_size)
    base_model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    return model, tokenizer


def get_prompt_response(example: dict[str, Any]) -> tuple[str, str]:
    messages = example.get("messages")
    if isinstance(messages, list):
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_parts.append(content)
        if user_parts or assistant_parts:
            return "\n".join(user_parts).strip(), "\n".join(assistant_parts).strip()

    prompt = example.get("instruction", example.get("input", example.get("question", "")))
    response = example.get("output", example.get("answer", ""))
    return str(prompt).strip(), str(response).strip()


def _tokenize_example(tokenizer, example, max_seq_len):
    prompt, response = get_prompt_response(example)
    if not response:
        raise ValueError("Example has no assistant response/output text.")

    prompt_text = prompt.rstrip() + "\n"
    full_text = prompt_text + response
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    labels = input_ids.clone()

    prompt_len = min(len(prompt_ids), labels.shape[1])
    labels[:, :prompt_len] = -100
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    return input_ids, attention_mask, labels


def _collect_grad(model, use_lora=True):
    grads = []
    for _, param in model.named_parameters():
        if param.grad is None:
            continue
        if use_lora and not param.requires_grad:
            continue
        grads.append(param.grad.detach().cpu().float().view(-1))
    return torch.cat(grads)


def _projection_hash_constants(seed: int) -> tuple[int, int, int, int]:
    prime = 2_147_483_647
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.randint(1, prime - 1, (4,), generator=generator, dtype=torch.int64)
    return tuple(int(v.item()) for v in values)


def project_gradient_count_sketch(
    gradient: torch.Tensor,
    *,
    output_dim: int,
    seed: int,
    chunk_size: int = 1_048_576,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if gradient.ndim != 1:
        raise ValueError("gradient must be a 1D tensor.")

    grad = gradient.detach().to(device="cpu", dtype=torch.float32)
    projected = torch.zeros(output_dim, dtype=torch.float32)
    prime = 2_147_483_647
    a_bucket, b_bucket, a_sign, b_sign = _projection_hash_constants(seed)

    numel = grad.numel()
    for start in range(0, numel, chunk_size):
        end = min(numel, start + chunk_size)
        chunk = grad[start:end]
        indices = torch.arange(start, end, dtype=torch.int64)
        buckets = ((indices * a_bucket + b_bucket) % prime) % output_dim
        sign_bits = ((indices * a_sign + b_sign) % prime) % 2
        signs = sign_bits.to(torch.float32).mul_(2.0).sub_(1.0)
        projected.index_add_(0, buckets, chunk * signs)

    projected = projected / max(1.0, float(output_dim) ** 0.5)
    return projected.to(dtype=dtype)


def _cache_dtype_from_name(dtype_name: str) -> torch.dtype:
    lowered = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if lowered not in mapping:
        raise ValueError(f"Unsupported projected feature dtype: {dtype_name}")
    return mapping[lowered]


def compute_per_example_gradient(model, tokenizer, example, device, max_seq_len,
                                  use_lora=True):
    model.zero_grad()
    input_ids, attention_mask, labels = _tokenize_example(tokenizer, example, max_seq_len)
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    g = _collect_grad(model, use_lora=use_lora)
    model.zero_grad()
    return g


def compute_projected_feature(
    model,
    tokenizer,
    example,
    device,
    max_seq_len,
    *,
    output_dim,
    projection_seed,
    use_lora=True,
    chunk_size: int = 1_048_576,
    output_dtype: torch.dtype = torch.float32,
):
    gradient = compute_per_example_gradient(
        model=model,
        tokenizer=tokenizer,
        example=example,
        device=device,
        max_seq_len=max_seq_len,
        use_lora=use_lora,
    )
    return project_gradient_count_sketch(
        gradient,
        output_dim=output_dim,
        seed=projection_seed,
        chunk_size=chunk_size,
        dtype=output_dtype,
    )


def project_preconditioner_count_sketch(
    preconditioner: torch.Tensor,
    *,
    output_dim: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
):
    if output_dim <= 0:
        raise ValueError("output_dim must be positive.")
    if preconditioner.ndim != 1:
        raise ValueError("preconditioner must be a 1D tensor.")

    p = preconditioner.detach().to(device="cpu", dtype=torch.float32)
    projected = torch.zeros(output_dim, dtype=torch.float32)
    prime = 2_147_483_647
    a_bucket, b_bucket, _, _ = _projection_hash_constants(seed)
    indices = torch.arange(p.numel(), dtype=torch.int64)
    buckets = ((indices * a_bucket + b_bucket) % prime) % output_dim
    # This is a sketch-space diagonal approximation to S diag(p) S^T that
    # keeps the preconditioner aligned with the same projected coordinates.
    projected.index_add_(0, buckets, p)
    projected = projected / float(output_dim)
    return projected.to(dtype=dtype)


def compute_projected_target_gradient(
    model,
    tokenizer,
    target_exemplars,
    device,
    max_seq_len,
    *,
    output_dim,
    projection_seed,
    use_lora=True,
    chunk_size: int = 1_048_576,
    output_dtype: torch.dtype = torch.float32,
    normalize: bool = False,
):
    if not target_exemplars:
        raise ValueError("target_exemplars must not be empty.")

    model.eval()
    accumulated = None

    for example in tqdm(target_exemplars, desc="Computing projected target gradient"):
        feature = compute_projected_feature(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=max_seq_len,
            output_dim=output_dim,
            projection_seed=projection_seed,
            use_lora=use_lora,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
        accumulated = feature if accumulated is None else accumulated + feature

    projected = accumulated / float(len(target_exemplars))
    if normalize:
        projected = F.normalize(projected, dim=0)
    return projected.to(dtype=output_dtype)

# Diagonal approximation
def compute_projected_reference_fisher(
    model,
    tokenizer,
    reference_exemplars,
    device,
    max_seq_len,
    *,
    output_dim,
    projection_seed,
    use_lora=True,
    chunk_size: int = 1_048_576,
    output_dtype: torch.dtype = torch.float32,
):
    if not reference_exemplars:
        raise ValueError("reference_exemplars must not be empty.")

    raw_weights = [example.get("reference_fisher_weight") for example in reference_exemplars]
    if any(weight is not None for weight in raw_weights):
        if not all(weight is not None for weight in raw_weights):
            raise ValueError(
                "reference_fisher_weight must be present on every reference example "
                "or on none of them."
            )
        weights = torch.tensor([float(weight) for weight in raw_weights], dtype=torch.float32)
        if bool((weights < 0.0).any()):
            raise ValueError("reference_fisher_weight values must be nonnegative.")
        weight_total = float(weights.sum().item())
        if weight_total <= 0.0:
            raise ValueError("reference_fisher_weight values must have positive total mass.")
        weights = weights / weight_total
    else:
        weights = torch.full((len(reference_exemplars),), 1.0 / float(len(reference_exemplars)))

    model.eval()
    fisher_sum = None

    for row_idx, example in enumerate(tqdm(reference_exemplars, desc="Computing projected reference Fisher")):
        feature = compute_projected_feature(
            model=model,
            tokenizer=tokenizer,
            example=example,
            device=device,
            max_seq_len=max_seq_len,
            output_dim=output_dim,
            projection_seed=projection_seed,
            use_lora=use_lora,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
        squared = feature.square() * weights[row_idx].to(device=feature.device, dtype=feature.dtype)
        fisher_sum = squared if fisher_sum is None else fisher_sum + squared

    return fisher_sum.to(dtype=output_dtype)
