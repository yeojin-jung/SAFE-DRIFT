"""Per-stage FLOPs and peak-GPU-memory accounting.

Reported compute is analytic, not profiler-measured. Two families of estimate
are used:

* Transformer forward/backward passes are counted with the standard
  ``6 * N_params * N_tokens`` approximation, split into its parts so the
  assumptions stay visible: ``2 * N * T`` for the forward pass, ``4 * N * T``
  for a backward pass that materializes every weight gradient, and
  ``2 * N * T`` for a LoRA backward pass, which still backpropagates
  activation gradients through the frozen trunk but only forms weight
  gradients for the adapters. The unembedding matmul and the sequence-length
  quadratic attention term are added explicitly rather than folded into ``N``,
  because both are non-negligible at the vocabulary sizes (~100-150K) and
  sequence lengths used here.
* Selection and low-rank algebra are counted exactly from matmul/decomposition
  shapes (``2mnk`` for a dense matmul).

Peak memory is captured per stage by resetting ``torch.cuda.reset_peak_memory_stats``
on entry and reading ``torch.cuda.max_memory_allocated`` on exit, so stages do
not inherit each other's high-water marks. A process-wide peak is tracked
separately.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


BYTES_PER_GIB = 1024.0**3

# Multipliers applied to the forward-pass FLOP count.
BACKWARD_MULTIPLIER_FULL = 2.0
"""Backward with dense weight gradients: 2x forward (input grads + weight grads)."""

BACKWARD_MULTIPLIER_LORA = 1.0
"""LoRA backward: activation grads flow through the trunk (1x forward); adapter
weight grads are O(r) and folded into the trunk term rather than counted."""


# ---------------------------------------------------------------------------
# Global pass accounting
# ---------------------------------------------------------------------------


@dataclass
class PassCounter:
    """Running totals for model passes issued inside a stage."""

    forward_backward_passes: int = 0
    forward_passes: int = 0
    tokens: int = 0
    padded_tokens: int = 0

    def reset(self) -> None:
        self.forward_backward_passes = 0
        self.forward_passes = 0
        self.tokens = 0
        self.padded_tokens = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "forward_backward_passes": int(self.forward_backward_passes),
            "forward_passes": int(self.forward_passes),
            "tokens": int(self.tokens),
            "padded_tokens": int(self.padded_tokens),
        }

    def delta_since(self, baseline: dict[str, int]) -> dict[str, int]:
        current = self.snapshot()
        return {key: current[key] - int(baseline.get(key, 0)) for key in current}


GRADIENT_PASS_COUNTER = PassCounter()
"""Incremented by ``utils.extract_gradients.compute_per_example_gradient``.

Every per-example gradient path in the codebase (Adam preconditioner warmup,
random-sketch features, low-rank Fisher/basis construction) funnels through
that function, so this counter covers all of them.
"""


def record_forward_backward(tokens: int, padded_tokens: int | None = None) -> None:
    GRADIENT_PASS_COUNTER.forward_backward_passes += 1
    GRADIENT_PASS_COUNTER.tokens += int(tokens)
    GRADIENT_PASS_COUNTER.padded_tokens += int(tokens if padded_tokens is None else padded_tokens)


# ---------------------------------------------------------------------------
# Model shape description
# ---------------------------------------------------------------------------


@dataclass
class ModelShape:
    """Parameter counts and architecture dimensions needed for FLOPs estimates."""

    model_name: str | None = None
    params_total: int = 0
    params_trainable: int = 0
    params_embedding: int = 0
    params_non_embedding: int = 0
    num_hidden_layers: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params_total": int(self.params_total),
            "params_trainable": int(self.params_trainable),
            "params_embedding": int(self.params_embedding),
            "params_non_embedding": int(self.params_non_embedding),
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
        }


def _is_embedding_param(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed_tokens", "wte", "word_embeddings", "lm_head"))


def describe_model(model, model_name: str | None = None) -> ModelShape:
    """Count parameters of a (possibly PEFT-wrapped) causal LM."""
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    config = getattr(base, "config", None) or getattr(model, "config", None)

    params_total = 0
    params_trainable = 0
    params_embedding = 0
    for name, param in model.named_parameters():
        count = param.numel()
        params_total += count
        if param.requires_grad:
            params_trainable += count
        if _is_embedding_param(name):
            params_embedding += count

    return ModelShape(
        model_name=model_name,
        params_total=params_total,
        params_trainable=params_trainable,
        params_embedding=params_embedding,
        params_non_embedding=max(0, params_total - params_embedding),
        num_hidden_layers=getattr(config, "num_hidden_layers", None),
        hidden_size=getattr(config, "hidden_size", None),
        vocab_size=getattr(config, "vocab_size", None),
    )


# ---------------------------------------------------------------------------
# FLOPs models
# ---------------------------------------------------------------------------


def dense_matmul_flops(m: int, n: int, k: int) -> float:
    """FLOPs for an (m x k) @ (k x n) product, counting a multiply-add as 2."""
    return 2.0 * float(m) * float(n) * float(k)


def symmetric_eigh_flops(n: int) -> float:
    """Rough cost of a dense symmetric eigendecomposition (~(9-10) n^3)."""
    return 9.0 * float(n) ** 3


def thin_svd_flops(m: int, n: int) -> float:
    """Rough cost of a thin SVD of an (m x n) matrix with m >= n (~6 m n^2 + 20 n^3)."""
    tall, short = (m, n) if m >= n else (n, m)
    return 6.0 * float(tall) * float(short) ** 2 + 20.0 * float(short) ** 3


def causal_lm_forward_flops(
    shape: ModelShape,
    *,
    tokens: int,
    mean_sequence_length: float | None = None,
) -> dict[str, float]:
    """Forward-pass FLOPs, itemized.

    ``trunk`` is the ``2 * N_non_embedding * T`` term. ``unembedding`` is the
    logits projection, which the trunk term excludes. ``attention`` is the
    sequence-length quadratic term, approximated as
    ``2 * n_layers * d_model * T * mean_seq_len`` after halving for causal
    masking. Returns zeros for any term whose shape metadata is unavailable.
    """
    tokens = float(tokens)
    trunk = 2.0 * float(shape.params_non_embedding) * tokens

    unembedding = 0.0
    if shape.hidden_size and shape.vocab_size:
        unembedding = 2.0 * float(shape.hidden_size) * float(shape.vocab_size) * tokens

    attention = 0.0
    if shape.num_hidden_layers and shape.hidden_size and mean_sequence_length:
        attention = (
            2.0
            * float(shape.num_hidden_layers)
            * float(shape.hidden_size)
            * tokens
            * float(mean_sequence_length)
        )

    return {
        "trunk": trunk,
        "unembedding": unembedding,
        "attention": attention,
        "total": trunk + unembedding + attention,
    }


def causal_lm_flops(
    shape: ModelShape,
    *,
    tokens: int,
    mode: str,
    mean_sequence_length: float | None = None,
) -> dict[str, Any]:
    """Total FLOPs for ``tokens`` tokens under ``mode``.

    ``mode`` is one of ``forward``, ``forward_backward_lora``, or
    ``forward_backward_full``. The last reproduces the usual ``6 N T``
    convention; the LoRA variant uses ``4 N T``.
    """
    multipliers = {
        "forward": 1.0,
        "forward_backward_lora": 1.0 + BACKWARD_MULTIPLIER_LORA,
        "forward_backward_full": 1.0 + BACKWARD_MULTIPLIER_FULL,
    }
    if mode not in multipliers:
        raise ValueError(f"Unsupported FLOPs mode: {mode}")
    multiplier = multipliers[mode]

    forward = causal_lm_forward_flops(shape, tokens=tokens, mean_sequence_length=mean_sequence_length)
    return {
        "mode": mode,
        "multiplier_on_forward": multiplier,
        "tokens": int(tokens),
        "mean_sequence_length": mean_sequence_length,
        "forward_flops": forward["total"],
        "forward_flops_breakdown": {
            "trunk": forward["trunk"],
            "unembedding": forward["unembedding"],
            "attention": forward["attention"],
        },
        "total_flops": multiplier * forward["total"],
        "n_params_used": int(shape.params_non_embedding),
        "convention": (
            "6*N*T" if mode == "forward_backward_full" else "4*N*T" if mode == "forward_backward_lora" else "2*N*T"
        ),
    }


def gradient_extraction_flops(
    shape: ModelShape,
    counts: dict[str, int],
    *,
    mean_sequence_length: float | None = None,
    use_lora: bool = True,
) -> dict[str, Any]:
    """FLOPs for a batch of per-example gradient extractions."""
    mode = "forward_backward_lora" if use_lora else "forward_backward_full"
    tokens = int(counts.get("tokens", 0))
    if mean_sequence_length is None and counts.get("forward_backward_passes"):
        mean_sequence_length = tokens / float(counts["forward_backward_passes"])
    estimate = causal_lm_flops(shape, tokens=tokens, mode=mode, mean_sequence_length=mean_sequence_length)
    estimate["forward_backward_passes"] = int(counts.get("forward_backward_passes", 0))
    return estimate


# ---------------------------------------------------------------------------
# Peak GPU memory
# ---------------------------------------------------------------------------


_PROCESS_PEAK_ALLOCATED = 0
_PROCESS_PEAK_RESERVED = 0


def cuda_available(device: torch.device | str | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def reset_peak_memory(device: torch.device | str | None = None) -> None:
    if not cuda_available(device):
        return
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def read_peak_memory(device: torch.device | str | None = None) -> dict[str, Any]:
    """Read and fold the current peak into the process-wide high-water mark."""
    global _PROCESS_PEAK_ALLOCATED, _PROCESS_PEAK_RESERVED
    if not cuda_available(device):
        return {
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
            "peak_memory_allocated_gib": None,
            "peak_memory_reserved_gib": None,
        }
    torch.cuda.synchronize()
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    _PROCESS_PEAK_ALLOCATED = max(_PROCESS_PEAK_ALLOCATED, allocated)
    _PROCESS_PEAK_RESERVED = max(_PROCESS_PEAK_RESERVED, reserved)
    return {
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
        "peak_memory_allocated_gib": allocated / BYTES_PER_GIB,
        "peak_memory_reserved_gib": reserved / BYTES_PER_GIB,
    }


def process_peak_memory() -> dict[str, Any]:
    """Highest per-stage peak seen so far, plus device totals."""
    if not torch.cuda.is_available():
        return {
            "process_peak_memory_allocated_bytes": None,
            "process_peak_memory_reserved_bytes": None,
            "process_peak_memory_allocated_gib": None,
            "process_peak_memory_reserved_gib": None,
            "device_name": None,
            "device_total_memory_bytes": None,
        }
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "process_peak_memory_allocated_bytes": _PROCESS_PEAK_ALLOCATED,
        "process_peak_memory_reserved_bytes": _PROCESS_PEAK_RESERVED,
        "process_peak_memory_allocated_gib": _PROCESS_PEAK_ALLOCATED / BYTES_PER_GIB,
        "process_peak_memory_reserved_gib": _PROCESS_PEAK_RESERVED / BYTES_PER_GIB,
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
    }


# ---------------------------------------------------------------------------
# Stage profiling
# ---------------------------------------------------------------------------


@dataclass
class StageRecord:
    """Mutable handle handed to the body of :func:`profile_stage`."""

    stage: str
    flops: float | None = None
    flops_detail: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def set_flops(self, total: float, **detail: Any) -> None:
        self.flops = float(total)
        self.flops_detail.update(detail)

    def add_flops(self, total: float, key: str, **detail: Any) -> None:
        self.flops = float(total) if self.flops is None else self.flops + float(total)
        self.flops_detail[key] = {"flops": float(total), **detail}


@contextmanager
def profile_stage(
    stage: str,
    out: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    reset_memory: bool = True,
) -> Iterator[StageRecord]:
    """Time a stage, capture its peak GPU memory, and collect its FLOPs estimate.

    Writes a ``compute_<stage>`` entry into ``out``. Pass ``reset_memory=False``
    for a nested stage that should not clobber the enclosing stage's high-water
    mark.
    """
    if reset_memory:
        reset_peak_memory(device)
    baseline = GRADIENT_PASS_COUNTER.snapshot()
    record = StageRecord(stage=stage)
    start = time.perf_counter()
    try:
        yield record
    finally:
        elapsed = time.perf_counter() - start
        payload: dict[str, Any] = {
            "stage": stage,
            "wallclock_seconds": elapsed,
            "flops": record.flops,
            "flops_detail": record.flops_detail,
            "model_passes": GRADIENT_PASS_COUNTER.delta_since(baseline),
            **read_peak_memory(device),
            **record.extra,
        }
        out[f"compute_{stage}"] = payload


def empty_stage(stage: str, *, reason: str = "cache_hit") -> dict[str, Any]:
    """A zero-cost stage record, for cache hits and skipped work."""
    return {
        "stage": stage,
        "wallclock_seconds": 0.0,
        "flops": 0.0,
        "flops_detail": {},
        "model_passes": {"forward_backward_passes": 0, "forward_passes": 0, "tokens": 0, "padded_tokens": 0},
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
        "peak_memory_allocated_gib": None,
        "peak_memory_reserved_gib": None,
        "skipped_reason": reason,
    }


def summarize_stages(stages: dict[str, Any]) -> dict[str, Any]:
    """Roll per-stage records up into totals for a run."""
    total_flops = 0.0
    any_flops = False
    total_seconds = 0.0
    peak_allocated: int | None = None
    peak_reserved: int | None = None
    peak_stage: str | None = None

    for name, payload in stages.items():
        if not isinstance(payload, dict):
            continue
        flops = payload.get("flops")
        if flops is not None:
            total_flops += float(flops)
            any_flops = True
        total_seconds += float(payload.get("wallclock_seconds") or 0.0)
        allocated = payload.get("peak_memory_allocated_bytes")
        if allocated is not None and (peak_allocated is None or allocated > peak_allocated):
            peak_allocated = int(allocated)
            peak_stage = name
        reserved = payload.get("peak_memory_reserved_bytes")
        if reserved is not None and (peak_reserved is None or reserved > peak_reserved):
            peak_reserved = int(reserved)

    return {
        "total_flops": total_flops if any_flops else None,
        "total_wallclock_seconds": total_seconds,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_allocated_gib": None if peak_allocated is None else peak_allocated / BYTES_PER_GIB,
        "peak_memory_reserved_bytes": peak_reserved,
        "peak_memory_reserved_gib": None if peak_reserved is None else peak_reserved / BYTES_PER_GIB,
        "peak_memory_stage": peak_stage,
        "stages": sorted(stages.keys()),
    }
