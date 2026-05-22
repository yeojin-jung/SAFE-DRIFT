from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - optional dependency for adapter evals
    PeftModel = None  # type: ignore


@dataclass(frozen=True)
class BiasExample:
    bias_type: str
    stereotype_text: str
    anti_text: str
    unrelated_text: str


@dataclass(frozen=True)
class BehaviorMetrics:
    ss: float
    lms: float
    icat: float
    mean_bias_margin: float
    mean_lm_margin: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    lowered = str(dtype_name).strip().lower()
    if lowered == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
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


def parse_layers(layers: str) -> list[int]:
    return [int(value.strip()) for value in str(layers).split(",") if value.strip()]


def load_examples(path: str | Path) -> list[BiasExample]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of StereoSet triplets in {path}.")
    examples: list[BiasExample] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        examples.append(
            BiasExample(
                bias_type=str(record.get("bias_type", "")).strip().lower(),
                stereotype_text=str(record.get("stereotype", "")).strip(),
                anti_text=str(record.get("anti_stereotype", record.get("anti-stereotype", ""))).strip(),
                unrelated_text=str(record.get("unrelated", "")).strip(),
            )
        )
    return examples


def filter_examples(
    examples: list[BiasExample],
    *,
    domain: str,
    max_examples: int | None,
) -> list[BiasExample]:
    domain = str(domain).strip().lower()
    filtered = [example for example in examples if domain in {"", "all"} or example.bias_type == domain]
    if max_examples is not None:
        filtered = filtered[: int(max_examples)]
    return filtered


def compute_behavior_metrics(
    stereotype_scores: list[float],
    anti_scores: list[float],
    unrelated_scores: list[float],
) -> BehaviorMetrics:
    if not stereotype_scores:
        return BehaviorMetrics(ss=0.0, lms=0.0, icat=0.0, mean_bias_margin=0.0, mean_lm_margin=0.0)
    bias_wins = [float(s > a) for s, a in zip(stereotype_scores, anti_scores, strict=True)]
    lm_wins = [
        float(max(s, a) > u)
        for s, a, u in zip(stereotype_scores, anti_scores, unrelated_scores, strict=True)
    ]
    ss = 100.0 * sum(bias_wins) / len(bias_wins)
    lms = 100.0 * sum(lm_wins) / len(lm_wins)
    icat = lms * min(ss, 100.0 - ss) / 50.0
    mean_bias_margin = sum(s - a for s, a in zip(stereotype_scores, anti_scores, strict=True)) / len(stereotype_scores)
    mean_lm_margin = (
        sum(max(s, a) - u for s, a, u in zip(stereotype_scores, anti_scores, unrelated_scores, strict=True))
        / len(stereotype_scores)
    )
    return BehaviorMetrics(
        ss=ss,
        lms=lms,
        icat=icat,
        mean_bias_margin=mean_bias_margin,
        mean_lm_margin=mean_lm_margin,
    )


class CausalLMEvaluator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        batch_size: int,
        max_length: int,
        trust_remote_code: bool,
        adapter_path: str | None = None,
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if adapter_path is not None:
            if PeftModel is None:
                raise RuntimeError("peft is required to evaluate a saved adapter.")
            model = PeftModel.from_pretrained(model, adapter_path)
        self.model = model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: list[str], *, desc: str) -> list[float]:
        scores: list[float] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**encoded).logits[:, :-1, :]
            labels = encoded["input_ids"][:, 1:]
            mask = encoded["attention_mask"][:, 1:].float()
            token_log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            scores.extend((token_log_probs * mask).sum(dim=1).detach().float().cpu().tolist())
        return scores


NUMERIC_METRICS = [
    "SS",
    "LMS",
    "ICAT",
    "mean_bias_margin",
    "mean_lm_margin",
]

BEHAVIOR_METRIC_ATTRS = {
    "SS": "ss",
    "LMS": "lms",
    "ICAT": "icat",
    "mean_bias_margin": "mean_bias_margin",
    "mean_lm_margin": "mean_lm_margin",
}


def _layer_tag(layer_requested: Any) -> str:
    return f"layer{str(layer_requested).replace('-', 'neg')}"


def _score_behavior_metrics(
    *,
    model_name_or_path: str,
    filtered_examples,
    layers: str,
    device: str | None,
    torch_dtype: str,
    batch_size: int,
    max_length: int,
    trust_remote_code: bool,
    adapter_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    device_obj = choose_device(device)
    dtype = choose_dtype(torch_dtype, device_obj)
    layer_list = parse_layers(layers)

    evaluator = CausalLMEvaluator(
        model_name_or_path=model_name_or_path,
        device=device_obj,
        torch_dtype=dtype,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        adapter_path=adapter_path,
    )

    s_texts = [ex.stereotype_text for ex in filtered_examples]
    a_texts = [ex.anti_text for ex in filtered_examples]
    u_texts = [ex.unrelated_text for ex in filtered_examples]

    s_scores = evaluator.score_texts(s_texts, desc=f"{Path(model_name_or_path).name}: stereotype")
    a_scores = evaluator.score_texts(a_texts, desc=f"{Path(model_name_or_path).name}: anti")
    u_scores = evaluator.score_texts(u_texts, desc=f"{Path(model_name_or_path).name}: unrelated")
    behavior = compute_behavior_metrics(s_scores, a_scores, u_scores)

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for requested_layer in layer_list:
        layer_tag = _layer_tag(requested_layer)
        row = {
            "layer_requested": requested_layer,
            "layer_resolved": requested_layer,
            "n_examples": len(filtered_examples),
        }
        for metric in NUMERIC_METRICS:
            value = float(getattr(behavior, BEHAVIOR_METRIC_ATTRS[metric]))
            row[metric] = value
            metrics[f"bias_disentangle_{layer_tag}_{metric}"] = value
        rows.append(row)
    return rows, metrics


def evaluate_saved_adapter(
    *,
    base_model_name_or_path: str,
    adapter_path: str | Path,
    data_path: str | Path,
    domain: str = "gender",
    layers: str = "-1",
    alpha: float = 0.5,
    geometry: str = "sand-e",
    direction_split: float = 0.5,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 512,
    max_examples: int | None = None,
    torch_dtype: str = "auto",
    device: str | None = None,
    trust_remote_code: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    del alpha, geometry, direction_split
    set_seed(seed)
    examples = load_examples(str(data_path))
    filtered = filter_examples(examples, domain=domain, max_examples=max_examples)
    if not filtered:
        raise ValueError(f"No examples available for bias disentangle evaluation after filtering domain={domain!r}.")

    base_rows, base_metrics = _score_behavior_metrics(
        model_name_or_path=base_model_name_or_path,
        filtered_examples=filtered,
        layers=layers,
        device=device,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )
    ft_rows, ft_metrics = _score_behavior_metrics(
        model_name_or_path=base_model_name_or_path,
        filtered_examples=filtered,
        layers=layers,
        device=device,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
        adapter_path=str(adapter_path),
    )

    metrics: dict[str, float] = {}
    for metric_key, value in base_metrics.items():
        suffix = metric_key[len("bias_disentangle_") :]
        metrics[f"bias_disentangle_{suffix}_base"] = float(value)
    for metric_key, value in ft_metrics.items():
        suffix = metric_key[len("bias_disentangle_") :]
        metrics[f"bias_disentangle_{suffix}_ft"] = float(value)

    for metric_key, value in base_metrics.items():
        suffix = metric_key[len("bias_disentangle_") :]
        layer_tag, metric_name = suffix.split("_", 1)
        base_value = float(value)
        ft_value = float(ft_metrics[metric_key])
        metrics[f"bias_disentangle_{layer_tag}_delta_{metric_name}"] = float(ft_value - base_value)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps({"base": base_rows, "ft": ft_rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "summary_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return metrics


def evaluate_base_model(
    *,
    base_model_name_or_path: str,
    data_path: str | Path,
    domain: str = "gender",
    layers: str = "-1",
    alpha: float = 0.5,
    geometry: str = "sand-e",
    direction_split: float = 0.5,
    seed: int = 42,
    batch_size: int = 8,
    max_length: int = 512,
    max_examples: int | None = None,
    torch_dtype: str = "auto",
    device: str | None = None,
    trust_remote_code: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    del alpha, geometry, direction_split
    set_seed(seed)
    examples = load_examples(str(data_path))
    filtered = filter_examples(examples, domain=domain, max_examples=max_examples)
    if not filtered:
        raise ValueError(f"No examples available for bias disentangle evaluation after filtering domain={domain!r}.")

    rows, raw_metrics = _score_behavior_metrics(
        model_name_or_path=base_model_name_or_path,
        filtered_examples=filtered,
        layers=layers,
        device=device,
        torch_dtype=torch_dtype,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=trust_remote_code,
    )

    metrics: dict[str, float] = {}
    for metric_key, value in raw_metrics.items():
        suffix = metric_key[len("bias_disentangle_") :]
        metrics[f"bias_disentangle_{suffix}_base"] = float(value)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        (output_dir / "summary_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return metrics
