from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore

try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - optional dependency
    snapshot_download = None  # type: ignore

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None  # type: ignore


def safe_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value).strip("_") or "value"


def file_fingerprint(path: str | Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    stem = safe_slug(resolved.stem)[:24]
    digest = hashlib.sha1(f"{resolved}:{stat.st_size}:{int(stat.st_mtime)}".encode("utf-8")).hexdigest()[:12]
    return f"{stem}_{digest}"


def dataset_fingerprint(dataset_name: str, subset: str, split: str, label: str) -> str:
    digest = hashlib.sha1(f"{dataset_name}:{subset}:{split}:{label}".encode("utf-8")).hexdigest()[:12]
    return f"{safe_slug(dataset_name)}_{safe_slug(subset)}_{safe_slug(split)}_{safe_slug(label)}_{digest}"


def load_records(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No records found in {resolved}")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [coerce_record(record) for record in parsed]
    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(coerce_record(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {resolved}") from exc
    if not records:
        raise ValueError(f"No records found in {resolved}")
    return records


def coerce_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Each record must be a JSON object.")

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", ""))
            if role:
                normalized_messages.append({"role": role, "content": content})
        if normalized_messages:
            normalized = dict(record)
            normalized["messages"] = normalized_messages
            return normalized

    prompt = str(record.get("instruction", record.get("input", record.get("question", "")))).strip()
    response = str(record.get("output", record.get("answer", ""))).strip()
    if not prompt and not response:
        raise ValueError("Record must contain either messages or instruction/output-style fields.")

    normalized = dict(record)
    normalized["messages"] = []
    if prompt:
        normalized["messages"].append({"role": "user", "content": prompt})
    if response:
        normalized["messages"].append({"role": "assistant", "content": response})
    return normalized


def get_prompt_response(record: dict[str, Any]) -> tuple[str, str]:
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in record.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = str(message.get("content", ""))
        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)
    prompt = "\n".join(user_parts).strip()
    response = "\n".join(assistant_parts).strip()
    if prompt or response:
        return prompt, response
    prompt = str(record.get("instruction", record.get("input", record.get("question", "")))).strip()
    response = str(record.get("output", record.get("answer", ""))).strip()
    return prompt, response


def to_instruction_output(record: dict[str, Any]) -> dict[str, str]:
    prompt, response = get_prompt_response(record)
    return {"instruction": prompt, "output": response}


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records_by_proportions(
    records: Sequence[Any],
    proportions: Sequence[float],
    *,
    seed: int = 42,
    shuffle: bool = True,
    group_key: str | None = None,
) -> list[list[Any]]:
    if not records:
        return [[] for _ in proportions]
    if not proportions:
        raise ValueError("proportions must not be empty.")
    if any(float(value) < 0.0 for value in proportions):
        raise ValueError("proportions must be non-negative.")
    total = float(sum(float(value) for value in proportions))
    if total <= 0.0:
        raise ValueError("At least one split proportion must be positive.")

    def _split_sequence(items: list[Any], local_seed: int) -> list[list[Any]]:
        num_records = len(items)
        normalized = [float(value) / total for value in proportions]
        raw_sizes = [value * num_records for value in normalized]
        base_sizes = [int(size) for size in raw_sizes]
        remainder = num_records - sum(base_sizes)
        if remainder > 0:
            fractional = sorted(
                ((raw_sizes[idx] - base_sizes[idx], idx) for idx in range(len(base_sizes))),
                key=lambda item: (-item[0], item[1]),
            )
            for _, idx in fractional[:remainder]:
                base_sizes[idx] += 1

        ordered = list(items)
        if shuffle:
            rng = random.Random(int(local_seed))
            rng.shuffle(ordered)

        result: list[list[Any]] = []
        cursor = 0
        for size in base_sizes:
            result.append(list(ordered[cursor : cursor + size]))
            cursor += size
        return result

    if group_key is None:
        return _split_sequence(list(records), seed)

    grouped: dict[str, list[Any]] = {}
    for record in records:
        if isinstance(record, dict):
            group_value = str(record.get(group_key, "__missing__"))
        else:
            group_value = "__missing__"
        grouped.setdefault(group_value, []).append(record)

    merged_splits: list[list[Any]] = [[] for _ in proportions]
    for group_index, (group_value, group_records) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        group_splits = _split_sequence(group_records, seed + group_index * 9973)
        for split_idx, split_records in enumerate(group_splits):
            merged_splits[split_idx].extend(split_records)
    return merged_splits


@dataclass
class StereoTriplet:
    id: str
    bias_type: str
    target: str
    context: str
    stereotype: str
    anti_stereotype: str
    unrelated: str
    subset: str


DEFAULT_STEREOSET_LABELS = {
    0: "anti-stereotype",
    1: "stereotype",
    2: "unrelated",
    3: "related",
}


def _label_to_str(label, feature) -> str:
    if isinstance(label, str):
        return label
    if isinstance(label, int) and label in DEFAULT_STEREOSET_LABELS:
        return DEFAULT_STEREOSET_LABELS[label]
    if feature is None:
        return str(label)
    try:
        return feature.int2str(int(label))
    except Exception:
        return str(label)


def _get_gold_label_feature(features) -> Any:
    try:
        f = features["sentences"]["gold_label"]
        if hasattr(f, "int2str"):
            return f
        if hasattr(f, "feature") and hasattr(f.feature, "int2str"):
            return f.feature
        return f
    except Exception:
        pass
    try:
        sent_feat = features["sentences"]
        if hasattr(sent_feat, "feature"):
            inner = sent_feat.feature
            if isinstance(inner, dict):
                return inner.get("gold_label")
            try:
                return inner["gold_label"]
            except Exception:
                return None
    except Exception:
        pass
    return None


def _iter_sentence_rows(sents) -> Iterable[dict]:
    if isinstance(sents, list):
        for s in sents:
            if isinstance(s, dict):
                yield s
        return
    if isinstance(sents, dict):
        cols = [k for k in ("gold_label", "sentence", "id", "labels") if k in sents]
        if not cols:
            return
        n = max((len(v) for v in sents.values() if isinstance(v, list)), default=0)
        for i in range(n):
            row = {}
            for k, v in sents.items():
                if isinstance(v, list):
                    row[k] = v[i] if i < len(v) else None
                else:
                    row[k] = v
            yield row


def parse_triplet(example: dict, subset: str, features=None) -> StereoTriplet:
    sents = example["sentences"]
    gold_feature = _get_gold_label_feature(features) if features is not None else None

    label2sent: Dict[str, str] = {}
    for s in _iter_sentence_rows(sents):
        lbl = _label_to_str(s["gold_label"], gold_feature)
        label2sent[lbl] = s["sentence"]

    required = ["stereotype", "anti-stereotype", "unrelated"]
    missing = [k for k in required if k not in label2sent]
    if missing:
        raise ValueError(f"Example {example.get('id')} missing labels {missing}")

    return StereoTriplet(
        id=example["id"],
        bias_type=example.get("bias_type", ""),
        target=example.get("target", ""),
        context=example.get("context", ""),
        stereotype=label2sent["stereotype"],
        anti_stereotype=label2sent["anti-stereotype"],
        unrelated=label2sent["unrelated"],
        subset=subset,
    )


def iter_triplets(ds, subset: str) -> Iterable[StereoTriplet]:
    features = getattr(ds, "features", None)
    for ex in ds:
        yield parse_triplet(ex, subset=subset, features=features)


def _hf_cache_hub_roots() -> list[Path]:
    candidates: list[Path] = []
    env_hf_home = os.environ.get("HF_HOME")
    env_hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    env_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    for raw in [env_hub_cache, env_hf_home, env_transformers_cache]:
        if not raw:
            continue
        root = Path(raw)
        if root.name == "hub":
            candidates.append(root)
        else:
            candidates.append(root / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    # Preserve order while removing duplicates.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _find_cached_stereoset_parquet(subset: str, split: str) -> Optional[Path]:
    relative = Path(subset) / f"{split}-00000-of-00001.parquet"
    for hub_root in _hf_cache_hub_roots():
        repo_root = hub_root / "datasets--McGill-NLP--stereoset" / "snapshots"
        if not repo_root.exists():
            continue
        for snapshot_dir in sorted(repo_root.iterdir(), reverse=True):
            candidate = snapshot_dir / relative
            if candidate.exists():
                return candidate
    return None


def _download_stereoset_snapshot(subset: str, split: str) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not installed; cannot download StereoSet from HF.")
    allow_patterns = [f"{subset}/{split}-*.parquet", "README.md"]
    try:
        snapshot_dir = snapshot_download(
            repo_id="McGill-NLP/stereoset",
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
    except Exception:
        snapshot_dir = snapshot_download(
            repo_id="McGill-NLP/stereoset",
            repo_type="dataset",
            allow_patterns=allow_patterns,
            local_files_only=False,
        )
    snapshot_path = Path(snapshot_dir)
    matches = sorted((snapshot_path / subset).glob(f"{split}-*.parquet"))
    if not matches:
        raise FileNotFoundError(
            f"StereoSet snapshot for subset={subset!r}, split={split!r} did not contain parquet files."
        )
    return matches[0]


def _resolve_stereoset_parquet(subset: str, split: str) -> Path:
    cached = _find_cached_stereoset_parquet(subset=subset, split=split)
    if cached is not None:
        return cached
    return _download_stereoset_snapshot(subset=subset, split=split)


def _load_stereoset_examples_from_parquet(subset: str, split: str) -> list[dict[str, Any]]:
    if pq is None:
        raise RuntimeError("pyarrow is not installed; cannot read StereoSet parquet files.")
    parquet_path = _resolve_stereoset_parquet(subset=subset, split=split)
    table = pq.read_table(parquet_path)
    return table.to_pylist()


def _load_stereoset_examples(subset: str, split: str) -> tuple[list[dict[str, Any]], Any]:
    if pq is not None:
        try:
            return _load_stereoset_examples_from_parquet(subset=subset, split=split), None
        except Exception:
            pass
    if load_dataset is None:
        raise RuntimeError(
            "Neither parquet loading nor datasets is available; cannot load StereoSet from HF."
        )
    ds = load_dataset("McGill-NLP/stereoset", subset, split=split)
    return list(ds), getattr(ds, "features", None)


def load_stereoset_triplet_records(
    *,
    subset: str = "intrasentence",
    split: str = "validation",
) -> list[dict[str, Any]]:
    examples, features = _load_stereoset_examples(subset=subset, split=split)
    records: list[dict[str, Any]] = []
    for example in examples:
        triplet = parse_triplet(example, subset=subset, features=features)
        records.append(
            {
                "id": triplet.id,
                "bias_type": triplet.bias_type,
                "target": triplet.target,
                "context": triplet.context,
                "stereotype": triplet.stereotype,
                "anti_stereotype": triplet.anti_stereotype,
                "unrelated": triplet.unrelated,
                "subset": triplet.subset,
            }
        )
    return records


def load_stereoset_records(
    *,
    subset: str = "intrasentence",
    split: str = "validation",
    label: str = "stereotype",
    output_format: str = "instruction",
) -> list[dict[str, Any]]:
    examples, features = _load_stereoset_examples(subset=subset, split=split)
    if label == "all":
        labels = ["stereotype", "anti-stereotype", "unrelated"]
    else:
        labels = [value.strip() for value in str(label).split(",") if value.strip()]
        if not labels:
            raise ValueError("label must be 'all' or a non-empty comma-separated list of StereoSet labels.")
    records: list[dict[str, Any]] = []
    for example in examples:
        triplet = parse_triplet(example, subset=subset, features=features)
        for lbl in labels:
            instruction = triplet.context.strip() if triplet.context else "Complete the sentence."
            output = {
                "stereotype": triplet.stereotype,
                "anti-stereotype": triplet.anti_stereotype,
                "unrelated": triplet.unrelated,
            }[lbl]
            if output_format == "messages":
                records.append(
                    {
                        "id": triplet.id,
                        "bias_type": triplet.bias_type,
                        "target": triplet.target,
                        "subset": triplet.subset,
                        "label": lbl,
                        "messages": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": output},
                        ],
                    }
                )
            else:
                records.append(
                    {
                        "id": triplet.id,
                        "bias_type": triplet.bias_type,
                        "target": triplet.target,
                        "subset": triplet.subset,
                        "label": lbl,
                        "instruction": instruction,
                        "output": output,
                    }
                )
    return records


def _require_datasets_loader(dataset_name: str) -> None:
    if load_dataset is None:
        raise RuntimeError(f"datasets is not installed; cannot load {dataset_name}.")


def load_gsm8k_records(
    *,
    subset: str = "main",
    split: str = "test",
) -> list[dict[str, Any]]:
    _require_datasets_loader("gsm8k")
    ds = load_dataset("gsm8k", subset, split=split)
    records: list[dict[str, Any]] = []
    for idx, example in enumerate(ds):
        question = str(example.get("question", "")).strip()
        answer = str(example.get("answer", "")).strip()
        if not question or not answer:
            continue
        records.append(
            {
                "id": str(example.get("id", idx)),
                "dataset": "gsm8k",
                "subset": subset,
                "split": split,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return records


def load_mmlu_records(
    *,
    subset: str = "all",
    split: str = "test",
) -> list[dict[str, Any]]:
    _require_datasets_loader("cais/mmlu")
    ds = load_dataset("cais/mmlu", subset, split=split)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    records: list[dict[str, Any]] = []
    for idx, example in enumerate(ds):
        question = str(example.get("question", "")).strip()
        choices = example.get("choices") or []
        answer = example.get("answer")
        if not question or not isinstance(choices, list) or answer is None:
            continue
        choice_lines = [f"{labels[i]}. {str(choice).strip()}" for i, choice in enumerate(choices)]
        prompt = "\n".join(
            [
                question,
                "",
                *choice_lines,
                "",
                "Answer with the single best option letter.",
            ]
        ).strip()
        if isinstance(answer, str):
            gold = answer.strip().upper()
        else:
            answer_index = int(answer)
            gold = labels[answer_index] if 0 <= answer_index < len(labels) else str(answer)
        records.append(
            {
                "id": str(example.get("id", idx)),
                "dataset": "mmlu",
                "subject": str(example.get("subject", subset)),
                "subset": subset,
                "split": split,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": gold},
                ],
            }
        )
    return records
