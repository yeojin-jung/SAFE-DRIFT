from __future__ import annotations

import csv
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


EPS = 1e-30


def _to_float_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(dtype=torch.float32, device="cpu")


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _record_key(record: dict[str, Any]) -> str:
    for key in ("id", "text_hash"):
        value = record.get(key)
        if value not in {None, ""}:
            return f"{key}:{value}"
    messages = record.get("messages")
    if isinstance(messages, list):
        return "messages:" + json.dumps(messages, sort_keys=True, ensure_ascii=False)
    prompt = record.get("prompt")
    if prompt not in {None, ""}:
        return f"prompt:{prompt}"
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def selected_indices_from_records(
    candidates: list[dict[str, Any]],
    selected_records: Iterable[dict[str, Any]],
) -> list[int]:
    buckets: dict[str, deque[int]] = defaultdict(deque)
    for index, record in enumerate(candidates):
        buckets[_record_key(record)].append(index)

    indices: list[int] = []
    missing = 0
    for record in selected_records:
        bucket = buckets.get(_record_key(record))
        if not bucket:
            missing += 1
            continue
        indices.append(bucket.popleft())
    if missing:
        raise ValueError(f"Could not map {missing} selected records back to the candidate pool.")
    return indices


def resolve_reference_rank(
    *,
    feature_dim: int,
    reference_fisher: torch.Tensor | None,
    metadata: dict[str, Any] | None = None,
    explicit_rank: int | None = None,
) -> int:
    if explicit_rank is not None:
        rank = int(explicit_rank)
        if 0 < rank <= feature_dim:
            return rank
    metadata = metadata or {}
    for key in (
        "low_rank_resolved_reference_rank",
        "low_rank_reference_rank",
        "K_R",
        "reference_rank",
    ):
        value = metadata.get(key)
        if value in {None, "auto"}:
            continue
        rank = int(value)
        if 0 < rank <= feature_dim:
            return rank
    if reference_fisher is not None and reference_fisher.numel() > 0:
        flat = reference_fisher.detach().float().flatten()
        nonzero = int((flat.abs() > 1e-12).sum().item())
        if 0 < nonzero <= feature_dim:
            return nonzero
        return min(int(flat.numel()), feature_dim)
    raise ValueError("Cannot resolve reference rank without metadata or reference_fisher.")


def block_split(x: torch.Tensor, k_r: int) -> tuple[torch.Tensor, torch.Tensor]:
    return x[..., :k_r], x[..., k_r:]


def summarize_tensor(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().cpu().flatten()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    arr = values.numpy()
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_selected(values: torch.Tensor, selected_indices: np.ndarray) -> dict[str, Any]:
    summary = {"all": summarize_tensor(values)}
    if selected_indices.size:
        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        summary["selected"] = summarize_tensor(values[index_tensor])
    else:
        summary["selected"] = summarize_tensor(values[:0])
    return summary


def energy_share(features: torch.Tensor, k_r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_block, task_block = block_split(features, k_r)
    ref_norm2 = (ref_block**2).sum(dim=-1)
    task_norm2 = (task_block**2).sum(dim=-1)
    share = ref_norm2 / (ref_norm2 + task_norm2).clamp_min(EPS)
    return share, ref_norm2.sqrt(), task_norm2.sqrt()


def preconditioner_share(features: torch.Tensor, preconditioner: torch.Tensor, k_r: int) -> torch.Tensor:
    ref_block, task_block = block_split(features, k_r)
    p_rr = preconditioner[:k_r, :k_r]
    p_tt = preconditioner[k_r:, k_r:]
    e_r = (ref_block @ p_rr * ref_block).sum(dim=-1)
    e_t = (task_block @ p_tt * task_block).sum(dim=-1)
    return e_r / (e_r + e_t).clamp_min(EPS)


def reference_load(features: torch.Tensor, reference_fisher: torch.Tensor, k_r: int) -> torch.Tensor:
    ref_block = features[..., :k_r]
    lamb = reference_fisher[:k_r].flatten()
    return (ref_block**2 * lamb).sum(dim=-1)


def influence_decomposition(
    targets: torch.Tensor,
    candidates: torch.Tensor,
    preconditioner: torch.Tensor,
    k_r: int,
    *,
    target_sample_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    n_targets = int(targets.shape[0])
    if n_targets <= 0:
        empty = torch.zeros(candidates.shape[0], dtype=torch.float32)
        return {"RR": empty, "cross": empty, "TT": empty, "total": empty, "signed_total": empty}
    count = min(max(1, int(target_sample_size)), n_targets)
    rng = random.Random(int(seed))
    indices = list(range(n_targets))
    rng.shuffle(indices)
    sampled = targets[indices[:count]]

    p_rr = preconditioner[:k_r, :k_r]
    p_rt = preconditioner[:k_r, k_r:]
    p_tr = preconditioner[k_r:, :k_r]
    p_tt = preconditioner[k_r:, k_r:]
    t_r, t_t = block_split(sampled, k_r)
    c_r, c_t = block_split(candidates, k_r)

    s_rr = t_r @ p_rr @ c_r.T
    s_rt = t_r @ p_rt @ c_t.T
    s_tr = t_t @ p_tr @ c_r.T
    s_tt = t_t @ p_tt @ c_t.T
    total = s_rr + s_rt + s_tr + s_tt
    return {
        "RR": s_rr.abs().mean(dim=0),
        "cross": (s_rt + s_tr).abs().mean(dim=0),
        "TT": s_tt.abs().mean(dim=0),
        "total": total.abs().mean(dim=0),
        "signed_total": total.mean(dim=0),
    }


def target_block_alignment(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    preconditioner: torch.Tensor,
    k_r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_mean = targets.mean(dim=0)
    target_r, target_t = block_split(target_mean, k_r)
    cand_r, cand_t = block_split(candidates, k_r)
    p_rr = preconditioner[:k_r, :k_r]
    p_tt = preconditioner[k_r:, k_r:]

    num_r = cand_r @ (p_rr @ target_r)
    num_t = cand_t @ (p_tt @ target_t)
    norm_c_r = torch.sqrt(((cand_r @ p_rr) * cand_r).sum(dim=-1).clamp_min(EPS))
    norm_c_t = torch.sqrt(((cand_t @ p_tt) * cand_t).sum(dim=-1).clamp_min(EPS))
    norm_t_r = torch.sqrt((target_r @ p_rr @ target_r).clamp_min(EPS))
    norm_t_t = torch.sqrt((target_t @ p_tt @ target_t).clamp_min(EPS))
    return num_r / (norm_c_r * norm_t_r).clamp_min(EPS), num_t / (norm_c_t * norm_t_t).clamp_min(EPS)


def damped_fisher_alignment(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    reference_fisher: torch.Tensor,
    k_r: int,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    target_mean = targets.mean(dim=0)
    target_r, target_t = block_split(target_mean, k_r)
    cand_r, cand_t = block_split(candidates, k_r)
    lamb = reference_fisher[:k_r].flatten()
    d_r = cand_r @ (target_r / (lamb + float(alpha)).clamp_min(EPS))
    d_t = (cand_t @ target_t) / max(float(alpha), EPS)
    diag = {
        "alpha": float(alpha),
        "d_R_std": float(d_r.std().item()) if d_r.numel() > 1 else 0.0,
        "d_T_std": float(d_t.std().item()) if d_t.numel() > 1 else 0.0,
        "d_R_median_abs": float(d_r.abs().median().item()),
        "d_T_median_abs": float(d_t.abs().median().item()),
    }
    diag["T_over_R_std_ratio"] = float(diag["d_T_std"] / max(diag["d_R_std"], EPS))
    return d_r, d_t, diag


def quadrant_fractions(x: torch.Tensor, y: torch.Tensor, selected_indices: np.ndarray | None = None) -> dict[str, float]:
    if selected_indices is not None:
        index_tensor = torch.as_tensor(selected_indices, dtype=torch.long)
        x = x[index_tensor]
        y = y[index_tensor]
    if x.numel() == 0:
        return {
            "T_pos_R_neg_ideal": 0.0,
            "T_pos_R_pos_useful_costly": 0.0,
            "T_neg_R_pos_pure_drift": 0.0,
            "T_neg_R_neg_misaligned": 0.0,
        }
    sx = x > 0
    sy = y > 0
    return {
        "T_pos_R_neg_ideal": float(((~sx) & sy).float().mean().item()),
        "T_pos_R_pos_useful_costly": float((sx & sy).float().mean().item()),
        "T_neg_R_pos_pure_drift": float((sx & (~sy)).float().mean().item()),
        "T_neg_R_neg_misaligned": float(((~sx) & (~sy)).float().mean().item()),
    }


def pca_block_energy_summary(candidates: torch.Tensor, k_r: int, n_components: int = 16) -> dict[str, Any]:
    if candidates.shape[0] < 2 or candidates.shape[1] < 2:
        return {"components": []}
    x = candidates - candidates.mean(dim=0, keepdim=True)
    _, singular_values, v_h = torch.linalg.svd(x, full_matrices=False)
    v = v_h.T
    count = min(int(n_components), int(v.shape[1]))
    rows = []
    for index in range(count):
        component = v[:, index]
        ref_energy = float((component[:k_r] ** 2).sum().item())
        task_energy = float((component[k_r:] ** 2).sum().item())
        rows.append(
            {
                "component": index + 1,
                "singular_value": float(singular_values[index].item()),
                "reference_energy_share": ref_energy / max(ref_energy + task_energy, EPS),
                "task_energy_share": task_energy / max(ref_energy + task_energy, EPS),
            }
        )
    return {"components": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_entanglement(
    *,
    candidate_features: torch.Tensor,
    target_feature: torch.Tensor,
    target_features: torch.Tensor | None,
    reference_fisher: torch.Tensor,
    selector_preconditioner: torch.Tensor | None,
    selected_indices: list[int],
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    reference_rank: int | None = None,
    target_sample_size: int = 128,
    seed: int = 42,
    selected_csv_max_rows: int = 5000,
    label: str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = _to_float_cpu(candidate_features)
    target_mean = _to_float_cpu(target_feature)
    targets = _to_float_cpu(target_features)
    fisher = _to_float_cpu(reference_fisher)
    if candidates is None or target_mean is None or fisher is None:
        raise ValueError("candidate_features, target_feature, and reference_fisher are required.")
    if candidates.ndim != 2:
        raise ValueError(f"candidate_features must be 2D, got shape={tuple(candidates.shape)}.")
    if target_mean.ndim == 1:
        target_mean = target_mean.unsqueeze(0)
    if targets is None:
        targets = target_mean
    elif targets.ndim == 1:
        targets = targets.unsqueeze(0)

    feature_dim = int(candidates.shape[1])
    k_r = resolve_reference_rank(
        feature_dim=feature_dim,
        reference_fisher=fisher,
        metadata=metadata,
        explicit_rank=reference_rank,
    )
    k_t = feature_dim - k_r
    if k_t < 0:
        raise ValueError(f"Resolved reference rank {k_r} exceeds feature dimension {feature_dim}.")

    if selector_preconditioner is None:
        preconditioner = torch.eye(feature_dim, dtype=torch.float32)
        preconditioner_source = "identity"
    else:
        preconditioner = _to_float_cpu(selector_preconditioner)
        if preconditioner is None:
            preconditioner = torch.eye(feature_dim, dtype=torch.float32)
            preconditioner_source = "identity"
        else:
            preconditioner_source = "selector"
    if preconditioner.ndim == 1:
        preconditioner = torch.diag(preconditioner)
    if preconditioner.shape != (feature_dim, feature_dim):
        raise ValueError(
            f"selector_preconditioner must have shape {(feature_dim, feature_dim)}, "
            f"got {tuple(preconditioner.shape)}."
        )

    selected_ordered: list[int] = []
    seen_selected: set[int] = set()
    for raw_index in selected_indices:
        index = int(raw_index)
        if index < 0 or index >= candidates.shape[0]:
            raise ValueError("Selected indices contain out-of-range candidate rows.")
        if index in seen_selected:
            continue
        seen_selected.add(index)
        selected_ordered.append(index)
    selected = np.array(selected_ordered, dtype=np.int64)

    rho_e_candidates, ref_norm, task_norm = energy_share(candidates, k_r)
    rho_e_targets, _, _ = energy_share(targets, k_r)
    rho_p_candidates = preconditioner_share(candidates, preconditioner, k_r)
    rho_p_targets = preconditioner_share(targets, preconditioner, k_r)
    load = reference_load(candidates, fisher, k_r)
    influence = influence_decomposition(
        targets,
        candidates,
        preconditioner,
        k_r,
        target_sample_size=target_sample_size,
        seed=seed,
    )
    align_r, align_t = target_block_alignment(candidates, targets, preconditioner, k_r)
    alpha_raw = (metadata or {}).get("low_rank_builder_alpha", 1.0e-3)
    alpha = 1.0e-3 if alpha_raw is None else float(alpha_raw)
    damped_r, damped_t, damped_diag = damped_fisher_alignment(
        candidates,
        targets,
        fisher,
        k_r,
        alpha=alpha,
    )

    summary: dict[str, Any] = {
        "label": label,
        "output_dir": str(output_dir),
        "K_R": int(k_r),
        "K_T": int(k_t),
        "n_candidates": int(candidates.shape[0]),
        "n_targets": int(targets.shape[0]),
        "n_selected": int(selected.size),
        "preconditioner_source": preconditioner_source,
        "rho_E_candidates": summarize_selected(rho_e_candidates, selected),
        "rho_E_targets": summarize_tensor(rho_e_targets),
        "rho_P_candidates": summarize_selected(rho_p_candidates, selected),
        "rho_P_targets": summarize_tensor(rho_p_targets),
        "reference_load": summarize_selected(load, selected),
        "reference_norm": summarize_selected(ref_norm, selected),
        "task_norm": summarize_selected(task_norm, selected),
        "influence_decomposition": {
            key: summarize_selected(value, selected)
            for key, value in influence.items()
        },
        "target_block_alignment": {
            "R": summarize_selected(align_r, selected),
            "T": summarize_selected(align_t, selected),
            "quadrants_all": quadrant_fractions(align_r, align_t),
            "quadrants_selected": quadrant_fractions(align_r, align_t, selected),
        },
        "damped_fisher_alignment": {
            "R": summarize_selected(damped_r, selected),
            "T": summarize_selected(damped_t, selected),
            "diagnostics": damped_diag,
            "quadrants_all": quadrant_fractions(damped_r, damped_t),
            "quadrants_selected": quadrant_fractions(damped_r, damped_t, selected),
        },
        "pca_block_energy": pca_block_energy_summary(candidates, k_r),
    }

    row_limit = max(0, int(selected_csv_max_rows))
    rows: list[dict[str, Any]] = []
    for selected_rank, candidate_index in enumerate(selected[:row_limit], start=1):
        idx = int(candidate_index)
        rows.append(
            {
                "selected_rank": selected_rank,
                "candidate_row": idx,
                "rho_E": float(rho_e_candidates[idx].item()),
                "rho_P": float(rho_p_candidates[idx].item()),
                "reference_load": float(load[idx].item()),
                "reference_norm": float(ref_norm[idx].item()),
                "task_norm": float(task_norm[idx].item()),
                "influence_RR": float(influence["RR"][idx].item()),
                "influence_cross": float(influence["cross"][idx].item()),
                "influence_TT": float(influence["TT"][idx].item()),
                "influence_total": float(influence["total"][idx].item()),
                "influence_signed_total": float(influence["signed_total"][idx].item()),
                "target_align_R": float(align_r[idx].item()),
                "target_align_T": float(align_t[idx].item()),
                "damped_align_R": float(damped_r[idx].item()),
                "damped_align_T": float(damped_t[idx].item()),
            }
        )
    selected_csv_path = output_dir / "selected_subset_entanglement_metrics.csv"
    _write_csv(selected_csv_path, rows)
    if rows:
        summary["selected_subset_metrics_csv"] = str(selected_csv_path)

    summary_path = output_dir / "entanglement_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return _json_safe(summary)


def headline_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    def get(path: list[str], default: Any = None) -> Any:
        cur: Any = summary
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    return {
        "summary_file": str(Path(str(summary["output_dir"])) / "entanglement_summary.json"),
        "n_selected": summary.get("n_selected"),
        "K_R": summary.get("K_R"),
        "K_T": summary.get("K_T"),
        "rho_E_selected_mean": get(["rho_E_candidates", "selected", "mean"]),
        "rho_P_selected_mean": get(["rho_P_candidates", "selected", "mean"]),
        "reference_load_selected_mean": get(["reference_load", "selected", "mean"]),
        "target_alignment_selected_ideal_fraction": get(
            ["target_block_alignment", "quadrants_selected", "T_pos_R_neg_ideal"],
        ),
        "damped_alignment_selected_ideal_fraction": get(
            ["damped_fisher_alignment", "quadrants_selected", "T_pos_R_neg_ideal"],
        ),
    }
