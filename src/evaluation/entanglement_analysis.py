"""
Entanglement analysis for the low-rank selector cache.

Reads selector_features.pt where the K = K_R + K_T column layout is:
    g = [ g^R (reference subspace, K_R dims) | g^T (residual / task subspace, K_T dims) ].

Computes four entanglement metrics and writes plots + a text summary into
./entanglement_analysis/ next to the cache.

    1. rho_E    energy share in reference block:        ||g^R||^2 / ||g||^2
    2. rho_P    preconditioner-weighted reference share: <g^R, P_RR g^R> / (<g^R, P_RR g^R> + <g^T, P_TT g^T>)
    3. influence decomposition for sampled targets:     |RR|, |cross|, |TT| as shares of |RR|+|cross|+|TT|
    4. PCA on candidate_features, energy of each PC in R-block vs T-block.

Run:
    python entanglement_analysis.py                       # analyze cache next to this script
    python entanglement_analysis.py --cache-dir <path>    # analyze any cache folder

Auto-discovers every *_selector_features.pt under <cache-dir>/selector_feature_cache/
and tags each one by its (K_R, K_T) and a short hash from the filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


_HASH_RE = re.compile(r"_[0-9a-f]{8,}")


def pretty_label(s: str, max_len: int = 36) -> str:
    """Strip the first 8+ char hex hash and everything after it. Fallback to truncation."""
    if not s:
        return "?"
    m = _HASH_RE.search(s)
    if m:
        s = s[: m.start()]
    s = s.rstrip("_")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def dataset_header(md: dict) -> str:
    return (f"candidate: {pretty_label(md.get('candidate', ''))}   |   "
            f"target: {pretty_label(md.get('target', ''))}   |   "
            f"reference: {pretty_label(md.get('reference', ''))}")


def discover_feature_files(cache_root: Path) -> dict[str, Path]:
    feat_dir = cache_root / "selector_feature_cache"
    if not feat_dir.is_dir():
        raise FileNotFoundError(f"no selector_feature_cache dir under {cache_root}")
    files = sorted(feat_dir.glob("*_selector_features.pt"))
    out: dict[str, Path] = {}
    for f in files:
        m_kr = re.search(r"KR(\d+)", f.name)
        m_kt = re.search(r"KT(\d+)", f.name)
        m_proj = re.search(r"_proj(\d+)_", f.name)
        if m_kr and m_kt:
            tag = f"lowrank_KR{m_kr.group(1)}_KT{m_kt.group(1)}"
        elif m_proj:
            tag = f"randomsketch_proj{m_proj.group(1)}"
        else:
            tag = f.stem[:40]
        # de-dup if same tag appears twice
        base = tag; n = 1
        while tag in out:
            n += 1
            tag = f"{base}_{n}"
        out[tag] = f
    return out


def resolved_block_ranks(md: dict) -> tuple[int | None, int | None]:
    k_r = md.get("K_R", md.get("low_rank_resolved_reference_rank"))
    k_t = md.get("K_T", md.get("low_rank_resolved_task_rank"))
    k_r = None if k_r is None else int(k_r)
    k_t = None if k_t is None else int(k_t)
    return k_r, k_t


def load_projected_preconditioner(cache_file: Path, payload: dict, md: dict) -> torch.Tensor | None:
    projected = payload.get("selector_preconditioner")
    if projected is not None:
        return projected

    preconditioner_cache_path = md.get("preconditioner_cache_path")
    if not preconditioner_cache_path:
        return None
    preconditioner_cache = Path(str(preconditioner_cache_path))
    if not preconditioner_cache.exists():
        return None

    pre_payload = torch.load(preconditioner_cache, map_location="cpu", weights_only=False)
    projected_dict = pre_payload.get("projected_preconditioners")
    if isinstance(projected_dict, dict):
        projected = projected_dict.get(cache_file.stem)
        if projected is not None:
            return projected
    projected = pre_payload.get("projected_preconditioner")
    return projected


def block_split(x: torch.Tensor, K_R: int):
    return x[..., :K_R], x[..., K_R:]


def energy_share(g: torch.Tensor, K_R: int):
    gR, gT = block_split(g, K_R)
    nR2 = (gR ** 2).sum(-1)
    nT2 = (gT ** 2).sum(-1)
    rho = nR2 / (nR2 + nT2).clamp_min(1e-30)
    return rho, nR2.sqrt(), nT2.sqrt()


def precond_share(g: torch.Tensor, P: torch.Tensor, K_R: int):
    """Share of g^T P g coming from the R block (using P_RR vs P_TT only).
    This is the natural 'how much does selector weigh each block for this point' metric."""
    gR, gT = block_split(g, K_R)
    P_RR = P[:K_R, :K_R]
    P_TT = P[K_R:, K_R:]
    e_R = (gR @ P_RR * gR).sum(-1)
    e_T = (gT @ P_TT * gT).sum(-1)
    return e_R / (e_R + e_T).clamp_min(1e-30)


def influence_decomp(targets: torch.Tensor, cands: torch.Tensor, P: torch.Tensor, K_R: int,
                     n_target_sample: int = 128, seed: int = 0):
    """For a sample of targets, return per-candidate magnitudes of RR / cross / TT / total
    averaged across the sampled targets."""
    rng = np.random.default_rng(seed)
    n_t = targets.shape[0]
    idx = rng.choice(n_t, size=min(n_target_sample, n_t), replace=False)
    t = targets[idx]

    P_RR = P[:K_R, :K_R]
    P_RT = P[:K_R, K_R:]
    P_TR = P[K_R:, :K_R]
    P_TT = P[K_R:, K_R:]

    tR, tT = block_split(t, K_R)
    cR, cT = block_split(cands, K_R)

    s_RR = tR @ P_RR @ cR.T
    s_RT = tR @ P_RT @ cT.T
    s_TR = tT @ P_TR @ cR.T
    s_TT = tT @ P_TT @ cT.T
    s_total = s_RR + s_RT + s_TR + s_TT

    # mean absolute magnitude per candidate across sampled targets
    return {
        "RR": s_RR.abs().mean(0),
        "cross": (s_RT + s_TR).abs().mean(0),
        "TT": s_TT.abs().mean(0),
        "total": s_total.abs().mean(0),
        "signed_total": s_total.mean(0),
    }, idx


def _quadrant_fractions(x: np.ndarray, y: np.ndarray) -> dict:
    sx = x > 0; sy = y > 0
    return {
        "T_pos_R_neg_ideal": float(((~sx) & sy).mean()),
        "T_pos_R_pos_useful_costly": float((sx & sy).mean()),
        "T_neg_R_pos_pure_drift": float((sx & (~sy)).mean()),
        "T_neg_R_neg_misaligned": float(((~sx) & (~sy)).mean()),
    }


def _jointplot_quadrants(out_path: Path, x: np.ndarray, y: np.ndarray,
                         color_vals: np.ndarray, *,
                         xlabel: str, ylabel: str, suptitle: str,
                         color_label: str, footer: str,
                         cmap: str = "viridis", color_vmin: float = 0.0, color_vmax: float = 1.0,
                         lim: float | None = None,
                         highlight_idx: np.ndarray | None = None,
                         highlight_label: str | None = None,
                         highlight_color: str = "#d62728"):
    """Shared jointplot rendering for the (cosine) and (damped-Fisher) variants.

    If `highlight_idx` is given, those candidates are plotted as red dots on top of
    the hexbin and added as a separate red histogram on each marginal — used to
    overlay e.g. the selector's actually-chosen top-K.
    """
    if lim is None:
        lim = max(abs(x).max(), abs(y).max()) * 1.05
        lim = max(lim, 0.05)

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        hspace=0.04, wspace=0.04,
        left=0.10, right=0.86, bottom=0.13, top=0.90,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    hb = ax_main.hexbin(
        x, y, C=color_vals, reduce_C_function=np.mean,
        gridsize=60, mincnt=2,
        cmap=cmap, vmin=color_vmin, vmax=color_vmax,
        extent=(-lim, lim, -lim, lim),
    )
    ax_main.axhline(0, color="k", lw=0.8)
    ax_main.axvline(0, color="k", lw=0.8)
    ax_main.set_xlim(-lim, lim)
    ax_main.set_ylim(-lim, lim)
    ax_main.grid(True, alpha=0.25)

    qbox = dict(facecolor="white", alpha=0.85, ec="0.7", boxstyle="round,pad=0.3")
    ax_main.text(0.03, 0.97, "task-relevant, low drift\n(T+, R−)   $\\bigstar$ ideal",
                 transform=ax_main.transAxes, ha="left", va="top", fontsize=9, bbox=qbox)
    ax_main.text(0.97, 0.97, "useful + drift cost\n(T+, R+)",
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=9, bbox=qbox)
    ax_main.text(0.97, 0.03, "pure drift\n(T−, R+)",
                 transform=ax_main.transAxes, ha="right", va="bottom", fontsize=9, bbox=qbox)
    ax_main.text(0.03, 0.03, "actively misaligned\n(T−, R−)",
                 transform=ax_main.transAxes, ha="left", va="bottom", fontsize=9, bbox=qbox)

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    bins_marg = np.linspace(-lim, lim, 81)
    ax_top.hist(x, bins=bins_marg, color="0.45", edgecolor="none")
    ax_top.axvline(0, color="k", lw=0.5)
    ax_top.set_yticks([])
    ax_top.tick_params(axis="x", labelbottom=False)
    for s in ("top", "right", "left"):
        ax_top.spines[s].set_visible(False)

    ax_right.hist(y, bins=bins_marg, color="0.45", edgecolor="none",
                  orientation="horizontal")
    ax_right.axhline(0, color="k", lw=0.5)
    ax_right.set_xticks([])
    ax_right.tick_params(axis="y", labelleft=False)
    for s in ("top", "right", "bottom"):
        ax_right.spines[s].set_visible(False)

    # selector top-K overlay (drawn AFTER the hexbin so it's visible on top)
    if highlight_idx is not None and len(highlight_idx) > 0:
        ax_main.scatter(
            x[highlight_idx], y[highlight_idx],
            s=18, c=highlight_color, alpha=0.9, edgecolors="white", linewidths=0.4,
            zorder=5,
        )
        # small in-axes legend marker so the red is captioned without colliding
        # with the bottom footer text
        ax_main.scatter([], [], s=24, c=highlight_color, edgecolors="white",
                        linewidths=0.4, label=highlight_label or "selector top-K")
        ax_main.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                       fontsize=8, framealpha=0.9, frameon=True)
        # add red marginal hist (slim, on top of the gray)
        ax_top.hist(x[highlight_idx], bins=bins_marg, color=highlight_color,
                    alpha=0.85, edgecolor="none")
        ax_right.hist(y[highlight_idx], bins=bins_marg, color=highlight_color,
                      alpha=0.85, edgecolor="none", orientation="horizontal")

    cbar_ax = fig.add_axes([0.89, 0.13, 0.022, 0.55])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label(color_label)

    fig.suptitle(suptitle, fontsize=10, y=0.965)
    fig.text(0.48, 0.04, footer, ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def reference_load(cands: torch.Tensor, ref_fisher: torch.Tensor, K_R: int) -> torch.Tensor:
    """ℓ_i := ‖F_R^{1/2} b_i‖² = <b_i^R, Λ_R b_i^R> = sum_k Λ_k (b_i^R)_k²

    The Fisher quadratic form on the candidate's R-block — how much "pressure"
    candidate i places on the protected reference geometry. Always ≥ 0.
    """
    cR = cands[..., :K_R]
    Lambda = ref_fisher[:K_R]
    return (cR ** 2 * Lambda).sum(-1)


def damped_fisher_alignment(cands: torch.Tensor, targets: torch.Tensor,
                            ref_fisher: torch.Tensor, K_R: int, alpha: float):
    """Compute the per-channel target influence under a damped reference-Fisher metric.

        d_i^R = <b_i^R, (Lambda_R + alpha I)^{-1} b_tar^R>
        d_i^T = alpha^{-1} <b_i^T, b_tar^T>

    Their sum equals the target influence delivered by candidate i under the natural-gradient
    update that solves   min_w  L_tar(w) + (1/2)(w-w0)^T F_R (w-w0)   with Tikhonov damping alpha.

    Returns (d_R, d_T) each of shape [N] and a dict of scale diagnostics.
    """
    g_bar = targets.mean(0)
    gR_bar, gT_bar = block_split(g_bar, K_R)
    cR, cT = block_split(cands, K_R)

    Lambda = ref_fisher[:K_R]
    inv_R = 1.0 / (Lambda + alpha)               # K_R-dim diagonal
    d_R = cR @ (inv_R * gR_bar)                  # [N]
    d_T = (cT @ gT_bar) / alpha                  # [N]

    diag = {
        "alpha": float(alpha),
        "d_R_std": float(d_R.std()),
        "d_T_std": float(d_T.std()),
        "d_R_median_abs": float(d_R.abs().median()),
        "d_T_median_abs": float(d_T.abs().median()),
        "T_over_R_std_ratio": float(d_T.std() / d_R.std().clamp_min(1e-30)),
    }
    return d_R, d_T, diag


def target_block_alignment(cands: torch.Tensor, targets: torch.Tensor, P: torch.Tensor, K_R: int):
    """For each candidate i, compute Adam-preconditioned cosine similarity between
    g_i and the mean target gradient g_bar, separately in the R block and the T block.

    Returns (cos_R, cos_T) each of shape [N].
    Cauchy-Schwarz under <.,P_RR.> guarantees both lie in [-1, 1].
    """
    g_bar = targets.mean(0)
    gR_bar, gT_bar = block_split(g_bar, K_R)
    P_RR = P[:K_R, :K_R]
    P_TT = P[K_R:, K_R:]
    cR, cT = block_split(cands, K_R)

    num_R = cR @ (P_RR @ gR_bar)
    num_T = cT @ (P_TT @ gT_bar)

    norm_cR = torch.sqrt(((cR @ P_RR) * cR).sum(-1).clamp_min(1e-30))
    norm_cT = torch.sqrt(((cT @ P_TT) * cT).sum(-1).clamp_min(1e-30))
    norm_tR = torch.sqrt((gR_bar @ P_RR @ gR_bar).clamp_min(torch.tensor(1e-30)))
    norm_tT = torch.sqrt((gT_bar @ P_TT @ gT_bar).clamp_min(torch.tensor(1e-30)))

    return num_R / (norm_cR * norm_tR), num_T / (norm_cT * norm_tT)


def pca_block_energy(cands: torch.Tensor, K_R: int, n_components: int = 32):
    """Run SVD on candidate_features. Return per-PC fraction of energy in R-block vs T-block."""
    X = cands - cands.mean(0, keepdim=True)
    # economy SVD on N x K, K is small (~320) so full svd is fine
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.T  # [K, K]
    n_components = min(n_components, V.shape[1])
    V = V[:, :n_components]
    e_R = (V[:K_R] ** 2).sum(0)
    e_T = (V[K_R:] ** 2).sum(0)
    return S[:n_components].numpy(), e_R.numpy(), e_T.numpy()


def summarize(arr: torch.Tensor) -> dict:
    a = arr.numpy()
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def analyze_one(tag: str, path: Path, out_root: Path, selector_top_k: int = 200):
    print(f"\n===== {tag} =====")
    print(f"  loading {path.name}")
    d = torch.load(path, map_location="cpu", weights_only=False)
    md = d["metadata"]
    K_R, K_T = resolved_block_ranks(md)
    if K_R is None or K_T is None:
        print(f"  skip: not a low-rank feature file (no K_R/K_T in metadata)")
        return None
    if K_R <= 0 or K_T <= 0:
        print(f"  skip: degenerate low-rank block split (K_R={K_R}, K_T={K_T})")
        return None
    print(f"  K_R={K_R}, K_T={K_T}, total K={K_R + K_T}")

    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cands = d["candidate_features"]
    targets = d.get("target_features")
    if targets is None:
        target_feature = d.get("target_feature")
        if target_feature is None:
            print("  skip: cache has neither target_features nor target_feature")
            return None
        targets = target_feature.unsqueeze(0)
    ref_fisher = d["reference_fisher"]
    P = load_projected_preconditioner(path, d, md)
    if P is None:
        print("  skip: no projected selector_preconditioner found in payload or sidecar cache")
        return None

    header = dataset_header(md)
    cand_short = pretty_label(md.get("candidate", ""))
    target_short = pretty_label(md.get("target", ""))
    ref_short = pretty_label(md.get("reference", ""))

    summary = {
        "K_R": K_R, "K_T": K_T,
        "n_candidates": cands.shape[0], "n_targets": targets.shape[0],
        "candidate_dataset": cand_short,
        "target_dataset": target_short,
        "reference_dataset": ref_short,
    }

    # ---- (1+2) energy share rho_E and preconditioner-weighted share rho_P ----
    rhoE_c, nR_c, nT_c = energy_share(cands, K_R)
    rhoE_t, nR_t, nT_t = energy_share(targets, K_R)
    summary["rho_E_candidates"] = summarize(rhoE_c)
    summary["rho_E_targets"] = summarize(rhoE_t)

    rhoP_c = precond_share(cands, P, K_R)
    rhoP_t = precond_share(targets, P, K_R)
    summary["rho_P_candidates"] = summarize(rhoP_c)
    summary["rho_P_targets"] = summarize(rhoP_t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    bins = np.linspace(0, 1, 50)
    cand_lbl = f"candidates (n={cands.shape[0]:,})"
    tgt_lbl = f"targets (n={targets.shape[0]:,})"

    ax = axes[0]
    ax.hist(rhoE_c.numpy(), bins=bins, alpha=0.55, label=cand_lbl, density=True, color="C0")
    ax.hist(rhoE_t.numpy(), bins=bins, alpha=0.55, label=tgt_lbl, density=True, color="C1")
    ax.axvline(K_R / (K_R + K_T), ls="--", color="k", lw=0.8,
               label=f"uniform-energy baseline ({K_R}/{K_R+K_T})")
    ax.set_xlabel(r"$\rho_E = \|g^R\|^2 / \|g\|^2$")
    ax.set_ylabel("density")
    ax.set_title(r"(a) raw energy share in reference subspace" "\n"
                 r"$1 \to$ all gradient mass in reference;  $0 \to$ all in residual")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.hist(rhoP_c.numpy(), bins=bins, alpha=0.55, label=cand_lbl, density=True, color="C0")
    ax.hist(rhoP_t.numpy(), bins=bins, alpha=0.55, label=tgt_lbl, density=True, color="C1")
    ax.set_xlabel(r"$\rho_P = \langle g^R, P_{RR} g^R\rangle\,/\,(\langle g^R, P_{RR} g^R\rangle + \langle g^T, P_{TT} g^T\rangle)$",
                  fontsize=9)
    ax.set_title(r"(b) preconditioner-weighted reference share" "\n"
                 r"reweighted by selector's actual metric $P$")
    ax.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.suptitle(f"Reference-subspace entanglement of candidate vs target gradients  [{tag}]\n{header}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "01_rho_E_and_rho_P.png", dpi=140)
    plt.close(fig)

    # ---- scatter ||g^R|| vs ||g^T|| -----------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(nR_c.numpy(), nT_c.numpy(), s=5, alpha=0.35, label=cand_lbl, color="C0",
               edgecolors="none")
    ax.scatter(nR_t.numpy(), nT_t.numpy(), s=10, alpha=0.55, label=tgt_lbl, color="C1",
               edgecolors="none")
    lim = max(nR_c.max().item(), nT_c.max().item(), nR_t.max().item(), nT_t.max().item()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.6, label=r"$\|g^R\|=\|g^T\|$")
    ax.set_xlabel(r"$\|g^R\|$   (reference block)")
    ax.set_ylabel(r"$\|g^T\|$   (residual / task block)")
    ax.set_title(f"Per-sample gradient norms in each block  [{tag}]\n{header}", fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "03_norm_scatter.png", dpi=140)
    plt.close(fig)

    # ---- (3) influence decomposition ----------------------------------------
    decomp, sampled_t = influence_decomp(targets, cands, P, K_R, n_target_sample=128, seed=0)
    rr = decomp["RR"]
    cross = decomp["cross"]
    tt = decomp["TT"]
    denom = (rr + cross + tt).clamp_min(1e-30)  # so the three shares sum to 1
    frac_RR = (rr / denom).numpy()
    frac_cross = (cross / denom).numpy()
    frac_TT = (tt / denom).numpy()

    summary["influence_decomp"] = {
        "n_targets_sampled": int(len(sampled_t)),
        "RR_share_mean": float(frac_RR.mean()),
        "cross_share_mean": float(frac_cross.mean()),
        "TT_share_mean": float(frac_TT.mean()),
        "RR_abs_mean": float(rr.mean()),
        "cross_abs_mean": float(cross.mean()),
        "TT_abs_mean": float(tt.mean()),
    }

    # rank candidates by signed total influence and stack-bar the top K
    signed = decomp["signed_total"]
    order = torch.argsort(signed, descending=True)
    K = 60
    top_idx = order[:K].numpy()
    bot_idx = order[-K:].numpy()
    pick = np.concatenate([top_idx, bot_idx])
    labels = ["top-K"] * K + ["bot-K"] * K

    # Selector top-K (used to overlay on figures 8 and 9)
    sel_K = max(1, min(int(selector_top_k), int(cands.shape[0])))
    selector_top_idx = order[:sel_K].numpy()
    summary["selector_top_k"] = sel_K

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, idx, title in [(axes[0], top_idx, f"top-{K} highest-influence candidates"),
                           (axes[1], bot_idx, f"bottom-{K} most-negative-influence candidates")]:
        x = np.arange(len(idx))
        rr_b = rr[idx].numpy()
        cr_b = cross[idx].numpy()
        tt_b = tt[idx].numpy()
        denom = (rr_b + cr_b + tt_b).clip(1e-30)
        ax.bar(x, rr_b / denom, label="|RR|  (reference $\\to$ reference)", color="C0", width=1.0)
        ax.bar(x, cr_b / denom, bottom=rr_b / denom,
               label="|cross|  (RT + TR)", color="C1", width=1.0)
        ax.bar(x, tt_b / denom, bottom=(rr_b + cr_b) / denom,
               label="|TT|  (residual $\\to$ residual)", color="C2", width=1.0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("candidate (ranked by signed influence)")
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, K - 0.5)
    axes[0].set_ylabel("share of |influence| per candidate")
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
    fig.suptitle(
        f"Block decomposition of selector influence  [{tag}]"
        f"   |   per-candidate stack averaged over {len(sampled_t)} sampled targets\n{header}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "04_influence_decomp.png", dpi=140)
    plt.close(fig)

    # histogram of per-candidate share distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(frac_RR, bins=40, alpha=0.55, label="|RR|  reference $\\to$ reference", density=True, color="C0")
    ax.hist(frac_cross, bins=40, alpha=0.55, label="|cross|  RT + TR", density=True, color="C1")
    ax.hist(frac_TT, bins=40, alpha=0.55, label="|TT|  residual $\\to$ residual", density=True, color="C2")
    ax.set_xlabel("share of |influence| per candidate")
    ax.set_ylabel("density (over candidates)")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")
    ax.set_title(
        f"Per-candidate distribution of block contributions to selector influence  [{tag}]\n{header}",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "05_influence_share_hist.png", dpi=140)
    plt.close(fig)

    # ---- preconditioner block heatmap + reference Fisher spectrum -----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    block_norms = np.array([
        [P[:K_R, :K_R].norm().item(), P[:K_R, K_R:].norm().item()],
        [P[K_R:, :K_R].norm().item(), P[K_R:, K_R:].norm().item()],
    ])
    im = axes[0].imshow(block_norms, cmap="viridis")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["R (reference)", "T (residual)"])
    axes[0].set_yticklabels(["R (reference)", "T (residual)"])
    vmax = block_norms.max()
    for i in range(2):
        for j in range(2):
            color = "white" if block_norms[i, j] < 0.6 * vmax else "black"
            axes[0].text(j, i, f"{block_norms[i,j]:.1f}", ha="center", va="center",
                         color=color, fontsize=11)
    axes[0].set_title(r"$\|P_{\mathrm{block}}\|_F$  of selector_preconditioner")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    ref_eig = ref_fisher[:K_R].numpy()
    axes[1].semilogy(np.sort(ref_eig)[::-1], lw=1.4)
    axes[1].set_xlabel("rank in reference basis")
    axes[1].set_ylabel("Fisher eigenvalue (log)")
    axes[1].set_title(f"reference Fisher spectrum  ($K_R$={K_R})")
    axes[1].grid(True, alpha=0.3, which="both")
    fig.suptitle(f"Basis structure: cross-block coupling and reference spectrum  [{tag}]\n{header}",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "06_preconditioner_and_spectrum.png", dpi=140)
    plt.close(fig)

    summary["preconditioner_block_norms_F"] = {
        "RR": float(block_norms[0, 0]),
        "RT": float(block_norms[0, 1]),
        "TR": float(block_norms[1, 0]),
        "TT": float(block_norms[1, 1]),
    }
    summary["reference_fisher_task_block_max"] = float(ref_fisher[K_R:].abs().max())
    summary["reference_fisher_reference_block"] = {
        "max": float(ref_fisher[:K_R].max()),
        "min": float(ref_fisher[:K_R].min()),
        "mean": float(ref_fisher[:K_R].mean()),
        "n_nonzero": int((ref_fisher[:K_R] != 0).sum()),
    }

    # ---- (4) PCA block energy -----------------------------------------------
    sing, e_R_pc, e_T_pc = pca_block_energy(cands, K_R, n_components=min(64, K_R + K_T))
    pcs = np.arange(len(sing))

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.1], "hspace": 0.12})
    width = 0.85
    ax_top = axes[0]
    ax_top.bar(pcs, e_R_pc, width=width, color="C0",
               label="share in R block (reference subspace)")
    ax_top.bar(pcs, e_T_pc, width=width, bottom=e_R_pc, color="C2",
               label="share in T block (residual subspace)")
    ax_top.set_ylabel("unit-PC energy share")
    ax_top.set_ylim(0, 1.0)
    ax_top.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=2,
                  fontsize=8, framealpha=0.95, borderaxespad=0)
    ax_top.set_title(
        f"PCA of candidate_features: where each PC lives  [{tag}]\n{header}",
        fontsize=9,
    )
    ax_top.grid(True, alpha=0.25, axis="y")

    ax_bot = axes[1]
    ax_bot.semilogy(pcs, sing, color="k", marker="o", ms=3, lw=0.9)
    ax_bot.set_ylabel("singular value (log)")
    ax_bot.set_xlabel("principal component (of candidate_features)")
    ax_bot.grid(True, alpha=0.3, which="both")
    ax_bot.set_xlim(-0.5, len(pcs) - 0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "07_pca_block_energy.png", dpi=140)
    plt.close(fig)

    summary["pca_top_pcs_R_share_mean_top10"] = float(e_R_pc[:10].mean())
    summary["pca_top_pcs_T_share_mean_top10"] = float(e_T_pc[:10].mean())

    # ---- (5) per-candidate cosine alignment with target's R vs target's T block ----
    cosR, cosT = target_block_alignment(cands, targets, P, K_R)
    cosR_np = cosR.numpy()
    cosT_np = cosT.numpy()
    quad_cos = _quadrant_fractions(cosR_np, cosT_np)
    summary["target_alignment_quadrants"] = quad_cos
    quad_T_pos_R_neg = quad_cos["T_pos_R_neg_ideal"]
    quad_T_pos_R_pos = quad_cos["T_pos_R_pos_useful_costly"]
    quad_T_neg_R_pos = quad_cos["T_neg_R_pos_pure_drift"]
    quad_T_neg_R_neg = quad_cos["T_neg_R_neg_misaligned"]

    # quadrant fractions of the selector top-K (overlay)
    quad_cos_topk = _quadrant_fractions(cosR_np[selector_top_idx], cosT_np[selector_top_idx])
    summary["target_alignment_quadrants_selector_topK"] = quad_cos_topk

    footer8 = (
        f"all candidates:  ideal (T+,R−)={quad_T_pos_R_neg:.1%}   |   "
        f"useful+costly (T+,R+)={quad_T_pos_R_pos:.1%}   |   "
        f"pure drift (T−,R+)={quad_T_neg_R_pos:.1%}   |   "
        f"misaligned (T−,R−)={quad_T_neg_R_neg:.1%}\n"
        f"selector top-{sel_K} (red):  ideal={quad_cos_topk['T_pos_R_neg_ideal']:.1%}   |   "
        f"useful+costly={quad_cos_topk['T_pos_R_pos_useful_costly']:.1%}   |   "
        f"pure drift={quad_cos_topk['T_neg_R_pos_pure_drift']:.1%}   |   "
        f"misaligned={quad_cos_topk['T_neg_R_neg_misaligned']:.1%}"
    )
    suptitle8 = (
        "Source of target improvement: reference-sensitive vs reference-orthogonal directions  "
        f"[{tag}]\n{header}"
    )
    _jointplot_quadrants(
        out_dir / "08_target_block_alignment.png",
        cosR_np, cosT_np, color_vals=rhoE_c.numpy(),
        xlabel=(
            r"$\cos_{P_{RR}}(\widehat g_i^R,\ \widehat g_{\mathrm{tar}}^R)$"
            "       (alignment with target in reference subspace)"
        ),
        ylabel=(
            r"$\cos_{P_{TT}}(\widehat g_i^T,\ \widehat g_{\mathrm{tar}}^T)$"
            "       (alignment with target in residual subspace)"
        ),
        suptitle=suptitle8,
        color_label=r"mean $\rho_E$ within cell  (1=cell mostly entangled)",
        footer=footer8,
        highlight_idx=selector_top_idx,
        highlight_label=f"selector top-{sel_K}",
    )

    # ---- (6) damped-Fisher influence accounting (d_R, d_T) -----------------
    alpha_meta = md.get("low_rank_builder_alpha", None)
    alpha = float(alpha_meta) if alpha_meta is not None else 1e-3
    d_R, d_T, d_diag = damped_fisher_alignment(cands, targets, ref_fisher, K_R, alpha)
    d_R_np = d_R.numpy(); d_T_np = d_T.numpy()

    # Per-axis stddev normalization, so the four-quadrant structure is readable.
    # Quadrant boundaries at 0 are scale-invariant, so quadrant fractions are unaffected.
    sR = d_R_np / (d_R_np.std() + 1e-30)
    sT = d_T_np / (d_T_np.std() + 1e-30)
    quad_d = _quadrant_fractions(sR, sT)
    summary["damped_fisher_alignment"] = {
        "alpha": alpha,
        "alpha_source": "low_rank_builder_alpha" if alpha_meta is not None else "default_1e-3",
        **d_diag,
        "quadrants": quad_d,
        "d_T_share_per_candidate_mean": float(
            (np.abs(d_T_np) / (np.abs(d_R_np) + np.abs(d_T_np) + 1e-30)).mean()
        ),
    }

    quad_d_topk = _quadrant_fractions(sR[selector_top_idx], sT[selector_top_idx])
    summary["damped_fisher_alignment"]["quadrants_selector_topK"] = quad_d_topk

    footer = (
        f"$\\alpha$={alpha:g}   |   "
        f"raw std$(d_i^R)$={d_diag['d_R_std']:.3g}, std$(d_i^T)$={d_diag['d_T_std']:.3g}, "
        f"ratio std$(d^T)$/std$(d^R)$={d_diag['T_over_R_std_ratio']:.1f}× "
        f"(axes z-scored per-axis for readability).\n"
        f"all candidates:  ideal={quad_d['T_pos_R_neg_ideal']:.1%}   |   "
        f"useful+costly={quad_d['T_pos_R_pos_useful_costly']:.1%}   |   "
        f"pure drift={quad_d['T_neg_R_pos_pure_drift']:.1%}   |   "
        f"misaligned={quad_d['T_neg_R_neg_misaligned']:.1%}\n"
        f"selector top-{sel_K} (red):  ideal={quad_d_topk['T_pos_R_neg_ideal']:.1%}   |   "
        f"useful+costly={quad_d_topk['T_pos_R_pos_useful_costly']:.1%}   |   "
        f"pure drift={quad_d_topk['T_neg_R_pos_pure_drift']:.1%}   |   "
        f"misaligned={quad_d_topk['T_neg_R_neg_misaligned']:.1%}"
    )
    suptitle9 = (
        "Damped-Fisher influence accounting:  "
        r"$d_i^R = (b_i^R)^\top(\Lambda_R+\alpha I)^{-1} b_{\mathrm{tar}}^R$,   "
        r"$d_i^T = \alpha^{-1}(b_i^T)^\top b_{\mathrm{tar}}^T$"
        f"   [{tag}]\n{header}"
    )
    # Clip view to the 99.5-percentile of |.| so a few extreme candidates don't
    # collapse the bulk of the cloud into a tiny center region.
    lim_d = float(np.percentile(np.concatenate([np.abs(sR), np.abs(sT)]), 99.5)) * 1.05
    lim_d = max(lim_d, 0.05)
    _jointplot_quadrants(
        out_dir / "09_damped_fisher_alignment.png",
        sR, sT, color_vals=rhoE_c.numpy(),
        xlabel=r"$d_i^R / \mathrm{std}(d^R)$       (target influence via reference subspace, budgeted)",
        ylabel=r"$d_i^T / \mathrm{std}(d^T)$       (target influence via residual subspace)",
        suptitle=suptitle9,
        color_label=r"$\rho_E$  (candidate's reference-block energy share)",
        footer=footer,
        lim=lim_d,
        highlight_idx=selector_top_idx,
        highlight_label=f"selector top-{sel_K}",
    )

    # ---- (7) reference load ℓ_i and (d_T, ℓ) jointplot --------------------
    # The "ideal" candidate has large d_T (target-relevant via residual block)
    # and small ℓ (low pressure on reference Fisher).
    load = reference_load(cands, ref_fisher, K_R).numpy()
    nz = load[load > 0]
    eps_load = float(nz.min()) * 0.1 if nz.size else 1e-30
    log_load = np.log10(np.maximum(load, eps_load))
    load_med = float(np.median(load))

    # Operational "ideal" set: positive d_T AND ℓ below pool-median
    ideal_mask_all = (d_T_np > 0) & (load < load_med)
    ideal_mask_topk = ideal_mask_all[selector_top_idx]
    summary["reference_load"] = {
        "median": load_med,
        "p10": float(np.percentile(load, 10)),
        "p90": float(np.percentile(load, 90)),
        "max": float(load.max()),
        "min_nonzero": float(nz.min()) if nz.size else 0.0,
        "ideal_share_all_pool": float(ideal_mask_all.mean()),
        "ideal_share_selector_topK": float(ideal_mask_topk.mean()),
        "median_ell_selector_topK": float(np.median(load[selector_top_idx])),
    }

    xlim10 = float(np.percentile(np.abs(sT), 99.5)) * 1.05
    xlim10 = max(xlim10, 0.5)
    y_lo = float(np.percentile(log_load, 0.5))
    y_hi = float(np.percentile(log_load, 99.5))

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        hspace=0.04, wspace=0.04,
        left=0.10, right=0.86, bottom=0.13, top=0.90,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    hb = ax_main.hexbin(
        sT, log_load, C=rhoE_c.numpy(), reduce_C_function=np.mean,
        gridsize=60, mincnt=2, cmap="viridis", vmin=0, vmax=1,
        extent=(-xlim10, xlim10, y_lo, y_hi),
    )
    ax_main.axvline(0, color="k", lw=0.7)
    ax_main.axhline(np.log10(load_med), color="k", lw=0.5, ls="--", alpha=0.6)
    ax_main.set_xlim(-xlim10, xlim10)
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.grid(True, alpha=0.25)

    # shade ideal region: lower-right (d^T > 0 AND ℓ < median)
    ax_main.axhspan(y_lo, np.log10(load_med), xmin=0.5, xmax=1.0,
                    facecolor="gold", alpha=0.12, zorder=0)

    qbox = dict(facecolor="white", alpha=0.85, ec="0.7", boxstyle="round,pad=0.3")
    ax_main.text(
        0.97, 0.03,
        "ideal\n$\\bigstar$  large $d^T$, small $\\ell$\n(target-relevant,\nlow reference pressure)",
        transform=ax_main.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(facecolor="gold", alpha=0.85, ec="0.7", boxstyle="round,pad=0.3"),
    )
    ax_main.text(0.97, 0.97, "useful but costly\n(large $d^T$, large $\\ell$)",
                 transform=ax_main.transAxes, ha="right", va="top", fontsize=9, bbox=qbox)
    ax_main.text(0.03, 0.97, "high reference pressure,\nweak target alignment",
                 transform=ax_main.transAxes, ha="left", va="top", fontsize=9, bbox=qbox)
    ax_main.text(0.03, 0.03, "low pressure but\nweak/negative target alignment",
                 transform=ax_main.transAxes, ha="left", va="bottom", fontsize=9, bbox=qbox)

    ax_main.set_xlabel(r"$d_i^T / \mathrm{std}(d^T)$       (target alignment via residual block)")
    ax_main.set_ylabel(r"$\log_{10} \ell_i$       (reference Fisher load $\langle b_i^R, \Lambda_R b_i^R\rangle$)")

    bins_x = np.linspace(-xlim10, xlim10, 81)
    bins_y = np.linspace(y_lo, y_hi, 81)
    ax_top.hist(sT, bins=bins_x, color="0.45", edgecolor="none")
    ax_top.set_yticks([])
    ax_top.tick_params(axis="x", labelbottom=False)
    for s in ("top", "right", "left"):
        ax_top.spines[s].set_visible(False)

    ax_right.hist(log_load, bins=bins_y, color="0.45", edgecolor="none",
                  orientation="horizontal")
    ax_right.set_xticks([])
    ax_right.tick_params(axis="y", labelleft=False)
    for s in ("top", "right", "bottom"):
        ax_right.spines[s].set_visible(False)

    if len(selector_top_idx):
        ax_main.scatter(
            sT[selector_top_idx], log_load[selector_top_idx],
            s=18, c="#d62728", alpha=0.9, edgecolors="white", linewidths=0.4, zorder=5,
        )
        ax_main.scatter([], [], s=24, c="#d62728", edgecolors="white",
                        linewidths=0.4, label=f"selector top-{sel_K}")
        ax_main.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                       fontsize=8, framealpha=0.9)
        ax_top.hist(sT[selector_top_idx], bins=bins_x, color="#d62728",
                    alpha=0.85, edgecolor="none")
        ax_right.hist(log_load[selector_top_idx], bins=bins_y, color="#d62728",
                      alpha=0.85, edgecolor="none", orientation="horizontal")

    cbar_ax = fig.add_axes([0.89, 0.13, 0.022, 0.55])
    cbar = fig.colorbar(hb, cax=cbar_ax)
    cbar.set_label(r"mean $\rho_E$ within cell")

    fig.suptitle(
        r"Reference load vs target alignment:  large $d_i^T$ + small $\ell_i$ = target-relevant, low reference pressure"
        f"   [{tag}]\n{header}",
        fontsize=10, y=0.965,
    )
    fig.text(
        0.48, 0.04,
        f"all candidates: ideal share (positive $d^T$ AND $\\ell$ < median)  =  {ideal_mask_all.mean():.1%}     |     "
        f"selector top-{sel_K}: ideal share  =  {ideal_mask_topk.mean():.1%}\n"
        f"median $\\ell$ over pool = {load_med:.2e}     |     "
        f"median $\\ell$ over selector top-{sel_K} = {np.median(load[selector_top_idx]):.2e}",
        ha="center", va="bottom", fontsize=9,
    )

    fig.savefig(out_dir / "10_reference_load_vs_dT.png", dpi=140)
    plt.close(fig)

    return summary


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


def preconditioner_share(features: torch.Tensor, preconditioner: torch.Tensor, k_r: int) -> torch.Tensor:
    return precond_share(features, preconditioner, k_r)


def influence_decomposition(
    targets: torch.Tensor,
    candidates: torch.Tensor,
    preconditioner: torch.Tensor,
    k_r: int,
    *,
    target_sample_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    if int(targets.shape[0]) <= 0:
        empty = torch.zeros(candidates.shape[0], dtype=torch.float32)
        return {"RR": empty, "cross": empty, "TT": empty, "total": empty, "signed_total": empty}
    decomp, _ = influence_decomp(
        targets,
        candidates,
        preconditioner,
        k_r,
        n_target_sample=target_sample_size,
        seed=seed,
    )
    return decomp


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
    singular_values, ref_energy, task_energy = pca_block_energy(candidates, k_r, n_components=n_components)
    rows = []
    for index, (singular, ref_share, task_share) in enumerate(zip(singular_values, ref_energy, task_energy, strict=True)):
        rows.append(
            {
                "component": index + 1,
                "singular_value": float(singular),
                "reference_energy_share": float(ref_share),
                "task_energy_share": float(task_share),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=str(Path(__file__).resolve().parent),
                        help="cache root containing selector_feature_cache/. Defaults to script dir.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output dir (defaults to <cache-dir>/entanglement_analysis)")
    parser.add_argument("--selector-top-k", type=int, default=200,
                        help="number of top-influence candidates to highlight on figs 8/9")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else cache_root / "entanglement_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    feature_files = discover_feature_files(cache_root)
    if not feature_files:
        print(f"no *_selector_features.pt found under {cache_root}/selector_feature_cache/")
        return
    print(f"discovered {len(feature_files)} feature file(s) in {cache_root}:")
    for tag, p in feature_files.items():
        print(f"  [{tag}] {p.name}")

    all_summary = {}
    for tag, path in feature_files.items():
        if not path.exists():
            print(f"skip {tag}: file missing")
            continue
        result = analyze_one(tag, path, out_root, selector_top_k=args.selector_top_k)
        if result is not None:
            all_summary[tag] = result

    summary_path = out_root / "entanglement_summary.json"
    summary_path.write_text(json.dumps(all_summary, indent=2))
    print(f"\nwrote {summary_path}")
    for tag, s in all_summary.items():
        print(f"\n[{tag}] highlights:")
        print(f"  rho_E candidates: mean={s['rho_E_candidates']['mean']:.3f}  median={s['rho_E_candidates']['median']:.3f}")
        print(f"  rho_E targets   : mean={s['rho_E_targets']['mean']:.3f}  median={s['rho_E_targets']['median']:.3f}")
        print(f"  rho_P candidates: mean={s['rho_P_candidates']['mean']:.3f}  median={s['rho_P_candidates']['median']:.3f}")
        print(f"  rho_P targets   : mean={s['rho_P_targets']['mean']:.3f}  median={s['rho_P_targets']['median']:.3f}")
        d = s["influence_decomp"]
        print(f"  influence shares (mean over candidates):  RR={d['RR_share_mean']:.3f}  cross={d['cross_share_mean']:.3f}  TT={d['TT_share_mean']:.3f}")


if __name__ == "__main__":
    main()
