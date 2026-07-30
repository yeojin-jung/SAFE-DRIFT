#!/usr/bin/env python3
"""Held-out reference KL backfill: KL(p_base || p_ft) on the held-out StereoSet
reference sentences, per finished cell. This is a scale-comparable behavioral
drift metric (nats/token) — unlike raw reference Fisher cost or a fixed rho,
it does not depend on gradient/Fisher numerical scale.

For each selector with a saved final_adapter it loads the base model once, wraps
it with the adapter, and uses PEFT `disable_adapter()` to obtain base vs
fine-tuned next-token distributions on the SAME sentences, then averages the
per-token forward KL. Writes `reference_heldout_kl` into summary.json.

Two independent modes, usable together in one invocation:
  --output-root/--model-tag  the original model_tag/selector/training_runs grid
                              (full/random/dsir/less/prismatic/safe, Qwen2.5 scales
                              or olmo7b).
  --replay-run LABEL=DIR      one or more replay-method runs (flat output dirs
                              from experiments/run_train_replay_subset_code.sbatch,
                              e.g. .../outputs/code_replay_k80_p10). The held-out
                              StereoSet texts are read from that run's own
                              run_config.json (`bias_eval_data_path`), so the KL
                              uses exactly the same held-out set that run's own
                              StereoSet bias eval used -- no path needs to be
                              re-specified.
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME_BY_TAG = {
    "qwen2_5_0_5b": "Qwen/Qwen2.5-0.5B", "qwen2_5_1_5b": "Qwen/Qwen2.5-1.5B",
    "qwen2_5_3b": "Qwen/Qwen2.5-3B", "qwen2_5_7b": "Qwen/Qwen2.5-7B", "qwen2_5_14b": "Qwen/Qwen2.5-14B",
    "olmo7b": "allenai/OLMo-2-1124-7B",
}
SELECTORS = ["full", "random", "dsir", "less", "prismatic", "safe"]


def extract_reference_texts(recs: list[dict]) -> list[str]:
    texts = []
    for r in recs:
        for k in ("stereotype", "anti_stereotype", "unrelated"):
            if isinstance(r.get(k), str):
                texts.append(r[k])
    return texts


def load_reference_texts_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return extract_reference_texts(json.loads(path.read_text()))


def load_reference_texts(model_dir: Path, selector: str) -> list[str]:
    # Prefer the selector's own held-out triplets; fall back to any sibling's.
    cands = [model_dir / selector / "reference_bias_eval_triplets.json",
             *sorted(model_dir.glob("*/reference_bias_eval_triplets.json")),
             model_dir.parent / "shared_bias_eval" / "reference_bias_eval_triplets.json"]
    for p in cands:
        texts = load_reference_texts_from_file(p)
        if texts:
            return texts
    return []


def replay_reference_texts(output_dir: Path) -> list[str]:
    """Reuse the exact StereoSet held-out file the replay run's own bias eval used.

    run_config.json is `vars(args)` from train_lora_sft.py, so it always carries
    the --bias-eval-data-path that run was submitted with (see
    experiments/run_train_replay_subset_code.sbatch's BIAS_EVAL_DATA_PATH, which
    in turn is the --stereoset-eval-output-file build_replay_subset.py wrote --
    already guaranteed disjoint from that run's training data).
    """
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text())
    bias_eval_data_path = config.get("bias_eval_data_path")
    if not bias_eval_data_path:
        return []
    return load_reference_texts_from_file(Path(bias_eval_data_path))


def find_adapter(model_dir: Path, selector: str) -> Path | None:
    tr = model_dir / selector / "training_runs"
    if not tr.exists():
        return None
    for run in sorted(tr.iterdir()):
        a = run / "final_adapter"
        if a.exists():
            return a
    return None


def summary_path(model_dir: Path, selector: str) -> Path | None:
    fs = sorted(glob.glob(str(model_dir / selector / "training_runs" / "*" / "summary.json")))
    return Path(fs[0]) if fs else None


@torch.no_grad()
def kl_for_adapter(base_name: str, adapter: Path, texts: list[str], *, device, dtype,
                   batch_size: int, max_length: int) -> float:
    tok = AutoTokenizer.from_pretrained(base_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype).to(device).eval()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()

    total_kl, total_tok = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits_ft = model(**enc).logits[:, :-1, :].float()
        with model.disable_adapter():
            logits_base = model(**enc).logits[:, :-1, :].float()
        logp_base = F.log_softmax(logits_base, dim=-1)
        logp_ft = F.log_softmax(logits_ft, dim=-1)
        p_base = logp_base.exp()
        kl = (p_base * (logp_base - logp_ft)).sum(-1)                 # [B, T-1] forward KL per token
        mask = enc["attention_mask"][:, 1:].to(kl.dtype)
        total_kl += float((kl * mask).sum())
        total_tok += int(mask.sum())
    del base, model
    torch.cuda.empty_cache()
    return total_kl / max(total_tok, 1)


def backfill_cell(label: str, sp: Path, adapter: Path, texts: list[str], base_name: str, *,
                   device, dtype, batch_size: int, max_length: int, force: bool) -> None:
    summ = json.loads(sp.read_text())
    if "reference_heldout_kl" in summ and not force:
        print(f"[skip] {label}: kl present ({summ['reference_heldout_kl']:.4f})")
        return
    if not texts:
        print(f"[skip] {label}: no reference texts")
        return
    kl = kl_for_adapter(base_name, adapter, texts, device=device, dtype=dtype,
                        batch_size=batch_size, max_length=max_length)
    summ["reference_heldout_kl"] = kl
    summ["reference_heldout_kl_n_texts"] = len(texts)
    sp.write_text(json.dumps(summ, indent=2, ensure_ascii=False) + "\n")
    print(f"[done] {label}: reference_heldout_kl = {kl:.5f} nats/token (n={len(texts)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default=None, help="Root for the model_tag/selector/training_runs grid.")
    ap.add_argument("--model-tag", default=None, choices=sorted(MODEL_NAME_BY_TAG))
    ap.add_argument("--selectors", nargs="*", default=None)
    ap.add_argument(
        "--replay-run",
        action="append",
        default=[],
        metavar="LABEL=OUTPUT_DIR",
        help="Backfill a replay-method run (flat output dir from "
        "run_train_replay_subset_code.sbatch, e.g. mylabel=/path/to/outputs/code_replay_k80_p10). "
        "Repeatable. Held-out StereoSet texts are read from that run's own run_config.json.",
    )
    ap.add_argument(
        "--replay-model-tag",
        default="olmo7b",
        choices=sorted(MODEL_NAME_BY_TAG),
        help="Base model tag for --replay-run entries (default: olmo7b, i.e. allenai/OLMo-2-1124-7B).",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (args.output_root and args.model_tag) and not args.replay_run:
        ap.error("Pass --output-root and --model-tag for the selector grid, and/or --replay-run for replay runs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if args.output_root and args.model_tag:
        model_dir = Path(args.output_root).resolve() / args.model_tag
        base_name = MODEL_NAME_BY_TAG[args.model_tag]
        selectors = args.selectors or SELECTORS
        for selector in selectors:
            sp = summary_path(model_dir, selector)
            adapter = find_adapter(model_dir, selector)
            label = f"{args.model_tag}/{selector}"
            if sp is None or adapter is None:
                print(f"[skip] {label}: no summary/adapter")
                continue
            texts = load_reference_texts(model_dir, selector)
            backfill_cell(label, sp, adapter, texts, base_name, device=device, dtype=dtype,
                          batch_size=args.batch_size, max_length=args.max_length, force=args.force)

    if args.replay_run:
        replay_base_name = MODEL_NAME_BY_TAG[args.replay_model_tag]
        for raw in args.replay_run:
            if "=" not in raw:
                print(f"[skip] --replay-run {raw!r}: expected LABEL=OUTPUT_DIR")
                continue
            label, output_dir_str = raw.split("=", 1)
            output_dir = Path(output_dir_str).resolve()
            sp = output_dir / "summary.json"
            adapter = output_dir / "final_adapter"
            if not sp.exists() or not adapter.exists():
                print(f"[skip] {label}: no summary/adapter under {output_dir}")
                continue
            texts = replay_reference_texts(output_dir)
            backfill_cell(label, sp, adapter, texts, replay_base_name, device=device, dtype=dtype,
                          batch_size=args.batch_size, max_length=args.max_length, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
