# CLAUDE.md

Project: SAFE-DRIFT — data selection for SFT that trades off target-task gain against
off-target ("reference") drift, formalized as a reference-Fisher-constrained update.
Currently in a NeurIPS rebuttal cycle.

## Rebuttal context

- Full reviews (3 reviewers, scores, weaknesses/questions/limitations with response
  boxes) + AC meta-review + prioritized rebuttal checklist:
  `manuscript/reviews.tex` (compiled: `manuscript/reviews.pdf`).
- Experiment-section plan (main + appendix deliverables: what to show, metrics,
  which setting, whether new runs/logging are needed, reviewer/AC tags):
  `manuscript/experiment_plan.tex` (compiled: `manuscript/experiment_plan.pdf`).
- AC's 5 priority asks (see `reviews.tex` Section "Meta-Review"): (1) computational
  complexity — now reporting in FLOPs + peak memory, not just wall-clock; (2) how
  accurately predicted quadratic drift matches realized drift over the *full*
  training trajectory; (3) reference-set coverage assumption / "SAFE" as an
  over-claimed title — being addressed by a colleague's mixture-of-references
  experiment (cross-referenced in the plan, not new work here); (4) missing
  forgetting- / off-target-shift-aware baselines beyond target-oriented selectors;
  (5) validation on other model architectures and scale.

## Method map (where things live)

- Core SAFE-DRIFT closed-form update (damped natural gradient; alpha search from
  rho/epsilon budgets): `src/safe_drift/safe_drift_update.py`
- Discrete subset selection (rank vs. greedy_marginal solvers; euclidean vs.
  reference-Fisher-whitened geometry): `src/safe_drift/safe_drift_selector.py`
- Low-rank Fisher + gradient basis construction (K_R, K_T, common basis,
  shrinkage): `src/utils/low_rank_fisher_builder.py`
- Entanglement diagnostics (rho_E, rho_P, influence decomposition, spectral
  energy): `src/evaluation/entanglement_analysis.py`
- StereoSet-style reference/bias eval (SS/LMS/ICAT): `src/evaluation/bias_disentangle.py`
- Main sweep/manifest driver (builds `selector_sft_sweep_manifest.json`):
  `scripts/run_selector_sft_sweep_code.py`
- SLURM launchers: `experiments/*.sbatch`

## Conventions

- Manuscript sources live in `manuscript/`; compile with `pdflatex` (may need two
  passes for cross-references/TOC).
- Manifest paths reference the cluster mount `/net/projects2/mercury/yeojin/...` —
  adjust if running from a different environment.
- `manuscript/experiments_checklist.tex` is an older, standing settings/checklist
  doc (rank grids, dataset splits per setting) — distinct from the rebuttal-specific
  plan in `experiment_plan.tex`.

## Active work

Implementing the rebuttal experiment plan (`manuscript/experiment_plan.tex`):
main items M1-M5 (settings table, main results + Pareto, model/scale ablation,
spectral/entanglement + rank ablation, predicted-vs-realized calibration) and
appendix items A1-A7 (forgetting baselines, optimizer ablation, FLOPs/memory
scaling, FLOPs-matched value comparison, reference-coverage cross-ref, optional
robustness/full-FT runs). Open decisions: which single forgetting baseline goes
in the main table (M2), and which setting hosts the calibration study (M5).
