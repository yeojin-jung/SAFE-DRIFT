# Invalid Reference-Composition Plots

The `solver_tradeoffs_*` plots in this directory that use the Setting-2
reference-composition sweep must not be interpreted as independent
composition results.

The run code shared one SAFE subset output directory across `B_T`, `B_K`,
`T_K`, and `B_T_K`, while the subset filename omitted the reference
composition. Later compositions therefore reused an earlier composition's
selected subset and selection metadata. All 27 seed/epsilon/gamma cells in
the collected data contain at least one exact cross-composition duplicate in
target accuracy, held-out KL, drift, and predicted Fisher cost.

The selector artifact key has been corrected in
`scripts/run_selector_sft_sweep_code.py`. The Setting-2 composition sweep
must be rerun before these figures are regenerated.
