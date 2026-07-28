# SAFE-DRIFT Setting 2 current results

Snapshot time: **2026-07-28 15:45 UTC**

This directory is a Git-friendly snapshot of the current Setting 2
MedQA/MedMCQA experiments. It contains the directly browsable aggregate
tables and figures, plus compressed raw run summaries and diagnostics with
their original cloud-relative paths.

## Snapshot status

- Completed run summaries: **341**
- Full-parameter SFT sweep: **5 successful of 81 scheduled**
- Historical full-SFT failures: **2 pre-fix runs**, retained for audit and
  requiring retry
- Paused SGD dispatch: **5 successful of 235 scheduled**
- Model ablation: queued, not started at snapshot time

The sweeps were still active when this snapshot was created. The status above
is therefore intentionally frozen at the stated timestamp.

## Contents

- `analysis/current_20260727/`: aggregate result tables, diagnostics, and
  publication plots assembled on 2026-07-27.
- `analysis/current_20260728_preliminary/`: the latest preliminary tables for
  the active priority experiments.
- `raw_snapshot/*_completed_runs.csv`: searchable one-row-per-completed-run
  index with target, drift, KL, constraint, timing, and FLOP fields.
- `raw_snapshot/*_inventory.csv`: inventory of every file under the five
  Setting 2 cloud output trees, including size and snapshot disposition.
- `raw_snapshot/*_summaries.tar.gz`: all 341 `summary.json` files plus run
  manifests, base evaluations, entanglement summaries, run configs, and
  completion markers.
- `raw_snapshot/*_diagnostics.tar.gz`: per-run training metrics, constraint
  trajectories, selected-subset entanglement tables, selected-gradient row
  metadata, and selected-index files.
- `raw_snapshot/*_dispatch.tar.gz`: JSON/JSONL dispatch definitions, worker
  status records, pause state, and queue metadata.
- `raw_snapshot/*_snapshot_status.json`: machine-readable snapshot counts.
- `CHECKSUMS.sha256`: SHA-256 checksums for every file in this export.

Archives preserve paths relative to:

```text
/lambda/nfs/mem/safe-drift/outputs
/lambda/nfs/mem/safe-drift/dispatch
```

For example:

```bash
tar -xzf raw_snapshot/setting2_current_20260728T1545Z_summaries.tar.gz
```

## Large artifacts

The Git snapshot intentionally excludes adapter weights, tokenizer copies,
caches, raw repeated selected-candidate text, binary gradient/feature tensors,
and duplicate per-run PNG/PDF files. These remain on the shared cloud
filesystem. Their exact paths and sizes are recorded in
`raw_snapshot/*_inventory.csv` under the corresponding `cloud_only_*`
category.

The aggregate figures in `analysis/` are included directly. The cloud roots
remain:

```text
/lambda/nfs/mem/safe-drift/outputs
/lambda/nfs/mem/safe-drift/shared_cache
```
