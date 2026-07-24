# Vast.ai Launch Notes

This folder contains a launcher for the Setting 2 reference-coverage sweep:

```bash
scripts/vastai/launch_reference_coverage.sh --ssh root@HOST --port PORT
```

The launcher assumes an existing Vast.ai instance with SSH enabled. It syncs the
local SAFE-DRIFT repo to the instance, creates a Vast-local config by replacing
`/lambda/nfs/mem/safe-drift` with `/workspace/safe-drift`, installs
`requirements/vast-train.txt`, prepares the datasets, builds command arrays, runs
cache prep, runs the 15 experiment jobs, and writes a reference-coverage CSV.

## Instance

Recommended starting point:

```text
image: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime or newer PyTorch CUDA image
disk: 300 GB minimum, 500 GB safer
gpu: A100 80GB / H100 preferred; 48GB cards may be tight
ssh: enabled, direct if available
```

If you use multiple GPUs, the launcher runs one experiment command per visible
GPU. Use `--gpus 1` to force serial execution.

## Authentication

If Hugging Face access is needed, log in on the Vast instance before launching:

```bash
ssh -i ~/.ssh/id_ed25519 -p PORT root@HOST
python -m pip install --upgrade huggingface_hub
huggingface-cli login
exit
```

Do not put API tokens in shell history or shared logs.

## Launch

From the local SAFE-DRIFT repo:

```bash
scripts/vastai/launch_reference_coverage.sh \
  --ssh root@HOST \
  --port PORT \
  --identity ~/.ssh/id_ed25519 \
  --remote-root /workspace/safe-drift \
  --gpus auto
```

The command prints a `tail -f` command for the remote log. Outputs are written
under:

```text
/workspace/safe-drift/outputs/setting_2_medqa_medmcqa_reference_coverage_v1
/workspace/safe-drift/dispatch/<experiment-id>/reference_coverage_results.csv
```

## Resume

Re-running the launcher is resume-friendly. The runner skips completed manifests
and the experiment config has `resume_completed_runs: true`.

Use `--skip-sync` when only resuming an already-synced repo:

```bash
scripts/vastai/launch_reference_coverage.sh \
  --ssh root@HOST \
  --port PORT \
  --skip-sync
```

## Pull Results

```bash
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p PORT" \
  root@HOST:/workspace/safe-drift/dispatch/ \
  vast_dispatch/

rsync -az -e "ssh -i ~/.ssh/id_ed25519 -p PORT" \
  root@HOST:/workspace/safe-drift/outputs/setting_2_medqa_medmcqa_reference_coverage_v1/ \
  vast_outputs/setting_2_medqa_medmcqa_reference_coverage_v1/
```
