#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/net/projects2/mercury/cdonnat/SAFE-DRIFT}"
VENV="${VENV:-$PROJECT_DIR/.venv}"
COMMANDS_FILE="${COMMANDS_FILE:?Set COMMANDS_FILE to a JSONL command manifest.}"

cd "$PROJECT_DIR"
mkdir -p logs .cache/huggingface .cache/matplotlib
source "$VENV/bin/activate"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_DIR/.cache/matplotlib}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

python scripts/cluster/run_array_command.py \
  --commands-file "$COMMANDS_FILE" \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}"
