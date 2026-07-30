#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH -t 04:00:00
#SBATCH -J backfill_reference_kl
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --constraint='a100|h100|h200'

set -euo pipefail

# Backfill KL(base || fine-tuned) on held-out StereoSet reference sentences into
# each run's summary.json (scripts/backfill_setting1_reference_kl.py). Supports
# the model_tag/selector/training_runs grid and/or one or more replay-method runs
# in the same invocation.
#
# Examples:
#   # Replay runs only:
#   sbatch --export=ALL,REPLAY_RUNS="k80=/path/outputs/code_replay_k80_p10 k90=/path/outputs/code_replay_k90_p10" \
#       experiments/run_backfill_reference_kl.sbatch
#
#   # Selector grid only:
#   sbatch --export=ALL,OUTPUT_ROOT=/path/to/outputs,MODEL_TAG=olmo7b \
#       experiments/run_backfill_reference_kl.sbatch
#
#   # Both in one job:
#   sbatch --export=ALL,OUTPUT_ROOT=/path/to/outputs,MODEL_TAG=olmo7b,REPLAY_RUNS="k80=/path/outputs/code_replay_k80_p10" \
#       experiments/run_backfill_reference_kl.sbatch

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/home/lcpandia/SAFE-DRIFT}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"
BACKFILL_SCRIPT="${BACKFILL_SCRIPT:-${MAIN_PROJECT_DIR}/scripts/backfill_setting1_reference_kl.py}"

OUTPUT_ROOT="${OUTPUT_ROOT:-}"
MODEL_TAG="${MODEL_TAG:-}"
SELECTORS="${SELECTORS:-}"

REPLAY_RUNS="${REPLAY_RUNS:-}"
REPLAY_MODEL_TAG="${REPLAY_MODEL_TAG:-olmo7b}"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-64}"
DTYPE="${DTYPE:-bfloat16}"
FORCE="${FORCE:-false}"

if [[ -z "${OUTPUT_ROOT}" && -z "${REPLAY_RUNS}" ]]; then
    echo "Set OUTPUT_ROOT (+ MODEL_TAG) for the selector grid and/or REPLAY_RUNS for replay runs." >&2
    exit 1
fi
if [[ -n "${OUTPUT_ROOT}" && -z "${MODEL_TAG}" ]]; then
    echo "OUTPUT_ROOT is set but MODEL_TAG is empty -- required together." >&2
    exit 1
fi

mkdir -p "${MAIN_PROJECT_DIR}/logs"
cd "${MAIN_PROJECT_DIR}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "project:     ${MAIN_PROJECT_DIR}"
echo "script:      ${BACKFILL_SCRIPT}"
if [[ -n "${OUTPUT_ROOT}" ]]; then
    echo "grid:        output_root=${OUTPUT_ROOT} model_tag=${MODEL_TAG} selectors=${SELECTORS:-<default>}"
fi
if [[ -n "${REPLAY_RUNS}" ]]; then
    echo "replay runs: ${REPLAY_RUNS} (base model tag: ${REPLAY_MODEL_TAG})"
fi
nvidia-smi || true

CMD=(python "${BACKFILL_SCRIPT}" --batch-size "${BATCH_SIZE}" --max-length "${MAX_LENGTH}" --dtype "${DTYPE}")

if [[ -n "${OUTPUT_ROOT}" ]]; then
    CMD+=(--output-root "${OUTPUT_ROOT}" --model-tag "${MODEL_TAG}")
    if [[ -n "${SELECTORS}" ]]; then
        read -r -a SELECTOR_ARRAY <<< "${SELECTORS}"
        CMD+=(--selectors "${SELECTOR_ARRAY[@]}")
    fi
fi

if [[ -n "${REPLAY_RUNS}" ]]; then
    read -r -a REPLAY_RUN_ARRAY <<< "${REPLAY_RUNS}"
    for entry in "${REPLAY_RUN_ARRAY[@]}"; do
        CMD+=(--replay-run "${entry}")
    done
    CMD+=(--replay-model-tag "${REPLAY_MODEL_TAG}")
fi

if [[ "${FORCE}" == "true" ]]; then
    CMD+=(--force)
fi

echo "command:     ${CMD[*]}"
"${CMD[@]}"

echo "finished: $(date)"
