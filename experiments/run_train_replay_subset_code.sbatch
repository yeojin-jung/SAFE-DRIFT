#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -J code_replay_train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --constraint='a100|h100|h200'

set -euo pipefail

# Train a LoRA adapter directly on a precomputed replay-mixed subset (from
# src/utils/build_replay_subset.py), skipping selector/gradient computation
# entirely -- selection is already done, this file just needs to be trained on.
# Training/eval hyperparameters mirror experiments/run_selector_sft_compare_group_code.sbatch
# so results stay comparable to the less/random/dsir/... baselines trained through
# that pipeline.
#
# Override any variable at submission time, for example:
#   sbatch --export=ALL,REPLAY_FILE=...,RUN_TAG=code_replay_k80 \
#       experiments/run_train_replay_subset_code.sbatch

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/home/lcpandia/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/scratch/lcpandia/clean_workspace}"
CLEAN_EXPERIMENTS_DIR="${CLEAN_PROJECT_DIR}/experiments"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${MAIN_PROJECT_DIR}/src/utils/train_lora_sft.py}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"

REPLAY_FILE="${REPLAY_FILE:?Set REPLAY_FILE to the replay-mixed jsonl produced by build_replay_subset.py.}"
TARGET_FILE="${TARGET_FILE:-${MAIN_PROJECT_DIR}/data/setting_1_code/humaneval.jsonl}"
OOD_EVAL_FILE="${OOD_EVAL_FILE:-${TARGET_FILE}}"

MODEL_NAME="${MODEL_NAME:-allenai/OLMo-2-1124-7B}"
RUN_TAG="${RUN_TAG:-code_replay_train_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_EXPERIMENTS_DIR}/outputs/${RUN_TAG}}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${OUTPUT_ROOT}/train_cache}"

# Same target-file split used during LESS selection, so val/final_test line up
# with the splits used across all other selector baselines.
TARGET_SPLIT_PROPORTIONS="${TARGET_SPLIT_PROPORTIONS:-0.34 0.33 0.33}"
TARGET_SPLIT_SEED="${TARGET_SPLIT_SEED:-42}"

TRAIN_MAX_SEQ_LENGTH="${TRAIN_MAX_SEQ_LENGTH:-1024}"
TRAIN_TORCH_DTYPE="${TRAIN_TORCH_DTYPE:-bf16}"
TRAIN_LORA_R="${TRAIN_LORA_R:-16}"
TRAIN_LORA_ALPHA="${TRAIN_LORA_ALPHA:-16}"
TRAIN_LORA_DROPOUT="${TRAIN_LORA_DROPOUT:-0.05}"
TRAIN_LORA_TARGET_MODULES="${TRAIN_LORA_TARGET_MODULES:-q_proj k_proj v_proj o_proj}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-20}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-}"

EVALUATOR="${EVALUATOR:-humaneval}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-32}"
HUMANEVAL_NUM_SAMPLES="${HUMANEVAL_NUM_SAMPLES:-1}"
HUMANEVAL_PASS_AT_KS="${HUMANEVAL_PASS_AT_KS:-1}"
HUMANEVAL_TEMPERATURE="${HUMANEVAL_TEMPERATURE:-0.8}"
HUMANEVAL_TOP_P="${HUMANEVAL_TOP_P:-0.95}"

mkdir -p "${CLEAN_EXPERIMENTS_DIR}/logs" "${OUTPUT_ROOT}" "${TRAIN_CACHE_DIR}"

cd "${MAIN_PROJECT_DIR}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"

read -r -a TRAIN_MODULE_ARRAY <<< "${TRAIN_LORA_TARGET_MODULES}"

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "project:     ${MAIN_PROJECT_DIR}"
echo "trainer:     ${TRAIN_SCRIPT}"
echo "replay file: ${REPLAY_FILE}"
echo "target file: ${TARGET_FILE}"
echo "ood eval:    ${OOD_EVAL_FILE}"
echo "run_tag:     ${RUN_TAG}"
echo "output root: ${OUTPUT_ROOT}"
echo "model:       ${MODEL_NAME}"
echo "lora:        r=${TRAIN_LORA_R}, alpha=${TRAIN_LORA_ALPHA}, modules=${TRAIN_LORA_TARGET_MODULES}"
nvidia-smi || true

CMD=(
    python "${TRAIN_SCRIPT}"
    --model-name-or-path "${MODEL_NAME}"
    --train-file "${REPLAY_FILE}"
    --validation-file "${TARGET_FILE}"
    --validation-split-proportions ${TARGET_SPLIT_PROPORTIONS}
    --validation-split-seed "${TARGET_SPLIT_SEED}"
    --validation-split-role val
    --eval-file "${TARGET_FILE}"
    --eval-split-proportions ${TARGET_SPLIT_PROPORTIONS}
    --eval-split-seed "${TARGET_SPLIT_SEED}"
    --eval-split-role final_test
    --ood-eval-file "${OOD_EVAL_FILE}"
    --output-dir "${OUTPUT_ROOT}"
    --cache-dir "${TRAIN_CACHE_DIR}"
    --max-seq-length "${TRAIN_MAX_SEQ_LENGTH}"
    --torch-dtype "${TRAIN_TORCH_DTYPE}"
    --lora-r "${TRAIN_LORA_R}"
    --lora-alpha "${TRAIN_LORA_ALPHA}"
    --lora-dropout "${TRAIN_LORA_DROPOUT}"
    --lora-target-modules "${TRAIN_MODULE_ARRAY[@]}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --seed "${SEED}"
    --logging-steps "${LOGGING_STEPS}"
    --eval-steps "${EVAL_STEPS}"
    --evaluator "${EVALUATOR}"
    --generation-max-new-tokens "${GENERATION_MAX_NEW_TOKENS}"
    --humaneval-num-samples "${HUMANEVAL_NUM_SAMPLES}"
    --humaneval-pass-at-ks ${HUMANEVAL_PASS_AT_KS}
    --humaneval-temperature "${HUMANEVAL_TEMPERATURE}"
    --humaneval-top-p "${HUMANEVAL_TOP_P}"
    --evaluate-base-model
    --evaluate-base-ood
    --gradient-checkpointing
    --low-cpu-mem-usage
)

if [[ -n "${MAX_STEPS}" ]]; then
    CMD+=(--max-steps "${MAX_STEPS}")
else
    CMD+=(--num-train-epochs "${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${EVAL_MAX_EXAMPLES}" ]]; then
    CMD+=(--eval-max-examples "${EVAL_MAX_EXAMPLES}")
fi

"${CMD[@]}"

echo "finished: $(date)"
