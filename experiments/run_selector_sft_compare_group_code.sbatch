#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -J code_selector_compare
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --constraint='a100|h100|h200'

set -euo pipefail

# Run code only from the main clone, while keeping all data, logs, caches, and
# experiment outputs under clean_workspace.
MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/net/projects2/mercury/yeojin/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/projects2/mercury/yeojin/clean_workspace}"
CLEAN_EXPERIMENTS_DIR="${CLEAN_PROJECT_DIR}/experiments"
MAIN_RUNNER="${MAIN_RUNNER:-${MAIN_PROJECT_DIR}/scripts/run_selector_sft_sweep_code.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${MAIN_PROJECT_DIR}/src/utils/train_lora_sft.py}"
CODE_DATA_DIR="${CODE_DATA_DIR:-${MAIN_PROJECT_DIR}/data/setting_1_code}"
CANDIDATE_FILE="${CANDIDATE_FILE:-${CODE_DATA_DIR}/codealpaca.jsonl}"
TARGET_FILE="${TARGET_FILE:-${CODE_DATA_DIR}/humaneval.jsonl}"
OOD_EVAL_FILE="${OOD_EVAL_FILE:-${CODE_DATA_DIR}/humaneval.jsonl}"
REFERENCE_FILE="${REFERENCE_FILE:-}"

MODEL_NAME="${MODEL_NAME:-allenai/OLMo-2-1124-7B}"
RUN_TAG="${RUN_TAG:-code_selector_compare_$(date +%Y%m%d_%H%M%S)}"
SELECTORS="${SELECTORS:-full random dsir less prismatic safe}"
SUBSET_PERCENTAGES="${SUBSET_PERCENTAGES:-5 15}"
SEED="${SEED:-42}"
CACHE_ROOT_WAS_SET="${CACHE_ROOT+x}"
FEATURE_CACHE_DIR_WAS_SET="${FEATURE_CACHE_DIR+x}"
TRAIN_CACHE_DIR_WAS_SET="${TRAIN_CACHE_DIR+x}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_EXPERIMENTS_DIR}/outputs/${RUN_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${OUTPUT_ROOT}/cache}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${CACHE_ROOT}/selector_feature_cache}"
TRAIN_CACHE_DIR="${TRAIN_CACHE_DIR:-${CACHE_ROOT}/train_cache}"
BIAS_DATA_PATH="${BIAS_DATA_PATH:-${OUTPUT_ROOT}/reference_bias_eval.json}"
BIAS_EVAL_FILE="${BIAS_EVAL_FILE:-}"
EVALUATOR="${EVALUATOR:-humaneval}"

REFERENCE_HF_SUBSET="${REFERENCE_HF_SUBSET:-intrasentence}"
REFERENCE_HF_SPLIT="${REFERENCE_HF_SPLIT:-validation}"
REFERENCE_HF_LABEL="${REFERENCE_HF_LABEL:-stereotype,anti-stereotype}"
REFERENCE_HF_FORMAT="${REFERENCE_HF_FORMAT:-instruction}"
TARGET_SPLIT_PROPORTIONS="${TARGET_SPLIT_PROPORTIONS:-0.34 0.33 0.33}"
TARGET_SPLIT_SEED="${TARGET_SPLIT_SEED:-42}"
REFERENCE_SPLIT_PROPORTIONS="${REFERENCE_SPLIT_PROPORTIONS:-0.5 0.5}"
REFERENCE_SPLIT_SEED="${REFERENCE_SPLIT_SEED:-42}"

# Keep selector-side gradient extraction light; this is the configuration that
# fit reliably in recent code runs.
SELECTION_LORA_R="${SELECTION_LORA_R:-2}"
SELECTION_LORA_ALPHA="${SELECTION_LORA_ALPHA:-4}"
SELECTION_LORA_DROPOUT="${SELECTION_LORA_DROPOUT:-0.0}"
SELECTION_LORA_TARGET_MODULES="${SELECTION_LORA_TARGET_MODULES:-q_proj}"
LESS_SIMILARITY="${LESS_SIMILARITY:-cosine}"
SELECTOR_FEATURE_METHOD="${SELECTOR_FEATURE_METHOD:-low_rank}"
SELECTOR_PRECONDITIONER="${SELECTOR_PRECONDITIONER:-adam}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
ADAM_WARMUP_STEPS="${ADAM_WARMUP_STEPS:-0.03}"
PRECONDITIONER_SPLIT_SEED="${PRECONDITIONER_SPLIT_SEED:-42}"
SELECTOR_PROJECTION_DIM="${SELECTOR_PROJECTION_DIM:-8192}"
SELECTOR_PROJECTION_SEED="${SELECTOR_PROJECTION_SEED:-13}"
SELECTOR_PROJECTION_CHUNK_SIZE="${SELECTOR_PROJECTION_CHUNK_SIZE:-1048576}"
SELECTOR_PROJECTION_CACHE_DTYPE="${SELECTOR_PROJECTION_CACHE_DTYPE:-float16}"

SAFE_ALPHA="${SAFE_ALPHA:-}"
SAFE_LEARNING_RATE="${SAFE_LEARNING_RATE:-1.0}"
SAFE_GEOMETRY="${SAFE_GEOMETRY:-reference}"
SAFE_SOLVER="${SAFE_SOLVER:-greedy_marginal}"
SAFE_SHORTLIST_SIZE="${SAFE_SHORTLIST_SIZE:-}"
SAFE_AVERAGE_BY_BUDGET="${SAFE_AVERAGE_BY_BUDGET:-true}"
SAFE_COST_C="${SAFE_COST_C:-}"
SAFE_EPSILON="${SAFE_EPSILON:-}"
REFERENCE_FISHER_MAX_EXAMPLES="${REFERENCE_FISHER_MAX_EXAMPLES:-1024}"
LOW_RANK_REFERENCE_RANK="${LOW_RANK_REFERENCE_RANK:-256}"
LOW_RANK_BUILDER_ALPHA="${LOW_RANK_BUILDER_ALPHA:-1e-3}"
LOW_RANK_DELTA="${LOW_RANK_DELTA:-0.01}"
LOW_RANK_MAX_REFERENCE_RANK="${LOW_RANK_MAX_REFERENCE_RANK:-}"
LOW_RANK_REFERENCE_SHRINKAGE="${LOW_RANK_REFERENCE_SHRINKAGE:-none}"
LOW_RANK_TASK_RANK="${LOW_RANK_TASK_RANK:-64}"
LOW_RANK_COMMON_RANK="${LOW_RANK_COMMON_RANK:-}"
LOW_RANK_AUTO_TASK_RANK="${LOW_RANK_AUTO_TASK_RANK:-false}"
LOW_RANK_DELTA_TASK="${LOW_RANK_DELTA_TASK:-0.01}"
LOW_RANK_MAX_TASK_RANK="${LOW_RANK_MAX_TASK_RANK:-}"
LOW_RANK_TARGET_MAX_EXAMPLES="${LOW_RANK_TARGET_MAX_EXAMPLES:-}"
LOW_RANK_CANDIDATE_MAX_EXAMPLES="${LOW_RANK_CANDIDATE_MAX_EXAMPLES:-}"
SAVE_ALL_TARGET_FEATURES="${SAVE_ALL_TARGET_FEATURES:-false}"

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
SKIP_TRAINING="${SKIP_TRAINING:-false}"
SKIP_FINAL_EVALUATION="${SKIP_FINAL_EVALUATION:-false}"
OVERWRITE_SUBSETS="${OVERWRITE_SUBSETS:-false}"
OVERWRITE_SELECTION_CACHE="${OVERWRITE_SELECTION_CACHE:-false}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-20}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-32}"
HUMANEVAL_NUM_SAMPLES="${HUMANEVAL_NUM_SAMPLES:-1}"
HUMANEVAL_PASS_AT_KS="${HUMANEVAL_PASS_AT_KS:-1}"
HUMANEVAL_TEMPERATURE="${HUMANEVAL_TEMPERATURE:-0.8}"
HUMANEVAL_TOP_P="${HUMANEVAL_TOP_P:-0.95}"
BENCHMARK_EVALS="${BENCHMARK_EVALS:-}"
VALIDATION_GRADIENT_MAX_EXAMPLES="${VALIDATION_GRADIENT_MAX_EXAMPLES:-256}"

BIAS_DOMAIN="${BIAS_DOMAIN:-all}"
BIAS_LAYERS="${BIAS_LAYERS:--1}"
BIAS_ALPHA="${BIAS_ALPHA:-0.5}"
BIAS_GEOMETRY="${BIAS_GEOMETRY:-sand-e}"
BIAS_DIRECTION_SPLIT="${BIAS_DIRECTION_SPLIT:-0.5}"
BIAS_BATCH_SIZE="${BIAS_BATCH_SIZE:-1}"
BIAS_MAX_LENGTH="${BIAS_MAX_LENGTH:-512}"
BIAS_MAX_EXAMPLES="${BIAS_MAX_EXAMPLES:-none}"
REFERENCE_SPLIT_GROUP_KEY="${REFERENCE_SPLIT_GROUP_KEY:-bias_type}"
RUN_BIAS_EVAL="${RUN_BIAS_EVAL:-false}"
BIAS_EVAL_SCRIPT="${BIAS_EVAL_SCRIPT:-${CLEAN_EXPERIMENTS_DIR}/scripts/run_bias_disentangle_array.sbatch}"
BIAS_RUN_LIST="${BIAS_RUN_LIST:-${OUTPUT_ROOT}/bias_eval_runs.jsonl}"
BIAS_EVAL_PARTITION="${BIAS_EVAL_PARTITION:-general}"
BIAS_EVAL_GPUS="${BIAS_EVAL_GPUS:-1}"
BIAS_EVAL_MEM="${BIAS_EVAL_MEM:-64G}"
BIAS_EVAL_CPUS="${BIAS_EVAL_CPUS:-4}"
BIAS_EVAL_TIME="${BIAS_EVAL_TIME:-06:00:00}"
BIAS_EVAL_CONSTRAINT="${BIAS_EVAL_CONSTRAINT:-a100|h100|h200}"
if [[ -z "${SUBMIT_SPLIT_JOBS+x}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        SUBMIT_SPLIT_JOBS=false
    else
        SUBMIT_SPLIT_JOBS=true
    fi
fi
SPLIT_BY_BUDGET="${SPLIT_BY_BUDGET:-false}"
CACHE_JOB_TIME="${CACHE_JOB_TIME:-04:00:00}"
CACHE_JOB_MEM="${CACHE_JOB_MEM:-128G}"
CACHE_JOB_CPUS="${CACHE_JOB_CPUS:-8}"
CACHE_JOB_GPUS="${CACHE_JOB_GPUS:-1}"
SELECTOR_JOB_TIME="${SELECTOR_JOB_TIME:-08:00:00}"
SELECTOR_JOB_MEM="${SELECTOR_JOB_MEM:-128G}"
SELECTOR_JOB_CPUS="${SELECTOR_JOB_CPUS:-8}"
SELECTOR_JOB_GPUS="${SELECTOR_JOB_GPUS:-1}"
SPLIT_JOB_PARTITION="${SPLIT_JOB_PARTITION:-general}"
SPLIT_JOB_CONSTRAINT="${SPLIT_JOB_CONSTRAINT:-a100|h100|h200}"

slugify() {
    printf '%s' "$1" | sed -E 's/[^A-Za-z0-9]+/_/g; s/^_+//; s/_+$//'
}

if [[ -n "${SAFE_COST_C}" ]]; then
    if [[ -n "${SAFE_ALPHA}" ]]; then
        echo "[safe] SAFE_COST_C was set, so ignoring explicit SAFE_ALPHA=${SAFE_ALPHA} and using cost-based mode."
    fi
    if [[ -z "${SAFE_EPSILON}" ]]; then
        echo "[safe] SAFE_COST_C requires SAFE_EPSILON to be set." >&2
        exit 1
    fi
    RESOLVED_SAFE_ALPHA=""
else
    RESOLVED_SAFE_ALPHA="${SAFE_ALPHA:-1e-3}"
fi

if [[ "${LOW_RANK_REFERENCE_RANK}" == "auto" ]]; then
    RESOLVED_LOW_RANK_REFERENCE_RANK=""
    RESOLVED_LOW_RANK_REFERENCE_RANK_TAG="auto"
else
    RESOLVED_LOW_RANK_REFERENCE_RANK="${LOW_RANK_REFERENCE_RANK}"
    RESOLVED_LOW_RANK_REFERENCE_RANK_TAG="${LOW_RANK_REFERENCE_RANK:-auto}"
fi

if [[ "${LOW_RANK_TASK_RANK}" == "auto" ]]; then
    RESOLVED_LOW_RANK_TASK_RANK=""
    RESOLVED_LOW_RANK_AUTO_TASK_RANK="true"
    RESOLVED_LOW_RANK_TASK_RANK_TAG="auto"
else
    RESOLVED_LOW_RANK_TASK_RANK="${LOW_RANK_TASK_RANK}"
    RESOLVED_LOW_RANK_AUTO_TASK_RANK="${LOW_RANK_AUTO_TASK_RANK}"
    RESOLVED_LOW_RANK_TASK_RANK_TAG="${LOW_RANK_TASK_RANK:-none}"
fi

MODEL_TAG="$(slugify "${MODEL_NAME}")"
DATA_TAG="code_sweep"
KR_TAG="KR$(slugify "${RESOLVED_LOW_RANK_REFERENCE_RANK_TAG}")"
KADD_TAG="Kadd$(slugify "${RESOLVED_LOW_RANK_TASK_RANK_TAG}")"
KCOMMON_TAG="Kcommon$(slugify "${LOW_RANK_COMMON_RANK:-auto}")"
SEED_TAG="seed$(slugify "${SEED}")_tsplit$(slugify "${TARGET_SPLIT_SEED}")_rsplit$(slugify "${REFERENCE_SPLIT_SEED}")_psplit$(slugify "${PRECONDITIONER_SPLIT_SEED:-42}")"
LR_TAG="lr$(slugify "${LEARNING_RATE}")"
PREC_TAG="selprecond$(slugify "${SELECTOR_PRECONDITIONER}")"
SELECTION_LORA_TAG="sellora_r$(slugify "${SELECTION_LORA_R}")_a$(slugify "${SELECTION_LORA_ALPHA}")_d$(slugify "${SELECTION_LORA_DROPOUT}")"
TRAIN_LORA_TAG="trainlora_r$(slugify "${TRAIN_LORA_R}")_a$(slugify "${TRAIN_LORA_ALPHA}")"
SHARED_CACHE_TAG="${DATA_TAG}_${MODEL_TAG}_${PREC_TAG}_${KR_TAG}_${KADD_TAG}_${KCOMMON_TAG}_${SEED_TAG}_${LR_TAG}_${SELECTION_LORA_TAG}_${TRAIN_LORA_TAG}"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${CLEAN_EXPERIMENTS_DIR}/shared_cache/${SHARED_CACHE_TAG}}"
if [[ -z "${CACHE_ROOT_WAS_SET}" ]]; then
    CACHE_ROOT="${SHARED_CACHE_ROOT}"
fi
if [[ -z "${FEATURE_CACHE_DIR_WAS_SET}" ]]; then
    FEATURE_CACHE_DIR="${CACHE_ROOT}/selector_feature_cache"
fi
if [[ -z "${TRAIN_CACHE_DIR_WAS_SET}" ]]; then
    TRAIN_CACHE_DIR="${CACHE_ROOT}/train_cache"
fi

if [[ "${SUBMIT_SPLIT_JOBS}" == "true" ]]; then
    submit_split_job() {
        local job_name="$1"
        local dependency="$2"
        local job_time="$3"
        local job_mem="$4"
        local job_cpus="$5"
        local job_gpus="$6"
        shift 6

        local dep_args=()
        if [[ -n "${dependency}" ]]; then
            dep_args=(--dependency="afterok:${dependency}")
        fi

        env \
            "MAIN_PROJECT_DIR=${MAIN_PROJECT_DIR}" \
            "CLEAN_PROJECT_DIR=${CLEAN_PROJECT_DIR}" \
            "MAIN_RUNNER=${MAIN_RUNNER}" \
            "TRAIN_SCRIPT=${TRAIN_SCRIPT}" \
            "MODEL_NAME=${MODEL_NAME}" \
            "CACHE_ROOT=${CACHE_ROOT}" \
            "FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}" \
            "TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}" \
            "SHARED_CACHE_ROOT=${SHARED_CACHE_ROOT}" \
            "SEED=${SEED}" \
            "TARGET_SPLIT_PROPORTIONS=${TARGET_SPLIT_PROPORTIONS}" \
            "TARGET_SPLIT_SEED=${TARGET_SPLIT_SEED}" \
            "REFERENCE_SPLIT_PROPORTIONS=${REFERENCE_SPLIT_PROPORTIONS}" \
            "REFERENCE_SPLIT_SEED=${REFERENCE_SPLIT_SEED}" \
            "REFERENCE_SPLIT_GROUP_KEY=${REFERENCE_SPLIT_GROUP_KEY}" \
            "CODE_DATA_DIR=${CODE_DATA_DIR}" \
            "CANDIDATE_FILE=${CANDIDATE_FILE}" \
            "TARGET_FILE=${TARGET_FILE}" \
            "OOD_EVAL_FILE=${OOD_EVAL_FILE}" \
            "REFERENCE_FILE=${REFERENCE_FILE}" \
            "REFERENCE_HF_SUBSET=${REFERENCE_HF_SUBSET}" \
            "REFERENCE_HF_SPLIT=${REFERENCE_HF_SPLIT}" \
            "REFERENCE_HF_LABEL=${REFERENCE_HF_LABEL}" \
            "REFERENCE_HF_FORMAT=${REFERENCE_HF_FORMAT}" \
            "SELECTION_LORA_R=${SELECTION_LORA_R}" \
            "SELECTION_LORA_ALPHA=${SELECTION_LORA_ALPHA}" \
            "SELECTION_LORA_DROPOUT=${SELECTION_LORA_DROPOUT}" \
            "SELECTION_LORA_TARGET_MODULES=${SELECTION_LORA_TARGET_MODULES}" \
            "SELECTOR_FEATURE_METHOD=${SELECTOR_FEATURE_METHOD}" \
            "SELECTOR_PRECONDITIONER=${SELECTOR_PRECONDITIONER}" \
            "LESS_SIMILARITY=${LESS_SIMILARITY}" \
            "ADAM_BETA2=${ADAM_BETA2}" \
            "ADAM_EPS=${ADAM_EPS}" \
            "ADAM_WARMUP_STEPS=${ADAM_WARMUP_STEPS}" \
            "PRECONDITIONER_SPLIT_SEED=${PRECONDITIONER_SPLIT_SEED}" \
            "SAFE_ALPHA=${RESOLVED_SAFE_ALPHA}" \
            "SAFE_LEARNING_RATE=${SAFE_LEARNING_RATE}" \
            "SAFE_GEOMETRY=${SAFE_GEOMETRY}" \
            "SAFE_SOLVER=${SAFE_SOLVER}" \
            "SAFE_SHORTLIST_SIZE=${SAFE_SHORTLIST_SIZE}" \
            "SAFE_AVERAGE_BY_BUDGET=${SAFE_AVERAGE_BY_BUDGET}" \
            "SAFE_COST_C=${SAFE_COST_C}" \
            "SAFE_EPSILON=${SAFE_EPSILON}" \
            "REFERENCE_FISHER_MAX_EXAMPLES=${REFERENCE_FISHER_MAX_EXAMPLES}" \
            "LOW_RANK_REFERENCE_RANK=${RESOLVED_LOW_RANK_REFERENCE_RANK}" \
            "LOW_RANK_BUILDER_ALPHA=${LOW_RANK_BUILDER_ALPHA}" \
            "LOW_RANK_DELTA=${LOW_RANK_DELTA}" \
            "LOW_RANK_MAX_REFERENCE_RANK=${LOW_RANK_MAX_REFERENCE_RANK}" \
            "LOW_RANK_REFERENCE_SHRINKAGE=${LOW_RANK_REFERENCE_SHRINKAGE}" \
            "LOW_RANK_TASK_RANK=${RESOLVED_LOW_RANK_TASK_RANK}" \
            "LOW_RANK_COMMON_RANK=${LOW_RANK_COMMON_RANK}" \
            "LOW_RANK_AUTO_TASK_RANK=${RESOLVED_LOW_RANK_AUTO_TASK_RANK}" \
            "LOW_RANK_DELTA_TASK=${LOW_RANK_DELTA_TASK}" \
            "LOW_RANK_MAX_TASK_RANK=${LOW_RANK_MAX_TASK_RANK}" \
            "LOW_RANK_TARGET_MAX_EXAMPLES=${LOW_RANK_TARGET_MAX_EXAMPLES}" \
            "LOW_RANK_CANDIDATE_MAX_EXAMPLES=${LOW_RANK_CANDIDATE_MAX_EXAMPLES}" \
            "TRAIN_MAX_SEQ_LENGTH=${TRAIN_MAX_SEQ_LENGTH}" \
            "TRAIN_TORCH_DTYPE=${TRAIN_TORCH_DTYPE}" \
            "TRAIN_LORA_R=${TRAIN_LORA_R}" \
            "TRAIN_LORA_ALPHA=${TRAIN_LORA_ALPHA}" \
            "TRAIN_LORA_DROPOUT=${TRAIN_LORA_DROPOUT}" \
            "TRAIN_LORA_TARGET_MODULES=${TRAIN_LORA_TARGET_MODULES}" \
            "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}" \
            "PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE}" \
            "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" \
            "LEARNING_RATE=${LEARNING_RATE}" \
            "WEIGHT_DECAY=${WEIGHT_DECAY}" \
            "WARMUP_RATIO=${WARMUP_RATIO}" \
            "LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE}" \
            "MAX_GRAD_NORM=${MAX_GRAD_NORM}" \
            "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}" \
            "MAX_STEPS=${MAX_STEPS}" \
            "LOGGING_STEPS=${LOGGING_STEPS}" \
            "EVAL_STEPS=${EVAL_STEPS}" \
            "EVAL_MAX_EXAMPLES=${EVAL_MAX_EXAMPLES}" \
            "GENERATION_MAX_NEW_TOKENS=${GENERATION_MAX_NEW_TOKENS}" \
            "HUMANEVAL_NUM_SAMPLES=${HUMANEVAL_NUM_SAMPLES}" \
            "HUMANEVAL_PASS_AT_KS=${HUMANEVAL_PASS_AT_KS}" \
            "HUMANEVAL_TEMPERATURE=${HUMANEVAL_TEMPERATURE}" \
            "HUMANEVAL_TOP_P=${HUMANEVAL_TOP_P}" \
            "EVALUATOR=${EVALUATOR}" \
            "SKIP_FINAL_EVALUATION=${SKIP_FINAL_EVALUATION}" \
            "BIAS_EVAL_FILE=${BIAS_EVAL_FILE}" \
            "VALIDATION_GRADIENT_MAX_EXAMPLES=${VALIDATION_GRADIENT_MAX_EXAMPLES}" \
            "BIAS_DOMAIN=${BIAS_DOMAIN}" \
            "BIAS_LAYERS=${BIAS_LAYERS}" \
            "BIAS_ALPHA=${BIAS_ALPHA}" \
            "BIAS_GEOMETRY=${BIAS_GEOMETRY}" \
            "BIAS_DIRECTION_SPLIT=${BIAS_DIRECTION_SPLIT}" \
            "BIAS_BATCH_SIZE=${BIAS_BATCH_SIZE}" \
            "BIAS_MAX_LENGTH=${BIAS_MAX_LENGTH}" \
            "BIAS_MAX_EXAMPLES=${BIAS_MAX_EXAMPLES}" \
            "BIAS_EVAL_SCRIPT=${BIAS_EVAL_SCRIPT}" \
            "BIAS_EVAL_PARTITION=${BIAS_EVAL_PARTITION}" \
            "BIAS_EVAL_GPUS=${BIAS_EVAL_GPUS}" \
            "BIAS_EVAL_MEM=${BIAS_EVAL_MEM}" \
            "BIAS_EVAL_CPUS=${BIAS_EVAL_CPUS}" \
            "BIAS_EVAL_TIME=${BIAS_EVAL_TIME}" \
            "BIAS_EVAL_CONSTRAINT=${BIAS_EVAL_CONSTRAINT}" \
            "$@" \
            sbatch \
            --partition="${SPLIT_JOB_PARTITION}" \
            --gres="gpu:${job_gpus}" \
            --mem="${job_mem}" \
            --cpus-per-task="${job_cpus}" \
            --time="${job_time}" \
            --job-name="${job_name}" \
            --constraint="${SPLIT_JOB_CONSTRAINT}" \
            "${dep_args[@]}" \
            --export=ALL \
            "$0" | awk '{print $4}'
    }

    DATE_TAG="$(date +%Y%m%d)"
    SAFE_COST_TAG="$(slugify "${SAFE_COST_C:-none}")"
    SAFE_EPS_TAG="$(slugify "${SAFE_EPSILON:-none}")"
    BASE_RUN_TAG="${DATA_TAG}_${KR_TAG}_${KADD_TAG}_${PREC_TAG}_rho${SAFE_COST_TAG}_eps${SAFE_EPS_TAG}_${LR_TAG}_${SEED_TAG}_${DATE_TAG}"

    read -r -a SELECTOR_ARRAY <<< "${SELECTORS}"
    read -r -a BUDGET_ARRAY <<< "${SUBSET_PERCENTAGES}"

    NEEDS_FEATURE_CACHE=false
    for selector in "${SELECTOR_ARRAY[@]}"; do
        case "${selector}" in
            less|prismatic|safe)
                NEEDS_FEATURE_CACHE=true
                ;;
        esac
    done

    CACHE_JOB_ID=""
    if [[ "${NEEDS_FEATURE_CACHE}" == "true" ]]; then
        FEATURE_CACHE_READY=false
        if [[ -d "${FEATURE_CACHE_DIR}" ]] && find "${FEATURE_CACHE_DIR}" -maxdepth 1 -type f -name '*_selector_features.pt' -print -quit | grep -q .; then
            FEATURE_CACHE_READY=true
        fi

        if [[ "${FEATURE_CACHE_READY}" == "true" ]]; then
            echo "cache warmup job: skipped; existing selector feature cache found in ${FEATURE_CACHE_DIR}"
            echo "shared cache root: ${SHARED_CACHE_ROOT}"
        else
            CACHE_RUN_TAG="code_cachewarmup_${BASE_RUN_TAG}"
            CACHE_JOB_ID="$(
                submit_split_job \
                    "code_cache_${KR_TAG}_${KADD_TAG}" \
                    "" \
                    "${CACHE_JOB_TIME}" \
                    "${CACHE_JOB_MEM}" \
                    "${CACHE_JOB_CPUS}" \
                    "${CACHE_JOB_GPUS}" \
                    "RUN_TAG=${CACHE_RUN_TAG}" \
                    "OUTPUT_ROOT=${CLEAN_EXPERIMENTS_DIR}/outputs/${CACHE_RUN_TAG}" \
                    "SELECTORS=less" \
                    "SUBSET_PERCENTAGES=5" \
                    "SKIP_TRAINING=true" \
                    "RUN_BIAS_EVAL=false" \
                    "SUBMIT_SPLIT_JOBS=false"
            )"
            echo "cache warmup job: ${CACHE_JOB_ID}"
            echo "shared cache root: ${SHARED_CACHE_ROOT}"
        fi
    else
        echo "cache warmup job: skipped; selectors do not require shared gradient features"
    fi

    for selector in "${SELECTOR_ARRAY[@]}"; do
        SELECTOR_DEPENDENCY=""
        case "${selector}" in
            less|prismatic|safe)
                SELECTOR_DEPENDENCY="${CACHE_JOB_ID}"
                ;;
        esac

        if [[ "${SPLIT_BY_BUDGET}" == "true" ]]; then
            for budget in "${BUDGET_ARRAY[@]}"; do
                BUDGET_TAG="pct$(slugify "${budget}")"
                SELECTOR_RUN_TAG="code_${selector}_${BUDGET_TAG}_${BASE_RUN_TAG}"
                JOB_ID="$(
                    submit_split_job \
                        "code_${selector}_${BUDGET_TAG}" \
                        "${SELECTOR_DEPENDENCY}" \
                        "${SELECTOR_JOB_TIME}" \
                        "${SELECTOR_JOB_MEM}" \
                        "${SELECTOR_JOB_CPUS}" \
                        "${SELECTOR_JOB_GPUS}" \
                        "RUN_TAG=${SELECTOR_RUN_TAG}" \
                        "OUTPUT_ROOT=${CLEAN_EXPERIMENTS_DIR}/outputs/${SELECTOR_RUN_TAG}" \
                        "SELECTORS=${selector}" \
                        "SUBSET_PERCENTAGES=${budget}" \
                        "SKIP_TRAINING=false" \
                        "RUN_BIAS_EVAL=${RUN_BIAS_EVAL}" \
                        "SUBMIT_SPLIT_JOBS=false"
                )"
                echo "${selector} ${budget}% job: ${JOB_ID}"
            done
        else
            SELECTOR_RUN_TAG="code_${selector}_${BASE_RUN_TAG}"
            JOB_ID="$(
                submit_split_job \
                    "code_${selector}_${KR_TAG}_${KADD_TAG}" \
                    "${SELECTOR_DEPENDENCY}" \
                    "${SELECTOR_JOB_TIME}" \
                    "${SELECTOR_JOB_MEM}" \
                    "${SELECTOR_JOB_CPUS}" \
                    "${SELECTOR_JOB_GPUS}" \
                    "RUN_TAG=${SELECTOR_RUN_TAG}" \
                    "OUTPUT_ROOT=${CLEAN_EXPERIMENTS_DIR}/outputs/${SELECTOR_RUN_TAG}" \
                    "SELECTORS=${selector}" \
                    "SUBSET_PERCENTAGES=${SUBSET_PERCENTAGES}" \
                    "SKIP_TRAINING=false" \
                    "RUN_BIAS_EVAL=${RUN_BIAS_EVAL}" \
                    "SUBMIT_SPLIT_JOBS=false"
            )"
            echo "${selector} job: ${JOB_ID}"
        fi
    done
    exit 0
fi

mkdir -p "${CLEAN_EXPERIMENTS_DIR}/logs"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CACHE_ROOT}" "${FEATURE_CACHE_DIR}" "${TRAIN_CACHE_DIR}"
mkdir -p "${CACHE_ROOT}/candidate_pools" "${CACHE_ROOT}/fisher" "${CACHE_ROOT}/metadata"
cat > "${CACHE_ROOT}/metadata/run_settings.env" <<EOF
MODEL_NAME=${MODEL_NAME}
DATA_TAG=${DATA_TAG}
SELECTOR_FEATURE_METHOD=${SELECTOR_FEATURE_METHOD}
SELECTOR_PRECONDITIONER=${SELECTOR_PRECONDITIONER}
SELECTION_LORA_R=${SELECTION_LORA_R}
SELECTION_LORA_ALPHA=${SELECTION_LORA_ALPHA}
SELECTION_LORA_TARGET_MODULES=${SELECTION_LORA_TARGET_MODULES}
LOW_RANK_REFERENCE_RANK=${RESOLVED_LOW_RANK_REFERENCE_RANK}
LOW_RANK_TASK_RANK=${RESOLVED_LOW_RANK_TASK_RANK}
LOW_RANK_COMMON_RANK=${LOW_RANK_COMMON_RANK}
LOW_RANK_REFERENCE_SHRINKAGE=${LOW_RANK_REFERENCE_SHRINKAGE}
LOW_RANK_BUILDER_ALPHA=${LOW_RANK_BUILDER_ALPHA}
SEED=${SEED}
TARGET_SPLIT_SEED=${TARGET_SPLIT_SEED}
REFERENCE_SPLIT_SEED=${REFERENCE_SPLIT_SEED}
PRECONDITIONER_SPLIT_SEED=${PRECONDITIONER_SPLIT_SEED:-42}
SAFE_ALPHA=${RESOLVED_SAFE_ALPHA}
SAFE_COST_C=${SAFE_COST_C}
SAFE_EPSILON=${SAFE_EPSILON}
LEARNING_RATE=${LEARNING_RATE}
CACHE_ROOT=${CACHE_ROOT}
FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR}
TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}
EOF

cd "${MAIN_PROJECT_DIR}"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV:-rl-synth}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"
export MAIN_PROJECT_DIR
export CLEAN_PROJECT_DIR
export CLEAN_EXPERIMENTS_DIR
export MAIN_RUNNER
export TRAIN_SCRIPT
export CODE_DATA_DIR
export CANDIDATE_FILE
export TARGET_FILE
export OOD_EVAL_FILE
export REFERENCE_FILE
export MODEL_NAME
export RUN_TAG
export SEED
export OUTPUT_ROOT
export CACHE_ROOT
export FEATURE_CACHE_DIR
export TRAIN_CACHE_DIR
export BIAS_DATA_PATH
export BIAS_EVAL_FILE
export TARGET_SPLIT_PROPORTIONS
export TARGET_SPLIT_SEED
export REFERENCE_SPLIT_PROPORTIONS
export REFERENCE_SPLIT_SEED
export REFERENCE_SPLIT_GROUP_KEY
export BIAS_DOMAIN
export BIAS_LAYERS
export BIAS_ALPHA
export BIAS_GEOMETRY
export BIAS_DIRECTION_SPLIT
export BIAS_BATCH_SIZE
export BIAS_MAX_LENGTH
export BIAS_MAX_EXAMPLES
export BIAS_RUN_LIST

read -r -a SELECTION_MODULE_ARRAY <<< "${SELECTION_LORA_TARGET_MODULES}"
read -r -a TRAIN_MODULE_ARRAY <<< "${TRAIN_LORA_TARGET_MODULES}"

echo "hostname: $(hostname)"
echo "date:     $(date)"
echo "main dir: ${MAIN_PROJECT_DIR}"
echo "clean dir:${CLEAN_PROJECT_DIR}"
echo "runner:   ${MAIN_RUNNER}"
echo "trainer:  ${TRAIN_SCRIPT}"
echo "candidate:${CANDIDATE_FILE}"
echo "target:   ${TARGET_FILE}"
echo "ood eval: ${OOD_EVAL_FILE}"
if [[ -n "${REFERENCE_FILE}" ]]; then
    echo "reference:${REFERENCE_FILE}"
else
    echo "reference: hf stereoset ${REFERENCE_HF_SUBSET}/${REFERENCE_HF_SPLIT} label=${REFERENCE_HF_LABEL}"
fi
echo "run_tag:  ${RUN_TAG}"
echo "selectors:${SELECTORS}"
echo "budgets:  ${SUBSET_PERCENTAGES}"
echo "cache:    ${CACHE_ROOT}"
echo "shared:   ${SHARED_CACHE_ROOT}"
echo "features: method=${SELECTOR_FEATURE_METHOD}, preconditioner=${SELECTOR_PRECONDITIONER}, K_R=${RESOLVED_LOW_RANK_REFERENCE_RANK_TAG}, K_add=${RESOLVED_LOW_RANK_TASK_RANK_TAG}, auto_K_add=${RESOLVED_LOW_RANK_AUTO_TASK_RANK}"
echo "sft:      train_lora_r=${TRAIN_LORA_R}, train_lora_alpha=${TRAIN_LORA_ALPHA}, modules=${TRAIN_LORA_TARGET_MODULES}"
echo "bias:     data=${BIAS_DATA_PATH}, domain=${BIAS_DOMAIN}, max_examples=${BIAS_MAX_EXAMPLES}"
if [[ -n "${BENCHMARK_EVALS}" && "${BENCHMARK_EVALS}" != "none" ]]; then
    echo "bench:    ${BENCHMARK_EVALS}"
else
    echo "bench:    disabled"
fi
if [[ -n "${BIAS_EVAL_FILE}" ]]; then
    echo "bias eval:${BIAS_EVAL_FILE}"
fi
nvidia-smi

CMD=(
    python "${MAIN_RUNNER}"
    --candidate-file "${CANDIDATE_FILE}"
    --target-file "${TARGET_FILE}"
    --ood-eval-file "${OOD_EVAL_FILE}"
    --target-split-proportions ${TARGET_SPLIT_PROPORTIONS}
    --target-split-seed "${TARGET_SPLIT_SEED}"
    --reference-split-proportions ${REFERENCE_SPLIT_PROPORTIONS}
    --reference-split-seed "${REFERENCE_SPLIT_SEED}"
    --reference-split-group-key "${REFERENCE_SPLIT_GROUP_KEY}"
    --reference-bias-eval-output-file "${BIAS_DATA_PATH}"
    --selectors ${SELECTORS}
    --subset-percentages ${SUBSET_PERCENTAGES}
    --seed "${SEED}"
    --output-dir "${OUTPUT_ROOT}"
    --selection-output-dir "${OUTPUT_ROOT}/subsets"
    --training-output-dir "${OUTPUT_ROOT}/training_runs"
    --feature-cache-dir "${FEATURE_CACHE_DIR}"
    --train-cache-dir "${TRAIN_CACHE_DIR}"
    --train-script "${TRAIN_SCRIPT}"
    --model-name "${MODEL_NAME}"
    --device cuda
    --max-seq-len 1024
    --lora-r "${SELECTION_LORA_R}"
    --lora-alpha "${SELECTION_LORA_ALPHA}"
    --lora-dropout "${SELECTION_LORA_DROPOUT}"
    --lora-target-modules "${SELECTION_MODULE_ARRAY[@]}"
    --selector-feature-method "${SELECTOR_FEATURE_METHOD}"
    --selector-preconditioner "${SELECTOR_PRECONDITIONER}"
    --less-similarity "${LESS_SIMILARITY}"
    --adam-beta2 "${ADAM_BETA2}"
    --adam-eps "${ADAM_EPS}"
    --adam-warmup-steps "${ADAM_WARMUP_STEPS}"
    --preconditioner-split-seed "${PRECONDITIONER_SPLIT_SEED}"
    --selector-projection-dim "${SELECTOR_PROJECTION_DIM}"
    --selector-projection-seed "${SELECTOR_PROJECTION_SEED}"
    --selector-projection-chunk-size "${SELECTOR_PROJECTION_CHUNK_SIZE}"
    --selector-projection-cache-dtype "${SELECTOR_PROJECTION_CACHE_DTYPE}"
    --safe-learning-rate "${SAFE_LEARNING_RATE}"
    --safe-geometry "${SAFE_GEOMETRY}"
    --safe-solver "${SAFE_SOLVER}"
    --reference-fisher-max-examples "${REFERENCE_FISHER_MAX_EXAMPLES}"
    --low-rank-builder-alpha "${LOW_RANK_BUILDER_ALPHA}"
    --low-rank-delta "${LOW_RANK_DELTA}"
    --low-rank-reference-shrinkage "${LOW_RANK_REFERENCE_SHRINKAGE}"
    --train-max-seq-length "${TRAIN_MAX_SEQ_LENGTH}"
    --train-torch-dtype "${TRAIN_TORCH_DTYPE}"
    --train-lora-r "${TRAIN_LORA_R}"
    --train-lora-alpha "${TRAIN_LORA_ALPHA}"
    --train-lora-dropout "${TRAIN_LORA_DROPOUT}"
    --train-lora-target-modules "${TRAIN_MODULE_ARRAY[@]}"
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --learning-rate "${LEARNING_RATE}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-ratio "${WARMUP_RATIO}"
    --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
    --max-grad-norm "${MAX_GRAD_NORM}"
    --gradient-checkpointing
    --low-cpu-mem-usage
    --logging-steps "${LOGGING_STEPS}"
    --eval-steps "${EVAL_STEPS}"
    --evaluator "${EVALUATOR}"
    --generation-max-new-tokens "${GENERATION_MAX_NEW_TOKENS}"
    --humaneval-num-samples "${HUMANEVAL_NUM_SAMPLES}"
    --humaneval-pass-at-ks ${HUMANEVAL_PASS_AT_KS}
    --humaneval-temperature "${HUMANEVAL_TEMPERATURE}"
    --humaneval-top-p "${HUMANEVAL_TOP_P}"
    --compute-validation-gradient
    --validation-gradient-max-examples "${VALIDATION_GRADIENT_MAX_EXAMPLES}"
    --compute-reference-fisher
)

if [[ -n "${BENCHMARK_EVALS}" && "${BENCHMARK_EVALS}" != "none" ]]; then
    read -r -a BENCHMARK_EVAL_ARRAY <<< "${BENCHMARK_EVALS}"
    CMD+=(--benchmark-evals "${BENCHMARK_EVAL_ARRAY[@]}")
fi

if [[ -n "${REFERENCE_FILE}" ]]; then
    CMD+=(--reference-file "${REFERENCE_FILE}")
    if [[ -n "${BIAS_EVAL_FILE}" ]]; then
        CMD+=(--bias-eval-file "${BIAS_EVAL_FILE}")
    fi
else
    CMD+=(
        --reference-hf-stereoset
        --reference-hf-subset "${REFERENCE_HF_SUBSET}"
        --reference-hf-split "${REFERENCE_HF_SPLIT}"
        --reference-hf-label "${REFERENCE_HF_LABEL}"
        --reference-hf-format "${REFERENCE_HF_FORMAT}"
    )
fi

if [[ -n "${RESOLVED_LOW_RANK_REFERENCE_RANK}" ]]; then
    CMD+=(--low-rank-reference-rank "${RESOLVED_LOW_RANK_REFERENCE_RANK}")
elif [[ "${RESOLVED_LOW_RANK_REFERENCE_RANK_TAG}" == "auto" ]]; then
    CMD+=(--low-rank-reference-rank "auto")
fi
if [[ -n "${LOW_RANK_MAX_REFERENCE_RANK}" ]]; then
    CMD+=(--low-rank-max-reference-rank "${LOW_RANK_MAX_REFERENCE_RANK}")
fi
if [[ -n "${RESOLVED_LOW_RANK_TASK_RANK}" ]]; then
    CMD+=(--low-rank-task-rank "${RESOLVED_LOW_RANK_TASK_RANK}")
elif [[ "${RESOLVED_LOW_RANK_TASK_RANK_TAG}" == "auto" ]]; then
    CMD+=(--low-rank-task-rank "auto")
fi
if [[ -n "${LOW_RANK_COMMON_RANK}" ]]; then
    CMD+=(--low-rank-common-rank "${LOW_RANK_COMMON_RANK}")
fi
if [[ "${RESOLVED_LOW_RANK_AUTO_TASK_RANK}" == "true" ]]; then
    CMD+=(--low-rank-auto-task-rank)
fi
CMD+=(--low-rank-delta-task "${LOW_RANK_DELTA_TASK}")
if [[ -n "${LOW_RANK_MAX_TASK_RANK}" ]]; then
    CMD+=(--low-rank-max-task-rank "${LOW_RANK_MAX_TASK_RANK}")
fi
if [[ -n "${LOW_RANK_TARGET_MAX_EXAMPLES}" ]]; then
    CMD+=(--low-rank-target-max-examples "${LOW_RANK_TARGET_MAX_EXAMPLES}")
fi
if [[ -n "${LOW_RANK_CANDIDATE_MAX_EXAMPLES}" ]]; then
    CMD+=(--low-rank-candidate-max-examples "${LOW_RANK_CANDIDATE_MAX_EXAMPLES}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
    CMD+=(--max-steps "${MAX_STEPS}")
else
    CMD+=(--num-train-epochs "${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${EVAL_MAX_EXAMPLES}" ]]; then
    CMD+=(--eval-max-examples "${EVAL_MAX_EXAMPLES}")
fi
if [[ "${SKIP_TRAINING}" == "true" ]]; then
    CMD+=(--skip-training)
fi
if [[ "${SKIP_FINAL_EVALUATION}" == "true" ]]; then
    CMD+=(--skip-final-evaluation)
fi
if [[ "${OVERWRITE_SUBSETS}" == "true" ]]; then
    CMD+=(--overwrite-subsets)
fi
if [[ "${OVERWRITE_SELECTION_CACHE}" == "true" ]]; then
    CMD+=(--overwrite-selection-cache)
fi
if [[ "${SAVE_ALL_TARGET_FEATURES}" == "true" ]]; then
    CMD+=(--save-all-target-features)
fi

if [[ -n "${SAFE_SHORTLIST_SIZE}" ]]; then
    CMD+=(--safe-shortlist-size "${SAFE_SHORTLIST_SIZE}")
fi
if [[ "${SAFE_AVERAGE_BY_BUDGET}" == "false" ]]; then
    CMD+=(--no-safe-average-by-budget)
fi
if [[ -n "${SAFE_ALPHA}" ]]; then
    CMD+=(--safe-alpha "${SAFE_ALPHA}")
fi
if [[ -n "${SAFE_COST_C}" ]]; then
    CMD+=(--safe-cost-c "${SAFE_COST_C}")
fi
if [[ -n "${SAFE_EPSILON}" ]]; then
    CMD+=(--safe-epsilon "${SAFE_EPSILON}")
fi

"${CMD[@]}"

if [[ "${RUN_BIAS_EVAL}" != "true" ]]; then
    echo "[bias-eval] skipped. Set RUN_BIAS_EVAL=true to submit independent BDI/ICAT array jobs after SFT."
    echo "finished: $(date)"
    exit 0
fi

python - <<'PY'
import json
import os
from pathlib import Path

manifest_path = Path(os.environ["OUTPUT_ROOT"]) / "selector_sft_sweep_manifest.json"
run_list_path = Path(os.environ["BIAS_RUN_LIST"])
run_list_path.parent.mkdir(parents=True, exist_ok=True)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = []
for run in manifest.get("runs", []):
    run_dir = Path(run["run_output_dir"])
    adapter_dir = run_dir / "final_adapter"
    if not adapter_dir.exists():
        print(f"[bias-eval] skip {run_dir.name}: no final_adapter")
        continue
    rows.append(
        {
            "selector": run.get("selector"),
            "method_tag": run.get("method_tag"),
            "subset_percentage": run.get("subset_percentage"),
            "subset_budget": run.get("subset_budget"),
            "run_output_dir": str(run_dir),
            "adapter_dir": str(adapter_dir),
            "bias_output_dir": str(run_dir / "bias_disentangle_all_heldout"),
        }
    )

with run_list_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[bias-eval] wrote {len(rows)} adapter eval tasks to {run_list_path}")
PY

BIAS_NUM_RUNS="$(python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["BIAS_RUN_LIST"])
if not path.exists():
    print(0)
else:
    print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
)"

if [[ "${BIAS_NUM_RUNS}" -le 0 ]]; then
    echo "[bias-eval] no completed adapters found; no eval jobs submitted."
    echo "finished: $(date)"
    exit 0
fi

BIAS_ARRAY_MAX=$((BIAS_NUM_RUNS - 1))
BIAS_JOB_ID="$(
    sbatch \
        --parsable \
        -p "${BIAS_EVAL_PARTITION}" \
        --gres="gpu:${BIAS_EVAL_GPUS}" \
        --mem="${BIAS_EVAL_MEM}" \
        --cpus-per-task="${BIAS_EVAL_CPUS}" \
        -t "${BIAS_EVAL_TIME}" \
        --constraint="${BIAS_EVAL_CONSTRAINT}" \
        --array="0-${BIAS_ARRAY_MAX}" \
        --export=ALL,MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR}",MODEL_NAME="${MODEL_NAME}",BIAS_DATA_PATH="${BIAS_DATA_PATH}",BIAS_RUN_LIST="${BIAS_RUN_LIST}",BIAS_DOMAIN="${BIAS_DOMAIN}",BIAS_LAYERS="${BIAS_LAYERS}",BIAS_ALPHA="${BIAS_ALPHA}",BIAS_GEOMETRY="${BIAS_GEOMETRY}",BIAS_DIRECTION_SPLIT="${BIAS_DIRECTION_SPLIT}",BIAS_BATCH_SIZE="${BIAS_BATCH_SIZE}",BIAS_MAX_LENGTH="${BIAS_MAX_LENGTH}",BIAS_MAX_EXAMPLES="${BIAS_MAX_EXAMPLES}" \
        "${BIAS_EVAL_SCRIPT}"
)"
echo "[bias-eval] submitted ${BIAS_NUM_RUNS} independent BDI/ICAT eval jobs as array job ${BIAS_JOB_ID}"
echo "finished: $(date)"
