#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -J safe_drift_s1_test
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --constraint='a100|h100|h200'

set -euo pipefail

# Single-command smoke run for Setting 1.
# Override any variable at submission time, for example:
#   sbatch --export=ALL,MAX_STEPS=20,SAFE_RHO=0.3 experiments/run_setting1_single_test.sbatch

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/net/projects2/mercury/yeojin/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/projects2/mercury/yeojin/clean_workspace2}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"

RUN_TAG="${RUN_TAG:-setting1_single_test_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_PROJECT_DIR}/experiments/outputs/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-${MAIN_PROJECT_DIR}/logs}"
TEST_CONFIG="${TEST_CONFIG:-${OUTPUT_ROOT}/setting1_single_test.yaml}"

MODEL_VARIANT_NAME="${MODEL_VARIANT_NAME:-olmo7b_gradient_equals_ft}"
GRADIENT_MODEL="${GRADIENT_MODEL:-allenai/OLMo-2-1124-7B}"
FINETUNE_MODEL="${FINETUNE_MODEL:-allenai/OLMo-2-1124-7B}"

SUBSET_PERCENTAGE="${SUBSET_PERCENTAGE:-5.0}"
SAFE_RHO="${SAFE_RHO:-1e-04}"
SAFE_EPSILON="${SAFE_EPSILON:-1.0}"
RUN_BASELINES="${RUN_BASELINES:-false}"

LOW_RANK_REFERENCE_RANK="${LOW_RANK_REFERENCE_RANK:-512}"
LOW_RANK_TASK_RANK="${LOW_RANK_TASK_RANK:-128}"
LOW_RANK_CANDIDATE_MAX_EXAMPLES="${LOW_RANK_CANDIDATE_MAX_EXAMPLES:-1000}"
REFERENCE_FISHER_MAX_EXAMPLES="${REFERENCE_FISHER_MAX_EXAMPLES:-512}"
VALIDATION_GRADIENT_MAX_EXAMPLES="${VALIDATION_GRADIENT_MAX_EXAMPLES:-128}"

TRAIN_TORCH_DTYPE="${TRAIN_TORCH_DTYPE:-bf16}"
MAX_STEPS="${MAX_STEPS:-10}"
EVAL_MAX_EXAMPLES="${EVAL_MAX_EXAMPLES:-8}"
GENERATION_MAX_NEW_TOKENS="${GENERATION_MAX_NEW_TOKENS:-64}"
HUMANEVAL_NUM_SAMPLES="${HUMANEVAL_NUM_SAMPLES:-10}"
HUMANEVAL_PASS_AT_KS="${HUMANEVAL_PASS_AT_KS:-[1, 10]}"

RUN_ENTANGLEMENT_ANALYSIS="${RUN_ENTANGLEMENT_ANALYSIS:-true}"
ENTANGLEMENT_SELECTOR_TOP_K="${ENTANGLEMENT_SELECTOR_TOP_K:-200}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

cd "${MAIN_PROJECT_DIR}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"

cat > "${TEST_CONFIG}" <<EOF
experiment:
  id: setting_1_single_test_${RUN_TAG}
  name: "Setting 1 single test run"

data:
  candidate_file: data/setting_1_code/codealpaca.jsonl
  target_file: data/setting_1_code/humaneval.jsonl
  ood_eval_file: data/setting_1_code/humaneval.jsonl
  reference_hf_stereoset: true
  reference_hf_subset: intrasentence
  reference_hf_split: validation
  reference_hf_label: stereotype,anti-stereotype
  reference_hf_format: instruction

output_root: ${OUTPUT_ROOT}
run_baselines: ${RUN_BASELINES}
subset_percentages: [${SUBSET_PERCENTAGE}]
seed: 42

model_variants:
  - name: ${MODEL_VARIANT_NAME}
    gradient_model: ${GRADIENT_MODEL}
    finetune_model: ${FINETUNE_MODEL}
    enabled: true

rank_grid:
  - name: KR${LOW_RANK_REFERENCE_RANK}_KT${LOW_RANK_TASK_RANK}
    K_R: ${LOW_RANK_REFERENCE_RANK}
    K_T: ${LOW_RANK_TASK_RANK}

safe:
  epsilon: ${SAFE_EPSILON}
  rho_values: [${SAFE_RHO}]

selection:
  lora_r: 2
  lora_alpha: 4
  lora_dropout: 0.0
  lora_target_modules: ["q_proj"]
  max_seq_len: 1024
  selector_feature_method: low_rank
  selector_preconditioner: adam
  adam_warmup_steps: 0.03
  reference_fisher_max_examples: ${REFERENCE_FISHER_MAX_EXAMPLES}
  validation_gradient_max_examples: ${VALIDATION_GRADIENT_MAX_EXAMPLES}
  low_rank_candidate_max_examples: ${LOW_RANK_CANDIDATE_MAX_EXAMPLES}
  low_rank_candidate_projection_chunk_rows: 256
  safe_geometry: reference
  safe_solver: greedy_marginal
  safe_learning_rate: 1.0

training:
  train_torch_dtype: ${TRAIN_TORCH_DTYPE}
  train_max_seq_length: 1024
  train_lora_r: 16
  train_lora_alpha: 16
  train_lora_dropout: 0.05
  train_lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-4
  lr_scheduler_type: linear
  max_grad_norm: 1.0
  max_steps: ${MAX_STEPS}
  eval_steps: 5
  logging_steps: 1

evaluation:
  evaluator: humaneval
  eval_max_examples: ${EVAL_MAX_EXAMPLES}
  generation_max_new_tokens: ${GENERATION_MAX_NEW_TOKENS}
  humaneval_num_samples: ${HUMANEVAL_NUM_SAMPLES}
  humaneval_pass_at_ks: ${HUMANEVAL_PASS_AT_KS}
  humaneval_temperature: 0.8
  humaneval_top_p: 0.95
  run_entanglement_analysis: ${RUN_ENTANGLEMENT_ANALYSIS}
  entanglement_selector_top_k: ${ENTANGLEMENT_SELECTOR_TOP_K}
EOF

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "project:     ${MAIN_PROJECT_DIR}"
echo "config:      ${TEST_CONFIG}"
echo "output root: ${OUTPUT_ROOT}"
echo "model:       ${GRADIENT_MODEL} -> ${FINETUNE_MODEL}"
echo "rank:        KR=${LOW_RANK_REFERENCE_RANK}, KT=${LOW_RANK_TASK_RANK}"
echo "subset/rho:  ${SUBSET_PERCENTAGE}% / ${SAFE_RHO}"
echo "max steps:   ${MAX_STEPS}"
nvidia-smi

python scripts/run_experiment_pipeline.py \
  --config "${TEST_CONFIG}" \
  --execute \
  --print-commands

echo "finished: $(date)"
