#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -J safe_drift_s1_rhos
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --constraint='a100|h100|h200'
#SBATCH --exclude=l001

set -euo pipefail

# Setting 1 KR=512, KT=128 SAFE-only rho sweep, repeated over three seeds.
# Override variables at submission time with, for example:
#   sbatch --export=ALL,RHO_VALUES=1.0e-4 \
#     experiments/run_setting1_KR512_KR128_rho_sweep.sbatch

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/net/projects2/mercury/yeojin/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/projects2/mercury/yeojin/clean_workspace2}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"

SEEDS=(42 52 62)
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0-$((${#SEEDS[@]} - 1))." >&2
  exit 2
fi
SEED="${SEEDS[${ARRAY_TASK_ID}]}"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    RUN_TAG="setting1_kr512_kt128_rho_sweep_${SLURM_ARRAY_JOB_ID}"
  else
    RUN_TAG="setting1_kr512_kt128_rho_sweep_manual_$(date +%Y%m%d_%H%M%S)"
  fi
fi
OUTPUT_PARENT_ROOT="${OUTPUT_PARENT_ROOT:-${CLEAN_PROJECT_DIR}/experiments/outputs/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_PARENT_ROOT}/seed${SEED}}"
RUN_CONFIG="${RUN_CONFIG:-${OUTPUT_ROOT}/setting_1_kr512_kt128_rho_sweep.yaml}"

RHO_VALUES="${RHO_VALUES:-1.0e-5, 5.0e-5, 1.0e-4, 5.0e-4}"
TRAIN_TORCH_DTYPE="${TRAIN_TORCH_DTYPE:-bf16}"
RUN_ENTANGLEMENT_ANALYSIS="${RUN_ENTANGLEMENT_ANALYSIS:-true}"
ENTANGLEMENT_SELECTOR_TOP_K="${ENTANGLEMENT_SELECTOR_TOP_K:-200}"

mkdir -p "${MAIN_PROJECT_DIR}/logs" "${OUTPUT_ROOT}"
cd "${MAIN_PROJECT_DIR}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cat > "${RUN_CONFIG}" <<EOF
experiment:
  id: setting_1_codealpaca_humaneval_stereoset_kr512_kt128_rho_sweep
  name: "Setting 1: CodeAlpaca / HumanEval / StereoSet (KR512 KT128 rho sweep)"

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
seed: ${SEED}
repeat_seeds: [${SEED}]
run_baselines: false

model_variants:
  - name: olmo7b_gradient_equals_ft
    gradient_model: allenai/OLMo-2-1124-7B
    finetune_model: allenai/OLMo-2-1124-7B
    enabled: true
  - name: olmo7b_to_instruct_transfer
    gradient_model: allenai/OLMo-2-1124-7B
    finetune_model: allenai/OLMo-2-1124-7B-Instruct
    enabled: false

rank_grid:
  - name: KR512_KT128
    K_R: 512
    K_T: 128

safe:
  epsilon: 1.0
  rho_values: [${RHO_VALUES}]

selection:
  feature_cache_dir: ${OUTPUT_PARENT_ROOT}/shared_selector_feature_cache/seed${SEED}/selector_feature_cache

training:
  train_torch_dtype: ${TRAIN_TORCH_DTYPE}
  train_lora_r: 16
  train_lora_alpha: 16
  train_lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

evaluation:
  evaluator: humaneval
  humaneval_num_samples: 10
  humaneval_pass_at_ks: [1, 10]
  humaneval_temperature: 0.8
  humaneval_top_p: 0.95
  run_entanglement_analysis: ${RUN_ENTANGLEMENT_ANALYSIS}
  entanglement_selector_top_k: ${ENTANGLEMENT_SELECTOR_TOP_K}
EOF

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "project:     ${MAIN_PROJECT_DIR}"
echo "array task:  ${ARRAY_TASK_ID}"
echo "seed:        ${SEED}"
echo "config:      ${RUN_CONFIG}"
echo "output root: ${OUTPUT_ROOT}"
echo "rho values:  [${RHO_VALUES}]"
echo "epsilon:     1.0"
echo "rank:        KR=512, KT=128"
echo "train dtype: ${TRAIN_TORCH_DTYPE}"
nvidia-smi

python scripts/run_experiment_pipeline.py \
  --config "${RUN_CONFIG}" \
  --execute \
  --print-commands

echo "finished: $(date)"
