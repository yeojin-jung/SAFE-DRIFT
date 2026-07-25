#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH --array=0-2
#SBATCH -J safe_drift_s1_olmo13b
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --constraint='a100|h100|h200'
#SBATCH --exclude=l001

set -euo pipefail

# M3 model/scale ablation: identical to
# experiments/run_setting1_KR512_KR128_rho_sweep.sbatch except that the
# gradient/finetune backbone is OLMo-2-13B instead of OLMo-2-7B, only a single
# rho is run, and only the 5% budget is run. Everything else -- setting 1 data,
# KR=512/KT=128, epsilon=1.0, LoRA config, evaluator, three seeds -- is
# unchanged, so the two runs are directly comparable.
#
# Requires allenai/OLMo-2-1124-13B in the HF cache: jobs run with
# HF_HUB_OFFLINE=1 and will fail at model load otherwise. HF_HUB_CACHE points at
# the project mount because the home quota cannot hold the 13B checkpoint; the
# 7B model and the StereoSet dataset are symlinked into that cache root.
#
# Override variables at submission time with, for example:
#   sbatch --export=ALL,RHO_VALUES=3.0e-5 \
#     experiments/run_setting1_olmo13b_KR512_KR128_rho1e5.sbatch

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/net/projects2/mercury/yeojin/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/projects2/mercury/yeojin/clean_workspace2}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"

GRADIENT_MODEL="${GRADIENT_MODEL:-allenai/OLMo-2-1124-13B}"
FINETUNE_MODEL="${FINETUNE_MODEL:-allenai/OLMo-2-1124-13B}"

SEEDS=(42 52 62)
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0-$((${#SEEDS[@]} - 1))." >&2
  exit 2
fi
SEED="${SEEDS[${ARRAY_TASK_ID}]}"

if [[ -z "${RUN_TAG:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    RUN_TAG="setting1_olmo13b_kr512_kt128_rho1e5_${SLURM_ARRAY_JOB_ID}"
  else
    RUN_TAG="setting1_olmo13b_kr512_kt128_rho1e5_manual_$(date +%Y%m%d_%H%M%S)"
  fi
fi
OUTPUT_PARENT_ROOT="${OUTPUT_PARENT_ROOT:-${CLEAN_PROJECT_DIR}/experiments/outputs/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_PARENT_ROOT}/seed${SEED}}"
RUN_CONFIG="${RUN_CONFIG:-${OUTPUT_ROOT}/setting_1_olmo13b_kr512_kt128_rho1e5.yaml}"

RHO_VALUES="${RHO_VALUES:-1.0e-5}"
SUBSET_PERCENTAGES="${SUBSET_PERCENTAGES:-5.0}"
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
export HF_HUB_CACHE="${HF_HUB_CACHE:-/net/projects2/mercury/yeojin/hf_cache/hub}"
# Legacy alias: utils.data_construction._hf_cache_hub_roots() (the offline
# StereoSet parquet fallback) reads HUGGINGFACE_HUB_CACHE/HF_HOME but not
# HF_HUB_CACHE, so set both to keep the model and dataset lookups consistent.
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
# OLMo-2-13B is loaded in float32 for gradient extraction (~51 GiB of weights);
# with backward activations that leaves ~13 GiB of headroom on an 80 GiB card,
# so avoid losing any of it to allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Fail fast with a clear message rather than deep inside transformers.
MODEL_CACHE_DIR="${HF_HUB_CACHE}/models--${GRADIENT_MODEL//\//--}"
if [[ ! -d "${MODEL_CACHE_DIR}" ]]; then
  echo "Model not found in offline HF cache: ${MODEL_CACHE_DIR}" >&2
  echo "Pre-fetch it before submitting (HF_HUB_OFFLINE=1 is set for this job)." >&2
  exit 3
fi

cat > "${RUN_CONFIG}" <<EOF
experiment:
  id: setting_1_codealpaca_humaneval_stereoset_olmo13b_kr512_kt128_rho1e5
  name: "Setting 1: CodeAlpaca / HumanEval / StereoSet (OLMo-2-13B, KR512 KT128, rho 1e-5)"

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
subset_percentages: [${SUBSET_PERCENTAGES}]

model_variants:
  - name: olmo13b_gradient_equals_ft
    gradient_model: ${GRADIENT_MODEL}
    finetune_model: ${FINETUNE_MODEL}
    enabled: true

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
echo "grad model:  ${GRADIENT_MODEL}"
echo "ft model:    ${FINETUNE_MODEL}"
echo "rho values:  [${RHO_VALUES}]"
echo "subset pct:  [${SUBSET_PERCENTAGES}]"
echo "hf cache:    ${HF_HUB_CACHE}"
echo "epsilon:     1.0"
echo "rank:        KR=512, KT=128"
echo "train dtype: ${TRAIN_TORCH_DTYPE}"
nvidia-smi

python scripts/run_experiment_pipeline.py \
  --config "${RUN_CONFIG}" \
  --execute \
  --print-commands

echo "finished: $(date)"
