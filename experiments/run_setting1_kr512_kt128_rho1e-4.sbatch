#!/bin/bash
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -J safe_drift_s1_rho1e4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --constraint='a100|h100|h200'

set -euo pipefail

# Full Setting 1 run with the standard configs, restricted to:
#   rho = 1e-04, epsilon = 1.0
#   K_R = 512, K_T = 128

MAIN_PROJECT_DIR="${MAIN_PROJECT_DIR:-/net/projects2/mercury/yeojin/SAFE-DRIFT}"
CLEAN_PROJECT_DIR="${CLEAN_PROJECT_DIR:-/net/projects2/mercury/yeojin/clean_workspace2}"
CONDA_ENV="${CONDA_ENV:-rl-synth}"

RUN_TAG="${RUN_TAG:-setting1_kr512_kt128_rho1e-4_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CLEAN_PROJECT_DIR}/experiments/outputs/${RUN_TAG}}"
RUN_CONFIG="${RUN_CONFIG:-${OUTPUT_ROOT}/setting_1_kr512_kt128_rho1e-4.yaml}"

mkdir -p "${OUTPUT_ROOT}"

cd "${MAIN_PROJECT_DIR}"

export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"
eval "$(~/bin/micromamba shell hook -s bash)"
set +u
micromamba activate "${CONDA_ENV}"
set -u

export PYTHONPATH="${MAIN_PROJECT_DIR}:${PYTHONPATH:-}"

cat > "${RUN_CONFIG}" <<EOF
experiment:
  id: setting_1_codealpaca_humaneval_stereoset_kr512_kt128_rho1e-4
  name: "Setting 1: CodeAlpaca / HumanEval / StereoSet (KR512 KT128 rho=1e-4)"

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

model_variants:
  - name: olmo7b_gradient_equals_ft
    gradient_model: allenai/OLMo-2-1124-7B
    finetune_model: allenai/OLMo-2-1124-7B
    enabled: true
  - name: olmo7b_to_instruct_transfer
    gradient_model: allenai/OLMo-2-1124-7B
    finetune_model: allenai/OLMo-2-1124-7B-Instruct
    enabled: true

rank_grid:
  - name: KR512_KT128
    K_R: 512
    K_T: 128

safe:
  epsilon: 1.0
  rho_values: [1.0e-4]

training:
  train_lora_r: 16
  train_lora_alpha: 16
  train_lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

evaluation:
  evaluator: humaneval
  humaneval_num_samples: 10
  humaneval_pass_at_ks: [1, 10]
  humaneval_temperature: 0.8
  humaneval_top_p: 0.95
EOF

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "project:     ${MAIN_PROJECT_DIR}"
echo "config:      ${RUN_CONFIG}"
echo "output root: ${OUTPUT_ROOT}"
echo "rho/eps:     1e-04 / 1.0"
echo "rank:        KR=512, KT=128"
nvidia-smi

python scripts/run_experiment_pipeline.py \
  --config "${RUN_CONFIG}" \
  --execute \
  --print-commands

echo "finished: $(date)"
