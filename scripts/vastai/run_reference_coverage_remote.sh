#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/workspace/safe-drift}"
REMOTE_REPO="${REMOTE_REPO:-${REMOTE_ROOT}/repo}"
EXPERIMENT_ID="${EXPERIMENT_ID:-reference_coverage_$(date +%Y%m%d_%H%M%S)}"
GPU_COUNT="${GPU_COUNT:-auto}"
USE_VENV="${USE_VENV:-1}"

CONFIG_SOURCE="${REMOTE_REPO}/configs/experiments/setting_2_medqa_medmcqa_reference_coverage.yaml"
CONFIG_REMOTE="${REMOTE_ROOT}/configs/setting_2_medqa_medmcqa_reference_coverage_vast.yaml"
DISPATCH_DIR="${REMOTE_ROOT}/dispatch/${EXPERIMENT_ID}"
ARRAY_DIR="${DISPATCH_DIR}/arrays"
MANIFEST="${DISPATCH_DIR}/pipeline_manifest.json"
STATE_DIR="${DISPATCH_DIR}/state"

export USE_TF=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${REMOTE_ROOT}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export MPLCONFIGDIR="${REMOTE_ROOT}/mplconfig"
export PYTHONPATH="${REMOTE_REPO}/src:${REMOTE_REPO}:${PYTHONPATH:-}"

mkdir -p "${REMOTE_ROOT}/configs" "${DISPATCH_DIR}" "${ARRAY_DIR}" "${STATE_DIR}" \
  "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${MPLCONFIGDIR}"

cd "${REMOTE_REPO}"

if [[ "${USE_VENV}" == "1" ]]; then
  if [[ ! -d .venv ]]; then
    python -m venv --system-site-packages .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/vast-train.txt

python - <<'PY'
import torch
print("[vast-remote] torch", torch.__version__, "cuda", torch.version.cuda)
print("[vast-remote] cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
PY

sed "s#/lambda/nfs/mem/safe-drift#${REMOTE_ROOT}#g" "${CONFIG_SOURCE}" > "${CONFIG_REMOTE}"

python scripts/run_experiment_pipeline.py \
  --config "${CONFIG_REMOTE}" \
  --manifest-out "${MANIFEST}"

python scripts/cluster/make_rebuttal_command_arrays.py \
  --manifest "${MANIFEST}" \
  --out-dir "${ARRAY_DIR}" \
  --project-dir "${REMOTE_REPO}" \
  --shared-cache-root "${REMOTE_ROOT}/shared_cache"

detect_gpu_count() {
  if [[ "${GPU_COUNT}" != "auto" ]]; then
    echo "${GPU_COUNT}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' '
  else
    echo "1"
  fi
}

jsonl_count() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "0"
    return
  fi
  python - "$file" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
}

run_jsonl_sequential() {
  local file="$1"
  local count
  count="$(jsonl_count "${file}")"
  for ((task_id=0; task_id<count; task_id++)); do
    python scripts/cluster/run_array_command.py \
      --commands-file "${file}" \
      --task-id "${task_id}"
  done
}

run_jsonl_gpu_batches() {
  local file="$1"
  local batch_size="$2"
  local count
  local batches
  count="$(jsonl_count "${file}")"
  if [[ "${count}" -eq 0 ]]; then
    return
  fi
  batches=$(( (count + batch_size - 1) / batch_size ))
  for ((task_id=0; task_id<batches; task_id++)); do
    python scripts/cluster/run_array_command.py \
      --commands-file "${file}" \
      --task-id "${task_id}" \
      --batch-size "${batch_size}" \
      --parallel
  done
}

GPUS="$(detect_gpu_count)"
if [[ "${GPUS}" -lt 1 ]]; then
  GPUS=1
fi
echo "[vast-remote] using ${GPUS} concurrent GPU command(s)"

echo "[vast-remote] preparing datasets"
run_jsonl_sequential "${ARRAY_DIR}/all_prepare_commands.jsonl"

echo "[vast-remote] preparing shared selector caches"
run_jsonl_gpu_batches "${ARRAY_DIR}/all_cache_prep_commands.jsonl" "${GPUS}"

echo "[vast-remote] running sweeps"
run_jsonl_gpu_batches "${ARRAY_DIR}/all_run_commands.jsonl" "${GPUS}"

echo "[vast-remote] collecting reference coverage results"
python scripts/collect_reference_coverage_results.py \
  --runs-root "${REMOTE_ROOT}/outputs/setting_2_medqa_medmcqa_reference_coverage_v1" \
  --out "${DISPATCH_DIR}/reference_coverage_results.csv"

echo "[vast-remote] complete"
echo "[vast-remote] manifest: ${MANIFEST}"
echo "[vast-remote] arrays: ${ARRAY_DIR}"
echo "[vast-remote] results: ${DISPATCH_DIR}/reference_coverage_results.csv"
