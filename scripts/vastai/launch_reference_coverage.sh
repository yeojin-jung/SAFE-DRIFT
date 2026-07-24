#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/vastai/launch_reference_coverage.sh --ssh root@HOST --port PORT [options]

Launch the Setting 2 reference-coverage sweep on an existing Vast.ai instance.

Required:
  --ssh USER@HOST          SSH target from the Vast instance page.
  --port PORT             External SSH port from the Vast instance page.

Options:
  --identity PATH         SSH private key. Defaults to ~/.ssh/id_ed25519.
  --remote-root PATH      Remote experiment root. Defaults to /workspace/safe-drift.
  --gpus N                Concurrent commands to run. Defaults to detected GPU count.
  --experiment-id ID      Remote dispatch directory tag. Defaults to timestamped run.
  --foreground            Run in the SSH session instead of nohup/background.
  --skip-sync             Do not rsync the local repo before launching.
  --no-venv               Use the remote Python environment directly.
  -h, --help              Show this help.

Before running, create/start a Vast instance with SSH enabled. If the OLMo model
or datasets require Hugging Face authentication, set HF_TOKEN on the Vast
instance before launching.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SSH_TARGET=""
SSH_PORT=""
IDENTITY="${HOME}/.ssh/id_ed25519"
REMOTE_ROOT="/workspace/safe-drift"
GPU_COUNT="auto"
EXPERIMENT_ID="reference_coverage_$(date +%Y%m%d_%H%M%S)"
FOREGROUND=0
SKIP_SYNC=0
USE_VENV=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh)
      SSH_TARGET="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --identity)
      IDENTITY="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --gpus)
      GPU_COUNT="$2"
      shift 2
      ;;
    --experiment-id)
      EXPERIMENT_ID="$2"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    --no-venv)
      USE_VENV=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${SSH_TARGET}" || -z "${SSH_PORT}" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "${IDENTITY}" ]]; then
  echo "SSH identity not found: ${IDENTITY}" >&2
  exit 2
fi

SSH_OPTS=(
  -i "${IDENTITY}"
  -p "${SSH_PORT}"
  -o IdentitiesOnly=yes
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=20
)

REMOTE_REPO="${REMOTE_ROOT}/repo"
REMOTE_SCRIPT="${REMOTE_ROOT}/run_reference_coverage_remote.sh"
REMOTE_LOG="${REMOTE_ROOT}/logs/${EXPERIMENT_ID}.log"

echo "[vast-launch] repo=${REPO_ROOT}"
echo "[vast-launch] target=${SSH_TARGET}:${SSH_PORT}"
echo "[vast-launch] remote_root=${REMOTE_ROOT}"

ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "mkdir -p '${REMOTE_REPO}' '${REMOTE_ROOT}/logs'"

if [[ "${SKIP_SYNC}" -eq 0 ]]; then
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude 'outputs/' \
    --exclude 'analysis_outputs/' \
    --exclude 'wandb/' \
    -e "ssh ${SSH_OPTS[*]}" \
    "${REPO_ROOT}/" \
    "${SSH_TARGET}:${REMOTE_REPO}/"
fi

rsync -az \
  -e "ssh ${SSH_OPTS[*]}" \
  "${SCRIPT_DIR}/run_reference_coverage_remote.sh" \
  "${SSH_TARGET}:${REMOTE_SCRIPT}"

ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "chmod +x '${REMOTE_SCRIPT}'"

REMOTE_ENV=(
  "REMOTE_ROOT='${REMOTE_ROOT}'"
  "REMOTE_REPO='${REMOTE_REPO}'"
  "EXPERIMENT_ID='${EXPERIMENT_ID}'"
  "GPU_COUNT='${GPU_COUNT}'"
  "USE_VENV='${USE_VENV}'"
)

REMOTE_CMD="cd '${REMOTE_REPO}' && ${REMOTE_ENV[*]} '${REMOTE_SCRIPT}'"
if [[ "${FOREGROUND}" -eq 1 ]]; then
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" "${REMOTE_CMD}"
else
  ssh "${SSH_OPTS[@]}" "${SSH_TARGET}" \
    "mkdir -p '$(dirname "${REMOTE_LOG}")' && nohup bash -lc ${REMOTE_CMD@Q} > '${REMOTE_LOG}' 2>&1 < /dev/null & echo \$!"
  echo "[vast-launch] started in background"
  echo "[vast-launch] log: ssh -i ${IDENTITY} -p ${SSH_PORT} ${SSH_TARGET} \"tail -f '${REMOTE_LOG}'\""
  echo "[vast-launch] outputs: ${REMOTE_ROOT}/outputs/setting_2_medqa_medmcqa_reference_coverage_v1"
fi
