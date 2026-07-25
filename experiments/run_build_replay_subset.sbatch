#!/bin/bash
#SBATCH --partition=general
#SBATCH --mem=16G
#SBATCH -J build_replay_subset
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# CPU-only job: mixes an existing LESS subset with a random reference/replay
# sample. Override any variable at submission time, for example:
#   sbatch --export=ALL,LESS_SUBSET_FILE=...,OUTPUT_FILE=...,K=80 \
#       experiments/run_build_replay_subset.sbatch



LESS_SUBSET_FILE="${LESS_SUBSET_FILE:?Set LESS_SUBSET_FILE to an existing LESS subset jsonl (e.g. from run_selector_sft_sweep_code.py).}"
OUTPUT_FILE="${OUTPUT_FILE:?Set OUTPUT_FILE for the combined replay subset jsonl.}"

# Reference/replay pool: either a local file, or fall back to HF StereoSet
# (setting 1's reference source, same as run_selector_sft_sweep_code.py's
# --reference-hf-stereoset).
REFERENCE_FILE="${REFERENCE_FILE:-}"
REFERENCE_HF_STEREOSET="${REFERENCE_HF_STEREOSET:-true}"
REFERENCE_HF_SUBSET="${REFERENCE_HF_SUBSET:-intrasentence}"
REFERENCE_HF_SPLIT="${REFERENCE_HF_SPLIT:-validation}"
REFERENCE_HF_LABEL="${REFERENCE_HF_LABEL:-stereotype,anti-stereotype}"
REFERENCE_HF_FORMAT="${REFERENCE_HF_FORMAT:-instruction}"

K="${K:-80}"
SEED="${SEED:-42}"
TOTAL_BUDGET="${TOTAL_BUDGET:-}"
DEDUPE_KEY="${DEDUPE_KEY:-id}"

cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets_v2}"
export TOKENIZERS_PARALLELISM=false

CMD=(
    python src/utils/build_replay_subset.py
    --less-subset-file "$LESS_SUBSET_FILE"
    --output-file "$OUTPUT_FILE"
    --k "$K"
    --seed "$SEED"
    --dedupe-key "$DEDUPE_KEY"
)

if [[ -n "$REFERENCE_FILE" ]]; then
    CMD+=(--reference-file "$REFERENCE_FILE")
elif [[ "$REFERENCE_HF_STEREOSET" == "true" ]]; then
    CMD+=(
        --reference-hf-stereoset
        --reference-hf-subset "$REFERENCE_HF_SUBSET"
        --reference-hf-split "$REFERENCE_HF_SPLIT"
        --reference-hf-label "$REFERENCE_HF_LABEL"
        --reference-hf-format "$REFERENCE_HF_FORMAT"
    )
else
    echo "Set REFERENCE_FILE or leave REFERENCE_HF_STEREOSET=true." >&2
    exit 1
fi

if [[ -n "$TOTAL_BUDGET" ]]; then
    CMD+=(--total-budget "$TOTAL_BUDGET")
fi

echo "hostname:    $(hostname)"
echo "date:        $(date)"
echo "less subset: $LESS_SUBSET_FILE"
echo "output:      $OUTPUT_FILE"
echo "k:           $K"
echo "seed:        $SEED"
if [[ -n "$REFERENCE_FILE" ]]; then
    echo "reference:   $REFERENCE_FILE"
else
    echo "reference:   hf stereoset $REFERENCE_HF_SUBSET/$REFERENCE_HF_SPLIT label=$REFERENCE_HF_LABEL"
fi

"${CMD[@]}"

echo "finished: $(date)"
