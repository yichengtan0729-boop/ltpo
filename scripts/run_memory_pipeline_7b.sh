#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-sc-likelihood-ratios}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-./artifacts/models/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-openai/gsm8k}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MEMORY_DATASET="${MEMORY_DATASET:-$DATASET}"
MEMORY_SPLIT="${MEMORY_SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
MEMORY_START="${MEMORY_START:-0}"
MEMORY_END="${MEMORY_END:--1}"
EVAL_START="${EVAL_START:-0}"
EVAL_END="${EVAL_END:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-auto}"
SOLVER_PROMPT_IDX="${SOLVER_PROMPT_IDX:-0}"
SEED="${SEED:-42}"
N_MEMORY_SAMPLES="${N_MEMORY_SAMPLES:-5}"
BUILD_MEMORY_TEMPERATURE="${BUILD_MEMORY_TEMPERATURE:-0.4}"
MEMORY_LTPO_TEMPERATURE="${MEMORY_LTPO_TEMPERATURE:-0.3}"
MEMORY_TOP_P="${MEMORY_TOP_P:-0.9}"
MIN_MEMORY_RELIABILITY="${MIN_MEMORY_RELIABILITY:-0.25}"
N_PROTOTYPES="${N_PROTOTYPES:-16}"
TOP_EXAMPLES_PER_PROTOTYPE="${TOP_EXAMPLES_PER_PROTOTYPE:-3}"
TOP_K_PROTOTYPES="${TOP_K_PROTOTYPES:-2}"
N_CANDIDATES="${N_CANDIDATES:-2}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

PYTHON_CMD=(conda run -n "$CONDA_ENV" python)

if [[ "$SKIP_DOWNLOAD" != "1" && ! -d "$MODEL_PATH" ]]; then
  mkdir -p "$(dirname "$MODEL_PATH")"
  MODEL_REPO="$MODEL_REPO" MODEL_PATH="$MODEL_PATH" "${PYTHON_CMD[@]}" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["MODEL_PATH"],
    resume_download=True,
)
PY
fi

MODEL_FOR_RUN="$MODEL_REPO"
if [[ -d "$MODEL_PATH" ]]; then
  MODEL_FOR_RUN="$MODEL_PATH"
fi

safe_name() {
  printf "%s" "$1" | sed -E 's#[\\/:"*?<>|[:space:]]+#_#g; s#^_+|_+$##g'
}

MODEL_TAG="$(safe_name "$(basename "$MODEL_FOR_RUN")")"
DATA_TAG="$(safe_name "$DATASET")"

MEMORY_DIR="$OUTPUT_DIR/memories"
PROTOTYPE_DIR="$OUTPUT_DIR/prototypes"
MEMORY_PATH="$MEMORY_DIR/$MODEL_TAG-$DATA_TAG-memory.jsonl"
PROTOTYPE_PATH="$PROTOTYPE_DIR/$MODEL_TAG-$DATA_TAG-prototypes.json"
VECTORIZER_PATH="${PROTOTYPE_PATH%.json}.vectorizer.pkl"

mkdir -p "$OUTPUT_DIR" "$MEMORY_DIR" "$PROTOTYPE_DIR"

"${PYTHON_CMD[@]}" main.py \
  --method build_memory \
  --dataset "$DATASET" \
  --dataset_split "$DATASET_SPLIT" \
  --memory_dataset "$MEMORY_DATASET" \
  --memory_split "$MEMORY_SPLIT" \
  --model_name_or_path "$MODEL_FOR_RUN" \
  --output_dir "$OUTPUT_DIR" \
  --memory_dir "$MEMORY_DIR" \
  --prototype_dir "$PROTOTYPE_DIR" \
  --memory_output_path "$MEMORY_PATH" \
  --start_data_idx "$MEMORY_START" \
  --end_data_idx "$MEMORY_END" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --solver_prompt_idx "$SOLVER_PROMPT_IDX" \
  --seed "$SEED" \
  --n_memory_samples "$N_MEMORY_SAMPLES" \
  --memory_temperature "$BUILD_MEMORY_TEMPERATURE" \
  --memory_top_p "$MEMORY_TOP_P" \
  --min_memory_reliability "$MIN_MEMORY_RELIABILITY" \
  --verbose 1

"${PYTHON_CMD[@]}" main.py \
  --method build_prototypes \
  --dataset "$DATASET" \
  --memory_dataset "$MEMORY_DATASET" \
  --model_name_or_path "$MODEL_FOR_RUN" \
  --output_dir "$OUTPUT_DIR" \
  --memory_dir "$MEMORY_DIR" \
  --prototype_dir "$PROTOTYPE_DIR" \
  --memory_output_path "$MEMORY_PATH" \
  --prototype_path "$PROTOTYPE_PATH" \
  --n_prototypes "$N_PROTOTYPES" \
  --top_examples_per_prototype "$TOP_EXAMPLES_PER_PROTOTYPE" \
  --seed "$SEED" \
  --verbose 1

"${PYTHON_CMD[@]}" main.py \
  --method memory_ltpo \
  --dataset "$DATASET" \
  --dataset_split "$DATASET_SPLIT" \
  --model_name_or_path "$MODEL_FOR_RUN" \
  --output_dir "$OUTPUT_DIR" \
  --memory_dir "$MEMORY_DIR" \
  --prototype_dir "$PROTOTYPE_DIR" \
  --prototype_path "$PROTOTYPE_PATH" \
  --vectorizer_path "$VECTORIZER_PATH" \
  --start_data_idx "$EVAL_START" \
  --end_data_idx "$EVAL_END" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --solver_prompt_idx "$SOLVER_PROMPT_IDX" \
  --seed "$SEED" \
  --top_k_prototypes "$TOP_K_PROTOTYPES" \
  --n_candidates "$N_CANDIDATES" \
  --memory_temperature "$MEMORY_LTPO_TEMPERATURE" \
  --memory_top_p "$MEMORY_TOP_P" \
  --verbose 1
