#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./artifacts/models/Qwen2.5-7B-Instruct}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
DATASET="${DATASET:-openai/gsm8k}"
MEMORY_SPLIT="${MEMORY_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
START_DATA_IDX="${START_DATA_IDX:-0}"
END_DATA_IDX="${END_DATA_IDX:--1}"
DISABLE_STEP_DECODER="${DISABLE_STEP_DECODER:-0}"
DISABLE_FAILURE_PENALTY="${DISABLE_FAILURE_PENALTY:-0}"
PYTHON="${PYTHON:-python}"

if [[ "$SKIP_DOWNLOAD" != "1" && ! -d "$MODEL_NAME_OR_PATH" ]]; then
  mkdir -p "$(dirname "$MODEL_NAME_OR_PATH")"
  huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_NAME_OR_PATH"
fi

safe_name() {
  local text="${1//\\//}"
  text="${text%/}"
  text="${text##*/}"
  text="${text//:/_}"
  printf "%s" "$text"
}

MODEL_TAG="$(safe_name "$MODEL_NAME_OR_PATH")"
DATA_TAG="$(safe_name "$DATASET")"

STEP_MEMORY_DIR="$OUTPUT_DIR/step_memories"
STEP_PROTOTYPE_DIR="$OUTPUT_DIR/step_prototypes"
STEP_DECODER_DIR="$OUTPUT_DIR/step_decoders"
STEP_MEMORY_PATH="$STEP_MEMORY_DIR/$MODEL_TAG-$DATA_TAG-step-memory.jsonl"
STEP_PROTOTYPE_PATH="$STEP_PROTOTYPE_DIR/$MODEL_TAG-$DATA_TAG-step-prototypes.json"
STEP_VECTORIZER_PATH="${STEP_PROTOTYPE_PATH%.json}.step.vectorizer.pkl"
STEP_DECODER_PATH="$STEP_DECODER_DIR/$MODEL_TAG-$DATA_TAG-step-decoder.pt"

mkdir -p "$OUTPUT_DIR" "$STEP_MEMORY_DIR" "$STEP_PROTOTYPE_DIR" "$STEP_DECODER_DIR"

"$PYTHON" main.py \
  --method build_step_memory \
  --dataset "$DATASET" \
  --memory_dataset "$DATASET" \
  --memory_split "$MEMORY_SPLIT" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --step_memory_dir "$STEP_MEMORY_DIR" \
  --step_prototype_dir "$STEP_PROTOTYPE_DIR" \
  --step_decoder_dir "$STEP_DECODER_DIR" \
  --step_memory_output_path "$STEP_MEMORY_PATH" \
  --start_data_idx "$START_DATA_IDX" \
  --end_data_idx "$END_DATA_IDX" \
  --num_thought_tokens 4 \
  --max_num_steps 5 \
  --lr 0.03 \
  --sigma 0.05 \
  --sigma_decay 0.99 \
  --step_conf_weight 0.4 \
  --step_align_weight 0.4 \
  --step_collapse_weight 0.2 \
  --step_decoder_weight 0.2 \
  --step_failure_weight 0.2 \
  --step_grounding_mix 0.5 \
  --max_new_tokens 1024 \
  --verbose 1

"$PYTHON" main.py \
  --method build_step_prototypes \
  --dataset "$DATASET" \
  --memory_dataset "$DATASET" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --step_memory_dir "$STEP_MEMORY_DIR" \
  --step_prototype_dir "$STEP_PROTOTYPE_DIR" \
  --step_memory_output_path "$STEP_MEMORY_PATH" \
  --step_prototype_path "$STEP_PROTOTYPE_PATH" \
  --n_step_prototypes_per_group 4 \
  --verbose 1

if [[ "$DISABLE_STEP_DECODER" != "1" ]]; then
  "$PYTHON" main.py \
    --method train_step_decoder \
    --dataset "$DATASET" \
    --memory_dataset "$DATASET" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --step_memory_dir "$STEP_MEMORY_DIR" \
    --step_prototype_dir "$STEP_PROTOTYPE_DIR" \
    --step_decoder_dir "$STEP_DECODER_DIR" \
    --step_memory_output_path "$STEP_MEMORY_PATH" \
    --step_prototype_path "$STEP_PROTOTYPE_PATH" \
    --step_decoder_path "$STEP_DECODER_PATH" \
    --num_thought_tokens 4 \
    --step_decoder_epochs 1 \
    --step_decoder_batch_size 1 \
    --verbose 1
fi

STEP_LTPO_FLAGS=()
if [[ "$DISABLE_STEP_DECODER" == "1" ]]; then
  STEP_LTPO_FLAGS+=(--disable_step_decoder)
fi
if [[ "$DISABLE_FAILURE_PENALTY" == "1" ]]; then
  STEP_LTPO_FLAGS+=(--disable_failure_penalty)
fi

"$PYTHON" main.py \
  --method step_memory_ltpo \
  --dataset "$DATASET" \
  --dataset_split "$EVAL_SPLIT" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --step_memory_dir "$STEP_MEMORY_DIR" \
  --step_prototype_dir "$STEP_PROTOTYPE_DIR" \
  --step_decoder_dir "$STEP_DECODER_DIR" \
  --step_prototype_path "$STEP_PROTOTYPE_PATH" \
  --step_vectorizer_path "$STEP_VECTORIZER_PATH" \
  --step_decoder_path "$STEP_DECODER_PATH" \
  --start_data_idx "$START_DATA_IDX" \
  --end_data_idx "$END_DATA_IDX" \
  --num_thought_tokens 4 \
  --max_num_steps 5 \
  --lr 0.03 \
  --sigma 0.05 \
  --sigma_decay 0.99 \
  --step_conf_weight 0.4 \
  --step_align_weight 0.4 \
  --step_collapse_weight 0.2 \
  --step_decoder_weight 0.2 \
  --step_failure_weight 0.2 \
  --step_grounding_mix 0.5 \
  --max_new_tokens 1024 \
  --verbose 1 \
  "${STEP_LTPO_FLAGS[@]}"
