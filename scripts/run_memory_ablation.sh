#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ./output
mkdir -p ./output/memories
mkdir -p ./output/prototypes
mkdir -p ./output/runs
mkdir -p ./output/logs

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATASET="openai/gsm8k"
PROTO_PATH="./output/prototypes/gsm8k_prototypes.json"

# full
python main.py \
  --method memory_ltpo \
  --dataset "$DATASET" \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir ./output \
  --prototype_path "$PROTO_PATH" \
  --top_k_prototypes 2 \
  --n_candidates 2 \
  --memory_temperature 0.3 \
  --memory_top_p 0.9 \
  --verbose 1

# w/o CT
python main.py \
  --method memory_ltpo \
  --dataset "$DATASET" \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir ./output \
  --prototype_path "$PROTO_PATH" \
  --top_k_prototypes 2 \
  --n_candidates 2 \
  --memory_temperature 0.3 \
  --memory_top_p 0.9 \
  --disable_ct \
  --verbose 1

# w/o copy penalty
python main.py \
  --method memory_ltpo \
  --dataset "$DATASET" \
  --model_name_or_path "$MODEL_NAME" \
  --output_dir ./output \
  --prototype_path "$PROTO_PATH" \
  --top_k_prototypes 2 \
  --n_candidates 2 \
  --memory_temperature 0.3 \
  --memory_top_p 0.9 \
  --disable_copy_penalty \
  --verbose 1
