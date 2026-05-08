#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ./output
mkdir -p ./output/memories
mkdir -p ./output/prototypes
mkdir -p ./output/runs
mkdir -p ./output/logs

python main.py \
  --method memory_ltpo \
  --dataset openai/gsm8k \
  --dataset_split test \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --memory_dir ./output/memories \
  --prototype_dir ./output/prototypes \
  --prototype_path ./output/prototypes/gsm8k_prototypes.json \
  --start_data_idx 0 \
  --end_data_idx -1 \
  --max_new_tokens 1024 \
  --device cuda \
  --dtype auto \
  --solver_prompt_idx 0 \
  --seed 42 \
  --top_k_prototypes 2 \
  --n_candidates 2 \
  --memory_temperature 0.3 \
  --memory_top_p 0.9 \
  --verbose 1
