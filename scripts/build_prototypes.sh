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
  --method build_prototypes \
  --dataset openai/gsm8k \
  --memory_dataset openai/gsm8k \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --output_dir ./output \
  --memory_dir ./output/memories \
  --prototype_dir ./output/prototypes \
  --memory_output_path ./output/memories/gsm8k_memory.jsonl \
  --prototype_path ./output/prototypes/gsm8k_prototypes.json \
  --n_prototypes 16 \
  --top_examples_per_prototype 3 \
  --prototype_max_features 4096 \
  --seed 42 \
  --verbose 1
