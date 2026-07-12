#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

WORK_DIR="${WORK_DIR:-$ROOT_DIR/outputs/synthetic}"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

python -m flocrad manifest \
  --dataset synthetic \
  --data-dir "$WORK_DIR/data" \
  --output-csv "$WORK_DIR/data/manifest.csv" \
  --synthetic-size "${SYNTHETIC_SIZE:-120}" \
  --seed 42

python -m flocrad pipeline \
  --manifest "$WORK_DIR/data/manifest.csv" \
  --work-dir "$WORK_DIR" \
  --device cpu \
  --image-backbone tiny \
  --image-pretrained false \
  --epochs "${EPOCHS:-2}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --examples-per-study 2 \
  --error-counts 1,2 \
  --clean-fraction 0.15 \
  --min-compound-count 2 \
  --num-workers 0

cat "$WORK_DIR/evaluation/metrics.json"
