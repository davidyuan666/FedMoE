#!/usr/bin/env bash
# Run RQ3 experiment (Hierarchical / Skill-aware aggregation) on a single machine with 3x RTX 3060.
# - Two GPUs act as workers (default: 0,1)
# - One GPU acts as aggregator (default: 2)
#
# Usage (defaults shown):
#   WORKER_GPUS=0,1 AGG_GPU=2 EXPERTS=python,sql \
#   DATASET=dataset/test.jsonl NUM_SAMPLES=100 NUM_ROUNDS=6 ROUND_DURATION=3 \
#   MODEL_DIM=768 LORA_RANK=16 \
#   ./scripts/run_rq3.sh
#
# Quick smoke test:
#   NUM_SAMPLES=10 NUM_ROUNDS=1 ROUND_DURATION=1 ./scripts/run_rq3.sh

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,200p' "$0"
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

# Parameters with defaults (can be overridden via env)
WORKER_GPUS="${WORKER_GPUS:-0,1}"
AGG_GPU="${AGG_GPU:-2}"
EXPERTS="${EXPERTS:-python,sql}"
DATASET="${DATASET:-dataset/test.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_ROUNDS="${NUM_ROUNDS:-6}"
ROUND_DURATION="${ROUND_DURATION:-3}"
MODEL_DIM="${MODEL_DIM:-768}"
LORA_RANK="${LORA_RANK:-16}"

mkdir -p logs

echo "[run_rq3.sh] Using:"
echo "  WORKER_GPUS=${WORKER_GPUS}  AGG_GPU=${AGG_GPU}  EXPERTS=${EXPERTS}"
echo "  DATASET=${DATASET}  NUM_SAMPLES=${NUM_SAMPLES}  NUM_ROUNDS=${NUM_ROUNDS}  ROUND_DURATION=${ROUND_DURATION}"
echo "  MODEL_DIM=${MODEL_DIM}  LORA_RANK=${LORA_RANK}"

"${PYTHON_BIN}" RQ3-experiment.py \
  --dataset "${DATASET}" \
  --num-samples "${NUM_SAMPLES}" \
  --model-dim "${MODEL_DIM}" \
  --lora-rank "${LORA_RANK}" \
  --num-rounds "${NUM_ROUNDS}" \
  --round-duration "${ROUND_DURATION}" \
  --gpu-workers "${WORKER_GPUS}" \
  --gpu-agg "${AGG_GPU}" \
  --experts "${EXPERTS}" | tee logs/run_rq3.out

