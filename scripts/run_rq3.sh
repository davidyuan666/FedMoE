#!/usr/bin/env bash
# Run RQ3 (Continual domain shift & rehearsal) on a single machine with 3x RTX 3060.
# Two GPUs act as workers (default: 0,1), one GPU acts as aggregator (default: 2).
#
# Usage (defaults shown):
#   WORKER_GPUS=0,1 AGG_GPU=2 \
#   PHASES="python,sql,docs" PHASE_ROUNDS=4 REHEARSAL=0.1 \
#   DATASET=dataset/test.jsonl NUM_SAMPLES=100 ROUND_DURATION=3 \
#   MODEL_DIM=768 LORA_RANK=16 \
#   ./scripts/run_rq3.sh
#
# Quick smoke test:
#   NUM_SAMPLES=20 PHASE_ROUNDS=2 ROUND_DURATION=1 ./scripts/run_rq3.sh

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
PHASES="${PHASES:-python,sql,docs}"
PHASE_ROUNDS="${PHASE_ROUNDS:-4}"
REHEARSAL="${REHEARSAL:-0.1}"
DATASET="${DATASET:-dataset/test.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
ROUND_DURATION="${ROUND_DURATION:-3}"
MODEL_DIM="${MODEL_DIM:-768}"
LORA_RANK="${LORA_RANK:-16}"

mkdir -p logs

echo "[run_rq3.sh] Using:"
echo "  WORKER_GPUS=${WORKER_GPUS}  AGG_GPU=${AGG_GPU}"
echo "  PHASES=${PHASES}  PHASE_ROUNDS=${PHASE_ROUNDS}  REHEARSAL=${REHEARSAL}"
echo "  DATASET=${DATASET}  NUM_SAMPLES=${NUM_SAMPLES}  ROUND_DURATION=${ROUND_DURATION}"
echo "  MODEL_DIM=${MODEL_DIM}  LORA_RANK=${LORA_RANK}"

"${PYTHON_BIN}" RQ3-experiment.py \
  --dataset "${DATASET}" \
  --num-samples "${NUM_SAMPLES}" \
  --model-dim "${MODEL_DIM}" \
  --lora-rank "${LORA_RANK}" \
  --round-duration "${ROUND_DURATION}" \
  --gpu-workers "${WORKER_GPUS}" \
  --gpu-agg "${AGG_GPU}" \
  --phases "${PHASES}" \
  --phase-rounds "${PHASE_ROUNDS}" \
  --rehearsal-ratio "${REHEARSAL}" | tee logs/run_rq3.out
