#!/usr/bin/env bash
# Run all RQ experiments (RQ1, RQ2, RQ3) sequentially on a single machine with 3x RTX 3060.
# Inherits the same env variables as the individual scripts.
#
# Example:
#   WORKER_GPUS=0,1 AGG_GPU=2 EXPERTS=python,sql \
#   DATASET=dataset/test.jsonl NUM_SAMPLES=100 MODEL_DIM=768 LORA_RANK=16 \
#   NUM_ROUNDS=6 ROUND_DURATION=3 \
#   COMPRESSIONS=1.0,0.5 SYNC_INTERVALS=2.0,5.0 \
#   ./scripts/run_all.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

echo "[run_all.sh] === RQ1 ==="
./scripts/run_rq1.sh

echo "[run_all.sh] === RQ2 ==="
./scripts/run_rq2.sh

echo "[run_all.sh] === RQ3 ==="
./scripts/run_rq3.sh

echo "[run_all.sh] Done. Logs in ./logs/"

