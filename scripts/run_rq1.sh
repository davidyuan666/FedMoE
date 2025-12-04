#!/usr/bin/env bash
# Run RQ1 experiment on a single machine with 3x RTX 3060.
# - Two GPUs act as workers (default: 0,1)
# - One GPU acts as aggregator (default: 2)
#
# Usage (defaults shown):
#   WORKER_GPUS=0,1 AGG_GPU=2 EXPERTS=python,sql \
#   DATASET=dataset/ds1000.jsonl NUM_SAMPLES=100 NUM_ROUNDS=10 ROUND_DURATION=5 \
#   MODEL_DIM=768 LORA_RANK=16 MIN_DATASET_SAMPLES=50 \
#   ./scripts/run_rq1.sh
#
# With worker-specific datasets (domain specialization):
#   WORKER_GPUS=0,1 AGG_GPU=2 EXPERTS=python,sql \
#   WORKER_DATASETS=domains/python.jsonl,domains/sql.jsonl \
#   NUM_SAMPLES=100 NUM_ROUNDS=10 ROUND_DURATION=5 \
#   ./scripts/run_rq1.sh
#
# Quick smoke test:
#   NUM_SAMPLES=10 NUM_ROUNDS=1 ROUND_DURATION=1 ./scripts/run_rq1.sh

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '1,120p' "$0"
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"

# Parameters with defaults (can be overridden via env)
WORKER_GPUS="${WORKER_GPUS:-0,1}"
AGG_GPU="${AGG_GPU:-2}"
EXPERTS="${EXPERTS:-python,sql}"
DATASET="${DATASET:-dataset/ds1000.jsonl}"
WORKER_DATASETS="${WORKER_DATASETS:-}"  # Optional: comma-separated dataset paths for each worker
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
ROUND_DURATION="${ROUND_DURATION:-5}"
MODEL_DIM="${MODEL_DIM:-768}"
LORA_RANK="${LORA_RANK:-16}"
MIN_DATASET_SAMPLES="${MIN_DATASET_SAMPLES:-50}"

mkdir -p logs

echo "[run_rq1.sh] Using:"
echo "  WORKER_GPUS=${WORKER_GPUS}  AGG_GPU=${AGG_GPU}  EXPERTS=${EXPERTS}"
echo "  DATASET=${DATASET}  NUM_SAMPLES=${NUM_SAMPLES}  NUM_ROUNDS=${NUM_ROUNDS}  ROUND_DURATION=${ROUND_DURATION}"
echo "  MODEL_DIM=${MODEL_DIM}  LORA_RANK=${LORA_RANK}  MIN_DATASET_SAMPLES=${MIN_DATASET_SAMPLES}"
if [ -n "${WORKER_DATASETS}" ]; then
  echo "  WORKER_DATASETS=${WORKER_DATASETS}"
fi

CMD="${PYTHON_BIN} RQ1-experiment.py \
  --dataset \"${DATASET}\" \
  --num-samples ${NUM_SAMPLES} \
  --model-dim ${MODEL_DIM} \
  --lora-rank ${LORA_RANK} \
  --num-rounds ${NUM_ROUNDS} \
  --round-duration ${ROUND_DURATION} \
  --gpu-workers \"${WORKER_GPUS}\" \
  --gpu-agg ${AGG_GPU} \
  --experts \"${EXPERTS}\" \
  --min-dataset-samples ${MIN_DATASET_SAMPLES}"

if [ -n "${WORKER_DATASETS}" ]; then
  CMD="${CMD} --worker-datasets \"${WORKER_DATASETS}\""
fi

# Preflight dependency checks
missing=()
for mod in peft transformers accelerate torch; do
  if ! ${PYTHON_BIN} -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${mod}') else 1)" >/dev/null 2>&1; then
    missing+=("${mod}")
  fi
done

if [ ${#missing[@]} -gt 0 ]; then
  echo "[run_rq1.sh] Missing Python modules: ${missing[*]}"
  echo "[run_rq1.sh] Please install them in your current Python environment, e.g.:"
  echo "  ${PYTHON_BIN} -m pip install -U ${missing[*]}"
  echo "  # On Windows with Python launcher: py -3 -m pip install -U ${missing[*]}"
  exit 1
fi

eval "${CMD}" | tee logs/run_rq1.out

