#!/bin/bash
# 启动分布式 Worker 客户端

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"

# 确定平台和路径
UNAME_OUT="$(uname -s)"
case "${UNAME_OUT}" in
    CYGWIN*|MINGW*|MSYS*)
        BIN_DIR="Scripts"
        PYTHON_PATH="$PROJECT_ROOT/$VENV_DIR/Scripts/python.exe"
        ;;
    *)
        BIN_DIR="bin"
        PYTHON_PATH="$PROJECT_ROOT/$VENV_DIR/bin/python"
        ;;
esac

# 解析参数
# 默认参数，可通过环境变量覆盖
DEFAULT_COORDINATOR_URL="${COORDINATOR_URL:-http://127.0.0.1:5000}"
DEFAULT_SPECIALTY="${SPECIALTY:-python_expert}"
DEFAULT_BASE_MODEL="${BASE_MODEL:-qwen/Qwen2-0.5B-Instruct}"
DEFAULT_SPEED_FACTOR="${SPEED_FACTOR:-1.0}"
DEFAULT_DURATION="${DURATION:-3600}"
DEFAULT_SYNC_INTERVAL="${SYNC_INTERVAL:-10.0}"
DEFAULT_USE_HF="${USE_HUGGINGFACE:-false}"
RUN_MODE="${RUN_MODE:-foreground}" # foreground | daemon

COORDINATOR_URL="$DEFAULT_COORDINATOR_URL"
WORKER_ID="${WORKER_ID:-}"
SPECIALTY="$DEFAULT_SPECIALTY"
DATASET="${DATASET:-}"
BASE_MODEL="$DEFAULT_BASE_MODEL"
SPEED_FACTOR="$DEFAULT_SPEED_FACTOR"
DURATION="$DEFAULT_DURATION"
SYNC_INTERVAL="$DEFAULT_SYNC_INTERVAL"
USE_HUGGINGFACE="$DEFAULT_USE_HF"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --coordinator-url)
            COORDINATOR_URL="$2"
            shift 2
            ;;
        --worker-id)
            WORKER_ID="$2"
            shift 2
            ;;
        --specialty)
            SPECIALTY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --speed-factor)
            SPEED_FACTOR="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --sync-interval)
            SYNC_INTERVAL="$2"
            shift 2
            ;;
        --use-huggingface)
            USE_HUGGINGFACE="true"
            shift
            ;;
        --daemon)
            RUN_MODE="daemon"
            shift
            ;;
        --foreground)
            RUN_MODE="foreground"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--coordinator-url URL] [--worker-id ID] [--specialty SPECIALTY] [其他选项]"
            exit 1
            ;;
    esac
done

# 如果未指定 specialty，使用默认
if [ -z "$SPECIALTY" ]; then
    SPECIALTY="$DEFAULT_SPECIALTY"
fi

# 自动生成 Worker ID（包含主机名、专家名和时间戳），避免手动输入
if [ -z "$WORKER_ID" ]; then
    HOSTNAME="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-')"
    HOSTNAME="${HOSTNAME:-worker}"
    SPECIALTY_TAG="$(echo "$SPECIALTY" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-')"
    TIMESTAMP="$(date +%Y%m%d%H%M%S)"
    WORKER_ID="${HOSTNAME}-${SPECIALTY_TAG}-${TIMESTAMP}"
fi

# 如果未指定数据集，尝试自动选择
if [ -z "$DATASET" ]; then
    if [ -f "$PROJECT_ROOT/dataset/${SPECIALTY}.jsonl" ]; then
        DATASET="$PROJECT_ROOT/dataset/${SPECIALTY}.jsonl"
    elif [ -f "$PROJECT_ROOT/dataset/${SPECIALTY%_*}.jsonl" ]; then
        DATASET="$PROJECT_ROOT/dataset/${SPECIALTY%_*}.jsonl"
    elif [ -f "$PROJECT_ROOT/dataset/test.jsonl" ]; then
        DATASET="$PROJECT_ROOT/dataset/test.jsonl"
    fi
fi

# 检查虚拟环境
if [ ! -f "$PYTHON_PATH" ]; then
    echo "错误: 虚拟环境不存在，请先运行: bash scripts/setup.sh"
    exit 1
fi

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"
LOG_FILE="$PROJECT_ROOT/logs/worker-${WORKER_ID}.log"
PID_FILE="$PROJECT_ROOT/logs/worker-${WORKER_ID}.pid"

if [ "$RUN_MODE" = "daemon" ]; then
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Worker $WORKER_ID 已经在后台运行 (PID: $PID)"
            exit 1
        else
            rm -f "$PID_FILE"
        fi
    fi
fi

echo "=========================================="
echo "启动 FedMoE Worker"
echo "=========================================="
echo "Worker ID: $WORKER_ID"
echo "Specialty: $SPECIALTY"
echo "Coordinator: $COORDINATOR_URL"
echo "Dataset: ${DATASET:-使用默认数据}"
echo "Sync Interval: ${SYNC_INTERVAL}s"
if [ "$RUN_MODE" = "daemon" ]; then
    echo "运行模式: 后台 (使用 stop_worker.sh 停止)"
else
    echo "运行模式: 前台 (Ctrl+C 停止，输出同步写入日志)"
fi
echo "日志: $LOG_FILE"
echo ""

# 构建命令
CMD=("$PYTHON_PATH" "run_worker.py"
    "--coordinator-url" "$COORDINATOR_URL"
    "--worker-id" "$WORKER_ID"
    "--specialty" "$SPECIALTY"
    "--base-model" "$BASE_MODEL"
    "--speed-factor" "$SPEED_FACTOR"
    "--duration" "$DURATION"
    "--sync-interval" "$SYNC_INTERVAL"
)

if [ -n "$DATASET" ]; then
    CMD+=("--dataset" "$DATASET")
fi

if [ "$USE_HUGGINGFACE" = "true" ]; then
    CMD+=("--use-huggingface")
fi

cd "$PROJECT_ROOT"

if [ "$RUN_MODE" = "daemon" ]; then
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    WORKER_PID=$!
    echo $WORKER_PID > "$PID_FILE"
    sleep 2
    if ps -p "$WORKER_PID" > /dev/null 2>&1; then
        echo "✓ Worker $WORKER_ID 已在后台启动 (PID: $WORKER_PID)"
        echo "  查看日志: tail -f $LOG_FILE"
        echo "  停止服务: bash scripts/stop_worker.sh --worker-id $WORKER_ID"
    else
        echo "✗ Worker 启动失败，请查看日志: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
else
    echo "正在以前台模式运行，按 Ctrl+C 停止。日志同步写入 $LOG_FILE"
    set -o pipefail
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Worker $WORKER_ID 已正常退出 (exit code 0)"
    else
        echo "Worker $WORKER_ID 异常退出 (exit code $EXIT_CODE)，查看日志: $LOG_FILE"
    fi
    exit $EXIT_CODE
fi

