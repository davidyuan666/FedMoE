#!/bin/bash
# 启动分布式 Coordinator 服务器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"
PID_FILE="$PROJECT_ROOT/coordinator.pid"
LOG_FILE="$PROJECT_ROOT/logs/coordinator.log"
RUN_MODE="${RUN_MODE:-foreground}" # foreground | daemon

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

# 检查是否已经在运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Coordinator 已经在运行 (PID: $PID)"
        exit 1
    else
        echo "清理旧的 PID 文件"
        rm -f "$PID_FILE"
    fi
fi

# 创建日志目录
mkdir -p "$PROJECT_ROOT/logs"

# 解析参数
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
DEBUG="${DEBUG:-false}"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
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
            echo "用法: $0 [--host HOST] [--port PORT] [--debug] [--daemon]"
            exit 1
            ;;
    esac
done

# 检查虚拟环境
if [ ! -f "$PYTHON_PATH" ]; then
    echo "错误: 虚拟环境不存在，请先运行: bash scripts/setup.sh"
    exit 1
fi

# 设置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "启动 FedMoE Coordinator"
echo "=========================================="
echo "地址: $HOST:$PORT"
echo "日志: $LOG_FILE"
if [ "$RUN_MODE" = "daemon" ]; then
    echo "运行模式: 后台 (使用 stop_coordinator.sh 停止)"
else
    echo "运行模式: 前台 (Ctrl+C 停止，输出同步写入日志)"
fi
echo ""

cd "$PROJECT_ROOT"
CMD=("$PYTHON_PATH" "run_coordinator.py" "--host" "$HOST" "--port" "$PORT")
if [ "$DEBUG" = "true" ]; then
    CMD+=("--debug")
fi

mkdir -p "$PROJECT_ROOT/logs"

if [ "$RUN_MODE" = "daemon" ]; then
    # 检查是否已经在运行（仅后台模式需要）
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "Coordinator 已经在后台运行 (PID: $PID)"
            exit 1
        else
            echo "清理旧的 PID 文件"
            rm -f "$PID_FILE"
        fi
    fi

    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    COORDINATOR_PID=$!
    echo $COORDINATOR_PID > "$PID_FILE"
    sleep 2
    if ps -p "$COORDINATOR_PID" > /dev/null 2>&1; then
        echo "✓ Coordinator 已在后台启动 (PID: $COORDINATOR_PID)"
        echo "  访问地址: http://$HOST:$PORT"
        echo "  查看日志: tail -f $LOG_FILE"
        echo "  停止服务: bash scripts/stop_coordinator.sh"
    else
        echo "✗ Coordinator 启动失败，请查看日志: $LOG_FILE"
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
        echo "Coordinator 已正常退出 (exit code 0)"
    else
        echo "Coordinator 异常退出 (exit code $EXIT_CODE)，查看日志: $LOG_FILE"
    fi
    exit $EXIT_CODE
fi

