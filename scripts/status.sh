#!/bin/bash
# 查看 FedMoE 服务状态

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "FedMoE 服务状态"
echo "=========================================="
echo ""

# 检查 Coordinator
COORDINATOR_PID_FILE="$PROJECT_ROOT/coordinator.pid"
if [ -f "$COORDINATOR_PID_FILE" ]; then
    PID=$(cat "$COORDINATOR_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Coordinator: 运行中 (PID: $PID)"
    else
        echo "✗ Coordinator: 已停止 (PID 文件存在但进程不存在)"
        rm -f "$COORDINATOR_PID_FILE"
    fi
else
    echo "✗ Coordinator: 未运行"
fi

echo ""

# 检查 Workers
WORKER_DIR="$PROJECT_ROOT/logs"
if [ -d "$WORKER_DIR" ]; then
    WORKER_PIDS=$(find "$WORKER_DIR" -name "worker-*.pid" 2>/dev/null)
    if [ -n "$WORKER_PIDS" ]; then
        echo "Workers:"
        for PID_FILE in $WORKER_PIDS; do
            WORKER_NAME=$(basename "$PID_FILE" .pid | sed 's/worker-//')
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "  ✓ $WORKER_NAME: 运行中 (PID: $PID)"
            else
                echo "  ✗ $WORKER_NAME: 已停止 (PID 文件存在但进程不存在)"
                rm -f "$PID_FILE"
            fi
        done
    else
        echo "✗ Workers: 无运行中的 Worker"
    fi
else
    echo "✗ Workers: 无运行中的 Worker"
fi

echo ""

