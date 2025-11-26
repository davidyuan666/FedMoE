#!/bin/bash
# 停止所有 FedMoE 服务

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "停止所有 FedMoE 服务"
echo "=========================================="
echo ""

# 停止所有 Workers
WORKER_DIR="$PROJECT_ROOT/logs"
if [ -d "$WORKER_DIR" ]; then
    WORKER_PIDS=$(find "$WORKER_DIR" -name "worker-*.pid" 2>/dev/null)
    if [ -n "$WORKER_PIDS" ]; then
        echo "正在停止 Workers..."
        for PID_FILE in $WORKER_PIDS; do
            WORKER_NAME=$(basename "$PID_FILE" .pid | sed 's/worker-//')
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "  停止 $WORKER_NAME (PID: $PID)..."
                kill "$PID" 2>/dev/null || true
            fi
            rm -f "$PID_FILE"
        done
        sleep 2
        echo "✓ 所有 Workers 已停止"
    fi
fi

# 停止 Coordinator
COORDINATOR_PID_FILE="$PROJECT_ROOT/coordinator.pid"
if [ -f "$COORDINATOR_PID_FILE" ]; then
    PID=$(cat "$COORDINATOR_PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "正在停止 Coordinator (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
        echo "✓ Coordinator 已停止"
    fi
    rm -f "$COORDINATOR_PID_FILE"
fi

echo ""
echo "所有服务已停止"

