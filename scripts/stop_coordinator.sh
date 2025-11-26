#!/bin/bash
# 停止分布式 Coordinator 服务器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_ROOT/coordinator.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Coordinator 未运行（未找到 PID 文件）"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Coordinator 未运行 (PID: $PID 不存在)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "正在停止 Coordinator (PID: $PID)..."
kill "$PID"

# 等待进程结束
for i in {1..10}; do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# 如果还在运行，强制杀死
if ps -p "$PID" > /dev/null 2>&1; then
    echo "强制停止 Coordinator..."
    kill -9 "$PID"
    sleep 1
fi

if ps -p "$PID" > /dev/null 2>&1; then
    echo "✗ 无法停止 Coordinator"
    exit 1
else
    echo "✓ Coordinator 已停止"
    rm -f "$PID_FILE"
fi

