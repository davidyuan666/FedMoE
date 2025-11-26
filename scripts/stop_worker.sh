#!/bin/bash
# 停止分布式 Worker 客户端

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

WORKER_ID=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --worker-id)
            WORKER_ID="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 --worker-id WORKER_ID"
            exit 1
            ;;
    esac
done

if [ -z "$WORKER_ID" ]; then
    echo "错误: 缺少 --worker-id 参数"
    echo "用法: $0 --worker-id WORKER_ID"
    exit 1
fi

PID_FILE="$PROJECT_ROOT/logs/worker-${WORKER_ID}.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Worker $WORKER_ID 未运行（未找到 PID 文件）"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Worker $WORKER_ID 未运行 (PID: $PID 不存在)"
    rm -f "$PID_FILE"
    exit 1
fi

echo "正在停止 Worker $WORKER_ID (PID: $PID)..."
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
    echo "强制停止 Worker..."
    kill -9 "$PID"
    sleep 1
fi

if ps -p "$PID" > /dev/null 2>&1; then
    echo "✗ 无法停止 Worker $WORKER_ID"
    exit 1
else
    echo "✓ Worker $WORKER_ID 已停止"
    rm -f "$PID_FILE"
fi

