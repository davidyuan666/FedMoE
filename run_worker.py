#!/usr/bin/env python3
"""
启动分布式 Worker 客户端
可以在不同的机器上运行，连接到远程 Coordinator
"""

from __future__ import annotations

import argparse
import threading
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fedmoe.distributed_worker import DistributedWorker


def main():
    parser = argparse.ArgumentParser(description="启动分布式 Worker 客户端")
    parser.add_argument(
        "--coordinator-url",
        type=str,
        required=True,
        help="Coordinator 服务器地址（如: http://192.168.1.100:5000）",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        required=True,
        help="Worker ID（如: Worker-1-GPU0）",
    )
    parser.add_argument(
        "--specialty",
        type=str,
        required=True,
        help="专家类型（如: python_expert, sql_expert, docs_expert）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSONL 数据集文件路径（例如: dataset/test.jsonl）",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="qwen/Qwen2-0.5B-Instruct",
        help="基础模型名称（默认: qwen/Qwen2-0.5B-Instruct）",
    )
    parser.add_argument(
        "--speed-factor",
        type=float,
        default=1.0,
        help="速度因子（默认: 1.0，越大越慢）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="训练持续时间（秒，默认: 3600，即1小时）",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="使用 HuggingFace 而不是 ModelScope",
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="允许使用系统代理访问 Coordinator（默认禁用代理）",
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=10.0,
        help="定时上传梯度的间隔（秒，默认: 10.0）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FedMoE - 分布式 Worker")
    print("=" * 60)
    print(f"Coordinator: {args.coordinator_url}")
    print(f"Worker ID: {args.worker_id}")
    print(f"Specialty: {args.specialty}")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset: {args.dataset or '使用默认数据'}")
    print(f"Speed Factor: {args.speed_factor}")
    print(f"Duration: {args.duration} 秒")
    print(f"Sync Interval: {args.sync_interval} 秒")
    print()

    # 创建 worker
    worker = DistributedWorker(
        coordinator_url=args.coordinator_url,
        worker_id=args.worker_id,
        specialty=args.specialty,
        speed_factor=args.speed_factor,
        base_model_name=args.base_model,
        training_data_path=args.dataset,
        use_modelscope=not args.use_huggingface,
        sync_interval=args.sync_interval,
        use_proxy=args.use_proxy,
    )

    # 运行 worker
    print(f"\n=== [Worker {args.worker_id} 开始训练] ===")
    stop_event = threading.Event()

    def run_worker():
        worker.run_loop(stop_event)

    thread = threading.Thread(target=run_worker)
    thread.start()

    try:
        # 等待指定时间
        time.sleep(args.duration)
        print(f"\n=== [训练时间到，停止 Worker {args.worker_id}] ===")
        stop_event.set()
        thread.join(timeout=10.0)
    except KeyboardInterrupt:
        print(f"\n\n=== [用户中断，停止 Worker {args.worker_id}] ===")
        stop_event.set()
        thread.join(timeout=10.0)

    print(f"Worker {args.worker_id} 已停止")


if __name__ == "__main__":
    main()

