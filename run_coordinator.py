#!/usr/bin/env python3
"""
启动分布式 Coordinator 服务器
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fedmoe.distributed_coordinator import DistributedCoordinator


def main():
    parser = argparse.ArgumentParser(description="启动分布式 Coordinator 服务器")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="服务器端口（默认: 5000）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式",
    )
    args = parser.parse_args()

    coordinator = DistributedCoordinator(host=args.host, port=args.port)
    try:
        coordinator.run(debug=args.debug)
    except KeyboardInterrupt:
        print("\n\nCoordinator 服务器已停止")


if __name__ == "__main__":
    main()

