#!/usr/bin/env python3
"""
获取门户模型（管理模型）的权重
这个模型整合了所有垂直领域专家的梯度更新
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _base64_to_numpy(data: str | dict) -> np.ndarray:
    """将 base64 字符串转换为 numpy 数组"""
    if isinstance(data, str):
        data = json.loads(data)
    
    arr_bytes = base64.b64decode(data["data"])
    arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
    return arr.reshape(data["shape"])


def get_gateway_weights(coordinator_url: str) -> tuple:
    """
    从 coordinator 获取门户模型的权重。
    
    Args:
        coordinator_url: Coordinator 服务器地址
        
    Returns:
        (lora_A, lora_B, version) 元组
    """
    try:
        response = requests.get(
            f"{coordinator_url}/get_gateway_weights",
            timeout=30,
        )
        if response.status_code != 200:
            raise ValueError(f"获取门户模型权重失败: {response.text}")

        data = response.json()
        lora_A = _base64_to_numpy(data["lora_A"])
        lora_B = _base64_to_numpy(data["lora_B"])
        version = data["version"]
        return lora_A, lora_B, version
    except Exception as e:
        print(f"错误: 获取门户模型权重失败: {e}")
        raise


def get_all_expert_weights(coordinator_url: str) -> dict:
    """
    从 coordinator 获取所有专家的权重。
    
    Args:
        coordinator_url: Coordinator 服务器地址
        
    Returns:
        字典，键为专家名称，值为(lora_A, lora_B, version)元组
    """
    try:
        response = requests.get(
            f"{coordinator_url}/get_all_expert_weights",
            timeout=30,
        )
        if response.status_code != 200:
            raise ValueError(f"获取专家权重失败: {response.text}")

        data = response.json()
        experts = {}
        for expert_name, expert_data in data["experts"].items():
            lora_A = _base64_to_numpy(expert_data["lora_A"])
            lora_B = _base64_to_numpy(expert_data["lora_B"])
            version = expert_data["version"]
            experts[expert_name] = (lora_A, lora_B, version)
        return experts
    except Exception as e:
        print(f"错误: 获取专家权重失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="获取门户模型（管理模型）的权重")
    parser.add_argument(
        "--coordinator-url",
        type=str,
        required=True,
        help="Coordinator 服务器地址（如: http://192.168.1.100:5000）",
    )
    parser.add_argument(
        "--list-experts",
        action="store_true",
        help="列出所有专家及其权重信息",
    )
    parser.add_argument(
        "--save-gateway",
        type=str,
        default=None,
        help="保存门户模型权重到文件（numpy格式）",
    )
    args = parser.parse_args()

    coordinator_url = args.coordinator_url.rstrip("/")

    print("=" * 60)
    print("FedMoE - 门户模型（管理模型）查询工具")
    print("=" * 60)
    print(f"Coordinator: {coordinator_url}")
    print()

    # 检查 coordinator 是否可用
    try:
        response = requests.get(f"{coordinator_url}/health", timeout=5)
        if response.status_code != 200:
            raise ConnectionError(f"Coordinator 健康检查失败: {response.status_code}")
    except Exception as e:
        print(f"错误: 无法连接到 Coordinator {coordinator_url}: {e}")
        sys.exit(1)

    # 获取门户模型权重
    try:
        print("正在获取门户模型权重...")
        lora_A, lora_B, version = get_gateway_weights(coordinator_url)
        print(f"\n✓ 门户模型（管理模型）")
        print(f"  版本: {version}")
        print(f"  LoRA A 形状: {lora_A.shape}")
        print(f"  LoRA B 形状: {lora_B.shape}")
        print(f"  LoRA A 范数: {np.linalg.norm(lora_A):.6f}")
        print(f"  LoRA B 范数: {np.linalg.norm(lora_B):.6f}")
        
        if args.save_gateway:
            np.savez(args.save_gateway, lora_A=lora_A, lora_B=lora_B, version=version)
            print(f"\n✓ 门户模型权重已保存到: {args.save_gateway}")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

    # 如果请求列出所有专家
    if args.list_experts:
        try:
            print("\n正在获取所有专家权重...")
            experts = get_all_expert_weights(coordinator_url)
            print(f"\n✓ 找到 {len(experts)} 个专家:")
            for expert_name, (lora_A, lora_B, version) in experts.items():
                print(f"\n  专家: {expert_name}")
                print(f"    版本: {version}")
                print(f"    LoRA A 形状: {lora_A.shape}")
                print(f"    LoRA B 形状: {lora_B.shape}")
                print(f"    LoRA A 范数: {np.linalg.norm(lora_A):.6f}")
                print(f"    LoRA B 范数: {np.linalg.norm(lora_B):.6f}")
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()

