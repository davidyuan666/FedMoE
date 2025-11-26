#!/usr/bin/env python3
"""
快速启动脚本：直接运行 Qwen 模型微调
确保在虚拟环境中运行
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 检查是否在虚拟环境中运行
def check_venv():
    """检查是否在虚拟环境中，如果不在则提示用户"""
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
    )
    
    if not in_venv:
        venv_path = Path(__file__).parent / ".venv"
        print("=" * 60)
        print("警告: 未检测到虚拟环境！")
        print("=" * 60)
        print()
        print("请先设置虚拟环境并安装依赖：")
        print()
        print("Windows PowerShell:")
        print("  .\\scripts\\setup.ps1")
        print()
        print("Linux/Mac Bash:")
        print("  bash scripts/setup.sh")
        print()
        print("或手动激活虚拟环境后运行：")
        if sys.platform == "win32":
            print(f"  .\\.venv\\Scripts\\Activate.ps1")
        else:
            print(f"  source .venv/bin/activate")
        print()
        print("然后安装依赖：")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)
    
    print(f"✓ 检测到虚拟环境: {os.environ.get('VIRTUAL_ENV', sys.prefix)}")

# 检查虚拟环境
check_venv()

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fedmoe import SimulationConfig, run_qwen_simulation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedMoE - Qwen 模型真实微调")
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
        "--duration",
        type=float,
        default=30.0,
        help="训练时长（秒，默认: 30.0）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FedMoE - Qwen 模型真实微调")
    print("=" * 60)
    print()
    
    # 使用默认配置，直接运行真实微调
    config = SimulationConfig(
        training_duration_s=args.duration,
        worker_specs=None,  # 使用默认 workers
        expert_names=None,  # 使用默认 experts
    )
    
    # 检查默认数据集路径
    training_data_path = args.dataset
    if training_data_path is None:
        default_dataset = project_root / "dataset" / "test.jsonl"
        if default_dataset.exists():
            training_data_path = str(default_dataset)
            print(f"自动检测到数据集: {training_data_path}")
    
    print(f"基础模型: {args.base_model}")
    print(f"模型来源: ModelScope")
    print(f"训练时长: {config.training_duration_s} 秒")
    if training_data_path:
        print(f"训练数据集: {training_data_path}")
    else:
        print("训练数据集: 使用默认示例数据")
    print()
    
    run_qwen_simulation(
        config,
        base_model_name=args.base_model,
        use_modelscope=True,
        training_data_path=training_data_path,
    )

