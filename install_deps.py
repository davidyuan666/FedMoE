#!/usr/bin/env python3
"""
安装依赖到虚拟环境的辅助脚本
确保所有依赖都安装到 .venv 虚拟环境中
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def find_venv_python():
    """查找虚拟环境中的 Python"""
    project_root = Path(__file__).parent
    venv_path = project_root / ".venv"
    
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        python_path = venv_path / "bin" / "python"
        pip_path = venv_path / "bin" / "pip"
    
    return venv_path, python_path, pip_path

def main():
    venv_path, python_path, pip_path = find_venv_python()
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    # 检查虚拟环境是否存在
    if not venv_path.exists():
        print("=" * 60)
        print("虚拟环境不存在！")
        print("=" * 60)
        print()
        print("请先创建虚拟环境：")
        print()
        if sys.platform == "win32":
            print("  .\\scripts\\setup.ps1")
        else:
            print("  bash scripts/setup.sh")
        print()
        sys.exit(1)
    
    # 检查 pip 是否存在
    if not pip_path.exists():
        print("=" * 60)
        print("虚拟环境中的 pip 不存在！")
        print("=" * 60)
        print()
        print("请先运行 setup 脚本：")
        print()
        if sys.platform == "win32":
            print("  .\\scripts\\setup.ps1")
        else:
            print("  bash scripts/setup.sh")
        print()
        sys.exit(1)
    
    print("=" * 60)
    print("安装依赖到虚拟环境")
    print("=" * 60)
    print(f"虚拟环境: {venv_path}")
    print(f"使用 pip: {pip_path}")
    print(f"依赖文件: {requirements_file}")
    print()
    
    # 使用虚拟环境中的 pip 安装依赖
    try:
        cmd = [str(pip_path), "install", "-r", str(requirements_file)]
        print(f"执行: {' '.join(cmd)}")
        print()
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 60)
        print("✓ 依赖安装完成！")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("✗ 依赖安装失败！")
        print("=" * 60)
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

