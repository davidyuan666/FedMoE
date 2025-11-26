"""
使用真实 Qwen 模型进行微调的 Worker 实现。
"""

from __future__ import annotations

import random
import threading
import time
from typing import List, Tuple

import numpy as np

from .coordinator import CentralCoordinator
from .qwen_finetune import (
    QwenLoRAExpert,
    load_jsonl_dataset,
    prepare_training_texts_from_jsonl,
)


class QwenWorker:
    """
    使用真实 Qwen 模型进行本地训练的 Worker。
    每个 worker 维护一个 QwenLoRAExpert 实例，执行真实的微调训练。
    """

    def __init__(
        self,
        worker_id: str,
        coordinator: CentralCoordinator,
        specialty: str,
        speed_factor: float,
        base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
        training_data: List[str] | None = None,
        training_data_path: str | None = None,
        use_modelscope: bool = True,
    ) -> None:
        """
        初始化 Qwen Worker。

        Args:
            worker_id: Worker ID
            coordinator: 中央协调器
            specialty: 专家类型（如 'python_expert'）
            speed_factor: 速度因子（影响训练时间）
            base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头）
            training_data: 本地训练数据（字符串列表或字典列表）
            training_data_path: JSONL 数据集文件路径（如果提供，会从此文件加载数据）
            use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        """
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.specialty = specialty
        self.speed_factor = speed_factor
        self.base_model_name = base_model_name

        # 加载训练数据
        if training_data_path:
            # 从 JSONL 文件加载数据
            try:
                jsonl_data = load_jsonl_dataset(training_data_path)
                training_data = prepare_training_texts_from_jsonl(jsonl_data, use_chat_template=True)
                print(f"[{worker_id}] 从 {training_data_path} 加载了 {len(training_data)} 条训练数据")
            except Exception as e:
                print(f"[{worker_id}] 警告: 无法从 {training_data_path} 加载数据: {e}")
                print(f"[{worker_id}] 使用默认训练数据")
                training_data = self._generate_default_training_data(specialty)
        elif training_data is None:
            # 如果没有提供训练数据，生成一些示例数据
            training_data = self._generate_default_training_data(specialty)

        self.training_data = training_data

        # 创建本地 Qwen 专家实例
        self.expert = QwenLoRAExpert(
            expert_name=f"{worker_id}_{specialty}",
            base_model_name=base_model_name,
            use_modelscope=use_modelscope,
        )

        print(
            f"Qwen Worker {self.worker_id} (Specialty: {self.specialty}, "
            f"Speed: {self.speed_factor:.1f}x, Data samples: {len(self.training_data)}) initialized."
        )

    def _generate_default_training_data(self, specialty: str) -> List[str]:
        """根据 specialty 生成默认训练数据。"""
        if "python" in specialty.lower():
            return [
                "def sort_list(items):\n    return sorted(items)",
                "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]",
                "def calculate_sum(data):\n    return sum(data)",
            ]
        elif "sql" in specialty.lower():
            return [
                "SELECT * FROM users WHERE age > 18",
                "SELECT name, email FROM customers ORDER BY name",
                "INSERT INTO products (name, price) VALUES ('item', 10.0)",
            ]
        elif "docs" in specialty.lower():
            return [
                "This function sorts a list of items in ascending order.",
                "The query retrieves all users older than 18 years.",
                "This module provides utilities for data processing.",
            ]
        else:
            return [
                "Example training data for general purpose.",
                "This is a sample training text.",
            ]

    def local_training_step(
        self, lora_A: np.ndarray, lora_B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行本地训练步骤，使用真实的 Qwen 模型进行微调。

        Args:
            lora_A: 全局 LoRA A 权重
            lora_B: 全局 LoRA B 权重

        Returns:
            (delta_A, delta_B) 权重更新
        """
        # 更新本地专家的权重以匹配全局权重
        # 获取当前权重，计算差值，然后应用更新
        current_A, current_B = self.expert.get_lora_weights()
        delta_A = lora_A - current_A
        delta_B = lora_B - current_B
        self.expert.update_lora_weights(delta_A, delta_B, learning_rate=1.0)

        # 根据速度因子调整训练参数
        # 速度因子越大，训练时间越长（模拟慢速设备）
        batch_size = max(1, int(4 / self.speed_factor))
        num_epochs = max(1, int(1 * self.speed_factor))

        train_time = random.uniform(2, 5) * self.speed_factor
        print(
            f"  [{self.worker_id}] Starting Qwen fine-tuning on {self.specialty}... "
            f"(Est. {train_time:.2f}s, batch_size={batch_size}, epochs={num_epochs})"
        )

        # 执行真实的微调
        delta_A, delta_B = self.expert.fine_tune_step(
            training_data=self.training_data,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=5e-5,
        )

        print(f"  [{self.worker_id}] Qwen fine-tuning complete.")
        return delta_A, delta_B

    def run_loop(self, stop_event: threading.Event) -> None:
        """
        Worker 主循环：拉取权重 -> 训练 -> 推送更新。
        """
        while not stop_event.is_set():
            try:
                # 从协调器拉取最新的全局权重
                global_A, global_B, global_version = self.coordinator.get_expert_weights(
                    self.specialty
                )

                # 执行本地训练
                lora_delta = self.local_training_step(global_A, global_B)

                # 模拟网络延迟
                network_delay = random.uniform(0.5, 2.0)
                time.sleep(network_delay)

                # 推送更新到协调器
                self.coordinator.push_expert_update(
                    self.specialty, lora_delta, global_version
                )

                # 等待一段时间再进行下一轮
                time.sleep(random.uniform(3, 8))

            except Exception as exc:  # pragma: no cover
                print(f"[{self.worker_id}] Error: {exc}")
                time.sleep(5)

        print(f"[{self.worker_id}] Shutting down.")

