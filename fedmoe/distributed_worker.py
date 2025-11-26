"""
分布式 Worker 客户端，可以连接到远程 Coordinator。
"""

from __future__ import annotations

import base64
import json
import random
import threading
import time
from typing import List, Tuple

import numpy as np
import requests

from .qwen_finetune import QwenLoRAExpert, load_jsonl_dataset, prepare_training_texts_from_jsonl


class DistributedWorker:
    """
    分布式 Worker，可以连接到远程 Coordinator。
    """

    def __init__(
        self,
        coordinator_url: str,
        worker_id: str,
        specialty: str,
        speed_factor: float,
        base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
        training_data_path: str | None = None,
        use_modelscope: bool = True,
        sync_interval: float = 10.0,
        use_proxy: bool = False,
    ):
        """
        初始化分布式 Worker。

        Args:
            coordinator_url: Coordinator 服务器地址（如 "http://192.168.1.100:5000"）
            worker_id: Worker ID
            specialty: 专家类型（如 'python_expert', 'sql_expert', 'docs_expert'）
            speed_factor: 速度因子（影响训练时间）
            base_model_name: 基础 Qwen 模型名称
            training_data_path: JSONL 数据集文件路径
            use_modelscope: 是否使用 ModelScope 加载模型
            sync_interval: 定时上传梯度的间隔（秒，默认10秒）
            use_proxy: 是否使用系统代理访问 Coordinator（默认: False）
        """
        self.coordinator_url = coordinator_url.rstrip("/")
        self.worker_id = worker_id
        self.specialty = specialty
        self.speed_factor = speed_factor
        self.base_model_name = base_model_name
        self.sync_interval = sync_interval
        self.use_proxy = use_proxy

        # 创建 HTTP 会话，默认禁用系统代理，避免本地调试端口被劫持
        self.session = requests.Session()
        if not use_proxy:
            self.session.trust_env = False

        # 检查 coordinator 是否可用
        try:
            response = self.session.get(f"{self.coordinator_url}/health", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Coordinator 健康检查失败: {response.status_code}")
            print(f"[{worker_id}] 成功连接到 Coordinator: {coordinator_url}")
        except Exception as e:
            raise ConnectionError(f"无法连接到 Coordinator {coordinator_url}: {e}")

        # 加载训练数据
        if training_data_path:
            try:
                jsonl_data = load_jsonl_dataset(training_data_path)
                training_data = prepare_training_texts_from_jsonl(jsonl_data, use_chat_template=True)
                print(f"[{worker_id}] 从 {training_data_path} 加载了 {len(training_data)} 条训练数据")
            except Exception as e:
                print(f"[{worker_id}] 警告: 无法从 {training_data_path} 加载数据: {e}")
                training_data = None
        else:
            training_data = None

        self.training_data = training_data

        # 创建本地 Qwen 专家实例
        self.expert = QwenLoRAExpert(
            expert_name=f"{worker_id}_{specialty}",
            base_model_name=base_model_name,
            use_modelscope=use_modelscope,
        )

        # 注册专家到 coordinator（如果还没有注册）
        self._ensure_expert_registered()

        print(
            f"[{worker_id}] 分布式 Worker 初始化完成 (Specialty: {specialty}, "
            f"Speed: {speed_factor:.1f}x, Data samples: {len(training_data) if training_data else 0})"
        )

    def _ensure_expert_registered(self):
        """确保专家已注册到 coordinator"""
        model_dim = self.expert.base_model.config.hidden_size
        lora_rank = self.expert.lora_rank

        try:
            response = self.session.post(
                f"{self.coordinator_url}/register_expert",
                json={
                    "expert_name": self.specialty,
                    "model_dim": int(model_dim),
                    "lora_rank": int(lora_rank),
                },
                timeout=10,
            )
            if response.status_code == 200:
                print(f"[{self.worker_id}] 专家 {self.specialty} 已注册到 Coordinator")
            else:
                print(f"[{self.worker_id}] 警告: 注册专家失败: {response.text}")
        except Exception as e:
            print(f"[{self.worker_id}] 警告: 注册专家时出错: {e}")

    def _get_expert_weights(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """从 coordinator 获取专家权重"""
        try:
            response = self.session.post(
                f"{self.coordinator_url}/get_expert_weights",
                json={"expert_name": self.specialty},
                timeout=30,
            )
            if response.status_code != 200:
                raise ValueError(f"获取权重失败: {response.text}")

            data = response.json()
            lora_A = _base64_to_numpy(data["lora_A"])
            lora_B = _base64_to_numpy(data["lora_B"])
            version = data["version"]
            return lora_A, lora_B, version
        except Exception as e:
            print(f"[{self.worker_id}] 错误: 获取权重失败: {e}")
            raise

    def _push_expert_update(
        self, lora_delta: Tuple[np.ndarray, np.ndarray], worker_model_version: int
    ) -> None:
        """推送专家更新到 coordinator"""
        delta_A, delta_B = lora_delta
        try:
            response = self.session.post(
                f"{self.coordinator_url}/push_expert_update",
                json={
                    "expert_name": self.specialty,
                    "lora_A": _numpy_to_base64(delta_A),
                    "lora_B": _numpy_to_base64(delta_B),
                    "worker_model_version": worker_model_version,
                },
                timeout=60,
            )
            if response.status_code != 200:
                raise ValueError(f"推送更新失败: {response.text}")
        except Exception as e:
            print(f"[{self.worker_id}] 错误: 推送更新失败: {e}")
            raise

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
        current_A, current_B = self.expert.get_lora_weights()

        # 检查形状是否匹配
        if current_A.shape != lora_A.shape or current_B.shape != lora_B.shape:
            print(
                f"  [{self.worker_id}] 警告: 权重形状不匹配！"
                f"当前: A={current_A.shape}, B={current_B.shape}, "
                f"全局: A={lora_A.shape}, B={lora_B.shape}"
            )
            return np.zeros_like(current_A), np.zeros_like(current_B)

        delta_A = lora_A - current_A
        delta_B = lora_B - current_B
        self.expert.update_lora_weights(delta_A, delta_B, learning_rate=1.0)

        # 根据速度因子调整训练参数
        batch_size = max(1, int(4 / self.speed_factor))
        num_epochs = max(1, int(1 * self.speed_factor))

        train_time = random.uniform(2, 5) * self.speed_factor
        print(
            f"  [{self.worker_id}] Starting Qwen fine-tuning on {self.specialty}... "
            f"(Est. {train_time:.2f}s, batch_size={batch_size}, epochs={num_epochs})"
        )

        # 执行真实的微调
        if self.training_data:
            delta_A, delta_B = self.expert.fine_tune_step(
                training_data=self.training_data,
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=5e-5,
            )
        else:
            # 如果没有训练数据，返回零更新
            delta_A = np.zeros_like(lora_A)
            delta_B = np.zeros_like(lora_B)

        print(f"  [{self.worker_id}] Qwen fine-tuning complete.")
        return delta_A, delta_B

    def run_loop(self, stop_event: threading.Event) -> None:
        """
        Worker 主循环：拉取权重 -> 训练 -> 定时推送更新。
        """
        last_sync_time = time.time()
        
        while not stop_event.is_set():
            try:
                # 从 coordinator 拉取最新的全局权重
                global_A, global_B, global_version = self._get_expert_weights()

                # 执行本地训练
                lora_delta = self.local_training_step(global_A, global_B)

                # 检查是否到了定时上传的时间
                current_time = time.time()
                time_since_last_sync = current_time - last_sync_time
                
                if time_since_last_sync >= self.sync_interval:
                    # 模拟网络延迟
                    network_delay = random.uniform(0.5, 2.0)
                    time.sleep(network_delay)

                    # 推送更新到 coordinator
                    self._push_expert_update(lora_delta, global_version)
                    last_sync_time = current_time
                    print(f"[{self.worker_id}] 已上传梯度更新到 Coordinator (间隔: {self.sync_interval}s)")
                else:
                    # 等待到下一个同步时间
                    remaining_time = self.sync_interval - time_since_last_sync
                    print(f"[{self.worker_id}] 等待 {remaining_time:.1f}s 后上传梯度...")
                    time.sleep(min(remaining_time, 1.0))  # 最多等待1秒，然后重新检查

            except Exception as exc:
                print(f"[{self.worker_id}] Error: {exc}")
                time.sleep(5)

        print(f"[{self.worker_id}] Shutting down.")


def _numpy_to_base64(arr: np.ndarray) -> str:
    """将 numpy 数组转换为 base64 字符串"""
    arr_bytes = arr.tobytes()
    arr_b64 = base64.b64encode(arr_bytes).decode("utf-8")
    return json.dumps({
        "data": arr_b64,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    })


def _base64_to_numpy(data: str | dict) -> np.ndarray:
    """将 base64 字符串转换为 numpy 数组"""
    if isinstance(data, str):
        data = json.loads(data)

    arr_bytes = base64.b64decode(data["data"])
    arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
    return arr.reshape(data["shape"])

