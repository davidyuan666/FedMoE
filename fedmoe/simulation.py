"""
Simulation helpers for orchestrating the three-phase FedMoE demo.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Tuple

from .agent import InferenceAgent
from .config import DEFAULT_EXPERTS, DEFAULT_WORKERS, LORA_RANK, MODEL_DIM
from .coordinator import CentralCoordinator
from .workers import HeterogeneousWorker
from .qwen_worker import QwenWorker


@dataclass
class SimulationConfig:
    """
    Aggregates knobs controlling the demo lifecycle.
    """

    training_duration_s: float = 15.0
    worker_specs: List[Tuple[str, str, float]] = None
    expert_names: List[str] = None

    def __post_init__(self) -> None:
        if self.worker_specs is None:
            self.worker_specs = DEFAULT_WORKERS
        if self.expert_names is None:
            self.expert_names = DEFAULT_EXPERTS


def initialize_system(config: SimulationConfig) -> Tuple[CentralCoordinator, List[HeterogeneousWorker]]:
    coordinator = CentralCoordinator()
    for expert_name in config.expert_names:
        coordinator.register_expert(expert_name, MODEL_DIM, LORA_RANK)
    workers = [
        HeterogeneousWorker(worker_id, coordinator, specialty, speed_factor)
        for worker_id, specialty, speed_factor in config.worker_specs
    ]
    return coordinator, workers


def run_training_phase(workers: List[HeterogeneousWorker], duration_s: float) -> None:
    print("\n--- [Phase 1: Starting Distributed Training] ---")
    stop_event = threading.Event()
    threads: List[threading.Thread] = []
    for worker in workers:
        thread = threading.Thread(target=worker.run_loop, args=(stop_event,))
        thread.start()
        threads.append(thread)

    time.sleep(duration_s)

    print("\n--- [Phase 2: Stopping Training] ---")
    stop_event.set()
    for thread in threads:
        thread.join(timeout=3.0)
    print("All workers stopped.")


def run_inference_phase(coordinator: CentralCoordinator) -> None:
    print("\n--- [Phase 3: Starting Inference Agent] ---")
    agent = InferenceAgent(coordinator)
    prompts = [
        "Write a python function to sort a list",
        "Write a python script to query the 'users' database",
        "How do I use the new python function 'calculate_metrics'?",
    ]
    for prompt in prompts:
        code = agent.generate_code(prompt)
        print(code)


def run_simulation(config: SimulationConfig = SimulationConfig()) -> None:
    coordinator, workers = initialize_system(config)
    run_training_phase(workers, config.training_duration_s)
    run_inference_phase(coordinator)


def initialize_qwen_system(
    config: SimulationConfig,
    base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
    use_modelscope: bool = True,
    training_data_path: str | None = None,
) -> Tuple[CentralCoordinator, List[QwenWorker]]:
    """
    初始化使用真实 Qwen 模型的系统。

    Args:
        config: 模拟配置
        base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头）
        use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        training_data_path: JSONL 数据集文件路径（如果提供，所有 worker 会使用此数据集）

    Returns:
        (coordinator, workers) 元组
    """
    coordinator = CentralCoordinator()
    for expert_name in config.expert_names:
        coordinator.register_expert(expert_name, MODEL_DIM, LORA_RANK)
    workers = [
        QwenWorker(
            worker_id,
            coordinator,
            specialty,
            speed_factor,
            base_model_name,
            training_data_path=training_data_path,
            use_modelscope=use_modelscope,
        )
        for worker_id, specialty, speed_factor in config.worker_specs
    ]
    return coordinator, workers


def run_qwen_simulation(
    config: SimulationConfig = SimulationConfig(),
    base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
    use_modelscope: bool = True,
    training_data_path: str | None = None,
) -> None:
    """
    运行使用真实 Qwen 模型的微调模拟。

    Args:
        config: 模拟配置
        base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头）
        use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        training_data_path: JSONL 数据集文件路径（如果提供，所有 worker 会使用此数据集）
    """
    print("\n=== [使用真实 Qwen 模型进行微调] ===")
    if training_data_path:
        print(f"使用数据集: {training_data_path}")
    coordinator, workers = initialize_qwen_system(
        config, base_model_name, use_modelscope, training_data_path
    )
    run_training_phase(workers, config.training_duration_s)
    run_inference_phase(coordinator)

