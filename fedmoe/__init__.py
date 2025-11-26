"""
FedMoE package entry.

Provides modular building blocks for federated mixture-of-experts simulations
that demonstrate cross-machine LoRA fine-tuning for Qwen-style models.
"""

from .config import (
    MODEL_DIM,
    LORA_RANK,
    SERVER_LR,
    STALENESS_DECAY,
    DEFAULT_WORKERS,
    DEFAULT_EXPERTS,
)
from .experts import ExpertModel
from .coordinator import CentralCoordinator
from .workers import HeterogeneousWorker
from .agent import InferenceAgent
from .config import SimulationConfig
from .qwen_finetune import initialize_qwen_system, run_qwen_simulation

__all__ = [
    "MODEL_DIM",
    "LORA_RANK",
    "SERVER_LR",
    "STALENESS_DECAY",
    "DEFAULT_WORKERS",
    "DEFAULT_EXPERTS",
    "ExpertModel",
    "CentralCoordinator",
    "SimulationConfig",
    "initialize_qwen_system",
    "run_qwen_simulation",
]

