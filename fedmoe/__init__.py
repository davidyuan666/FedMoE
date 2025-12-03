"""
FedMoE package entry (minimal).

Provides the core building blocks used by the RQ experiments:
- config: global hyperparameters
- experts: LoRA expert container
- coordinator: central aggregator
- workers: local worker simulation
"""

from .config import (
    LORA_RANK,
    SERVER_LR,
    STALENESS_DECAY,
)
from .experts import ExpertModel
from .coordinator import CentralCoordinator
from .workers import HeterogeneousWorker

__all__ = [
    "LORA_RANK",
    "SERVER_LR",
    "STALENESS_DECAY",
    "ExpertModel",
    "CentralCoordinator",
    "HeterogeneousWorker",
]

