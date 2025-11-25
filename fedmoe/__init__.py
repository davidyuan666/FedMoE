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
from .simulation import (
    SimulationConfig,
    initialize_system,
    run_training_phase,
    run_inference_phase,
    run_simulation,
)

__all__ = [
    "MODEL_DIM",
    "LORA_RANK",
    "SERVER_LR",
    "STALENESS_DECAY",
    "DEFAULT_WORKERS",
    "DEFAULT_EXPERTS",
    "ExpertModel",
    "CentralCoordinator",
    "HeterogeneousWorker",
    "InferenceAgent",
    "SimulationConfig",
    "initialize_system",
    "run_training_phase",
    "run_inference_phase",
    "run_simulation",
]

