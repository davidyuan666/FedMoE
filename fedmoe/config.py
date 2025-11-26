"""
Global hyperparameters and defaults for the FedMoE simulation.
"""

MODEL_DIM = 4096  # Base Qwen hidden size
LORA_RANK = 16  # Rank of each expert's LoRA adapter
SERVER_LR = 1.0  # Learning rate applied during aggregation
STALENESS_DECAY = 0.1  # Weight for staleness-aware updates

# Default worker blueprint definitions (id, specialty, speed factor)
DEFAULT_WORKERS = [
    ("Worker-1-Fast", "python_expert", 0.5),
    ("Worker-2-Slow", "python_expert", 2.0),
    ("Worker-3-SQL", "sql_expert", 1.0),
    ("Worker-4-Docs", "docs_expert", 1.5),
    ("Worker-5-Python", "python_expert", 1.0),
]

# Default expert names to register
DEFAULT_EXPERTS = ["python_expert", "sql_expert", "docs_expert"]


# Configuration for Qwen fine-tuning
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SimulationConfig:
    """
    Configuration for Qwen fine-tuning.
    """

    training_duration_s: float = 15.0
    worker_specs: List[Tuple[str, str, float]] = None
    expert_names: List[str] = None

    def __post_init__(self) -> None:
        if self.worker_specs is None:
            self.worker_specs = DEFAULT_WORKERS
        if self.expert_names is None:
            self.expert_names = DEFAULT_EXPERTS
