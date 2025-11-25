"""
Central coordinator responsible for aggregating and distributing expert updates.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import numpy as np

from .config import SERVER_LR, STALENESS_DECAY
from .experts import ExpertModel


class CentralCoordinator:
    """
    Manages expert registration, versioning, and staleness-aware gradient pushes.
    """

    def __init__(self) -> None:
        self.experts: Dict[str, ExpertModel] = {}
        self.lock = threading.Lock()
        print("Central Coordinator Initialized.")

    def register_expert(self, expert_name: str, model_dim: int, lora_rank: int) -> None:
        with self.lock:
            if expert_name not in self.experts:
                self.experts[expert_name] = ExpertModel(expert_name, model_dim, lora_rank)

    def get_expert_weights(self, expert_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
        with self.lock:
            if expert_name not in self.experts:
                raise ValueError(f"Expert {expert_name} not registered.")
            expert = self.experts[expert_name]
            return expert.lora_A.copy(), expert.lora_B.copy(), expert.version

    def push_expert_update(
        self,
        expert_name: str,
        lora_delta: Tuple[np.ndarray, np.ndarray],
        worker_model_version: int,
    ) -> None:
        with self.lock:
            if expert_name not in self.experts:
                return

            expert = self.experts[expert_name]
            staleness = expert.version - worker_model_version
            decay_factor = 1.0 / (1.0 + STALENESS_DECAY * staleness)
            delta_A, delta_B = lora_delta
            expert.lora_A += SERVER_LR * decay_factor * delta_A
            expert.lora_B += SERVER_LR * decay_factor * delta_B
            expert.version += 1

            print(
                f"[Coordinator] Updated {expert.name.upper()} (v{expert.version}) "
                f"from Worker (Staleness: {staleness}, Decay: {decay_factor:.2f})"
            )

