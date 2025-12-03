from __future__ import annotations

from typing import Dict, Optional, Tuple

import copy
import threading
import numpy as np

from .config import SERVER_LR, STALENESS_DECAY


class PeftCoordinator:
    """
    Aggregates LoRA adapter state dict deltas (PEFT) per expert.
    Stores and versions adapter state dicts (CPU tensors) and applies
    staleness-aware server updates.
    """

    def __init__(self, aggregator_device_id: Optional[int] = None) -> None:
        self.aggregator_device_id = aggregator_device_id
        self._adapters: Dict[str, Dict] = {}
        self._versions: Dict[str, int] = {}
        self._lock = threading.Lock()
        print(
            "[PeftCoordinator] Initialized "
            f"(Aggregator on GPU {self.aggregator_device_id})" if self.aggregator_device_id is not None else ""
        )

    def register_expert(self, expert_name: str) -> None:
        with self._lock:
            if expert_name not in self._adapters:
                self._adapters[expert_name] = {}
                self._versions[expert_name] = 0
                print(f"[PeftCoordinator] Registered expert: {expert_name}")

    def get_adapter_state(self, expert_name: str) -> Tuple[Dict, int]:
        with self._lock:
            if expert_name not in self._adapters:
                raise ValueError(f"Expert {expert_name} not registered")
            state = self._adapters[expert_name]
            version = self._versions[expert_name]
            # Make a deep copy to avoid in-place modifications by clients
            return copy.deepcopy(state), int(version)

    def push_adapter_delta(self, expert_name: str, delta_state: Dict, worker_model_version: int) -> None:
        with self._lock:
            if expert_name not in self._adapters:
                return
            current_version = self._versions[expert_name]
            staleness = current_version - worker_model_version
            decay = 1.0 / (1.0 + STALENESS_DECAY * staleness)

            base = self._adapters[expert_name]
            # Apply SERVER_LR * decay * delta
            for k, dv in delta_state.items():
                # dv is a torch.Tensor on CPU; we keep on CPU
                if k in base:
                    base[k] = base[k] + (SERVER_LR * decay) * dv
                else:
                    base[k] = (SERVER_LR * decay) * dv.clone()

            self._versions[expert_name] = current_version + 1
            print(
                f"[PeftCoordinator] Updated {expert_name} (v{self._versions[expert_name]}) "
                f"(Staleness: {staleness}, Decay: {decay:.2f})"
            )

