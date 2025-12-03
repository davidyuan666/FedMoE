"""
Worker abstractions that simulate heterogeneous cross-machine LoRA training.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Tuple, Optional

import numpy as np

from .coordinator import CentralCoordinator


class HeterogeneousWorker:
    """
    Simulates a GPU node with its own specialty, dataset, and performance profile.
    """

    def __init__(
        self,
        worker_id: str,
        coordinator: CentralCoordinator,
        specialty: str,
        speed_factor: float,
        device_id: Optional[int] = None,
    ) -> None:
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.specialty = specialty
        self.speed_factor = speed_factor
        self.device_id = device_id
        self.local_data = f"Simulated local data for {specialty}"
        print(
            f"Worker {self.worker_id} (Specialty: {self.specialty}, Speed: {self.speed_factor:.1f}x, "
            f"GPU: {self.device_id if self.device_id is not None else 'CPU'}) initialized."
        )

    def simulate_local_training(
        self, lora_A: np.ndarray, lora_B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_time = random.uniform(2, 5) * self.speed_factor
        print(
            f"  [{self.worker_id}] Starting training on {self.specialty} (GPU {self.device_id})... "
            f"(Est. {train_time:.2f}s)"
        )
        time.sleep(train_time)
        delta_A = np.random.randn(*lora_A.shape) * 0.05
        delta_B = np.random.randn(*lora_B.shape) * 0.05
        print(f"  [{self.worker_id}] Training complete.")
        return delta_A, delta_B

    def run_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                global_A, global_B, global_version = self.coordinator.get_expert_weights(self.specialty)
                lora_delta = self.simulate_local_training(global_A, global_B)
                network_delay = random.uniform(0.5, 2.0)
                time.sleep(network_delay)
                self.coordinator.push_expert_update(self.specialty, lora_delta, global_version)
                time.sleep(random.uniform(3, 8))
            except Exception as exc:  # pragma: no cover - simulation logging
                print(f"[{self.worker_id}] Error: {exc}")
                time.sleep(5)
        print(f"[{self.worker_id}] Shutting down.")
