"""
Helper script to launch the FedMoE simulation from a JSON config.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from fedmoe import DEFAULT_EXPERTS, DEFAULT_WORKERS, SimulationConfig, run_simulation


def load_config(path: Path) -> SimulationConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found.")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    training_duration = float(payload.get("training_duration_s", 15.0))
    experts = payload.get("experts", DEFAULT_EXPERTS)
    workers_data = payload.get("workers") or DEFAULT_WORKERS

    worker_specs: List[Tuple[str, str, float]] = []
    for entry in workers_data:
        worker_specs.append(
            (
                entry["id"],
                entry["specialty"],
                float(entry["speed"]),
            )
        )

    return SimulationConfig(
        training_duration_s=training_duration,
        worker_specs=worker_specs,
        expert_names=experts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedMoE with a JSON config file.")
    parser.add_argument(
        "--config",
        default="configs/sample_run.json",
        help="Path to simulation config JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    run_simulation(config)


if __name__ == "__main__":
    main()

