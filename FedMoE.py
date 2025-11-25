from __future__ import annotations

import argparse
import json
from typing import List, Tuple

from fedmoe import DEFAULT_EXPERTS, DEFAULT_WORKERS, SimulationConfig, run_simulation


def parse_worker_arg(worker_arg: str) -> Tuple[str, str, float]:
    """
    Parse CLI worker specs of the form 'worker_id:specialty:speed'.
    """
    parts = worker_arg.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Worker spec must be 'id:specialty:speed'.")
    worker_id, specialty, speed = parts
    try:
        speed_factor = float(speed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Speed must be numeric.") from exc
    return worker_id, specialty, speed_factor


def load_worker_specs(args: argparse.Namespace) -> List[Tuple[str, str, float]]:
    if args.worker_json:
        try:
            with open(args.worker_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return [(entry["id"], entry["specialty"], float(entry["speed"])) for entry in payload]
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(f"Invalid worker JSON: {exc}") from exc
    if args.worker:
        return [parse_worker_arg(spec) for spec in args.worker]
    return DEFAULT_WORKERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FedMoE Qwen-style federated simulation.")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Training phase duration (seconds).",
    )
    parser.add_argument(
        "--worker",
        action="append",
        metavar="ID:SPECIALTY:SPEED",
        help="Inline worker spec. Can be repeated.",
    )
    parser.add_argument(
        "--worker-json",
        help="Path to JSON file defining worker specs [{'id':..., 'specialty':..., 'speed':...}].",
    )
    parser.add_argument(
        "--experts",
        nargs="+",
        default=DEFAULT_EXPERTS,
        help="List of expert names to register.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_specs = load_worker_specs(args)
    config = SimulationConfig(training_duration_s=args.duration, worker_specs=worker_specs, expert_names=args.experts)
    run_simulation(config)


if __name__ == "__main__":
    main()