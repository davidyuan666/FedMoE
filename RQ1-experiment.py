#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from fedmoe.experiments import ExperimentConfig, RQ1Experiment


def main():
    parser = argparse.ArgumentParser(description="RQ1 - Federated vs Isolated (single-machine 3xRTX3060)")
    # Data
    parser.add_argument("--dataset", type=str, default="dataset/ds1000.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    # Model
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    # Training
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--round-duration", type=float, default=5.0)
    # Local GPU mapping
    parser.add_argument("--gpu-workers", type=str, default="0,1", help="comma-separated worker GPU ids, e.g., 0,1")
    parser.add_argument("--gpu-agg", type=int, default=2, help="aggregator GPU id, e.g., 2")
    # Experts (2 GPUs -> commonly 2 experts)
    parser.add_argument("--experts", type=str, default="python,sql", help="comma-separated expert names")
    args = parser.parse_args()

    # Ensure logs directory exists before experiment loggers are created
    Path("logs").mkdir(exist_ok=True)

    # Basic console logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    worker_gpu_ids = [int(x) for x in args.gpu_workers.split(",") if x.strip() != ""]
    expert_specialties = [x.strip() for x in args.experts.split(",") if x.strip() != ""]

    cfg = ExperimentConfig(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        model_dim=args.model_dim,
        lora_rank=args.lora_rank,
        num_rounds=args.num_rounds,
        round_duration=args.round_duration,
        num_experts=len(expert_specialties),
        expert_specialties=expert_specialties,
        worker_gpu_ids=worker_gpu_ids,
        aggregator_gpu_id=args.gpu_agg,
        use_gateway_model=True,
    )

    print(f"[RQ1] Workers on GPUs: {worker_gpu_ids}, Aggregator on GPU: {args.gpu_agg}")
    print(f"[RQ1] Experts: {expert_specialties}")

    exp = RQ1Experiment(cfg)
    results = exp.run()

    print("\n[RQ1] Comparison summary:")
    print(results.get("rq1_comparison", {}))

    return 0


if __name__ == "__main__":
    sys.exit(main())

