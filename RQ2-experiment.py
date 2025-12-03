#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from fedmoe.experiments import ExperimentConfig, RQ2Experiment


def main():
    parser = argparse.ArgumentParser(description="RQ2 - Communication vs Accuracy (single-machine 3xRTX3060)")
    # Data
    parser.add_argument("--dataset", type=str, default="dataset/test.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    # Model
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    # Training
    parser.add_argument("--num-rounds", type=int, default=6)
    parser.add_argument("--round-duration", type=float, default=3.0)
    # Local GPU mapping
    parser.add_argument("--gpu-workers", type=str, default="0,1", help="comma-separated worker GPU ids, e.g., 0,1")
    parser.add_argument("--gpu-agg", type=int, default=2, help="aggregator GPU id, e.g., 2")
    # Experts (2 GPUs -> commonly 2 experts)
    parser.add_argument("--experts", type=str, default="python,sql", help="comma-separated expert names")
    # Grid control (optional)
    parser.add_argument("--compressions", type=str, default="1.0,0.5", help="comma-separated compression ratios")
    parser.add_argument("--sync-intervals", type=str, default="2.0,5.0", help="comma-separated sync intervals")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
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

    print(f"[RQ2] Workers on GPUs: {worker_gpu_ids}, Aggregator on GPU: {args.gpu_agg}")
    print(f"[RQ2] Experts: {expert_specialties}")

    # Run grid
    exp = RQ2Experiment(cfg)

    compressions = [float(x) for x in args.compressions.split(",") if x.strip()]
    syncs = [float(x) for x in args.sync_intervals.split(",") if x.strip()]

    for c in compressions:
        for s in syncs:
            exp.run_with_config(compression_ratio=c, sync_interval=s)

    results = exp.analyze_trade_off()

    print("\n[RQ2] Pareto frontier:")
    for item in results.get("pareto_frontier", []):
        print(item)

    return 0


if __name__ == "__main__":
    sys.exit(main())

