#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from fedmoe.experiments import ExperimentConfig, RQ2Experiment


def main():
    parser = argparse.ArgumentParser(description="RQ2 - Specialization Granularity vs Quality (single-machine 3xRTX3060)")
    # Data
    parser.add_argument("--dataset", type=str, default="dataset/ds1000.jsonl")
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
    # Granularity sets: e.g., "generalist|python,sql|python,sql,docs"
    parser.add_argument("--sets", type=str, default="generalist|python,sql|python,sql,docs",
                        help="'|' separated expert sets; each set is comma-separated experts")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    worker_gpu_ids = [int(x) for x in args.gpu_workers.split(",") if x.strip() != ""]

    # Parse sets -> list of list[str]
    rq2_sets: list[list[str]] = []
    for seg in args.sets.split("|"):
        seg = seg.strip()
        if not seg:
            continue
        rq2_sets.append([x.strip() for x in seg.split(",") if x.strip()])

    # Use first set as default experts for config display; RQ2 class will iterate sets
    default_experts = rq2_sets[0] if rq2_sets else ["generalist"]

    cfg = ExperimentConfig(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        model_dim=args.model_dim,
        lora_rank=args.lora_rank,
        num_rounds=args.num_rounds,
        round_duration=args.round_duration,
        num_experts=len(default_experts),
        expert_specialties=default_experts,
        worker_gpu_ids=worker_gpu_ids,
        aggregator_gpu_id=args.gpu_agg,
        use_gateway_model=True,
        rq2_expert_sets=rq2_sets,
    )

    print(f"[RQ2] Workers on GPUs: {worker_gpu_ids}, Aggregator on GPU: {args.gpu_agg}")
    print(f"[RQ2] Sets: {rq2_sets}")

    exp = RQ2Experiment(cfg)
    results = exp.run()

    print("\n[RQ2] Ranking (best overall first):")
    for r in results.get("ranking", []):
        print({"setting": r.get("setting"), "overall": r.get("overall_quality"), "per_domain": r.get("per_domain")})

    return 0


if __name__ == "__main__":
    sys.exit(main())

