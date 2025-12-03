#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from fedmoe.experiments import ExperimentConfig, RQ3Experiment


def main():
    parser = argparse.ArgumentParser(description="RQ3 - Continual domain shift & rehearsal (single-machine 3xRTX3060)")
    # Data
    parser.add_argument("--dataset", type=str, default="dataset/test.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    # Model
    parser.add_argument("--model-dim", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=16)
    # Training timing
    parser.add_argument("--round-duration", type=float, default=3.0, help="seconds per round (sim speed)")
    # Local GPU mapping
    parser.add_argument("--gpu-workers", type=str, default="0,1", help="comma-separated worker GPU ids, e.g., 0,1")
    parser.add_argument("--gpu-agg", type=int, default=2, help="aggregator GPU id, e.g., 2")
    # Continual learning settings
    parser.add_argument("--phases", type=str, default="python,sql,docs",
                        help="comma-separated domain order, e.g., python,sql,docs")
    parser.add_argument("--phase-rounds", type=int, default=4, help="rounds per phase")
    parser.add_argument("--rehearsal-ratio", type=float, default=0.1,
                        help="proportion of rehearsal weight (0.0=no rehearsal)")
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    worker_gpu_ids = [int(x) for x in args.gpu_workers.split(",") if x.strip() != ""]
    phase_domains = [x.strip() for x in args.phases.split(",") if x.strip()]

    # Build config (expert_specialties unused here; keep for compatibility)
    cfg = ExperimentConfig(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        model_dim=args.model_dim,
        lora_rank=args.lora_rank,
        # generic rounds not used; per-phase rounds used instead
        num_rounds=args.phase_rounds,
        round_duration=args.round_duration,
        num_experts=len(phase_domains),
        expert_specialties=phase_domains,
        worker_gpu_ids=worker_gpu_ids,
        aggregator_gpu_id=args.gpu_agg,
        use_gateway_model=True,
        phase_domains=phase_domains,
        phase_rounds=args.phase_rounds,
        rehearsal_ratio=args.rehearsal_ratio,
    )

    print(f"[RQ3] Workers on GPUs: {worker_gpu_ids}, Aggregator on GPU: {args.gpu_agg}")
    print(f"[RQ3] Phases: {phase_domains}, phase_rounds={args.phase_rounds}, rehearsal_ratio={args.rehearsal_ratio}")

    exp = RQ3Experiment(cfg)
    results = exp.run()

    print("\n[RQ3] Results summary:")
    print({
        "naive_avg_forgetting": results.get("rq3_naive", {}).get("avg_forgetting"),
        "naive_final_overall": results.get("rq3_naive", {}).get("final_overall"),
        "rehearsal_avg_forgetting": results.get("rq3_rehearsal", {}).get("avg_forgetting"),
        "rehearsal_final_overall": results.get("rq3_rehearsal", {}).get("final_overall"),
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
