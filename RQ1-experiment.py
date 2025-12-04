#!/usr/bin/env python3
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fedmoe.experiments import ExperimentConfig, RQ1Experiment


def main():
    parser = argparse.ArgumentParser(description="RQ1 - Federated vs Isolated (single-machine 3xRTX3060)")
    # Data
    parser.add_argument("--dataset", type=str, default="dataset/ds1000.jsonl")
    parser.add_argument("--num-samples", type=int, default=100)
    # Worker-specific datasets (optional)
    parser.add_argument("--worker-datasets", type=str, default=None, 
                       help="comma-separated worker dataset paths, e.g., domains/pandas.jsonl,domains/sql.jsonl")
    parser.add_argument("--min-dataset-samples", type=int, default=50,
                       help="minimum non-empty samples required to enable finetuning on a dataset (per expert)")
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

    def count_non_empty_jsonl(p: Path) -> int:
        try:
            n = 0
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict) and obj:
                        n += 1
            return n
        except FileNotFoundError:
            return 0

    # Parse worker-specific datasets if provided
    worker_datasets = None
    if args.worker_datasets:
        dataset_paths = [x.strip() for x in args.worker_datasets.split(",") if x.strip() != ""]
        if len(dataset_paths) != len(expert_specialties):
            print(f"[RQ1] Warning: Number of datasets ({len(dataset_paths)}) != number of experts ({len(expert_specialties)})")
        # Map by position up to min length
        limit = min(len(dataset_paths), len(expert_specialties))
        raw_map = {spec: dataset_paths[i] for i, spec in enumerate(expert_specialties[:limit])}
        # Count and filter by min-dataset-samples
        eligible_specs: List[str] = []
        filtered_map: Dict[str, str] = {}
        print(f"[RQ1] Dataset sample threshold: >= {args.min_dataset_samples}")
        for spec, p in raw_map.items():
            path = Path(p)
            cnt = count_non_empty_jsonl(path)
            print(f"[RQ1] Dataset for expert '{spec}': {path} -> {cnt} valid items")
            if cnt >= args.min_dataset_samples:
                eligible_specs.append(spec)
                filtered_map[spec] = str(path)
            else:
                print(f"[RQ1] Skip finetuning for expert '{spec}' (insufficient data < {args.min_dataset_samples})")
        if not eligible_specs:
            print("[RQ1] Error: No expert has enough data to finetune. Abort.")
            return 1
        # Trim experts and GPUs to eligible set
        expert_specialties = eligible_specs
        worker_datasets = filtered_map
        if len(worker_gpu_ids) > len(expert_specialties):
            print(f"[RQ1] Trimming worker GPUs from {worker_gpu_ids} to match {len(expert_specialties)} experts")
            worker_gpu_ids = worker_gpu_ids[:len(expert_specialties)]

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
        worker_datasets=worker_datasets,
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

