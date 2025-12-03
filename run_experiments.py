#!/usr/bin/env python3
"""
Main entry point for running all three RQ experiments.

Usage:
    python run_experiments.py                    # Run with default config
    python run_experiments.py --num-rounds 20   # Custom number of rounds
    python run_experiments.py --num-samples 200 # Use more samples
"""

import argparse
import logging
import sys
from pathlib import Path

from fedmoe.experiments import ExperimentConfig, ExperimentOrchestrator


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "experiments.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive federated LoRA experiments (RQ1, RQ2, RQ3)"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/test.jsonl",
        help="Path to DS1000 dataset (JSONL format)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to use from dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-dim",
        type=int,
        default=768,
        help="Model dimension for LoRA"
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of training rounds"
    )
    parser.add_argument(
        "--round-duration",
        type=float,
        default=30.0,
        help="Duration of each round in seconds"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--num-experts",
        type=int,
        default=3,
        help="Number of experts"
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=5.0,
        help="Synchronization interval in seconds"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="experiment_report.txt",
        help="Output file for summary report"
    )
    
    # Experiment selection
    parser.add_argument(
        "--rq",
        type=str,
        choices=["all", "rq1", "rq2", "rq3"],
        default="all",
        help="Which experiments to run"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("main")
    
    logger.info("=" * 80)
    logger.info("FEDERATED LORA EXPERIMENTS")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Model dim: {args.model_dim}")
    logger.info(f"LoRA rank: {args.lora_rank}")
    logger.info(f"Num rounds: {args.num_rounds}")
    logger.info(f"Num experts: {args.num_experts}")
    logger.info("=" * 80)
    
    # Create config
    config = ExperimentConfig(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        model_dim=args.model_dim,
        lora_rank=args.lora_rank,
        num_rounds=args.num_rounds,
        round_duration=args.round_duration,
        num_experts=args.num_experts,
        sync_interval=args.sync_interval,
        use_gateway_model=True
    )
    
    # Run experiments
    orchestrator = ExperimentOrchestrator(config)
    
    try:
        results = orchestrator.run_all_experiments()
        
        # Save results
        orchestrator.save_results(args.output)
        logger.info(f"Results saved to {args.output}")
        
        # Generate and save report
        report = orchestrator.generate_summary_report()
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {args.report}")
        
        # Print summary
        print("\n" + report)
        
        logger.info("=" * 80)
        logger.info("EXPERIMENTS COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

