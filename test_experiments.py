#!/usr/bin/env python3
"""
Quick test script to verify experiment framework works correctly.

Usage:
    python test_experiments.py                    # Run quick test
    python test_experiments.py --verbose          # Verbose output
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test")


def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing imports...")
    try:
        from fedmoe.experiments import (
            ExperimentConfig,
            DS1000DataLoader,
            CodeQualityEvaluator,
            RQ1Experiment,
            RQ2Experiment,
            RQ3Experiment,
            ExperimentOrchestrator
        )
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test dataset loading"""
    logger.info("Testing dataset loading...")
    try:
        from fedmoe.experiments import DS1000DataLoader
        
        loader = DS1000DataLoader("dataset/test.jsonl", num_samples=10)
        
        if not loader.data:
            logger.error("✗ No data loaded")
            return False
        
        logger.info(f"✓ Loaded {len(loader.data)} samples")
        
        # Test splitting by specialty
        specialty_data = loader.split_by_specialty(["python", "sql", "docs"])
        logger.info(f"✓ Split data into specialties: {list(specialty_data.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False


def test_code_evaluation():
    """Test code quality evaluation"""
    logger.info("Testing code evaluation...")
    try:
        from fedmoe.experiments import CodeQualityEvaluator
        
        evaluator = CodeQualityEvaluator()
        
        # Test valid code
        gen_code = "result = df.iloc[List]"
        ref_code = "result = df.iloc[List]"
        metrics = evaluator.evaluate_code(gen_code, ref_code)
        
        if metrics.exact_match != 1.0:
            logger.error("✗ Exact match should be 1.0 for identical code")
            return False
        
        logger.info(f"✓ Code evaluation working: exact_match={metrics.exact_match}")
        
        # Test batch evaluation
        gen_codes = [gen_code, "x = 1"]
        ref_codes = [ref_code, "x = 1"]
        exact, exec_pass, sim = evaluator.batch_evaluate(gen_codes, ref_codes)
        
        logger.info(f"✓ Batch evaluation: exact={exact:.2f}, exec_pass={exec_pass:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Code evaluation failed: {e}")
        return False


def test_config():
    """Test experiment configuration"""
    logger.info("Testing configuration...")
    try:
        from fedmoe.experiments import ExperimentConfig
        
        config = ExperimentConfig(
            num_rounds=2,
            num_samples=10,
            num_experts=2
        )
        
        logger.info(f"✓ Config created: {config.num_rounds} rounds, "
                   f"{config.num_samples} samples, {config.num_experts} experts")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration failed: {e}")
        return False


def test_rq1_experiment():
    """Test RQ1 experiment (quick version)"""
    logger.info("Testing RQ1 experiment...")
    try:
        from fedmoe.experiments import ExperimentConfig, RQ1Experiment
        
        config = ExperimentConfig(
            num_rounds=1,
            num_samples=5,
            num_experts=2,
            expert_specialties=["python", "sql"],
            round_duration=1.0
        )
        
        rq1 = RQ1Experiment(config)
        
        # Run federated training
        logger.info("  Running federated training...")
        fed_results = rq1.run_federated_training()
        
        if not fed_results:
            logger.error("✗ No federated results")
            return False
        
        logger.info(f"✓ Federated training completed: {len(fed_results)} rounds")
        
        # Run isolated training
        logger.info("  Running isolated training...")
        iso_results = rq1.run_isolated_training()
        
        if not iso_results:
            logger.error("✗ No isolated results")
            return False
        
        logger.info(f"✓ Isolated training completed: {len(iso_results)} rounds")
        
        # Compare
        comparison = rq1.compare_results()
        logger.info(f"✓ Comparison: {comparison}")
        
        return True
    except Exception as e:
        logger.error(f"✗ RQ1 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rq2_experiment():
    """Test RQ2 experiment (quick version)"""
    logger.info("Testing RQ2 experiment...")
    try:
        from fedmoe.experiments import ExperimentConfig, RQ2Experiment
        
        config = ExperimentConfig(
            num_rounds=1,
            num_samples=5,
            num_experts=2,
            expert_specialties=["python", "sql"],
            round_duration=1.0
        )
        
        rq2 = RQ2Experiment(config)
        
        # Run with one configuration
        logger.info("  Running with compression=1.0, sync_interval=2.0...")
        results = rq2.run_with_config(compression_ratio=1.0, sync_interval=2.0)
        
        if not results:
            logger.error("✗ No results")
            return False
        
        logger.info(f"✓ RQ2 training completed: {len(results)} rounds")
        
        # Analyze
        analysis = rq2.analyze_trade_off()
        logger.info(f"✓ Analysis: {len(analysis['configurations'])} configurations")
        
        return True
    except Exception as e:
        logger.error(f"✗ RQ2 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rq3_experiment():
    """Test RQ3 experiment (quick version)"""
    logger.info("Testing RQ3 experiment...")
    try:
        from fedmoe.experiments import ExperimentConfig, RQ3Experiment
        
        config = ExperimentConfig(
            num_rounds=1,
            num_samples=5,
            num_experts=2,
            expert_specialties=["python", "sql"],
            round_duration=1.0
        )
        
        rq3 = RQ3Experiment(config)
        
        # Run flat aggregation
        logger.info("  Running flat aggregation...")
        flat_results = rq3.run_flat_aggregation()
        
        if not flat_results:
            logger.error("✗ No flat results")
            return False
        
        logger.info(f"✓ Flat aggregation completed: {len(flat_results)} rounds")
        
        # Compare strategies
        comparison = rq3.compare_aggregation_strategies()
        logger.info(f"✓ Comparison: {list(comparison.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ RQ3 experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestrator():
    """Test experiment orchestrator"""
    logger.info("Testing orchestrator...")
    try:
        from fedmoe.experiments import ExperimentConfig, ExperimentOrchestrator
        
        config = ExperimentConfig(
            num_rounds=1,
            num_samples=5,
            num_experts=2,
            expert_specialties=["python", "sql"],
            round_duration=1.0
        )
        
        orchestrator = ExperimentOrchestrator(config)
        
        # Generate summary (without running full experiments)
        report = orchestrator.generate_summary_report()
        
        if not report:
            logger.error("✗ No report generated")
            return False
        
        logger.info(f"✓ Report generated: {len(report)} characters")
        
        return True
    except Exception as e:
        logger.error(f"✗ Orchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 70)
    logger.info("EXPERIMENT FRAMEWORK TEST SUITE")
    logger.info("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Code Evaluation", test_code_evaluation),
        ("Configuration", test_config),
        ("RQ1 Experiment", test_rq1_experiment),
        ("RQ2 Experiment", test_rq2_experiment),
        ("RQ3 Experiment", test_rq3_experiment),
        ("Orchestrator", test_orchestrator),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info("")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 70)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

