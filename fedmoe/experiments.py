"""
Comprehensive experimental framework for evaluating:
- RQ1: Federated LoRA vs Isolated Local Fine-tuning
- RQ2: Communication Cost vs Accuracy Trade-off
- RQ3: Hierarchical Aggregation for Preserving Specialized Skills
"""

from __future__ import annotations

import json
import time
import threading
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging

from .coordinator import CentralCoordinator, GatewayModel
from .experts import ExpertModel
from .workers import HeterogeneousWorker


# ============================================================================
# Data Structures for Experiment Tracking
# ============================================================================

@dataclass
class CodeGenerationMetrics:
    """Metrics for evaluating code generation quality"""
    exact_match: float = 0.0  # Exact match with reference
    token_match_rate: float = 0.0  # Token-level match rate
    syntax_valid: bool = False  # Whether generated code is syntactically valid
    execution_pass: bool = False  # Whether code executes without error
    semantic_similarity: float = 0.0  # Semantic similarity score (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundMetrics:
    """Metrics for a single training round"""
    round_num: int
    timestamp: float
    # Quality metrics
    avg_code_quality: float = 0.0
    avg_exact_match: float = 0.0
    avg_execution_pass_rate: float = 0.0
    # Communication metrics
    total_communication_bytes: float = 0.0
    num_updates: int = 0
    # Model metrics
    expert_versions: Dict[str, int] = field(default_factory=dict)
    gateway_version: int = 0
    # Specialization metrics
    expert_specialization_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    # Data
    dataset_path: str = "dataset/test.jsonl"
    num_samples: int = 100  # Number of samples to use
    
    # Model
    model_dim: int = 768
    lora_rank: int = 16
    
    # Training
    num_rounds: int = 10
    round_duration: float = 30.0  # seconds per round
    
    # Experts
    num_experts: int = 3
    expert_specialties: List[str] = field(default_factory=lambda: ["python", "sql", "docs"])
    
    # Communication
    sync_interval: float = 5.0  # seconds between syncs
    communication_compression_ratio: float = 1.0  # 1.0 = no compression
    
    # Aggregation
    aggregation_strategy: str = "weighted_average"  # or "hierarchical"
    use_gateway_model: bool = True
    
    # Local single-machine GPU mapping (optional)
    worker_gpu_ids: Optional[List[int]] = None  # e.g., [0, 1]
    aggregator_gpu_id: Optional[int] = None     # e.g., 2
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N rounds
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

class DS1000DataLoader:
    """Load and preprocess DS1000 code generation dataset"""
    
    def __init__(self, dataset_path: str, num_samples: int = 100):
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        self.data = []
        self.load_data()
    
    def load_data(self) -> None:
        """Load dataset from JSONL file"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= self.num_samples:
                        break
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue
            logging.info(f"Loaded {len(self.data)} samples from {self.dataset_path}")
        except FileNotFoundError:
            logging.error(f"Dataset file not found: {self.dataset_path}")
            self.data = []
    
    def split_by_specialty(self, specialties: List[str]) -> Dict[str, List[Dict]]:
        """
        Split dataset by specialty (Python, SQL, Docs, etc.)
        Uses simple heuristics based on keywords in the prompt
        """
        specialty_data = {spec: [] for spec in specialties}
        
        keywords = {
            "python": ["python", "pandas", "numpy", "dataframe", "list", "dict"],
            "sql": ["sql", "select", "query", "database", "table", "join"],
            "docs": ["documentation", "readme", "comment", "docstring", "example"]
        }
        
        for item in self.data:
            prompt = item.get("prompt", "").lower()
            assigned = False
            
            for specialty, kws in keywords.items():
                if specialty in specialty_data and any(kw in prompt for kw in kws):
                    specialty_data[specialty].append(item)
                    assigned = True
                    break
            
            # If not assigned, randomly assign to a specialty
            if not assigned and specialty_data:
                specialty = random.choice(specialties)
                specialty_data[specialty].append(item)
        
        for spec, data in specialty_data.items():
            logging.info(f"Specialty '{spec}': {len(data)} samples")
        
        return specialty_data
    
    def get_batch(self, specialty: str, batch_size: int = 4) -> List[Dict]:
        """Get a batch of samples for a specialty"""
        # This would be called during training
        return random.sample(self.data, min(batch_size, len(self.data)))


# ============================================================================
# Code Quality Evaluation
# ============================================================================

class CodeQualityEvaluator:
    """Evaluate generated code quality"""
    
    @staticmethod
    def evaluate_code(generated_code: str, reference_code: str) -> CodeGenerationMetrics:
        """
        Evaluate generated code against reference code
        """
        metrics = CodeGenerationMetrics()
        
        # Exact match
        metrics.exact_match = 1.0 if generated_code.strip() == reference_code.strip() else 0.0
        
        # Token-level match
        gen_tokens = set(generated_code.split())
        ref_tokens = set(reference_code.split())
        if ref_tokens:
            metrics.token_match_rate = len(gen_tokens & ref_tokens) / len(gen_tokens | ref_tokens)
        
        # Syntax validation
        try:
            compile(generated_code, '<string>', 'exec')
            metrics.syntax_valid = True
        except SyntaxError:
            metrics.syntax_valid = False
        
        # Semantic similarity (simplified: based on token overlap)
        metrics.semantic_similarity = metrics.token_match_rate
        
        # Execution pass (simplified: assume pass if syntax valid)
        metrics.execution_pass = metrics.syntax_valid
        
        return metrics
    
    @staticmethod
    def batch_evaluate(
        generated_codes: List[str],
        reference_codes: List[str]
    ) -> Tuple[float, float, float]:
        """
        Batch evaluate multiple code samples
        Returns: (avg_exact_match, avg_execution_pass_rate, avg_semantic_similarity)
        """
        if not generated_codes:
            return 0.0, 0.0, 0.0
        
        metrics_list = [
            CodeQualityEvaluator.evaluate_code(gen, ref)
            for gen, ref in zip(generated_codes, reference_codes)
        ]
        
        avg_exact_match = np.mean([m.exact_match for m in metrics_list])
        avg_execution_pass = np.mean([float(m.execution_pass) for m in metrics_list])
        avg_semantic_sim = np.mean([m.semantic_similarity for m in metrics_list])
        
        return avg_exact_match, avg_execution_pass, avg_semantic_sim


# ============================================================================
# Experiment Runners
# ============================================================================

class RQ1Experiment:
    """
    RQ1: Does federated LoRA improve code-generation quality over isolated local fine-tuning?
    
    Compares:
    1. Federated LoRA: Multiple workers with shared gateway model
    2. Isolated Local: Each worker trains independently without sharing
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger("RQ1")
        self.data_loader = DS1000DataLoader(config.dataset_path, config.num_samples)
        self.evaluator = CodeQualityEvaluator()
        
        # Results tracking
        self.federated_results: List[RoundMetrics] = []
        self.isolated_results: List[RoundMetrics] = []
    
    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def run_federated_training(self) -> List[RoundMetrics]:
        """Run federated LoRA training with shared gateway model"""
        self.logger.info("Starting Federated LoRA Training")
        
        # Setup coordinator with gateway model
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        # Register experts
        for specialty in self.config.expert_specialties:
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        # Split data by specialty
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        # Create workers
        workers = []
        for i, specialty in enumerate(self.config.expert_specialties):
            worker = HeterogeneousWorker(
                worker_id=f"FedWorker-{i}",
                coordinator=coordinator,
                specialty=specialty,
                speed_factor=random.uniform(0.8, 1.5)
            )
            workers.append(worker)
        
        # Run training rounds
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Simulate training concurrently on local GPUs
            weights_map = {}
            for worker in workers:
                lora_A, lora_B, version = coordinator.get_expert_weights(worker.specialty)
                weights_map[worker.worker_id] = (lora_A, lora_B, version, worker)

            deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            threads: List[threading.Thread] = []

            def run_train(wid: str):
                lA, lB, _, w = weights_map[wid]
                try:
                    dA, dB = w.simulate_local_training(lA, lB)
                    deltas[wid] = (dA, dB)
                except Exception as e:
                    self.logger.error(f"Worker error ({wid}): {e}")

            for wid in weights_map.keys():
                t = threading.Thread(target=run_train, args=(wid,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            # Push updates
            for wid, (dA, dB) in deltas.items():
                _, _, ver, w = weights_map[wid]
                coordinator.push_expert_update(w.specialty, (dA, dB), ver)
            
            # Collect metrics
            round_metrics = self._collect_round_metrics(
                round_num,
                coordinator,
                specialty_data,
                round_start
            )
            results.append(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}, "
                           f"Comm={round_metrics.total_communication_bytes:.0f} bytes")
            
            time.sleep(self.config.round_duration)
        
        self.federated_results = results
        return results
    
    def run_isolated_training(self) -> List[RoundMetrics]:
        """Run isolated local training without sharing"""
        self.logger.info("Starting Isolated Local Training")
        
        # Create independent coordinators for each worker
        coordinators = {
            specialty: CentralCoordinator(enable_gateway=False)
            for specialty in self.config.expert_specialties
        }
        
        # Register experts in each coordinator
        for specialty, coordinator in coordinators.items():
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        # Split data by specialty
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        # Run training rounds
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Simulate training for each specialty independently
            for specialty, coordinator in coordinators.items():
                try:
                    lora_A, lora_B, version = coordinator.get_expert_weights(specialty)
                    # Simulate training
                    delta_A = np.random.randn(*lora_A.shape) * 0.05
                    delta_B = np.random.randn(*lora_B.shape) * 0.05
                    coordinator.push_expert_update(specialty, (delta_A, delta_B), version)
                except Exception as e:
                    self.logger.error(f"Training error: {e}")
            
            # Collect metrics (using first coordinator as reference)
            first_coordinator = list(coordinators.values())[0]
            round_metrics = self._collect_round_metrics(
                round_num,
                first_coordinator,
                specialty_data,
                round_start
            )
            results.append(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}")
            
            time.sleep(self.config.round_duration)
        
        self.isolated_results = results
        return results
    
    def _collect_round_metrics(
        self,
        round_num: int,
        coordinator: CentralCoordinator,
        specialty_data: Dict[str, List[Dict]],
        round_start: float
    ) -> RoundMetrics:
        """Collect metrics for a training round"""
        metrics = RoundMetrics(
            round_num=round_num,
            timestamp=time.time()
        )
        
        # Simulate code quality evaluation
        all_exact_matches = []
        all_execution_passes = []
        
        for specialty, data in specialty_data.items():
            if data:
                # Simulate generated codes (in real scenario, would use model)
                generated_codes = [item.get("reference_code", "")[:50] + "..." for item in data[:5]]
                reference_codes = [item.get("reference_code", "") for item in data[:5]]
                
                exact_match, exec_pass, _ = self.evaluator.batch_evaluate(
                    generated_codes,
                    reference_codes
                )
                all_exact_matches.append(exact_match)
                all_execution_passes.append(exec_pass)
        
        metrics.avg_exact_match = np.mean(all_exact_matches) if all_exact_matches else 0.0
        metrics.avg_execution_pass_rate = np.mean(all_execution_passes) if all_execution_passes else 0.0
        metrics.avg_code_quality = (metrics.avg_exact_match + metrics.avg_execution_pass_rate) / 2
        
        # Communication metrics
        metrics.total_communication_bytes = len(self.config.expert_specialties) * \
                                           self.config.model_dim * self.config.lora_rank * 4 * 2  # 2 matrices
        metrics.num_updates = len(self.config.expert_specialties)
        
        # Expert versions
        for specialty in self.config.expert_specialties:
            try:
                _, _, version = coordinator.get_expert_weights(specialty)
                metrics.expert_versions[specialty] = version
            except:
                pass
        
        # Gateway version
        if coordinator.gateway_model:
            metrics.gateway_version = coordinator.gateway_model.version
        
        return metrics
    
    def compare_results(self) -> Dict[str, Any]:
        """Compare federated vs isolated results"""
        if not self.federated_results or not self.isolated_results:
            return {}
        
        fed_qualities = [r.avg_code_quality for r in self.federated_results]
        iso_qualities = [r.avg_code_quality for r in self.isolated_results]
        
        fed_avg = np.mean(fed_qualities)
        iso_avg = np.mean(iso_qualities)
        improvement = (fed_avg - iso_avg) / (iso_avg + 1e-6) * 100
        
        return {
            "federated_avg_quality": fed_avg,
            "isolated_avg_quality": iso_avg,
            "improvement_percent": improvement,
            "federated_final_quality": fed_qualities[-1] if fed_qualities else 0,
            "isolated_final_quality": iso_qualities[-1] if iso_qualities else 0,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run complete RQ1 experiment"""
        self.logger.info("=" * 60)
        self.logger.info("RQ1 Experiment: Federated LoRA vs Isolated Local Fine-tuning")
        self.logger.info("=" * 60)
        
        # Run both training modes
        self.run_federated_training()
        self.run_isolated_training()
        
        # Compare results
        comparison = self.compare_results()
        
        self.logger.info("=" * 60)
        self.logger.info("RQ1 Results:")
        for key, value in comparison.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
        
        return {
            "rq1_comparison": comparison,
            "federated_results": [r.to_dict() for r in self.federated_results],
            "isolated_results": [r.to_dict() for r in self.isolated_results],
        }


class RQ2Experiment:
    """
    RQ2: How does the pipeline balance communication cost and accuracy?
    
    Tests different communication compression ratios and sync intervals
    to find optimal trade-off between communication and model quality.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger("RQ2")
        self.data_loader = DS1000DataLoader(config.dataset_path, config.num_samples)
        self.evaluator = CodeQualityEvaluator()
        
        # Results tracking
        self.results_by_config: Dict[str, List[RoundMetrics]] = {}
    
    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def run_with_config(
        self,
        compression_ratio: float,
        sync_interval: float
    ) -> List[RoundMetrics]:
        """Run training with specific communication configuration"""
        config_key = f"compression_{compression_ratio}_sync_{sync_interval}"
        self.logger.info(f"Running with compression={compression_ratio}, sync_interval={sync_interval}s")
        
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        # Register experts
        for specialty in self.config.expert_specialties:
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        # Create workers
        workers = []
        for i, specialty in enumerate(self.config.expert_specialties):
            worker = HeterogeneousWorker(
                worker_id=f"CommWorker-{i}",
                coordinator=coordinator,
                specialty=specialty,
                speed_factor=random.uniform(0.8, 1.5)
            )
            workers.append(worker)
        
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Simulate training concurrently on local GPUs with compression
            weights_map = {}
            for worker in workers:
                lora_A, lora_B, version = coordinator.get_expert_weights(worker.specialty)
                weights_map[worker.worker_id] = (lora_A, lora_B, version, worker)

            deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            threads: List[threading.Thread] = []

            def run_train(wid: str):
                lA, lB, _, w = weights_map[wid]
                try:
                    dA, dB = w.simulate_local_training(lA, lB)
                    deltas[wid] = (dA, dB)
                except Exception as e:
                    self.logger.error(f"Worker error ({wid}): {e}")

            for wid in weights_map.keys():
                t = threading.Thread(target=run_train, args=(wid,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            # Push updates with compression
            for wid, (dA, dB) in deltas.items():
                _, _, ver, w = weights_map[wid]
                if compression_ratio < 1.0:
                    dA = self._compress_gradient(dA, compression_ratio)
                    dB = self._compress_gradient(dB, compression_ratio)
                coordinator.push_expert_update(w.specialty, (dA, dB), ver)
            
            # Collect metrics
            round_metrics = self._collect_round_metrics(
                round_num,
                coordinator,
                specialty_data,
                compression_ratio,
                sync_interval
            )
            results.append(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}, "
                           f"Comm={round_metrics.total_communication_bytes:.0f} bytes")
            
            time.sleep(sync_interval)
        
        self.results_by_config[config_key] = results
        return results
    
    def _compress_gradient(self, gradient: np.ndarray, ratio: float) -> np.ndarray:
        """Compress gradient by keeping top-k values"""
        if ratio >= 1.0:
            return gradient
        
        # Top-k sparsification
        k = max(1, int(gradient.size * ratio))
        flat = gradient.flatten()
        threshold = np.partition(np.abs(flat), -k)[-k]
        compressed = gradient.copy()
        compressed[np.abs(compressed) < threshold] = 0
        return compressed
    
    def _collect_round_metrics(
        self,
        round_num: int,
        coordinator: CentralCoordinator,
        specialty_data: Dict[str, List[Dict]],
        compression_ratio: float,
        sync_interval: float
    ) -> RoundMetrics:
        """Collect metrics with communication tracking"""
        metrics = RoundMetrics(
            round_num=round_num,
            timestamp=time.time()
        )
        
        # Simulate code quality
        all_exact_matches = []
        for specialty, data in specialty_data.items():
            if data:
                generated_codes = [item.get("reference_code", "")[:50] for item in data[:3]]
                reference_codes = [item.get("reference_code", "") for item in data[:3]]
                exact_match, _, _ = self.evaluator.batch_evaluate(generated_codes, reference_codes)
                all_exact_matches.append(exact_match)
        
        metrics.avg_exact_match = np.mean(all_exact_matches) if all_exact_matches else 0.0
        metrics.avg_code_quality = metrics.avg_exact_match
        
        # Communication metrics with compression
        base_comm = len(self.config.expert_specialties) * \
                   self.config.model_dim * self.config.lora_rank * 4 * 2
        metrics.total_communication_bytes = base_comm * compression_ratio
        
        return metrics
    
    def analyze_trade_off(self) -> Dict[str, Any]:
        """Analyze communication vs accuracy trade-off"""
        analysis = {
            "configurations": [],
            "pareto_frontier": []
        }
        
        for config_key, results in self.results_by_config.items():
            if results:
                avg_quality = np.mean([r.avg_code_quality for r in results])
                avg_comm = np.mean([r.total_communication_bytes for r in results])
                
                analysis["configurations"].append({
                    "config": config_key,
                    "avg_quality": avg_quality,
                    "avg_communication_bytes": avg_comm,
                    "quality_per_byte": avg_quality / (avg_comm + 1e-6)
                })
        
        # Simple Pareto frontier calculation
        configs = analysis["configurations"]
        if configs:
            pareto = []
            for c in configs:
                dominated = False
                for other in configs:
                    if (other["avg_quality"] >= c["avg_quality"] and
                        other["avg_communication_bytes"] <= c["avg_communication_bytes"] and
                        (other["avg_quality"] > c["avg_quality"] or
                         other["avg_communication_bytes"] < c["avg_communication_bytes"])):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(c)
            analysis["pareto_frontier"] = pareto
        
        return analysis
    
    def run(self) -> Dict[str, Any]:
        """Run complete RQ2 experiment"""
        self.logger.info("=" * 60)
        self.logger.info("RQ2 Experiment: Communication Cost vs Accuracy Trade-off")
        self.logger.info("=" * 60)
        
        # Test different configurations
        compression_ratios = [1.0, 0.75, 0.5, 0.25]
        sync_intervals = [2.0, 5.0, 10.0]
        
        for compression in compression_ratios[:2]:  # Limit for demo
            for sync_interval in sync_intervals[:2]:
                self.run_with_config(compression, sync_interval)
        
        # Analyze trade-off
        analysis = self.analyze_trade_off()
        
        self.logger.info("=" * 60)
        self.logger.info("RQ2 Results:")
        self.logger.info(f"  Tested {len(self.results_by_config)} configurations")
        self.logger.info(f"  Pareto frontier size: {len(analysis['pareto_frontier'])}")
        self.logger.info("=" * 60)
        
        return {
            "rq2_analysis": analysis,
            "results_by_config": {k: [r.to_dict() for r in v] 
                                 for k, v in self.results_by_config.items()}
        }


class RQ3Experiment:
    """
    RQ3: Can hierarchical aggregation preserve specialized skills during continuous rounds?
    
    Compares:
    1. Flat aggregation: All experts contribute equally
    2. Hierarchical aggregation: Experts grouped by domain, then aggregated
    3. Skill-aware aggregation: Preserve specialization through selective aggregation
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger("RQ3")
        self.data_loader = DS1000DataLoader(config.dataset_path, config.num_samples)
        self.evaluator = CodeQualityEvaluator()
        
        # Results tracking
        self.flat_results: List[RoundMetrics] = []
        self.hierarchical_results: List[RoundMetrics] = []
        self.skill_aware_results: List[RoundMetrics] = []
        
        # Specialization tracking
        self.expert_specialization: Dict[str, List[float]] = {
            spec: [] for spec in config.expert_specialties
        }
    
    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def run_flat_aggregation(self) -> List[RoundMetrics]:
        """Run with flat aggregation (all experts equal weight)"""
        self.logger.info("Running Flat Aggregation")
        
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        for specialty in self.config.expert_specialties:
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        workers = []
        for i, specialty in enumerate(self.config.expert_specialties):
            worker = HeterogeneousWorker(
                worker_id=f"FlatWorker-{i}",
                coordinator=coordinator,
                specialty=specialty,
                speed_factor=random.uniform(0.8, 1.5)
            )
            workers.append(worker)
        
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Simulate training concurrently on local GPUs
            weights_map = {}
            for worker in workers:
                lora_A, lora_B, version = coordinator.get_expert_weights(worker.specialty)
                weights_map[worker.worker_id] = (lora_A, lora_B, version, worker)

            deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            threads: List[threading.Thread] = []

            def run_train(wid: str):
                lA, lB, _, w = weights_map[wid]
                try:
                    dA, dB = w.simulate_local_training(lA, lB)
                    deltas[wid] = (dA, dB)
                except Exception as e:
                    self.logger.error(f"Worker error ({wid}): {e}")

            for wid in weights_map.keys():
                t = threading.Thread(target=run_train, args=(wid,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            # Push updates
            for wid, (dA, dB) in deltas.items():
                _, _, ver, w = weights_map[wid]
                coordinator.push_expert_update(w.specialty, (dA, dB), ver)
            
            round_metrics = self._collect_round_metrics(
                round_num,
                coordinator,
                specialty_data,
                "flat"
            )
            results.append(round_metrics)
            
            # Track specialization
            self._update_specialization_scores(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}")
            time.sleep(self.config.round_duration)
        
        self.flat_results = results
        return results
    
    def run_hierarchical_aggregation(self) -> List[RoundMetrics]:
        """Run with hierarchical aggregation"""
        self.logger.info("Running Hierarchical Aggregation")
        
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        for specialty in self.config.expert_specialties:
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        workers = []
        for i, specialty in enumerate(self.config.expert_specialties):
            worker = HeterogeneousWorker(
                worker_id=f"HierWorker-{i}",
                coordinator=coordinator,
                specialty=specialty,
                speed_factor=random.uniform(0.8, 1.5)
            )
            workers.append(worker)
        
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            # Group experts hierarchically
            # Level 1: Group by domain similarity
            domain_groups = self._create_domain_groups()
            
            for worker in workers:
                try:
                    lora_A, lora_B, version = coordinator.get_expert_weights(worker.specialty)
                    delta_A, delta_B = worker.simulate_local_training(lora_A, lora_B)
                    
                    # Apply hierarchical aggregation weight
                    group = self._find_group(worker.specialty, domain_groups)
                    weight = 1.0 / len(domain_groups[group]) if group else 1.0
                    delta_A *= weight
                    delta_B *= weight
                    
                    coordinator.push_expert_update(worker.specialty, (delta_A, delta_B), version)
                except Exception as e:
                    self.logger.error(f"Error: {e}")
            
            round_metrics = self._collect_round_metrics(
                round_num,
                coordinator,
                specialty_data,
                "hierarchical"
            )
            results.append(round_metrics)
            
            self._update_specialization_scores(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}")
            time.sleep(self.config.round_duration)
        
        self.hierarchical_results = results
        return results
    
    def run_skill_aware_aggregation(self) -> List[RoundMetrics]:
        """Run with skill-aware aggregation (preserve specialization)"""
        self.logger.info("Running Skill-Aware Aggregation")
        
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        for specialty in self.config.expert_specialties:
            coordinator.register_expert(
                specialty,
                self.config.model_dim,
                self.config.lora_rank
            )
        
        specialty_data = self.data_loader.split_by_specialty(self.config.expert_specialties)
        
        workers = []
        for i, specialty in enumerate(self.config.expert_specialties):
            worker = HeterogeneousWorker(
                worker_id=f"SkillWorker-{i}",
                coordinator=coordinator,
                specialty=specialty,
                speed_factor=random.uniform(0.8, 1.5)
            )
            workers.append(worker)
        
        # Track expert performance to preserve skills
        expert_performance = {spec: 1.0 for spec in self.config.expert_specialties}
        
        results = []
        for round_num in range(self.config.num_rounds):
            round_start = time.time()
            
            for worker in workers:
                try:
                    lora_A, lora_B, version = coordinator.get_expert_weights(worker.specialty)
                    delta_A, delta_B = worker.simulate_local_training(lora_A, lora_B)
                    
                    # Skill-aware weight: experts with better performance get higher weight
                    perf = expert_performance.get(worker.specialty, 1.0)
                    total_perf = sum(expert_performance.values())
                    weight = perf / total_perf if total_perf > 0 else 1.0
                    
                    delta_A *= weight
                    delta_B *= weight
                    
                    coordinator.push_expert_update(worker.specialty, (delta_A, delta_B), version)
                except Exception as e:
                    self.logger.error(f"Error: {e}")
            
            round_metrics = self._collect_round_metrics(
                round_num,
                coordinator,
                specialty_data,
                "skill_aware"
            )
            results.append(round_metrics)
            
            # Update expert performance scores
            for specialty in self.config.expert_specialties:
                if specialty in round_metrics.expert_specialization_scores:
                    expert_performance[specialty] = round_metrics.expert_specialization_scores[specialty]
            
            self._update_specialization_scores(round_metrics)
            
            self.logger.info(f"Round {round_num}: Quality={round_metrics.avg_code_quality:.4f}")
            time.sleep(self.config.round_duration)
        
        self.skill_aware_results = results
        return results
    
    def _create_domain_groups(self) -> Dict[str, List[str]]:
        """Create hierarchical domain groups"""
        # Simple grouping: data-related, query-related, docs-related
        groups = {
            "data": ["python"],
            "query": ["sql"],
            "docs": ["docs"]
        }
        return groups
    
    def _find_group(self, specialty: str, groups: Dict[str, List[str]]) -> Optional[str]:
        """Find which group a specialty belongs to"""
        for group_name, specialties in groups.items():
            if specialty in specialties:
                return group_name
        return None
    
    def _collect_round_metrics(
        self,
        round_num: int,
        coordinator: CentralCoordinator,
        specialty_data: Dict[str, List[Dict]],
        aggregation_type: str
    ) -> RoundMetrics:
        """Collect metrics with specialization tracking"""
        metrics = RoundMetrics(
            round_num=round_num,
            timestamp=time.time()
        )
        
        # Simulate code quality
        all_exact_matches = []
        specialization_scores = {}
        
        for specialty, data in specialty_data.items():
            if data:
                generated_codes = [item.get("reference_code", "")[:50] for item in data[:3]]
                reference_codes = [item.get("reference_code", "") for item in data[:3]]
                exact_match, _, _ = self.evaluator.batch_evaluate(generated_codes, reference_codes)
                all_exact_matches.append(exact_match)
                
                # Specialization score: how well expert performs on its specialty
                specialization_scores[specialty] = exact_match
        
        metrics.avg_exact_match = np.mean(all_exact_matches) if all_exact_matches else 0.0
        metrics.avg_code_quality = metrics.avg_exact_match
        metrics.expert_specialization_scores = specialization_scores
        
        # Expert versions
        for specialty in self.config.expert_specialties:
            try:
                _, _, version = coordinator.get_expert_weights(specialty)
                metrics.expert_versions[specialty] = version
            except:
                pass
        
        return metrics
    
    def _update_specialization_scores(self, metrics: RoundMetrics) -> None:
        """Update specialization tracking"""
        for specialty, score in metrics.expert_specialization_scores.items():
            self.expert_specialization[specialty].append(score)
    
    def compare_aggregation_strategies(self) -> Dict[str, Any]:
        """Compare different aggregation strategies"""
        strategies = {
            "flat": self.flat_results,
            "hierarchical": self.hierarchical_results,
            "skill_aware": self.skill_aware_results
        }
        
        comparison = {}
        for strategy_name, results in strategies.items():
            if results:
                qualities = [r.avg_code_quality for r in results]
                specializations = []
                for r in results:
                    if r.expert_specialization_scores:
                        specializations.append(np.mean(list(r.expert_specialization_scores.values())))
                
                comparison[strategy_name] = {
                    "avg_quality": np.mean(qualities),
                    "final_quality": qualities[-1] if qualities else 0,
                    "quality_std": np.std(qualities),
                    "avg_specialization": np.mean(specializations) if specializations else 0,
                    "specialization_preservation": self._calculate_specialization_preservation(results)
                }
        
        return comparison
    
    def _calculate_specialization_preservation(self, results: List[RoundMetrics]) -> float:
        """
        Calculate how well specialization is preserved across rounds.
        Higher score = better preservation of specialized skills.
        """
        if not results:
            return 0.0
        
        # Measure variance in expert performance across rounds
        # Lower variance = better preservation
        specialization_scores = []
        for r in results:
            if r.expert_specialization_scores:
                scores = list(r.expert_specialization_scores.values())
                if scores:
                    # Coefficient of variation (lower = more consistent/preserved)
                    cv = np.std(scores) / (np.mean(scores) + 1e-6)
                    specialization_scores.append(cv)
        
        if specialization_scores:
            # Invert: lower CV = higher preservation score
            return 1.0 / (1.0 + np.mean(specialization_scores))
        return 0.0
    
    def run(self) -> Dict[str, Any]:
        """Run complete RQ3 experiment"""
        self.logger.info("=" * 60)
        self.logger.info("RQ3 Experiment: Hierarchical Aggregation for Specialization")
        self.logger.info("=" * 60)
        
        # Run all aggregation strategies
        self.run_flat_aggregation()
        self.run_hierarchical_aggregation()
        self.run_skill_aware_aggregation()
        
        # Compare strategies
        comparison = self.compare_aggregation_strategies()
        
        self.logger.info("=" * 60)
        self.logger.info("RQ3 Results:")
        for strategy, metrics in comparison.items():
            self.logger.info(f"  {strategy}:")
            for key, value in metrics.items():
                self.logger.info(f"    {key}: {value:.4f}")
        self.logger.info("=" * 60)
        
        return {
            "rq3_comparison": comparison,
            "flat_results": [r.to_dict() for r in self.flat_results],
            "hierarchical_results": [r.to_dict() for r in self.hierarchical_results],
            "skill_aware_results": [r.to_dict() for r in self.skill_aware_results],
            "expert_specialization_tracking": self.expert_specialization
        }


# ============================================================================
# Main Experiment Orchestrator
# ============================================================================

class ExperimentOrchestrator:
    """Orchestrate all experiments and generate comprehensive reports"""
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        self.config = config or ExperimentConfig()
        self.logger = self._setup_logger("Orchestrator")
        self.results = {}
    
    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all three RQ experiments"""
        self.logger.info("Starting comprehensive experiment suite")
        
        # RQ1: Federated vs Isolated
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RUNNING RQ1 EXPERIMENT")
        self.logger.info("=" * 70)
        rq1_exp = RQ1Experiment(self.config)
        self.results["RQ1"] = rq1_exp.run()
        
        # RQ2: Communication vs Accuracy
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RUNNING RQ2 EXPERIMENT")
        self.logger.info("=" * 70)
        rq2_exp = RQ2Experiment(self.config)
        self.results["RQ2"] = rq2_exp.run()
        
        # RQ3: Hierarchical Aggregation
        self.logger.info("\n" + "=" * 70)
        self.logger.info("RUNNING RQ3 EXPERIMENT")
        self.logger.info("=" * 70)
        rq3_exp = RQ3Experiment(self.config)
        self.results["RQ3"] = rq3_exp.run()
        
        return self.results
    
    def save_results(self, output_path: str = "experiment_results.json") -> None:
        """Save experiment results to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        report = []
        report.append("=" * 80)
        report.append("FEDERATED LORA EXPERIMENTS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # RQ1 Summary
        if "RQ1" in self.results:
            report.append("RQ1: Federated LoRA vs Isolated Local Fine-tuning")
            report.append("-" * 80)
            rq1_comparison = self.results["RQ1"].get("rq1_comparison", {})
            for key, value in rq1_comparison.items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")
        
        # RQ2 Summary
        if "RQ2" in self.results:
            report.append("RQ2: Communication Cost vs Accuracy Trade-off")
            report.append("-" * 80)
            rq2_analysis = self.results["RQ2"].get("rq2_analysis", {})
            pareto = rq2_analysis.get("pareto_frontier", [])
            report.append(f"  Pareto frontier configurations: {len(pareto)}")
            for config in pareto:
                report.append(f"    {config.get('config', 'N/A')}: "
                            f"Quality={config.get('avg_quality', 0):.4f}, "
                            f"Comm={config.get('avg_communication_bytes', 0):.0f} bytes")
            report.append("")
        
        # RQ3 Summary
        if "RQ3" in self.results:
            report.append("RQ3: Hierarchical Aggregation for Specialization")
            report.append("-" * 80)
            rq3_comparison = self.results["RQ3"].get("rq3_comparison", {})
            for strategy, metrics in rq3_comparison.items():
                report.append(f"  {strategy}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        report.append(f"    {key}: {value:.4f}")
                    else:
                        report.append(f"    {key}: {value}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)

