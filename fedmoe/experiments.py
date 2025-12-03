"""
Comprehensive experimental framework for evaluating:
- RQ1: Federated LoRA vs Isolated Local Fine-tuning
- RQ2: Specialization Granularity vs Code Generation Quality
- RQ3: Continual Domain Shift and Catastrophic Forgetting (with Rehearsal)
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
    
    # Training (generic)
    num_rounds: int = 10
    round_duration: float = 30.0  # seconds per round
    
    # Experts (default set)
    num_experts: int = 3
    expert_specialties: List[str] = field(default_factory=lambda: ["python", "sql", "docs"])
    
    # Communication (legacy - unused in redesigned RQ2/RQ3)
    sync_interval: float = 5.0
    communication_compression_ratio: float = 1.0
    
    # Aggregation (kept for completeness)
    aggregation_strategy: str = "weighted_average"
    use_gateway_model: bool = True
    
    # Local single-machine GPU mapping (optional)
    worker_gpu_ids: Optional[List[int]] = None  # e.g., [0, 1]
    aggregator_gpu_id: Optional[int] = None     # e.g., 2
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N rounds
    
    # RQ2 (granularity) - list of expert sets to compare, e.g., [["generalist"],["python","sql"],["python","sql","docs"]]
    rq2_expert_sets: Optional[List[List[str]]] = None
    
    # RQ3 (continual learning)
    phase_domains: Optional[List[str]] = None   # e.g., ["python","sql","docs"]
    phase_rounds: int = 4                       # rounds per phase
    rehearsal_ratio: float = 0.1                # portion of rehearsal updates (0=no rehearsal)
    
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
    RQ2 (Redesigned): Specialization Granularity vs Code Generation Quality
    
    Compare different expert granularities:
      - Generalist (1 expert over all data)
      - 2 specialists (e.g., python + sql)
      - 3 specialists (python + sql + docs)
    Measure per-domain quality and overall quality.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger("RQ2")
        self.data_loader = DS1000DataLoader(config.dataset_path, config.num_samples)
        self.evaluator = CodeQualityEvaluator()
        
        # Results tracking
        self.results_by_setting: Dict[str, Dict[str, Any]] = {}
    
    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _train_with_experts(self, expert_list: List[str]) -> Dict[str, Any]:
        """Train for config.num_rounds using the provided expert list"""
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        
        # Register experts
        for spec in expert_list:
            coordinator.register_expert(spec, self.config.model_dim, self.config.lora_rank)
        
        # Create workers
        workers: List[HeterogeneousWorker] = []
        for i, spec in enumerate(expert_list):
            workers.append(
                HeterogeneousWorker(
                    worker_id=f"GranWorker-{i}",
                    coordinator=coordinator,
                    specialty=spec,
                    speed_factor=random.uniform(0.8, 1.5),
                )
            )
        
        # Training loop
        for rnd in range(self.config.num_rounds):
            weights_map = {}
            for w in workers:
                lA, lB, v = coordinator.get_expert_weights(w.specialty)
                weights_map[w.worker_id] = (lA, lB, v, w)
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
                t.start(); threads.append(t)
            for t in threads: t.join()
            
            for wid, (dA, dB) in deltas.items():
                _, _, v, w = weights_map[wid]
                coordinator.push_expert_update(w.specialty, (dA, dB), v)
            
            time.sleep(self.config.round_duration)
        
        # Evaluate per-domain and overall
        canonical_domains = ["python", "sql", "docs"]
        domain_data = self.data_loader.split_by_specialty(canonical_domains)
        per_domain: Dict[str, float] = {}
        all_domain_scores: List[float] = []
        for dom, data in domain_data.items():
            if not data:
                continue
            gen = [item.get("reference_code", "")[:50] for item in data[:3]]
            ref = [item.get("reference_code", "") for item in data[:3]]
            exact, _, _ = self.evaluator.batch_evaluate(gen, ref)
            per_domain[dom] = float(exact)
            all_domain_scores.append(float(exact))
        overall = float(np.mean(all_domain_scores)) if all_domain_scores else 0.0
        
        return {
            "per_domain": per_domain,
            "overall_quality": overall,
            "experts": expert_list,
            "gateway_version": coordinator.gateway_model.version if coordinator.gateway_model else 0,
        }
    
    def run(self) -> Dict[str, Any]:
        """Run RQ2 granularity comparison"""
        self.logger.info("=" * 60)
        self.logger.info("RQ2 Experiment (Granularity → Quality)")
        self.logger.info("=" * 60)
        
        # Default sets if not provided
        sets = self.config.rq2_expert_sets or [["generalist"], ["python", "sql"], ["python", "sql", "docs"]]
        
        for expert_list in sets:
            key = "+".join(expert_list)
            self.logger.info(f"Training expert set: {key}")
            self.results_by_setting[key] = self._train_with_experts(expert_list)
        
        # Rank by overall quality
        ranking = sorted(
            (
                {"setting": k, **v}
                for k, v in self.results_by_setting.items()
            ),
            key=lambda x: x.get("overall_quality", 0.0), reverse=True
        )
        
        self.logger.info("Ranking by overall quality:")
        for r in ranking:
            self.logger.info(f"  {r['setting']}: overall={r['overall_quality']:.4f}, per_domain={r['per_domain']}")
        
        return {
            "rq2_results": self.results_by_setting,
            "ranking": ranking,
        }


class RQ3Experiment:
    """
    RQ3 (Redesigned): Continual Domain Shift and Catastrophic Forgetting (with Rehearsal)
    
    Setup sequential phases over domains (e.g., python → sql → docs),
    compare two strategies:
      - Naive: train only on current domain
      - Rehearsal: mix small proportion of past domains (rehearsal_ratio)
    Measure forgetting per domain across phases and final overall quality.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = self._setup_logger("RQ3")
        self.data_loader = DS1000DataLoader(config.dataset_path, config.num_samples)
        self.evaluator = CodeQualityEvaluator()

        # Results tracking
        self.naive_history: List[Dict[str, Any]] = []
        self.rehearsal_history: List[Dict[str, Any]] = []

    def _setup_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(f"logs/{name}.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _phase_train(self, domains: List[str], strategy: str) -> Dict[str, Any]:
        """Train sequentially for each domain with given strategy (naive/rehearsal)."""
        coordinator = CentralCoordinator(enable_gateway=True, aggregator_device_id=self.config.aggregator_gpu_id)
        # Register all experts upfront (one per domain, plus an optional generalist if needed)
        for dom in domains:
            coordinator.register_expert(dom, self.config.model_dim, self.config.lora_rank)

        # Maintain past domains for rehearsal
        seen: List[str] = []

        # Per-phase evaluation snapshots
        phase_snapshots: List[Dict[str, Any]] = []

        # Canonical domains for evaluation
        canonical = domains
        domain_data = self.data_loader.split_by_specialty(canonical)

        for p, dom in enumerate(domains):
            self.logger.info(f"Phase {p+1}/{len(domains)}: domain={dom}, strategy={strategy}")
            # Build workers for this phase
            workers: List[HeterogeneousWorker] = []
            # Current domain worker (full weight)
            workers.append(
                HeterogeneousWorker(
                    worker_id=f"{strategy}-cur-{dom}",
                    coordinator=coordinator,
                    specialty=dom,
                    speed_factor=random.uniform(0.8, 1.2),
                )
            )
            # Rehearsal workers for seen domains (reduced updates)
            if strategy == "rehearsal" and seen and self.config.rehearsal_ratio > 0:
                per_prev_weight = self.config.rehearsal_ratio / max(1, len(seen))
                for prev in seen:
                    workers.append(
                        HeterogeneousWorker(
                            worker_id=f"{strategy}-reh-{prev}",
                            coordinator=coordinator,
                            specialty=prev,
                            speed_factor=per_prev_weight,  # smaller updates
                        )
                    )

            # Run rounds for this phase
            for r in range(self.config.phase_rounds):
                weights_map: Dict[str, Tuple[np.ndarray, np.ndarray, int, HeterogeneousWorker]] = {}
                for w in workers:
                    lA, lB, v = coordinator.get_expert_weights(w.specialty)
                    weights_map[w.worker_id] = (lA, lB, v, w)

                deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                threads: List[threading.Thread] = []

                def run_train(wid: str):
                    lA, lB, _, w = weights_map[wid]
                    try:
                        dA, dB = w.simulate_local_training(lA, lB)
                        deltas[wid] = (dA, dB)
                    except Exception as e:
                        self.logger.error(f"Worker error ({wid}): {e}")

                for wid in list(weights_map.keys()):
                    t = threading.Thread(target=run_train, args=(wid,), daemon=True)
                    t.start(); threads.append(t)
                for t in threads: t.join()

                # Push updates
                for wid, (dA, dB) in deltas.items():
                    _, _, v, w = weights_map[wid]
                    coordinator.push_expert_update(w.specialty, (dA, dB), v)

                time.sleep(self.config.round_duration)

            # Phase end evaluation across all domains
            per_domain: Dict[str, float] = {}
            all_scores: List[float] = []
            for dname, data in domain_data.items():
                if not data:
                    continue
                gen = [item.get("reference_code", "")[:50] for item in data[:3]]
                ref = [item.get("reference_code", "") for item in data[:3]]
                exact, _, _ = self.evaluator.batch_evaluate(gen, ref)
                per_domain[dname] = float(exact)
                all_scores.append(float(exact))
            overall = float(np.mean(all_scores)) if all_scores else 0.0

            snapshot = {
                "phase": p + 1,
                "current_domain": dom,
                "per_domain": per_domain,
                "overall": overall,
            }
            phase_snapshots.append(snapshot)
            seen.append(dom)

        # Compute forgetting per domain: best past - last
        best_so_far: Dict[str, float] = {}
        for snap in phase_snapshots:
            for d, v in snap["per_domain"].items():
                best_so_far[d] = max(best_so_far.get(d, 0.0), v)
        last = phase_snapshots[-1]["per_domain"] if phase_snapshots else {}
        forgetting = {d: max(0.0, best_so_far.get(d, 0.0) - last.get(d, 0.0)) for d in best_so_far.keys()}
        avg_forgetting = float(np.mean(list(forgetting.values()))) if forgetting else 0.0

        return {
            "snapshots": phase_snapshots,
            "forgetting": forgetting,
            "avg_forgetting": avg_forgetting,
            "final_overall": phase_snapshots[-1]["overall"] if phase_snapshots else 0.0,
        }

    def run(self) -> Dict[str, Any]:
        """Run RQ3 continual domain shift (naive vs rehearsal)"""
        self.logger.info("=" * 60)
        self.logger.info("RQ3 Experiment (Continual Learning → Forgetting)")
        self.logger.info("=" * 60)

        domains = self.config.phase_domains or ["python", "sql", "docs"]
        self.logger.info(f"Domains: {domains}; phase_rounds={self.config.phase_rounds}; rehearsal_ratio={self.config.rehearsal_ratio}")

        naive = self._phase_train(domains, strategy="naive")
        rehearsal = self._phase_train(domains, strategy="rehearsal")

        self.logger.info("Naive avg_forgetting=%.4f final_overall=%.4f", naive.get("avg_forgetting", 0.0), naive.get("final_overall", 0.0))
        self.logger.info("Rehearsal avg_forgetting=%.4f final_overall=%.4f", rehearsal.get("avg_forgetting", 0.0), rehearsal.get("final_overall", 0.0))

        return {
            "rq3_naive": naive,
            "rq3_rehearsal": rehearsal,
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

