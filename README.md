# FedMoE
Federated LoRA Experts for Multi-GPU Qwen Fine-Tuning

## Overview
FedMoE illustrates how an open-source base model such as Qwen-7B can be fine-tuned across many machines and GPUs without sharing raw data. Instead of training a single monolith, each domain (Python, SQL, docs, etc.) owns a lightweight LoRA “expert” whose gradients are aggregated by a central coordinator and redistributed to every worker. This enables synchronous (staleness-aware) model updates, gradient sharing, and rapid specialization across heterogeneous hardware.

`FedMoE.py` ships a runnable end-to-end simulation that captures the control flow you would need before wiring it into real training loops.

### Module Layout
- `fedmoe/config.py`: global hyperparameters plus default expert/worker blueprints.
- `fedmoe/experts.py`: LoRA expert abstraction (can be replaced with real Qwen adapters).
- `fedmoe/coordinator.py`: versioned aggregation server with staleness-aware updates.
- `fedmoe/workers.py`: heterogeneous worker loops that simulate multi-GPU fine-tuning.
- `fedmoe/agent.py`: prompt router + collaborative inference agent.
- `fedmoe/simulation.py`: orchestration helpers for training/inference phases.
- `FedMoE.py`: CLI entrypoint to configure workers/experts and launch the simulation.

## Key Features
- **Expert-centric LoRA blocks**: each specialty keeps its own rank-limited LoRA adapter, ready to be attached to a Qwen checkpoint during fine-tuning.
- **Central coordinator**: aggregates worker gradients, applies staleness decay, bumps expert versions, and broadcasts fresh weights.
- **Heterogeneous workers**: simulate machines with different data, GPU speeds, and network delays to mimic real federated fleets.
- **Routing-aware inference agent**: inspects prompts, fetches the right experts, and stitches their generations to demonstrate cross-domain collaboration.
- **Extensible to real Qwen models**: the same abstractions map cleanly onto vLLM / Transformers code where `lora_A`/`lora_B` would be real tensors.

## Architecture
1. **CentralCoordinator**
   - Registers experts (`python_expert`, `sql_expert`, `docs_expert`, …).
   - Stores LoRA weights, versions, and applies updates with `SERVER_LR` and `STALENESS_DECAY`.
2. **HeterogeneousWorker**
   - Pulls the latest weights for its specialty.
   - Runs local training on private corpora (simulated by random deltas) and pushes updates back.
   - Models different GPU speeds via `speed_factor` and random network latency.
3. **InferenceAgent**
   - Routes user prompts to the relevant experts via regex-based intent detection.
   - Locks coordinator state while performing inference to keep weight reads consistent.

## Running the Simulation
1. Install dependencies (Python 3.9+ with `numpy`).
2. Quickstart (default settings):
   ```bash
   python FedMoE.py
   ```
   - `--duration 30` to keep the distributed loop alive longer.
   - `--worker WorkerX:python_expert:0.8` to add extra inline workers.
   - `--worker-json workers.json` to load a fleet definition (see schema in CLI help).
3. Config-driven run script:
   ```bash
   python scripts/run_fedmoe.py --config configs/sample_run.json
   ```
   - Customize the JSON file to describe cross-domain worker fleets or expert sets for experiments in the paper.
4. Observe three phases:
   - Expert registration and worker spin-up.
   - 15 seconds of asynchronous distributed training with versioned updates.
   - Inference agent collaboration on sample tasks (pure Python, Python+SQL, Python+Docs).

## Mapping to Real Qwen Fine-Tuning
The provided loop is deliberately lightweight so you can swap the simulated logic with real training code:

- Replace `simulate_local_training` with an actual LoRA fine-tune step that loads a Qwen checkpoint plus the worker’s current expert weights.
- Persist `lora_A` / `lora_B` to shared storage or RPC responses so freshly aggregated adapters can be reapplied to worker batches.
- Expand `route_task` to consume telemetry (perplexity, domain tags) or plug in a policy model for expert selection.
- Integrate secure communication or differential privacy before pushing gradients for production deployments.

## Roadmap Ideas
- Add more expert types (vision, speech) and dynamic expert allocation.
- Support hierarchical aggregation layers (edge → regional → central) for large federations.
- Swap the regex router with a planner that reasons over tool descriptions.
- Benchmark real Qwen adapters with mixed precision and fused optimizers.

## License
MIT – customize the orchestration to fit your own multi-GPU, cross-domain fine-tuning workflows.
