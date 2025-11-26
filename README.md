# FedMoE
Federated LoRA Experts for Multi-GPU Qwen Fine-Tuning

## Overview
FedMoE illustrates how an open-source base model such as Qwen-7B can be fine-tuned across many machines and GPUs without sharing raw data. Instead of training a single monolith, each domain (Python, SQL, docs, etc.) owns a lightweight LoRA “expert” whose gradients are aggregated by a central coordinator and redistributed to every worker. This enables synchronous (staleness-aware) model updates, gradient sharing, and rapid specialization across heterogeneous hardware.

`run_qwen_finetune.py` provides a simple entry point to run real Qwen model fine-tuning with federated LoRA experts.

### Module Layout
- `fedmoe/config.py`: global hyperparameters plus default expert/worker blueprints.
- `fedmoe/experts.py`: LoRA expert abstraction (can be replaced with real Qwen adapters).
- `fedmoe/coordinator.py`: versioned aggregation server with staleness-aware updates.
- `fedmoe/workers.py`: heterogeneous worker loops that simulate multi-GPU fine-tuning.
- `fedmoe/qwen_worker.py`: real Qwen model worker implementation for fine-tuning.
- `fedmoe/qwen_finetune.py`: real Qwen LoRA fine-tuning implementation.
- `fedmoe/agent.py`: prompt router + collaborative inference agent.
- `fedmoe/simulation.py`: orchestration helpers for training/inference phases.
- `run_qwen_finetune.py`: CLI entrypoint to run real Qwen fine-tuning with federated LoRA experts.

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

## Running Real Qwen Fine-Tuning

**重要提示：所有依赖必须安装到虚拟环境中！**

#### 方式 1：一键启动（推荐）⭐

直接运行快速启动脚本，它会自动检查并设置虚拟环境：

```bash
# Windows PowerShell（自动检查并创建虚拟环境）
.\scripts\run_qwen.ps1

# Linux/Mac Bash（自动检查并创建虚拟环境）
bash scripts/run_qwen.sh
```

**说明：** 这些脚本会自动：
1. 检查虚拟环境是否存在
2. 如果不存在，自动运行 `setup.sh/setup.ps1` 创建虚拟环境并安装依赖
3. 然后直接运行 Qwen 微调

#### 方式 2：手动分步（可选）

如果你想手动控制每个步骤：

```bash
# 步骤1：创建虚拟环境并安装依赖
# Windows PowerShell
.\scripts\setup.ps1

# Linux/Mac Bash
bash scripts/setup.sh

# 步骤2：运行 Qwen 微调
# Windows PowerShell
.\scripts\run_qwen.ps1

# Linux/Mac Bash
bash scripts/run_qwen.sh
```

#### 方式 3：使用 Python 脚本（需要先激活虚拟环境）

```bash
# 先激活虚拟环境
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate

# 然后运行
python run_qwen_finetune.py
```

**使用命令行参数：**

```bash
# 指定不同的 Qwen 模型
python run_qwen_finetune.py --base-model qwen/Qwen2-1.5B-Instruct

# 指定训练时长
python run_qwen_finetune.py --duration 60

# 指定数据集路径
python run_qwen_finetune.py --dataset dataset/test.jsonl

# 组合使用多个参数
python run_qwen_finetune.py --base-model qwen/Qwen2-0.5B-Instruct --duration 60 --dataset dataset/test.jsonl
```

**注意事项：**
- 真实微调模式需要安装额外的依赖：`torch`, `transformers`, `peft`, `accelerate`
- 首次运行会自动下载 Qwen 模型（需要网络连接）
- 确保有足够的 GPU 内存（推荐至少 8GB）或使用 CPU 模式（较慢）
- 训练时间取决于模型大小和硬件配置

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
