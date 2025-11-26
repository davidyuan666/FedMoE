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

### 系统架构

FedMoE 采用分布式联邦学习架构，支持多机器、多GPU的协同训练：

1. **CentralCoordinator (协调器)**
   - 注册和管理多个专家模型 (`python_expert`, `sql_expert`, `docs_expert`, …)
   - 存储 LoRA 权重、版本控制，应用 `SERVER_LR` 和 `STALENESS_DECAY` 进行更新
   - **门户模型 (Gateway Model)**: 自动聚合所有专家的梯度更新，形成统一的管理模型

2. **DistributedWorker (分布式工作节点)**
   - 从协调器拉取最新的全局权重
   - 使用本地数据集进行微调（支持不同机器使用不同数据集）
   - **定时上传梯度**: 按照配置的间隔（默认10秒）上传梯度更新
   - 支持异构硬件（不同的 GPU 速度和网络延迟）

3. **Gateway Model (门户/管理模型)**
   - 整合所有垂直领域专家的梯度更新
   - 使用加权平均策略聚合所有专家的知识
   - 可以随时查询和导出，用于推理或部署

详细的架构图请参考 [ARCHITECTURE.md](ARCHITECTURE.md)（包含 Graphviz DOT 语言描述的完整架构图）。

## 快速开始

### 分布式部署

#### 1. 启动 Coordinator

```bash
# 使用脚本启动（推荐）
bash scripts/start_coordinator.sh

# 或指定参数
bash scripts/start_coordinator.sh --host 0.0.0.0 --port 5000
```

#### 2. 在不同机器上启动 Workers

```bash
# 使用默认参数（自动连接本机 5000 端口，并生成 Worker ID）
bash scripts/start_worker.sh

# 机器1：Python 专家
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-1-Python \
    --specialty python_expert \
    --dataset /path/to/python_dataset.jsonl \
    --sync-interval 10.0

# 机器2：SQL 专家
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-2-SQL \
    --specialty sql_expert \
    --dataset /path/to/sql_dataset.jsonl \
    --sync-interval 10.0
```

#### 3. 查看状态和停止服务

```bash
# 查看所有服务状态
bash scripts/status.sh

# 停止所有服务
bash scripts/stop_all.sh

# 停止单个服务
bash scripts/stop_coordinator.sh
bash scripts/stop_worker.sh --worker-id Worker-1-Python
```

更多脚本使用说明请参考 [scripts/README.md](scripts/README.md)。

### 查询门户模型

```bash
# 查看门户模型信息
python get_gateway_model.py --coordinator-url http://192.168.1.100:5000

# 保存门户模型
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000 \
    --save-gateway gateway_model.npz \
    --list-experts
```

详细的门户模型使用说明请参考 [GATEWAY_MODEL_USAGE.md](GATEWAY_MODEL_USAGE.md)。

## Running Real Qwen Fine-Tuning

**重要提示：所有依赖必须安装到虚拟环境中！**

#### 方式 1：快速运行（推荐）

1. **准备虚拟环境**（仅首次需要）  
   ```bash
   # Windows PowerShell
   .\scripts\setup.ps1

   # Linux/Mac Bash
   bash scripts/setup.sh
   ```
2. **激活虚拟环境**  
   ```bash
   # Windows
   .\.venv\Scripts\Activate.ps1

   # Linux/Mac
   source .venv/bin/activate
   ```
3. **运行 Qwen 微调**  
   ```bash
   python run_qwen_finetune.py
   ```

#### 方式 2：直接使用 Python 脚本（已激活虚拟环境）

```bash
# 在激活虚拟环境后直接运行
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

## 文档

- [ARCHITECTURE.md](ARCHITECTURE.md): 系统架构图（Graphviz DOT 格式）
- [DISTRIBUTED_SETUP.md](DISTRIBUTED_SETUP.md): 分布式部署详细指南
- [GATEWAY_MODEL_USAGE.md](GATEWAY_MODEL_USAGE.md): 门户模型使用指南
- [scripts/README.md](scripts/README.md): 启动脚本使用说明

## License
MIT – customize the orchestration to fit your own multi-GPU, cross-domain fine-tuning workflows.
