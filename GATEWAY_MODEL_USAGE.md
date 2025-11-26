# 门户模型（管理模型）使用指南

## 概述

门户模型（Gateway Model）是一个整合了所有垂直领域专家梯度更新的统一管理模型。它自动聚合来自不同机器、不同数据集的专家更新，形成一个具备所有模型知识的综合模型。

## 架构说明

```
┌─────────────────────────────────────────────────────────┐
│              Coordinator (协调器)                        │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ python_expert│  │  sql_expert  │  │ docs_expert  │ │
│  │  (垂直领域1) │  │  (垂直领域2) │  │  (垂直领域3) │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                            │
│                    ┌───────▼────────┐                    │
│                    │  Gateway Model │                    │
│                    │  (门户/管理模型) │                    │
│                    │  整合所有更新    │                    │
│                    └────────────────┘                    │
└─────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    ┌────┴────┐          ┌─────┴─────┐        ┌─────┴─────┐
    │Worker-1 │          │ Worker-2  │        │ Worker-3  │
    │GPU机器1 │          │ GPU机器2  │        │ GPU机器3  │
    │数据集A  │          │ 数据集B   │        │ 数据集C   │
    └─────────┘          └───────────┘        └───────────┘
```

## 工作流程

1. **分布式训练**：
   - 不同机器上的 Worker 使用不同的数据集进行微调
   - 每个 Worker 定时（默认每10秒）上传梯度更新到 Coordinator

2. **专家模型更新**：
   - Coordinator 接收每个专家的梯度更新
   - 根据版本差异应用衰减因子（staleness decay）
   - 更新对应的专家模型

3. **门户模型聚合**：
   - 每当有专家更新时，Coordinator 自动将更新聚合到门户模型
   - 使用加权平均策略整合所有专家的梯度
   - 门户模型版本号递增

4. **获取管理模型**：
   - 使用 `get_gateway_model.py` 工具可以随时查询门户模型的权重
   - 可以保存门户模型用于后续推理或部署

## 使用示例

### 1. 启动 Coordinator

```bash
python run_coordinator.py --host 0.0.0.0 --port 5000
```

### 2. 在不同机器上启动 Worker

**机器1 - Python 专家：**
```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-1-Python \
    --specialty python_expert \
    --dataset /data/python_dataset.jsonl \
    --sync-interval 10.0
```

**机器2 - SQL 专家：**
```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-2-SQL \
    --specialty sql_expert \
    --dataset /data/sql_dataset.jsonl \
    --sync-interval 10.0
```

**机器3 - 文档专家：**
```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-3-Docs \
    --specialty docs_expert \
    --dataset /data/docs_dataset.jsonl \
    --sync-interval 10.0
```

### 3. 查询门户模型

**查看门户模型信息：**
```bash
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000
```

**保存门户模型：**
```bash
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000 \
    --save-gateway gateway_model.npz
```

**查看所有专家和门户模型：**
```bash
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000 \
    --list-experts
```

### 4. 加载门户模型（Python 代码示例）

```python
import numpy as np

# 加载保存的门户模型
data = np.load('gateway_model.npz')
lora_A = data['lora_A']
lora_B = data['lora_B']
version = data['version']

print(f"门户模型版本: {version}")
print(f"LoRA A 形状: {lora_A.shape}")
print(f"LoRA B 形状: {lora_B.shape}")

# 可以将这些权重应用到 Qwen 模型进行推理
```

## 配置参数

### Worker 参数

- `--sync-interval`: 定时上传梯度的间隔（秒）
  - 默认值：10.0
  - 建议范围：5-30 秒
  - 较小的值：更频繁的同步，但增加网络负载
  - 较大的值：减少网络负载，但同步延迟更高

### 聚合策略

门户模型支持两种聚合策略（在代码中配置）：

1. **weighted_average**（加权平均，默认）：
   - 根据专家数量平均分配权重
   - 适合所有专家同等重要的场景

2. **sum**（求和）：
   - 直接累加所有专家的更新
   - 适合需要累积所有知识的场景

## 优势

1. **统一管理**：一个模型整合所有垂直领域的知识
2. **自动聚合**：无需手动操作，系统自动整合更新
3. **版本控制**：可以追踪门户模型的更新历史
4. **灵活部署**：可以随时导出门户模型用于推理
5. **分布式友好**：支持多机器、多GPU的分布式训练

## 注意事项

1. **网络稳定性**：确保 Coordinator 和 Worker 之间的网络连接稳定
2. **同步频率**：根据训练速度和网络条件调整 `sync-interval`
3. **模型维度**：所有专家必须使用相同的基础模型和 LoRA 配置
4. **资源管理**：门户模型会占用额外的内存，但通常很小（LoRA 参数）

## API 端点

Coordinator 提供以下 HTTP API：

- `GET /get_gateway_weights`: 获取门户模型权重
- `GET /get_all_expert_weights`: 获取所有专家的权重
- `GET /list_experts`: 列出所有已注册的专家
- `POST /push_expert_update`: 推送专家更新（Worker 使用）
- `POST /get_expert_weights`: 获取指定专家的权重（Worker 使用）

## 故障排查

1. **门户模型未初始化**：
   - 确保至少有一个 Worker 已注册专家
   - 检查 Coordinator 日志

2. **权重形状不匹配**：
   - 确保所有 Worker 使用相同的基础模型
   - 检查 LoRA 配置是否一致

3. **更新未聚合**：
   - 检查 Coordinator 的 `enable_gateway` 参数
   - 查看 Coordinator 日志确认更新是否被接收

