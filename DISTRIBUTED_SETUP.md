# 分布式 FedMoE 使用指南

本指南说明如何在多台机器上运行分布式 Qwen 模型微调。

## 架构概述

- **Coordinator（协调器）**：运行在一台机器上，负责聚合所有 worker 的梯度更新
- **Worker（工作节点）**：运行在不同机器上，每台机器可以运行一个或多个 worker，使用不同的数据集进行微调

## 使用步骤

### 1. 安装依赖

在所有机器上安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 启动 Coordinator 服务器

在**一台机器**上（可以是任意一台，建议选择网络稳定的机器）启动 Coordinator：

```bash
# 默认在 0.0.0.0:5000 启动
python run_coordinator.py

# 或指定地址和端口
python run_coordinator.py --host 192.168.1.100 --port 5000
```

**重要**：记下 Coordinator 的 IP 地址，worker 需要连接到这个地址。

### 3. 在不同机器上启动 Worker

在**每台 GPU 机器**上启动 Worker，连接到 Coordinator：

#### 机器 1：微调 docs 数据集

```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-1-Docs \
    --specialty docs_expert \
    --dataset /path/to/docs_dataset.jsonl \
    --base-model qwen/Qwen2-0.5B-Instruct \
    --speed-factor 1.0 \
    --duration 3600 \
    --sync-interval 10.0
```

#### 机器 2：微调 python 数据集

```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-2-Python \
    --specialty python_expert \
    --dataset /path/to/python_dataset.jsonl \
    --base-model qwen/Qwen2-0.5B-Instruct \
    --speed-factor 1.0 \
    --duration 3600 \
    --sync-interval 10.0
```

#### 机器 3：微调 SQL 数据集

```bash
python run_worker.py \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-3-SQL \
    --specialty sql_expert \
    --dataset /path/to/sql_dataset.jsonl \
    --base-model qwen/Qwen2-0.5B-Instruct \
    --speed-factor 1.0 \
    --duration 3600 \
    --sync-interval 10.0
```

**注意**：`--sync-interval` 参数控制梯度上传的频率。所有 worker 会按照这个间隔定时上传梯度更新。建议根据训练速度和网络条件调整这个值：
- 较小的值（如 5 秒）：更频繁的同步，但会增加网络负载
- 较大的值（如 30 秒）：减少网络负载，但同步延迟更高

### 4. 参数说明

#### Coordinator 参数

- `--host`: 服务器地址（默认: 0.0.0.0，监听所有网络接口）
- `--port`: 服务器端口（默认: 5000）
- `--debug`: 启用调试模式

#### Worker 参数

- `--coordinator-url`: **必需**，Coordinator 服务器地址（如: http://192.168.1.100:5000）
- `--worker-id`: **必需**，Worker 的唯一标识符
- `--specialty`: **必需**，专家类型（如: python_expert, sql_expert, docs_expert）
- `--dataset`: 数据集文件路径（JSONL 格式）
- `--base-model`: 基础模型名称（默认: qwen/Qwen2-0.5B-Instruct）
- `--speed-factor`: 速度因子，越大训练越慢（默认: 1.0）
- `--duration`: 训练持续时间（秒，默认: 3600，即1小时）
- `--use-huggingface`: 使用 HuggingFace 而不是 ModelScope
- `--sync-interval`: **定时上传梯度的间隔（秒，默认: 10.0）**。所有 worker 会按照这个间隔定时上传梯度更新到 coordinator

## 工作流程

1. **初始化**：每个 worker 连接到 coordinator，注册自己的专家类型
2. **训练循环**：
   - Worker 从 coordinator 拉取最新的全局权重
   - Worker 使用本地数据集进行微调
   - Worker **定时**将梯度更新推送到 coordinator（默认每10秒）
   - Coordinator 聚合所有 worker 的更新，更新全局模型
   - Coordinator 同时将更新聚合到**门户模型（管理模型）**，整合所有垂直领域模型的梯度
3. **同步**：每隔指定时间（通过 `--sync-interval` 配置，默认10秒），worker 会拉取最新的全局权重并上传梯度更新

## 门户模型（管理模型）

系统会自动创建一个**门户模型（Gateway Model）**，用于聚合所有垂直领域专家的梯度更新。这个模型整合了所有专家的知识，形成一个统一的管理模型。

### 获取门户模型权重

使用提供的工具脚本获取门户模型的权重：

```bash
# 获取门户模型信息
python get_gateway_model.py --coordinator-url http://192.168.1.100:5000

# 获取门户模型并保存到文件
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000 \
    --save-gateway gateway_model.npz

# 同时列出所有专家的权重信息
python get_gateway_model.py \
    --coordinator-url http://192.168.1.100:5000 \
    --list-experts
```

### 门户模型的特点

- **自动聚合**：每当有专家更新时，门户模型会自动聚合这些更新
- **加权平均**：使用加权平均策略整合所有专家的梯度
- **统一管理**：一个模型整合了所有垂直领域的知识
- **版本控制**：门户模型也有版本号，可以追踪更新历史

## 注意事项

1. **网络要求**：
   - 确保所有机器可以访问 coordinator 的 IP 地址和端口
   - 如果使用防火墙，需要开放 coordinator 的端口

2. **数据集**：
   - 每台机器可以使用不同的数据集
   - 数据集格式必须是 JSONL，每行包含 `prompt` 和 `reference_code` 字段

3. **模型融合**：
   - 所有 worker 共享同一个专家类型时，它们的更新会被聚合
   - 不同专家类型（如 python_expert, sql_expert）的更新是独立的

4. **门户模型（管理模型）**：
   - 系统自动创建一个门户模型，整合所有垂直领域专家的梯度更新
   - 使用 `get_gateway_model.py` 工具可以查询和保存门户模型的权重
   - 门户模型使用加权平均策略聚合所有专家的更新

5. **推理融合**：
   - 当前版本支持训练时的参数融合
   - 推理时的模型融合功能可以后续添加

## 示例场景

假设你有 3 台 GPU 机器：

- **机器 A**（192.168.1.10）：运行 Coordinator + Worker-1（docs_expert）
- **机器 B**（192.168.1.11）：运行 Worker-2（python_expert）
- **机器 C**（192.168.1.12）：运行 Worker-3（sql_expert）

### 步骤 1：在机器 A 启动 Coordinator

```bash
python run_coordinator.py --host 0.0.0.0 --port 5000
```

### 步骤 2：在机器 A 启动 Worker-1

```bash
python run_worker.py \
    --coordinator-url http://localhost:5000 \
    --worker-id Worker-1-Docs \
    --specialty docs_expert \
    --dataset /data/docs.jsonl
```

### 步骤 3：在机器 B 启动 Worker-2

```bash
python run_worker.py \
    --coordinator-url http://192.168.1.10:5000 \
    --worker-id Worker-2-Python \
    --specialty python_expert \
    --dataset /data/python.jsonl
```

### 步骤 4：在机器 C 启动 Worker-3

```bash
python run_worker.py \
    --coordinator-url http://192.168.1.10:5000 \
    --worker-id Worker-3-SQL \
    --specialty sql_expert \
    --dataset /data/sql.jsonl
```

## 故障排查

1. **无法连接到 Coordinator**：
   - 检查 coordinator 是否正在运行
   - 检查网络连接和防火墙设置
   - 确认 coordinator 的 IP 地址和端口正确

2. **权重形状不匹配**：
   - 确保所有 worker 使用相同的基础模型
   - 检查模型是否正确加载

3. **训练速度慢**：
   - 检查 GPU 是否可用
   - 调整 `--speed-factor` 参数
   - 检查数据集大小和批次大小

