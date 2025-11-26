# FedMoE 系统架构（简化版）

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator (协调器)                       │
│                     (HTTP API Server)                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         CentralCoordinator (核心协调逻辑)              │    │
│  │                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │    │
│  │  │python_expert │  │  sql_expert   │  │docs_expert│ │    │
│  │  │ (垂直领域1)   │  │  (垂直领域2)   │  │(垂直领域3)│ │    │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘ │    │
│  │         │                  │                  │      │    │
│  │         └──────────────────┼──────────────────┘      │    │
│  │                            │                          │    │
│  │                    ┌───────▼────────┐                 │    │
│  │                    │ Gateway Model  │                 │    │
│  │                    │  (门户/管理模型) │                 │    │
│  │                    │ 整合所有专家梯度  │                 │    │
│  │                    └────────────────┘                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
         │                    │                    │
    ┌────┴────┐          ┌─────┴─────┐        ┌─────┴─────┐
    │Worker-1 │          │ Worker-2  │        │ Worker-3  │
    │GPU机器1 │          │ GPU机器2  │        │ GPU机器3  │
    │         │          │           │        │           │
    │Dataset A│          │Dataset B  │        │Dataset C  │
    │         │          │           │        │           │
    │定时上传  │          │定时上传    │        │定时上传    │
    │梯度(10s)│          │梯度(10s)  │        │梯度(10s)  │
    └─────────┘          └───────────┘        └───────────┘
```

## 数据流

```
Worker 训练循环:
  1. 拉取全局权重 (HTTP GET)
     ↓
  2. 本地微调 (使用本地数据集)
     ↓
  3. 等待定时间隔 (sync_interval)
     ↓
  4. 上传梯度更新 (HTTP POST)
     ↓
  回到步骤 1

Coordinator 处理:
  接收梯度更新
    ↓
  应用 Staleness Decay (处理版本差异)
    ↓
  更新专家模型
    ↓
  聚合到门户模型 (加权平均)
    ↓
  等待下次更新
```

## 文件结构

```
FedMoE/
├── fedmoe/                    # 核心模块
│   ├── coordinator.py        # CentralCoordinator + GatewayModel
│   ├── distributed_coordinator.py  # HTTP API 服务器
│   ├── distributed_worker.py # 分布式 Worker (定时上传)
│   ├── qwen_finetune.py      # QwenLoRAExpert (真实模型)
│   └── ...
├── scripts/                   # 启动脚本
│   ├── start_coordinator.sh  # 启动 Coordinator
│   ├── start_worker.sh       # 启动 Worker
│   ├── stop_coordinator.sh   # 停止 Coordinator
│   ├── stop_worker.sh        # 停止 Worker
│   ├── status.sh             # 查看状态
│   └── stop_all.sh           # 停止所有服务
├── run_coordinator.py        # Coordinator 入口
├── run_worker.py             # Worker 入口
├── get_gateway_model.py      # 查询门户模型
└── ...
```

## 关键特性

1. **分布式训练**: 不同机器上的 GPU 使用不同数据集进行微调
2. **定时同步**: Worker 按照配置的间隔（默认10秒）定时上传梯度
3. **门户模型**: 自动聚合所有专家的梯度，形成统一的管理模型
4. **版本控制**: 支持 staleness-aware 更新，处理网络延迟和版本差异
5. **灵活部署**: 支持多机器、多GPU的分布式部署

## 快速命令参考

```bash
# 启动 Coordinator
bash scripts/start_coordinator.sh

# 启动 Worker
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-1-Python \
    --specialty python_expert \
    --dataset /path/to/dataset.jsonl \
    --sync-interval 10.0

# 查看状态
bash scripts/status.sh

# 查询门户模型
python get_gateway_model.py --coordinator-url http://192.168.1.100:5000

# 停止所有服务
bash scripts/stop_all.sh
```

