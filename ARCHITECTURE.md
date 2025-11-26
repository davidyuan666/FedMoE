# FedMoE 系统架构图

## 系统架构概览

本文档使用 Graphviz DOT 语言描述 FedMoE 分布式联邦学习系统的架构。

## 完整架构图

```dot
digraph FedMoE {
    rankdir=TB;
    node [shape=box, style=rounded];
    
    subgraph cluster_coordinator {
        label="Coordinator (协调器)";
        style=filled;
        color=lightblue;
        
        DC [label="DistributedCoordinator\n(HTTP API Server)", shape=ellipse];
        CC [label="CentralCoordinator\n(核心协调逻辑)"];
        GM [label="GatewayModel\n(门户/管理模型)\n整合所有专家梯度"];
        
        subgraph cluster_experts {
            label="专家模型 (Expert Models)";
            PE [label="python_expert\n(Python 垂直领域)"];
            SE [label="sql_expert\n(SQL 垂直领域)"];
            DE [label="docs_expert\n(文档垂直领域)"];
        }
        
        DC -> CC;
        CC -> PE;
        CC -> SE;
        CC -> DE;
        CC -> GM;
        PE -> GM [style=dashed, label="梯度聚合"];
        SE -> GM [style=dashed, label="梯度聚合"];
        DE -> GM [style=dashed, label="梯度聚合"];
    }
    
    subgraph cluster_worker1 {
        label="Worker-1 (GPU 机器 1)";
        style=filled;
        color=lightgreen;
        
        DW1 [label="DistributedWorker\n(分布式客户端)"];
        QE1 [label="QwenLoRAExpert\n(本地专家实例)"];
        DS1 [label="Dataset A\n(数据集 A)"];
        
        DW1 -> QE1;
        DS1 -> QE1 [style=dashed];
    }
    
    subgraph cluster_worker2 {
        label="Worker-2 (GPU 机器 2)";
        style=filled;
        color=lightgreen;
        
        DW2 [label="DistributedWorker\n(分布式客户端)"];
        QE2 [label="QwenLoRAExpert\n(本地专家实例)"];
        DS2 [label="Dataset B\n(数据集 B)"];
        
        DW2 -> QE2;
        DS2 -> QE2 [style=dashed];
    }
    
    subgraph cluster_worker3 {
        label="Worker-3 (GPU 机器 3)";
        style=filled;
        color=lightgreen;
        
        DW3 [label="DistributedWorker\n(分布式客户端)"];
        QE3 [label="QwenLoRAExpert\n(本地专家实例)"];
        DS3 [label="Dataset C\n(数据集 C)"];
        
        DW3 -> QE3;
        DS3 -> QE3 [style=dashed];
    }
    
    DW1 -> DC [label="HTTP\n定时上传梯度\n(sync_interval)"];
    DW2 -> DC [label="HTTP\n定时上传梯度\n(sync_interval)"];
    DW3 -> DC [label="HTTP\n定时上传梯度\n(sync_interval)"];
    
    DC -> DW1 [label="HTTP\n拉取全局权重"];
    DC -> DW2 [label="HTTP\n拉取全局权重"];
    DC -> DW3 [label="HTTP\n拉取全局权重"];
    
    QE1 -> PE [style=dashed, label="对应"];
    QE2 -> SE [style=dashed, label="对应"];
    QE3 -> DE [style=dashed, label="对应"];
}
```

## 数据流图

```dot
digraph DataFlow {
    rankdir=LR;
    node [shape=box];
    
    subgraph cluster_training_loop {
        label="训练循环 (每个 Worker)";
        
        START [label="开始训练"];
        PULL [label="1. 拉取全局权重\n(get_expert_weights)"];
        TRAIN [label="2. 本地微调\n(使用本地数据集)"];
        WAIT [label="3. 等待定时间隔\n(sync_interval)"];
        PUSH [label="4. 上传梯度更新\n(push_expert_update)"];
        
        START -> PULL;
        PULL -> TRAIN;
        TRAIN -> WAIT;
        WAIT -> PUSH;
        PUSH -> PULL [label="循环"];
    }
    
    subgraph cluster_aggregation {
        label="Coordinator 聚合";
        
        RECEIVE [label="接收梯度更新"];
        UPDATE_EXPERT [label="更新专家模型\n(应用 staleness decay)"];
        AGGREGATE_GATEWAY [label="聚合到门户模型\n(加权平均)"];
        
        RECEIVE -> UPDATE_EXPERT;
        UPDATE_EXPERT -> AGGREGATE_GATEWAY;
    }
    
    PUSH -> RECEIVE [label="HTTP POST"];
    RECEIVE -> PULL [label="下次拉取时\n获取最新权重"];
}
```

## 组件关系图

```dot
graph ComponentRelations {
    rankdir=TB;
    node [shape=box];
    
    subgraph cluster_core {
        label="核心模块";
        
        CC [label="CentralCoordinator\ncoordinator.py"];
        GM [label="GatewayModel\ncoordinator.py"];
        EM [label="ExpertModel\nexperts.py"];
    }
    
    subgraph cluster_distributed {
        label="分布式模块";
        
        DC [label="DistributedCoordinator\ndistributed_coordinator.py"];
        DW [label="DistributedWorker\ndistributed_worker.py"];
    }
    
    subgraph cluster_qwen {
        label="Qwen 模型模块";
        
        QE [label="QwenLoRAExpert\nqwen_finetune.py"];
        QW [label="QwenWorker\nqwen_worker.py"];
    }
    
    subgraph cluster_scripts {
        label="启动脚本";
        
        SC [label="start_coordinator.sh"];
        SW [label="start_worker.sh"];
        ST [label="status.sh"];
        SA [label="stop_all.sh"];
    }
    
    DC -> CC;
    DW -> DC [label="HTTP"];
    DW -> QE;
    CC -> EM;
    CC -> GM;
    QE -> EM [style=dashed, label="对应"];
    
    SC -> DC;
    SW -> DW;
}
```

## 文件结构图

```dot
digraph FileStructure {
    rankdir=TB;
    node [shape=folder];
    
    ROOT [label="FedMoE/"];
    
    subgraph cluster_fedmoe {
        label="fedmoe/";
        
        COORD [label="coordinator.py\nCentralCoordinator\nGatewayModel"];
        DCOORD [label="distributed_coordinator.py\nDistributedCoordinator\n(HTTP API)"];
        DWORKER [label="distributed_worker.py\nDistributedWorker\n(定时上传梯度)"];
        QFINETUNE [label="qwen_finetune.py\nQwenLoRAExpert\n(真实模型微调)"];
        QWORKER [label="qwen_worker.py\nQwenWorker"];
        EXPERTS [label="experts.py\nExpertModel"];
        CONFIG [label="config.py\n配置参数"];
    }
    
    subgraph cluster_scripts {
        label="scripts/";
        
        START_C [label="start_coordinator.sh"];
        STOP_C [label="stop_coordinator.sh"];
        START_W [label="start_worker.sh"];
        STOP_W [label="stop_worker.sh"];
        STATUS [label="status.sh"];
        STOP_ALL [label="stop_all.sh"];
    }
    
    subgraph cluster_root {
        label="根目录";
        
        RUN_C [label="run_coordinator.py"];
        RUN_W [label="run_worker.py"];
        GET_GW [label="get_gateway_model.py"];
    }
    
    ROOT -> cluster_fedmoe;
    ROOT -> cluster_scripts;
    ROOT -> cluster_root;
}
```

## 部署架构图

```dot
digraph Deployment {
    rankdir=LR;
    node [shape=box, style=rounded];
    
    subgraph cluster_machine1 {
        label="机器 1 (Coordinator + Worker)";
        style=filled;
        color=lightblue;
        
        COORD_M1 [label="Coordinator\n:5000"];
        WORKER1 [label="Worker-1-Docs\ndocs_expert\nDataset: docs.jsonl"];
    }
    
    subgraph cluster_machine2 {
        label="机器 2 (Worker)";
        style=filled;
        color=lightgreen;
        
        WORKER2 [label="Worker-2-Python\npython_expert\nDataset: python.jsonl"];
    }
    
    subgraph cluster_machine3 {
        label="机器 3 (Worker)";
        style=filled;
        color=lightgreen;
        
        WORKER3 [label="Worker-3-SQL\nsql_expert\nDataset: sql.jsonl"];
    }
    
    WORKER1 -> COORD_M1 [label="HTTP\n定时同步\n(sync_interval=10s)"];
    WORKER2 -> COORD_M1 [label="HTTP\n定时同步\n(sync_interval=10s)"];
    WORKER3 -> COORD_M1 [label="HTTP\n定时同步\n(sync_interval=10s)"];
    
    COORD_M1 -> WORKER1 [label="HTTP\n返回全局权重"];
    COORD_M1 -> WORKER2 [label="HTTP\n返回全局权重"];
    COORD_M1 -> WORKER3 [label="HTTP\n返回全局权重"];
}
```

## 梯度聚合流程

```dot
digraph GradientAggregation {
    rankdir=TB;
    node [shape=box];
    
    subgraph cluster_workers {
        label="Workers (不同机器)";
        
        W1 [label="Worker-1\n上传梯度 Δ₁"];
        W2 [label="Worker-2\n上传梯度 Δ₂"];
        W3 [label="Worker-3\n上传梯度 Δ₃"];
    }
    
    subgraph cluster_coordinator {
        label="Coordinator";
        
        RECEIVE [label="接收梯度更新"];
        DECAY [label="应用 Staleness Decay\n(处理版本差异)"];
        UPDATE [label="更新专家模型\nExpert += LR × decay × Δ"];
        AGGREGATE [label="聚合到门户模型\nGateway = weighted_avg(Δ₁, Δ₂, Δ₃)"];
    }
    
    W1 -> RECEIVE;
    W2 -> RECEIVE;
    W3 -> RECEIVE;
    
    RECEIVE -> DECAY;
    DECAY -> UPDATE;
    UPDATE -> AGGREGATE;
    
    AGGREGATE -> GATEWAY [label="门户模型\n整合所有知识"];
    
    GATEWAY [label="Gateway Model\n(管理模型)\n具备所有垂直领域的梯度更新", shape=ellipse, style=filled, color=lightyellow];
}
```

## 使用说明

### 生成图片

要生成这些架构图的图片，需要安装 Graphviz：

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
# 下载安装: https://graphviz.org/download/
```

然后使用以下命令生成图片：

```bash
# 生成完整架构图
dot -Tpng -o architecture.png ARCHITECTURE.md

# 或者使用 neato, fdp, sfdp, twopi, circo 等布局引擎
neato -Tpng -o architecture.png ARCHITECTURE.md
```

### 在线查看

也可以使用在线工具查看：
- https://dreampuf.github.io/GraphvizOnline/
- https://edotor.net/

直接将 DOT 代码复制到这些工具中即可查看和编辑。

