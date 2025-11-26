# FedMoE 启动脚本使用指南

本目录包含用于启动、停止和管理 FedMoE 分布式系统的脚本。

## 脚本列表

### Coordinator 管理

- **`start_coordinator.sh`**: 启动 Coordinator 服务器
- **`stop_coordinator.sh`**: 停止 Coordinator 服务器

### Worker 管理

- **`start_worker.sh`**: 启动 Worker 客户端
- **`stop_worker.sh`**: 停止指定的 Worker

### 系统管理

- **`status.sh`**: 查看所有服务的运行状态
- **`stop_all.sh`**: 停止所有服务（Coordinator + 所有 Workers）

## 使用示例

### 1. 启动 Coordinator

脚本默认以**前台模式**运行（实时输出 + Ctrl+C 停止）。如需后台运行，请加 `--daemon`。

```bash
# 使用默认配置（0.0.0.0:5000，前台模式）
bash scripts/start_coordinator.sh

# 指定地址和端口
HOST=192.168.1.100 PORT=5000 bash scripts/start_coordinator.sh

# 后台运行（配合 stop/status 脚本）
bash scripts/start_coordinator.sh --daemon

# 启用调试模式
bash scripts/start_coordinator.sh --debug
```

### 2. 启动 Worker

```bash
# 最简用法（全部使用默认值）
bash scripts/start_worker.sh

# 常用参数
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-1-Python \
    --specialty python_expert \
    --dataset /path/to/python_dataset.jsonl \
    --base-model qwen/Qwen2-0.5B-Instruct \
    --speed-factor 1.0 \
    --duration 3600 \
    --sync-interval 10.0
```

默认行为：

- `--coordinator-url`: 默认 `http://127.0.0.1:5000`
- `--specialty`: 默认 `python_expert`
- `--worker-id`: 自动生成（`{hostname}-{specialty}-{timestamp}`）
- `--dataset`: 自动搜索 `dataset/{specialty}.jsonl`、`dataset/test.jsonl`
- 其他参数使用 `scripts/start_worker.sh` 顶部的默认值

可以通过环境变量覆盖默认值（无需修改脚本）：

```bash
export COORDINATOR_URL=http://192.168.1.100:5000
export SPECIALTY=sql_expert
export DATASET=/data/sql_dataset.jsonl
bash scripts/start_worker.sh
```

### 3. 查看服务状态

```bash
bash scripts/status.sh
```

输出示例：
```
==========================================
FedMoE 服务状态
==========================================

✓ Coordinator: 运行中 (PID: 12345)

Workers:
  ✓ Worker-1-Python: 运行中 (PID: 12346)
  ✓ Worker-2-SQL: 运行中 (PID: 12347)
  ✓ Worker-3-Docs: 运行中 (PID: 12348)
```

### 4. 停止服务

```bash
# 停止 Coordinator
bash scripts/stop_coordinator.sh

# 停止指定的 Worker
bash scripts/stop_worker.sh --worker-id Worker-1-Python

# 停止所有服务
bash scripts/stop_all.sh
```

## 日志文件

所有服务的日志文件保存在 `logs/` 目录：

- `logs/coordinator.log`: Coordinator 日志
- `logs/worker-{WORKER_ID}.log`: 各 Worker 的日志

查看日志：

```bash
# 查看 Coordinator 日志
tail -f logs/coordinator.log

# 查看 Worker 日志
tail -f logs/worker-Worker-1-Python.log
```

## PID 文件

脚本使用 PID 文件来跟踪运行中的进程：

- `coordinator.pid`: Coordinator 的 PID
- `logs/worker-{WORKER_ID}.pid`: 各 Worker 的 PID

如果进程异常退出，PID 文件可能残留，可以手动删除。

## 环境变量

可以通过环境变量设置默认值：

```bash
# 设置虚拟环境目录
export VENV_DIR=.venv

# 设置 Coordinator 地址和端口
export HOST=0.0.0.0
export PORT=5000
export DEBUG=false
```

## 完整部署示例

### 机器 1：启动 Coordinator

```bash
cd /path/to/FedMoE
bash scripts/start_coordinator.sh --host 0.0.0.0 --port 5000
```

### 机器 1：启动 Worker-1 (Docs)

```bash
cd /path/to/FedMoE
bash scripts/start_worker.sh \
    --coordinator-url http://localhost:5000 \
    --worker-id Worker-1-Docs \
    --specialty docs_expert \
    --dataset /data/docs.jsonl \
    --sync-interval 10.0
```

### 机器 2：启动 Worker-2 (Python)

```bash
cd /path/to/FedMoE
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-2-Python \
    --specialty python_expert \
    --dataset /data/python.jsonl \
    --sync-interval 10.0
```

### 机器 3：启动 Worker-3 (SQL)

```bash
cd /path/to/FedMoE
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5000 \
    --worker-id Worker-3-SQL \
    --specialty sql_expert \
    --dataset /data/sql.jsonl \
    --sync-interval 10.0
```

### 查看状态

在任何机器上运行：

```bash
bash scripts/status.sh
```

### 停止所有服务

```bash
# 在每台机器上分别停止
bash scripts/stop_worker.sh --worker-id Worker-1-Docs
bash scripts/stop_worker.sh --worker-id Worker-2-Python
bash scripts/stop_worker.sh --worker-id Worker-3-SQL

# 在 Coordinator 机器上停止
bash scripts/stop_coordinator.sh
```

## 故障排查

### 1. 端口被占用

如果 Coordinator 端口被占用，可以更改端口：

```bash
bash scripts/start_coordinator.sh --port 5001
```

然后 Worker 连接到新端口：

```bash
bash scripts/start_worker.sh \
    --coordinator-url http://192.168.1.100:5001 \
    ...
```

### 2. 无法连接到 Coordinator

检查：
- Coordinator 是否正在运行：`bash scripts/status.sh`
- 网络连接：`curl http://192.168.1.100:5000/health`
- 防火墙设置

### 3. Worker 启动失败

检查：
- 虚拟环境是否正确设置
- 依赖是否已安装
- 日志文件中的错误信息

### 4. 进程无法停止

如果正常停止失败，可以强制停止：

```bash
# 查找进程
ps aux | grep run_coordinator.py
ps aux | grep run_worker.py

# 强制杀死
kill -9 <PID>
```

然后清理 PID 文件：

```bash
rm -f coordinator.pid
rm -f logs/worker-*.pid
```

