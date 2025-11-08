import numpy as np
import threading
import time
import random

# --- 超参数 (Hyperparameters) ---
NUM_WORKERS = 5             # 模拟的客户端（代码仓库）数量
LORA_RANK = 8               # 模拟的 LoRA 秩
MODEL_DIM = 4096            # 模拟的 LLM 基础维度
STALENESS_DECAY = 0.1       # 陈旧度衰减因子 (对应论文中的 decay() 函数)
SERVER_LR = 1.0             # 服务器端聚合学习率

# ----------------------------------------
# 1. 中央服务器 (Aggregator)
# ----------------------------------------

class FedCodeGenServer:
    """
    模拟 FedCodeGen 的中央聚合服务器。
    它异步接收 LoRA 增量，并使用“陈旧度感知”策略聚合它们。
    """
    def __init__(self, model_dim, lora_rank):
        print(f"Initializing FedCodeGen Server (Model Dim: {model_dim}, LoRA Rank: {lora_rank})...")
        
        # 1. 模拟的全局 LoRA 权重 (A 和 B 矩阵)
        # 在真实场景中，基础模型 (W_global) 是冻结的，无需存储在服务器上
        self.lora_shape_A = (model_dim, lora_rank)
        self.lora_shape_B = (lora_rank, model_dim)
        
        self.global_lora_weights_A = np.zeros(self.lora_shape_A)
        self.global_lora_weights_B = np.zeros(self.lora_shape_B)
        
        # 2. 模型版本，用于跟踪陈旧度
        self.model_version = 0
        
        # 3. 线程锁，用于处理并发的异步更新
        self.lock = threading.Lock()

    def get_global_model_state(self):
        """
        客户端调用此方法来“拉取”最新的全局 LoRA 权重和版本。
        """
        with self.lock:
            # 返回权重的副本，防止多线程问题
            return (self.global_lora_weights_A.copy(), 
                    self.global_lora_weights_B.copy(), 
                    self.model_version)

    def receive_lora_update(self, worker_id, lora_delta, worker_model_version):
        """
        核心的异步更新方法。
        Worker 完成训练后调用此方法。
        """
        with self.lock:
            # 1. 计算陈旧度 (Staleness)
            # staleness = 当前服务器版本 - worker 训练时所用的版本
            staleness = self.model_version - worker_model_version
            if staleness < 0: 
                staleness = 0 # 理论上不应发生，但作为保护

            # 2. 计算陈旧度衰减 (Staleness-Aware Aggregation)
            # 论文中的 decay(tau_i) 函数
            decay_factor = 1.0 / (1.0 + STALENESS_DECAY * staleness)
            
            # 3. 应用更新
            # W_global_t+1 = W_global_t + lr * decay * Delta_i
            delta_A, delta_B = lora_delta
            self.global_lora_weights_A += SERVER_LR * decay_factor * delta_A
            self.global_lora_weights_B += SERVER_LR * decay_factor * delta_B
            
            # 4. 增加全局模型版本
            self.model_version += 1
            
            print(f"[Server] Received update from {worker_id} (Staleness: {staleness}). "
                  f"Applied decay: {decay_factor:.3f}. New global version: {self.model_version}")

# ----------------------------------------
# 2. 客户端 (Worker)
# ----------------------------------------

class FedCodeGenWorker:
    """
    模拟一个 FedCodeGen 客户端 (一个隔离的代码仓库)。
    它在异构的网络和计算条件下运行。
    """
    def __init__(self, worker_id, server, model_dim, lora_rank):
        self.worker_id = worker_id
        self.server = server
        self.model_dim = model_dim
        self.lora_rank = lora_rank
        
        # 模拟本地代码仓库数据
        self.local_code_repo = f"Simulated proprietary code for project {worker_id}"
        print(f"[{self.worker_id}] Initialized. Ready to train on: '{self.local_code_repo}'")

    def simulate_local_training(self, lora_A, lora_B):
        """
        模拟在本地代码仓库上进行 LoRA 微调。
        """
        print(f"[{self.worker_id}] Starting local training...")
        
        # 模拟异构计算能力 (训练耗时不同)
        train_time = random.uniform(2, 6)
        time.sleep(train_time)
        
        # --- 模拟 LoRA 训练 ---
        # 1. 复制全局权重
        local_lora_A = lora_A.copy()
        local_lora_B = lora_B.copy()
        
        # 2. 模拟本地梯度更新 (例如，基于 worker_id 产生特定偏向的梯度)
        # 这是一个简化的模拟，代表模型在本地数据上学到了东西
        local_gradient_A = (np.random.randn(self.model_dim, self.lora_rank) * 0.1) + (int(self.worker_id[-1]) * 0.02)
        local_gradient_B = (np.random.randn(self.lora_rank, self.model_dim) * 0.1) - (int(self.worker_id[-1]) * 0.02)
        
        local_lr = 0.01 # 本地学习率
        local_lora_A += local_gradient_A * local_lr
        local_lora_B += local_gradient_B * local_lr
        
        # 3. 计算 LoRA 增量 (Delta)
        # Delta = W_local_t+1 - W_global_t
        delta_A = local_lora_A - lora_A
        delta_B = local_lora_B - lora_B
        
        print(f"[{self.worker_id}] Finished local training in {train_time:.2f}s.")
        return (delta_A, delta_B)

    def run_loop(self):
        """
        Worker 的主循环：拉取 -> 训练 -> 推送
        """
        while True:
            try:
                # 1. 拉取 (Pull)
                (global_A, global_B, global_version) = self.server.get_global_model_state()
                print(f"[{self.worker_id}] Pulled global model version {global_version}")
                
                # 2. 训练 (Train)
                lora_delta = self.simulate_local_training(global_A, global_B)
                
                # 3. 推送 (Push)
                # 模拟异构网络延迟 (上传耗时不同)
                upload_delay = random.uniform(0.5, 2.0)
                time.sleep(upload_delay)
                
                self.server.receive_lora_update(
                    self.worker_id, 
                    lora_delta, 
                    global_version  # 关键：推送更新时，附带上当初拉取时的模型版本
                )
                
                # 模拟 Worker 的空闲时间
                time.sleep(random.uniform(5, 10))
                
            except Exception as e:
                print(f"[{self.worker_id}] Error: {e}. Retrying in 10s...")
                time.sleep(10)

# ----------------------------------------
# 3. 运行模拟
# ----------------------------------------

if __name__ == "__main__":
    
    # 1. 初始化服务器
    server = FedCodeGenServer(MODEL_DIM, LORA_RANK)
    
    # 2. 初始化所有 Workers
    workers = []
    for i in range(NUM_WORKERS):
        worker = FedCodeGenWorker(
            worker_id=f"Worker-{i}", 
            server=server,
            model_dim=MODEL_DIM,
            lora_rank=LORA_RANK
        )
        workers.append(worker)

    # 3. 在单独的线程中启动所有 Workers (模拟异步)
    threads = []
    for worker in workers:
        # daemon=True 确保主程序退出时线程也会退出
        t = threading.Thread(target=worker.run_loop, daemon=True) 
        t.start()
        threads.append(t)

    # 4. 保持主程序运行
    print(f"\n--- FedCodeGen Simulation Started with {NUM_WORKERS} Asynchronous Workers ---")
    print("--- (Press Ctrl+C to stop) ---")
    try:
        while True:
            # 我们可以每 10 秒检查一次全局模型的收敛情况
            time.sleep(10)
            (A, B, version) = server.get_global_model_state()
            print(f"\n--- [Main Thread Monitor] Global Version: {version}. "
                  f"Weight Norm (A): {np.linalg.norm(A):.4f} ---")
            
    except KeyboardInterrupt:
        print("\nShutting down simulation...")