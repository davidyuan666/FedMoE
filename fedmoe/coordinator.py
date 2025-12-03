"""
Central coordinator responsible for aggregating and distributing expert updates.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple, Optional

import numpy as np

from .config import SERVER_LR, STALENESS_DECAY
from .experts import ExpertModel


class GatewayModel:
    """
    门户模型（汇总模型），聚合所有专家的梯度更新。
    这个模型整合了所有垂直领域模型的更新，形成一个统一的管理模型。
    """
    
    def __init__(self, model_dim: int, lora_rank: int) -> None:
        self.model_dim = model_dim
        self.lora_rank = lora_rank
        self.lora_A = np.zeros((model_dim, lora_rank))
        self.lora_B = np.zeros((lora_rank, model_dim))
        self.version = 0
        self.expert_contributions: Dict[str, float] = {}  # 记录每个专家的贡献权重
        print("[GatewayModel] 门户模型已初始化")
    
    def aggregate_expert_updates(
        self, 
        expert_updates: Dict[str, Tuple[np.ndarray, np.ndarray]],
        aggregation_strategy: str = "weighted_average"
    ) -> None:
        """
        聚合所有专家的更新到门户模型。
        
        Args:
            expert_updates: 字典，键为专家名称，值为(lora_A_delta, lora_B_delta)元组
            aggregation_strategy: 聚合策略，可选 'weighted_average'（加权平均）或 'sum'（求和）
        """
        if not expert_updates:
            return
        
        if aggregation_strategy == "weighted_average":
            # 加权平均：根据专家数量平均分配权重
            num_experts = len(expert_updates)
            weight = 1.0 / num_experts
            
            aggregated_A = np.zeros_like(self.lora_A)
            aggregated_B = np.zeros_like(self.lora_B)
            
            for expert_name, (delta_A, delta_B) in expert_updates.items():
                # 检查形状是否匹配
                if delta_A.shape == self.lora_A.shape and delta_B.shape == self.lora_B.shape:
                    aggregated_A += weight * delta_A
                    aggregated_B += weight * delta_B
                    self.expert_contributions[expert_name] = weight
                else:
                    print(f"[GatewayModel] 警告: 专家 {expert_name} 的权重形状不匹配，跳过")
            
            # 更新门户模型
            self.lora_A += SERVER_LR * aggregated_A
            self.lora_B += SERVER_LR * aggregated_B
            
        elif aggregation_strategy == "sum":
            # 直接求和所有更新
            for expert_name, (delta_A, delta_B) in expert_updates.items():
                if delta_A.shape == self.lora_A.shape and delta_B.shape == self.lora_B.shape:
                    self.lora_A += SERVER_LR * delta_A
                    self.lora_B += SERVER_LR * delta_B
                    self.expert_contributions[expert_name] = 1.0
        
        self.version += 1
        print(f"[GatewayModel] 门户模型已更新 (v{self.version})，聚合了 {len(expert_updates)} 个专家的更新")
    
    def get_weights(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """获取门户模型的权重"""
        return self.lora_A.copy(), self.lora_B.copy(), self.version


class CentralCoordinator:
    """
    Manages expert registration, versioning, and staleness-aware gradient pushes.
    同时管理门户模型，用于聚合所有专家的梯度更新。
    """

    def __init__(self, enable_gateway: bool = True, aggregator_device_id: Optional[int] = None) -> None:
        self.experts: Dict[str, ExpertModel] = {}
        self.gateway_model: Optional[GatewayModel] = None
        self.enable_gateway = enable_gateway
        self.aggregator_device_id = aggregator_device_id
        self.lock = threading.Lock()
        print(
            "Central Coordinator Initialized. "
            f"{('(Aggregator on GPU ' + str(self.aggregator_device_id) + ')') if self.aggregator_device_id is not None else ''}"
        )
        if enable_gateway:
            print("[Coordinator] 门户模型功能已启用")

    def register_expert(self, expert_name: str, model_dim: int, lora_rank: int) -> None:
        with self.lock:
            if expert_name not in self.experts:
                self.experts[expert_name] = ExpertModel(expert_name, model_dim, lora_rank)
                # 如果启用了门户模型且尚未初始化，则初始化
                if self.enable_gateway and self.gateway_model is None:
                    self.gateway_model = GatewayModel(model_dim, lora_rank)
                    print(f"[Coordinator] 门户模型已初始化 (model_dim={model_dim}, lora_rank={lora_rank})")

    def get_expert_weights(self, expert_name: str) -> Tuple[np.ndarray, np.ndarray, int]:
        with self.lock:
            if expert_name not in self.experts:
                raise ValueError(f"Expert {expert_name} not registered.")
            expert = self.experts[expert_name]
            return expert.lora_A.copy(), expert.lora_B.copy(), expert.version

    def push_expert_update(
        self,
        expert_name: str,
        lora_delta: Tuple[np.ndarray, np.ndarray],
        worker_model_version: int,
    ) -> None:
        with self.lock:
            if expert_name not in self.experts:
                return

            expert = self.experts[expert_name]
            staleness = expert.version - worker_model_version
            decay_factor = 1.0 / (1.0 + STALENESS_DECAY * staleness)
            delta_A, delta_B = lora_delta
            expert.lora_A += SERVER_LR * decay_factor * delta_A
            expert.lora_B += SERVER_LR * decay_factor * delta_B
            expert.version += 1

            print(
                f"[Coordinator] Updated {expert.name.upper()} (v{expert.version}) "
                f"from Worker (Staleness: {staleness}, Decay: {decay_factor:.2f})"
            )
            
            # 如果启用了门户模型，将专家更新也聚合到门户模型
            # 注意：门户模型使用衰减后的更新，与专家模型保持一致
            # aggregate_expert_updates 内部会应用 SERVER_LR，所以这里只传入衰减后的增量
            if self.enable_gateway and self.gateway_model is not None:
                expert_updates = {
                    expert_name: (
                        decay_factor * delta_A, 
                        decay_factor * delta_B
                    )
                }
                self.gateway_model.aggregate_expert_updates(expert_updates, aggregation_strategy="weighted_average")
    
    def get_gateway_weights(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        获取门户模型（管理模型）的权重。
        这个模型整合了所有垂直领域模型的更新。
        
        Returns:
            (lora_A, lora_B, version) 元组
        """
        with self.lock:
            if self.gateway_model is None:
                raise ValueError("门户模型未初始化，请先注册至少一个专家")
            return self.gateway_model.get_weights()
    
    def get_all_expert_weights(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, int]]:
        """
        获取所有专家的权重。
        
        Returns:
            字典，键为专家名称，值为(lora_A, lora_B, version)元组
        """
        with self.lock:
            return {
                name: (expert.lora_A.copy(), expert.lora_B.copy(), expert.version)
                for name, expert in self.experts.items()
            }

