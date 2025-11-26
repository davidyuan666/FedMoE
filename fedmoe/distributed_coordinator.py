"""
分布式 Coordinator 服务器，提供 HTTP API 供远程 worker 连接。
"""

from __future__ import annotations

import base64
import json
import threading
from typing import Dict, Tuple

import numpy as np
from flask import Flask, jsonify, request

from .config import SERVER_LR, STALENESS_DECAY
from .coordinator import CentralCoordinator
from .experts import ExpertModel


class DistributedCoordinator:
    """
    分布式 Coordinator，提供 HTTP API 接口。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000):
        """
        初始化分布式 Coordinator。

        Args:
            host: 服务器地址
            port: 服务器端口
        """
        self.coordinator = CentralCoordinator()
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()
        print(f"[DistributedCoordinator] 初始化完成，将在 {host}:{port} 启动服务器")

    def _setup_routes(self):
        """设置 Flask 路由"""

        @self.app.route("/health", methods=["GET"])
        def health():
            """健康检查"""
            return jsonify({"status": "ok"})

        @self.app.route("/register_expert", methods=["POST"])
        def register_expert():
            """注册专家"""
            data = request.json
            expert_name = data.get("expert_name")
            model_dim = data.get("model_dim")
            lora_rank = data.get("lora_rank")

            if not all([expert_name, model_dim, lora_rank]):
                return jsonify({"error": "Missing required fields"}), 400

            self.coordinator.register_expert(expert_name, model_dim, lora_rank)
            return jsonify({"status": "registered", "expert_name": expert_name})

        @self.app.route("/get_expert_weights", methods=["POST"])
        def get_expert_weights():
            """获取专家权重"""
            data = request.json
            expert_name = data.get("expert_name")

            if not expert_name:
                return jsonify({"error": "Missing expert_name"}), 400

            try:
                lora_A, lora_B, version = self.coordinator.get_expert_weights(expert_name)
                # 将 numpy 数组序列化为 base64
                return jsonify({
                    "lora_A": _numpy_to_base64(lora_A),
                    "lora_B": _numpy_to_base64(lora_B),
                    "version": version,
                })
            except ValueError as e:
                return jsonify({"error": str(e)}), 404

        @self.app.route("/push_expert_update", methods=["POST"])
        def push_expert_update():
            """推送专家更新"""
            data = request.json
            expert_name = data.get("expert_name")
            lora_A_b64 = data.get("lora_A")
            lora_B_b64 = data.get("lora_B")
            worker_model_version = data.get("worker_model_version")

            if not all([expert_name, lora_A_b64, lora_B_b64, worker_model_version is not None]):
                return jsonify({"error": "Missing required fields"}), 400

            try:
                lora_A = _base64_to_numpy(lora_A_b64)
                lora_B = _base64_to_numpy(lora_B_b64)
                self.coordinator.push_expert_update(
                    expert_name, (lora_A, lora_B), worker_model_version
                )
                return jsonify({"status": "updated"})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/list_experts", methods=["GET"])
        def list_experts():
            """列出所有专家"""
            experts = list(self.coordinator.experts.keys())
            return jsonify({"experts": experts})

        @self.app.route("/get_gateway_weights", methods=["GET"])
        def get_gateway_weights():
            """获取门户模型（管理模型）的权重，整合了所有专家的梯度更新"""
            try:
                lora_A, lora_B, version = self.coordinator.get_gateway_weights()
                return jsonify({
                    "lora_A": _numpy_to_base64(lora_A),
                    "lora_B": _numpy_to_base64(lora_B),
                    "version": version,
                })
            except ValueError as e:
                return jsonify({"error": str(e)}), 404
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/get_all_expert_weights", methods=["GET"])
        def get_all_expert_weights():
            """获取所有专家的权重"""
            try:
                all_weights = self.coordinator.get_all_expert_weights()
                result = {}
                for expert_name, (lora_A, lora_B, version) in all_weights.items():
                    result[expert_name] = {
                        "lora_A": _numpy_to_base64(lora_A),
                        "lora_B": _numpy_to_base64(lora_B),
                        "version": version,
                    }
                return jsonify({"experts": result})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def run(self, debug: bool = False):
        """运行服务器"""
        print(f"\n=== [分布式 Coordinator 服务器启动] ===")
        print(f"地址: http://{self.host}:{self.port}")
        print(f"等待 worker 连接...\n")
        self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)


def _numpy_to_base64(arr: np.ndarray) -> str:
    """将 numpy 数组转换为 base64 字符串"""
    arr_bytes = arr.tobytes()
    arr_b64 = base64.b64encode(arr_bytes).decode("utf-8")
    return json.dumps({
        "data": arr_b64,
        "shape": arr.shape,
        "dtype": str(arr.dtype),
    })


def _base64_to_numpy(data: str | dict) -> np.ndarray:
    """将 base64 字符串转换为 numpy 数组"""
    if isinstance(data, str):
        data = json.loads(data)
    
    arr_bytes = base64.b64decode(data["data"])
    arr = np.frombuffer(arr_bytes, dtype=np.dtype(data["dtype"]))
    return arr.reshape(data["shape"])

