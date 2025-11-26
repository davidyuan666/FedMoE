"""
真实的 Qwen 模型微调实现，使用 LoRA 进行参数更新。

这个模块提供了与模拟系统兼容的真实模型微调功能，可以替换模拟的训练逻辑。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# 尝试导入 ModelScope，如果失败则使用 HuggingFace
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    snapshot_download = None

from .config import DEFAULT_EXPERTS, DEFAULT_WORKERS, LORA_RANK, MODEL_DIM, SimulationConfig


def load_jsonl_dataset(file_path: str | Path) -> List[Dict]:
    """
    从 JSONL 文件加载数据集。

    Args:
        file_path: JSONL 文件路径

    Returns:
        数据列表，每个元素是一个包含 'prompt' 和 'reference_code' 的字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "prompt" in item and "reference_code" in item:
                    data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 跳过无效的 JSON 行: {e}")
                continue

    print(f"从 {file_path} 加载了 {len(data)} 条数据")
    return data


def format_qwen_training_text(prompt: str, reference_code: str, use_chat_template: bool = True) -> str:
    """
    将 prompt 和 reference_code 格式化为 Qwen 训练格式。

    Args:
        prompt: 问题提示
        reference_code: 参考代码（答案）
        use_chat_template: 是否使用 tokenizer 的 chat_template（如果可用）

    Returns:
        格式化后的训练文本
    """
    # 如果使用 chat_template，需要在 tokenizer 中处理
    # 这里先返回简单的格式，在 fine_tune_step 中使用 tokenizer 的 chat_template
    if use_chat_template:
        # 返回原始数据，让 tokenizer 的 chat_template 处理
        return {"prompt": prompt, "reference_code": reference_code}
    else:
        # 使用简单的格式
        return f"{prompt}\n\n{reference_code}"


def prepare_training_texts_from_jsonl(
    jsonl_data: List[Dict], use_chat_template: bool = True
) -> List[str | Dict]:
    """
    从 JSONL 数据准备训练文本。

    Args:
        jsonl_data: JSONL 数据列表
        use_chat_template: 是否使用 chat_template 格式

    Returns:
        格式化后的训练文本列表
    """
    training_texts = []
    for item in jsonl_data:
        prompt = item.get("prompt", "")
        reference_code = item.get("reference_code", "")
        if prompt and reference_code:
            formatted = format_qwen_training_text(prompt, reference_code, use_chat_template)
            training_texts.append(formatted)
    return training_texts


class QwenLoRAExpert:
    """
    基于真实 Qwen 模型的 LoRA 专家实现。
    每个专家对应一个专门的 LoRA 适配器，可以独立训练和更新。
    """

    def __init__(
        self,
        expert_name: str,
        base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
        lora_rank: int = LORA_RANK,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_modelscope: bool = True,
    ) -> None:
        """
        初始化 Qwen LoRA 专家。

        Args:
            expert_name: 专家名称（如 'python_expert', 'sql_expert'）
            base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头，如 'qwen/Qwen2-0.5B-Instruct'）
            lora_rank: LoRA 的秩
            lora_alpha: LoRA 的 alpha 参数
            lora_dropout: LoRA dropout 率
            device: 运行设备
            use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        """
        self.expert_name = expert_name
        self.base_model_name = base_model_name
        self.device = device
        self.lora_rank = lora_rank
        self.version = 0
        self.use_modelscope = use_modelscope and MODELSCOPE_AVAILABLE

        print(f"[QwenExpert] 初始化专家: {expert_name} (设备: {device})")
        if self.use_modelscope:
            print(f"[QwenExpert] 使用 ModelScope 加载模型: {base_model_name}")
        else:
            print(f"[QwenExpert] 使用 HuggingFace 加载模型: {base_model_name}")

        # 加载基础模型和 tokenizer
        model_path = base_model_name
        if self.use_modelscope:
            # 使用 ModelScope 下载模型到本地缓存
            try:
                model_path = snapshot_download(base_model_name, cache_dir=None)
                print(f"[QwenExpert] ModelScope 模型已下载到: {model_path}")
            except Exception as e:
                print(f"[QwenExpert] ModelScope 下载失败，回退到 HuggingFace: {e}")
                model_path = base_model_name
                self.use_modelscope = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            trust_remote_code=True,
        )

        # 配置 LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的注意力层
            bias="none",
        )

        # 应用 LoRA 到模型
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        print(f"[QwenExpert] {expert_name} 初始化完成")

    def get_lora_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取当前 LoRA 权重 (lora_A 和 lora_B)。
        只提取 q_proj 层的权重，确保与 coordinator 的形状一致。
        q_proj 层的输入维度是 hidden_size，输出维度也是 hidden_size（对于 Qwen 模型）。

        Returns:
            (lora_A, lora_B) 的 numpy 数组元组，形状为 (hidden_size, rank) 和 (rank, hidden_size)
        """
        model_dim = self.base_model.config.hidden_size
        lora_A_list = []
        lora_B_list = []

        # 只收集 q_proj 层的权重（确保与 coordinator 维度一致）
        # q_proj 的输入是 hidden_size，输出也是 hidden_size（对于 Qwen）
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "q_proj" in name:
                A_weight = module.lora_A["default"].weight.data.cpu().numpy()
                B_weight = module.lora_B["default"].weight.data.cpu().numpy()
                
                # 只收集形状为 (hidden_size, rank) 和 (rank, hidden_size) 的层
                # 这确保与 coordinator 中的维度一致
                if A_weight.shape == (model_dim, self.lora_rank) and B_weight.shape == (self.lora_rank, model_dim):
                    lora_A_list.append(A_weight)
                    lora_B_list.append(B_weight)

        if len(lora_A_list) == 0:
            # 如果没有找到匹配的层，返回零矩阵
            return (
                np.zeros((model_dim, self.lora_rank)),
                np.zeros((self.lora_rank, model_dim)),
            )

        # 聚合所有 q_proj 层的权重（简单平均）
        lora_A = np.mean(lora_A_list, axis=0)
        lora_B = np.mean(lora_B_list, axis=0)

        return lora_A, lora_B

    def update_lora_weights(
        self, lora_A: np.ndarray, lora_B: np.ndarray, learning_rate: float = 1.0
    ) -> None:
        """
        更新 LoRA 权重。
        只更新 q_proj 层，确保与 coordinator 的维度一致。
        注意：lora_A 和 lora_B 是增量（delta），会被添加到当前权重上。

        Args:
            lora_A: lora_A 权重增量，形状应为 (hidden_size, rank)
            lora_B: lora_B 权重增量，形状应为 (rank, hidden_size)
            learning_rate: 学习率（用于缩放更新）
        """
        model_dim = self.base_model.config.hidden_size
        expected_shape_A = (model_dim, self.lora_rank)
        expected_shape_B = (self.lora_rank, model_dim)
        
        # 检查输入形状
        if lora_A.shape != expected_shape_A or lora_B.shape != expected_shape_B:
            print(
                f"[QwenExpert] 警告: {self.expert_name} 权重形状不匹配！"
                f"期望: A={expected_shape_A}, B={expected_shape_B}, "
                f"实际: A={lora_A.shape}, B={lora_B.shape}"
            )
            return
        
        lora_A_tensor = torch.from_numpy(lora_A).to(self.device)
        lora_B_tensor = torch.from_numpy(lora_B).to(self.device)

        updated_count = 0
        # 只更新 q_proj 层，确保与 coordinator 维度一致
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "q_proj" in name:
                # 获取当前权重
                current_A = module.lora_A["default"].weight.data
                current_B = module.lora_B["default"].weight.data

                # 只更新形状完全匹配的层
                if current_A.shape == lora_A_tensor.shape and current_B.shape == lora_B_tensor.shape:
                    module.lora_A["default"].weight.data = (
                        current_A + learning_rate * lora_A_tensor
                    )
                    module.lora_B["default"].weight.data = (
                        current_B + learning_rate * lora_B_tensor
                    )
                    updated_count += 1

        if updated_count == 0:
            print(f"[QwenExpert] 警告: {self.expert_name} 没有找到形状匹配的 q_proj 层进行更新")
            print(f"  期望形状: A={expected_shape_A}, B={expected_shape_B}")

        self.version += 1
        if updated_count > 0:
            print(f"[QwenExpert] {self.expert_name} 权重已更新 (版本: {self.version}, 更新了 {updated_count} 个 q_proj 层)")

    def fine_tune_step(
        self,
        training_data: List[str | Dict],
        batch_size: int = 4,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        max_length: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一个微调步骤，返回权重更新。

        Args:
            training_data: 训练数据（字符串列表或包含 'prompt' 和 'reference_code' 的字典列表）
            batch_size: 批次大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            max_length: 最大序列长度

        Returns:
            (delta_A, delta_B) 权重更新
        """
        print(f"[QwenExpert] {self.expert_name} 开始微调...")

        # 获取训练前的权重
        lora_A_before, lora_B_before = self.get_lora_weights()

        # 准备训练数据
        if not training_data:
            # 如果没有数据，返回零更新
            print(f"[QwenExpert] {self.expert_name} 没有训练数据，跳过微调")
            return (
                np.zeros_like(lora_A_before),
                np.zeros_like(lora_B_before),
            )

        # 处理训练数据：如果是字典格式（包含 prompt 和 reference_code），使用 chat_template
        formatted_texts = []
        has_chat_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

        for item in training_data:
            if isinstance(item, dict) and "prompt" in item and "reference_code" in item:
                # 使用 chat_template 格式化对话数据
                if has_chat_template:
                    messages = [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["reference_code"]},
                    ]
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                else:
                    # 如果没有 chat_template，使用简单格式
                    formatted_text = f"{item['prompt']}\n\n{item['reference_code']}"
                formatted_texts.append(formatted_text)
            elif isinstance(item, str):
                # 直接使用字符串
                formatted_texts.append(item)
            else:
                print(f"[QwenExpert] 警告: 跳过不支持的数据格式: {type(item)}")

        if not formatted_texts:
            print(f"[QwenExpert] {self.expert_name} 没有有效的训练数据，跳过微调")
            return (
                np.zeros_like(lora_A_before),
                np.zeros_like(lora_B_before),
            )

        # 编码数据
        encoded = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # 创建数据集
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.input_ids = encodings["input_ids"]
                self.attention_mask = encodings["attention_mask"]

            def __getitem__(self, idx):
                return {
                    "input_ids": self.input_ids[idx],
                    "attention_mask": self.attention_mask[idx],
                    "labels": self.input_ids[idx].clone(),
                }

            def __len__(self):
                return len(self.input_ids)

        dataset = SimpleDataset(encoded)

        # 训练参数
        training_args = TrainingArguments(
            output_dir=f"./tmp/{self.expert_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_strategy="no",  # 不保存检查点
            report_to="none",
            remove_unused_columns=False,
        )

        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        # 执行训练
        trainer.train()

        # 获取训练后的权重
        lora_A_after, lora_B_after = self.get_lora_weights()

        # 计算更新
        delta_A = lora_A_after - lora_A_before
        delta_B = lora_B_after - lora_B_before

        print(f"[QwenExpert] {self.expert_name} 微调完成")
        return delta_A, delta_B

    def generate(self, prompt: str, max_length: int = 256) -> str:
        """
        使用当前专家模型生成文本。

        Args:
            prompt: 输入提示
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def save_expert(self, save_path: str) -> None:
        """
        保存专家模型。

        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[QwenExpert] {self.expert_name} 已保存到 {save_path}")

    def load_expert(self, load_path: str) -> None:
        """
        加载专家模型。

        Args:
            load_path: 加载路径
        """
        self.model = PeftModel.from_pretrained(self.base_model, load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"[QwenExpert] {self.expert_name} 已从 {load_path} 加载")


def create_qwen_expert(
    expert_name: str,
    base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
    lora_rank: int = LORA_RANK,
    use_modelscope: bool = True,
) -> QwenLoRAExpert:
    """
    创建 Qwen LoRA 专家的便捷函数。

    Args:
        expert_name: 专家名称
        base_model_name: 基础模型名称（ModelScope 格式：小写开头）
        lora_rank: LoRA 秩
        use_modelscope: 是否使用 ModelScope 加载模型（默认 True）

    Returns:
        QwenLoRAExpert 实例
    """
    return QwenLoRAExpert(
        expert_name=expert_name,
        base_model_name=base_model_name,
        lora_rank=lora_rank,
        use_modelscope=use_modelscope,
    )


# 导入必要的模块用于系统初始化
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coordinator import CentralCoordinator
    from .qwen_worker import QwenWorker


def initialize_qwen_system(
    config: SimulationConfig,
    base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
    use_modelscope: bool = True,
    training_data_path: str | None = None,
) -> tuple:
    """
    初始化使用真实 Qwen 模型的系统。

    Args:
        config: 配置
        base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头）
        use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        training_data_path: JSONL 数据集文件路径（如果提供，所有 worker 会使用此数据集）

    Returns:
        (coordinator, workers) 元组
    """
    from .coordinator import CentralCoordinator
    from .qwen_worker import QwenWorker
    
    # 先创建一个临时专家来获取实际模型的 hidden_size
    # 这样可以确保 coordinator 使用正确的维度
    temp_expert = QwenLoRAExpert(
        expert_name="temp",
        base_model_name=base_model_name,
        use_modelscope=use_modelscope,
    )
    actual_model_dim = temp_expert.base_model.config.hidden_size
    print(f"[QwenFineTune] 检测到模型 hidden_size: {actual_model_dim}")
    
    # 清理临时专家（释放模型内存）
    del temp_expert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    coordinator = CentralCoordinator()
    for expert_name in config.expert_names:
        coordinator.register_expert(expert_name, actual_model_dim, LORA_RANK)
    workers = [
        QwenWorker(
            worker_id,
            coordinator,
            specialty,
            speed_factor,
            base_model_name,
            training_data_path=training_data_path,
            use_modelscope=use_modelscope,
        )
        for worker_id, specialty, speed_factor in config.worker_specs
    ]
    return coordinator, workers


def run_qwen_simulation(
    config: SimulationConfig = SimulationConfig(),
    base_model_name: str = "qwen/Qwen2-0.5B-Instruct",
    use_modelscope: bool = True,
    training_data_path: str | None = None,
) -> None:
    """
    运行使用真实 Qwen 模型的微调。

    Args:
        config: 配置
        base_model_name: 基础 Qwen 模型名称（ModelScope 格式：小写开头）
        use_modelscope: 是否使用 ModelScope 加载模型（默认 True）
        training_data_path: JSONL 数据集文件路径（如果提供，所有 worker 会使用此数据集）
    """
    import threading
    import time
    
    print("\n=== [使用真实 Qwen 模型进行微调] ===")
    if training_data_path:
        print(f"使用数据集: {training_data_path}")
    coordinator, workers = initialize_qwen_system(
        config, base_model_name, use_modelscope, training_data_path
    )
    
    # 运行训练阶段
    print("\n--- [Phase 1: Starting Distributed Training] ---")
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for worker in workers:
        thread = threading.Thread(target=worker.run_loop, args=(stop_event,))
        thread.start()
        threads.append(thread)

    time.sleep(config.training_duration_s)

    print("\n--- [Phase 2: Stopping Training] ---")
    stop_event.set()
    for thread in threads:
        thread.join(timeout=3.0)
    print("All workers stopped.")
    print("\n=== [微调完成] ===")

