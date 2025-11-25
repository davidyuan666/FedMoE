"""
真实的 Qwen 模型微调实现，使用 LoRA 进行参数更新。

这个模块提供了与模拟系统兼容的真实模型微调功能，可以替换模拟的训练逻辑。
"""

from __future__ import annotations

import os
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

from .config import LORA_RANK, MODEL_DIM


class QwenLoRAExpert:
    """
    基于真实 Qwen 模型的 LoRA 专家实现。
    每个专家对应一个专门的 LoRA 适配器，可以独立训练和更新。
    """

    def __init__(
        self,
        expert_name: str,
        base_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        lora_rank: int = LORA_RANK,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """
        初始化 Qwen LoRA 专家。

        Args:
            expert_name: 专家名称（如 'python_expert', 'sql_expert'）
            base_model_name: 基础 Qwen 模型名称
            lora_rank: LoRA 的秩
            lora_alpha: LoRA 的 alpha 参数
            lora_dropout: LoRA dropout 率
            device: 运行设备
        """
        self.expert_name = expert_name
        self.base_model_name = base_model_name
        self.device = device
        self.lora_rank = lora_rank
        self.version = 0

        print(f"[QwenExpert] 初始化专家: {expert_name} (设备: {device})")

        # 加载基础模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
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

        Returns:
            (lora_A, lora_B) 的 numpy 数组元组
        """
        lora_A_list = []
        lora_B_list = []

        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # 获取 LoRA 权重
                A_weight = module.lora_A["default"].weight.data.cpu().numpy()
                B_weight = module.lora_B["default"].weight.data.cpu().numpy()

                # 对于多个 LoRA 层，我们需要聚合或选择第一个
                # 这里简化处理，取第一个找到的层
                if len(lora_A_list) == 0:
                    lora_A_list.append(A_weight)
                    lora_B_list.append(B_weight)

        if len(lora_A_list) == 0:
            # 如果没有找到 LoRA 层，返回零矩阵
            model_dim = self.base_model.config.hidden_size
            return (
                np.zeros((model_dim, self.lora_rank)),
                np.zeros((self.lora_rank, model_dim)),
            )

        # 聚合所有 LoRA 层的权重（简单平均）
        lora_A = np.mean(lora_A_list, axis=0)
        lora_B = np.mean(lora_B_list, axis=0)

        return lora_A, lora_B

    def update_lora_weights(
        self, lora_A: np.ndarray, lora_B: np.ndarray, learning_rate: float = 1.0
    ) -> None:
        """
        更新 LoRA 权重。

        Args:
            lora_A: 新的 lora_A 权重
            lora_B: 新的 lora_B 权重
            learning_rate: 学习率（用于缩放更新）
        """
        lora_A_tensor = torch.from_numpy(lora_A).to(self.device)
        lora_B_tensor = torch.from_numpy(lora_B).to(self.device)

        # 更新所有 LoRA 层
        for name, module in self.model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # 获取当前权重
                current_A = module.lora_A["default"].weight.data
                current_B = module.lora_B["default"].weight.data

                # 计算更新（确保形状匹配）
                if current_A.shape == lora_A_tensor.shape:
                    module.lora_A["default"].weight.data = (
                        current_A + learning_rate * lora_A_tensor
                    )
                if current_B.shape == lora_B_tensor.shape:
                    module.lora_B["default"].weight.data = (
                        current_B + learning_rate * lora_B_tensor
                    )

        self.version += 1
        print(f"[QwenExpert] {self.expert_name} 权重已更新 (版本: {self.version})")

    def fine_tune_step(
        self,
        training_data: List[str],
        batch_size: int = 4,
        num_epochs: int = 1,
        learning_rate: float = 5e-5,
        max_length: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行一个微调步骤，返回权重更新。

        Args:
            training_data: 训练数据（字符串列表）
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

        # 编码数据
        encoded = self.tokenizer(
            training_data,
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
    base_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    lora_rank: int = LORA_RANK,
) -> QwenLoRAExpert:
    """
    创建 Qwen LoRA 专家的便捷函数。

    Args:
        expert_name: 专家名称
        base_model_name: 基础模型名称
        lora_rank: LoRA 秩

    Returns:
        QwenLoRAExpert 实例
    """
    return QwenLoRAExpert(
        expert_name=expert_name,
        base_model_name=base_model_name,
        lora_rank=lora_rank,
    )

