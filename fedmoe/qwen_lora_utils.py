from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

try:
    import bitsandbytes as bnb  # noqa: F401
    HAS_BNB = True
except Exception:
    HAS_BNB = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


@dataclass
class QwenLoraCfg:
    base_model: str = "Qwen/Qwen2-0.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True
    max_seq_len: int = 2048
    lr: float = 2e-4
    local_steps: int = 5
    batch_size: int = 1


def load_qwen_lora(cfg: QwenLoraCfg, device: str) -> Tuple[torch.nn.Module, AutoTokenizer, LoraConfig]:
    quantization_config = None
    kwargs = {}
    if cfg.use_4bit and HAS_BNB:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=None,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return model, tokenizer, lora_config


def get_adapter_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # Only LoRA adapter parameters (requires PEFT wrappers)
    sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if "lora_" in k or "lora_A" in k or "lora_B" in k or "lora_down" in k or "lora_up" in k:
            out[k] = v.detach().to("cpu").clone()
    return out


def state_dict_delta(after: Dict[str, torch.Tensor], before: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = set(after.keys()) & set(before.keys())
    return {k: (after[k] - before[k]) for k in keys}


def apply_delta_inplace(state: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], alpha: float = 1.0) -> None:
    for k, dv in delta.items():
        if k in state:
            state[k] = state[k] + alpha * dv
        else:
            state[k] = alpha * dv.clone()


def load_adapter_state_inplace(model: torch.nn.Module, adapter_state: Dict[str, torch.Tensor], device: Optional[str] = None) -> None:
    # Move tensors and load matching keys
    model_sd = model.state_dict()
    to_load: Dict[str, torch.Tensor] = {}
    for k, v in adapter_state.items():
        if k in model_sd:
            to_load[k] = v.to(model_sd[k].dtype)
    model_sd.update(to_load)
    model.load_state_dict(model_sd, strict=False)


def format_supervised_prompt(prompt: str, reference: str) -> str:
    # Minimal SFT-style formatting
    return (
        f"Problem:\n{prompt.strip()}\n\n"
        f"Answer:\n{reference.strip()}\n"
    )

