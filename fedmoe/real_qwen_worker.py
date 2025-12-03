from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random
import torch

from .peft_coordinator import PeftCoordinator
from .qwen_lora_utils import (
    QwenLoraCfg,
    load_qwen_lora,
    get_adapter_state_dict,
    state_dict_delta,
    load_adapter_state_inplace,
    format_supervised_prompt,
)


@dataclass
class RealWorkerCfg:
    device_id: Optional[int] = 0
    seed: int = 42


class RealQwenWorker:
    """
    Real Qwen LoRA worker that pulls current adapter state, performs local SFT-style
    updates for a small number of steps on its domain data, and pushes PEFT adapter deltas.
    """

    def __init__(
        self,
        worker_id: str,
        coordinator: PeftCoordinator,
        specialty: str,
        qwen_cfg: QwenLoraCfg,
        worker_cfg: Optional[RealWorkerCfg] = None,
    ) -> None:
        self.worker_id = worker_id
        self.coordinator = coordinator
        self.specialty = specialty
        self.qwen_cfg = qwen_cfg
        self.worker_cfg = worker_cfg or RealWorkerCfg()

        device = f"cuda:{self.worker_cfg.device_id}" if torch.cuda.is_available() else "cpu"
        self.model, self.tok, self.lora_conf = load_qwen_lora(self.qwen_cfg, device=device)
        self.device = device
        torch.manual_seed(self.worker_cfg.seed + (self.worker_cfg.device_id or 0))
        print(f"[RealQwenWorker] {self.worker_id} init on {self.device} (expert={self.specialty})")

    def _batch_from_items(self, items: List[Dict]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        for it in items:
            prompt = it.get("prompt", "")
            ref = it.get("reference_code", "")
            texts.append(format_supervised_prompt(prompt, ref))
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.qwen_cfg.max_seq_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # For SFT, labels are the same as input_ids
        enc["labels"] = enc["input_ids"].clone()
        return enc

    def local_train(self, domain_items: List[Dict]) -> Tuple[int, Dict]:
        """Return (worker_version, delta_adapter_state)."""
        # Pull latest adapter state & load
        current_state, version = self.coordinator.get_adapter_state(self.specialty)
        if current_state:
            load_adapter_state_inplace(self.model, current_state, device=self.device)

        before = get_adapter_state_dict(self.model)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.qwen_cfg.lr)

        self.model.train()
        for _ in range(self.qwen_cfg.local_steps):
            batch = random.sample(domain_items, min(self.qwen_cfg.batch_size, len(domain_items))) if domain_items else []
            if not batch:
                break
            enc = self._batch_from_items(batch)
            optim.zero_grad(set_to_none=True)
            out = self.model(**enc)
            loss = out.loss
            loss.backward()
            optim.step()

        after = get_adapter_state_dict(self.model)
        delta = state_dict_delta(after, before)
        return version, delta

    def push_update(self, delta: Dict, worker_version: int) -> None:
        self.coordinator.push_adapter_delta(self.specialty, delta, worker_version)

