"""
Inference agent that routes prompts to relevant experts and stitches outputs.
"""

from __future__ import annotations

import re
from typing import List

from .coordinator import CentralCoordinator


class InferenceAgent:
    """
    Performs prompt-aware routing and collaborative inference with experts.
    """

    def __init__(self, coordinator: CentralCoordinator) -> None:
        self.coordinator = coordinator
        print("\nInference Agent Initialized. Ready to route tasks.")

    def route_task(self, prompt: str) -> List[str]:
        prompt_lower = prompt.lower()
        required_experts: List[str] = []

        if re.search(r"python|pandas|numpy|def ", prompt_lower):
            required_experts.append("python_expert")
        if re.search(r"sql|database|query|select", prompt_lower):
            required_experts.append("sql_expert")
        if re.search(r"docstring|explain|comment|how to", prompt_lower):
            required_experts.append("docs_expert")
        if not required_experts:
            required_experts.append("python_expert")

        print(f"[Agent] Routing prompt to: {required_experts}")
        return required_experts

    def generate_code(self, prompt: str) -> str:
        print(f"\n[Agent] Received new task: '{prompt}'")
        required_experts = self.route_task(prompt)
        final_code_parts = [f"# Task: {prompt}\n", f"# Agent decided to use: {', '.join(required_experts)}\n" + ("-" * 30)]

        for expert_name in required_experts:
            try:
                with self.coordinator.lock:
                    expert = self.coordinator.experts[expert_name]
                prompt_part = f"Generate {expert_name} part for: {prompt}"
                final_code_parts.append(expert.simulate_inference(prompt_part))
            except Exception as exc:
                final_code_parts.append(f"\n--- [Error calling {expert_name}: {exc}] ---\n")

        return "\n".join(final_code_parts)

