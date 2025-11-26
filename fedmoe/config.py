"""
Core hyperparameters shared across the FedMoE runtime.
"""

# Rank of each expert's LoRA adapter
LORA_RANK = 16

# Learning rate applied during aggregation on the coordinator
SERVER_LR = 1.0

# Weight for staleness-aware updates
STALENESS_DECAY = 0.1
