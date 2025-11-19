"""
LLM Training Framework using MLX
Optimized for Apple Silicon (M1/M2/M3/M4) with auto-detection
"""

import os

# Suppress transformers warning about missing PyTorch/TensorFlow
# We only use transformers for tokenizers, not models (we use MLX for models)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

__version__ = "0.2.0"

from llm_training.config import Config
from llm_training.dataset import MarkdownDataset, prepare_dataset
from llm_training.training import Trainer
from llm_training.evaluation import Evaluator
from llm_training.finetuning import LoRAFineTuner

__all__ = [
    "Config",
    "MarkdownDataset",
    "prepare_dataset",
    "Trainer",
    "Evaluator",
    "LoRAFineTuner",
]
