"""
LLM Training Framework
Optimized for M3 MacBook Pro with 16GB RAM
"""

__version__ = "0.1.0"

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
