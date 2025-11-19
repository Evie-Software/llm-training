"""
Configuration management for LLM training with MLX.
Supports YAML config files with defaults optimized for M3 MacBook Pro (16GB RAM).
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class DataConfig:
    """Dataset configuration."""

    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    file_extensions: List[str] = field(default_factory=lambda: [".md", ".mdx"])
    max_length: int = 512  # Maximum sequence length (recommend ≤2048 for 16GB RAM)
    train_test_split: float = 0.9
    validation_split: float = 0.05
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "mlx-community/gpt2"  # MLX-compatible model
    cache_dir: str = "models/cache"
    output_dir: str = "models/output"
    max_length: int = 512

    # Popular MLX models:
    # - "mlx-community/gpt2" (124M params)
    # - "mlx-community/gpt2-medium" (355M params)
    # - "mlx-community/Llama-2-7b-mlx" (7B params - requires more RAM)
    # - "mlx-community/Mistral-7B-v0.1-mlx"


@dataclass
class TrainingConfig:
    """Training configuration optimized for M3 with 16GB RAM using MLX."""

    # Output
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"

    # Training parameters
    num_train_epochs: int = 3
    batch_size: int = 4  # MLX can handle larger batches efficiently
    gradient_accumulation_steps: int = 4  # Effective batch size = 16

    # Optimization
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Evaluation and saving
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 2  # Keep only 2 checkpoints to save space
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"

    # Logging
    logging_steps: int = 100
    report_to: str = "none"  # "wandb" or "none"

    # Other
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None

    # MLX-specific settings
    grad_checkpoint: bool = True  # Use gradient checkpointing for memory efficiency


@dataclass
class FineTuningConfig:
    """Fine-tuning configuration using LoRA with MLX."""

    # LoRA parameters
    lora_layers: int = 16  # Number of layers to apply LoRA to
    lora_rank: int = 8  # Rank of LoRA matrices
    lora_alpha: int = 16  # LoRA scaling factor
    lora_dropout: float = 0.0  # Dropout for LoRA layers

    # Training
    num_train_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "models/finetuned"
    logging_dir: str = "logs/finetuning"


@dataclass
class Config:
    """Main configuration class."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            finetuning=FineTuningConfig(**config_dict.get("finetuning", {})),
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "finetuning": asdict(self.finetuning),
        }

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def get_default(cls) -> "Config":
        """Get default configuration optimized for M3 MacBook Pro."""
        return cls()

    def validate(self) -> None:
        """Validate configuration parameters."""
        # Check paths exist or can be created
        for path in [
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.model.cache_dir,
            self.model.output_dir,
            self.training.output_dir,
            self.training.logging_dir,
        ]:
            Path(path).mkdir(parents=True, exist_ok=True)

        # Validate batch sizes
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.training.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")

        # Validate splits
        if not (0 < self.data.train_test_split < 1):
            raise ValueError("Train/test split must be between 0 and 1")
        if not (0 < self.data.validation_split < 1):
            raise ValueError("Validation split must be between 0 and 1")

        # Validate sequence lengths (critical for 16GB M3)
        if self.data.max_length > 2048:
            raise ValueError(
                f"max_length ({self.data.max_length}) exceeds 2048 tokens. "
                "For 16GB M3 MacBook, sequences >2048 tokens can cause OOM errors. "
                "Reduce max_length to ≤2048 for stable training."
            )
        if self.data.max_length > 1024:
            print(
                f"⚠ WARNING: max_length is {self.data.max_length}. "
                "For 16GB RAM, consider using ≤1024 for better stability."
            )
        if self.model.max_length > 2048:
            raise ValueError(
                f"model.max_length ({self.model.max_length}) exceeds 2048 tokens. "
                "For 16GB M3 MacBook, reduce to ≤2048."
            )

        print("✓ Configuration validated successfully")


def create_default_config(output_path: str = "configs/default.yaml") -> Config:
    """Create and save default configuration file."""
    config = Config.get_default()
    config.to_yaml(output_path)
    print(f"Default configuration saved to {output_path}")
    return config


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    config.validate()
