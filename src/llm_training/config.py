"""
Configuration management for LLM training with MLX.
Supports YAML config files with auto-detection for optimal defaults based on system RAM.
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
    max_length: int = 0  # 0 = auto-detect based on available RAM
    train_test_split: float = 0.9
    validation_split: float = 0.05
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = "mlx-community/gpt2"  # MLX-compatible model
    cache_dir: str = "models/cache"
    output_dir: str = "models/output"
    max_length: int = 0  # 0 = auto-detect based on available RAM

    # Popular MLX models by size:
    # - "mlx-community/distilgpt2" (82M params - good for 8-16GB RAM)
    # - "mlx-community/gpt2" (124M params - good for 16-32GB RAM)
    # - "mlx-community/gpt2-medium" (355M params - needs 24GB+ RAM)
    # - "mlx-community/gpt2-large" (774M params - needs 32GB+ RAM)
    # - "mlx-community/Llama-2-7b-mlx" (7B params - needs 48GB+ RAM)


@dataclass
class TrainingConfig:
    """Training configuration for MLX on Apple Silicon with auto-detection."""

    # Output
    output_dir: str = "checkpoints"
    logging_dir: str = "logs"

    # Training parameters (0 = auto-detect based on available RAM)
    num_train_epochs: int = 3
    batch_size: int = 0  # Auto-detect: 1 (8GB), 2 (16GB), 4 (32GB), 8 (64GB+)
    gradient_accumulation_steps: int = 0  # Auto-detect to maintain effective batch size of 16

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

    # Training (0 = auto-detect)
    num_train_epochs: int = 3
    batch_size: int = 0
    gradient_accumulation_steps: int = 0
    learning_rate: float = 1e-4

    # Output
    output_dir: str = "models/finetuned"
    logging_dir: str = "logs/finetuning"


@dataclass
class Config:
    """Main configuration class with auto-detection for system-specific settings."""

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
        """
        Get default configuration with auto-detected settings for current system.

        Returns optimized defaults based on detected RAM:
        - 8GB: batch_size=1, max_length=256
        - 16GB: batch_size=2, max_length=512
        - 32GB: batch_size=4, max_length=1024
        - 64GB+: batch_size=8, max_length=2048
        """
        config = cls()
        config._apply_auto_detection()
        return config

    def _apply_auto_detection(self) -> None:
        """Apply auto-detection for 0-valued parameters."""
        try:
            from llm_training.utils import get_system_capabilities

            caps = get_system_capabilities()
            rec = caps["recommended"]

            # Apply auto-detected values only if set to 0
            if self.data.max_length == 0:
                self.data.max_length = rec["max_length"]
            if self.model.max_length == 0:
                self.model.max_length = rec["max_length"]
            if self.training.batch_size == 0:
                self.training.batch_size = rec["batch_size"]
            if self.training.gradient_accumulation_steps == 0:
                self.training.gradient_accumulation_steps = rec["gradient_accumulation_steps"]
            if self.finetuning.batch_size == 0:
                self.finetuning.batch_size = rec["batch_size"]
            if self.finetuning.gradient_accumulation_steps == 0:
                self.finetuning.gradient_accumulation_steps = rec["gradient_accumulation_steps"]

        except ImportError:
            # Fallback to conservative defaults if utils not available
            if self.data.max_length == 0:
                self.data.max_length = 512
            if self.model.max_length == 0:
                self.model.max_length = 512
            if self.training.batch_size == 0:
                self.training.batch_size = 2
            if self.training.gradient_accumulation_steps == 0:
                self.training.gradient_accumulation_steps = 8
            if self.finetuning.batch_size == 0:
                self.finetuning.batch_size = 2
            if self.finetuning.gradient_accumulation_steps == 0:
                self.finetuning.gradient_accumulation_steps = 8

    def validate(self) -> None:
        """Validate configuration parameters with system-specific checks."""
        # Apply auto-detection first
        self._apply_auto_detection()

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

        # Validate sequence lengths based on detected system RAM
        try:
            from llm_training.utils import get_system_capabilities

            caps = get_system_capabilities()
            total_ram_gb = caps["total_ram_gb"]
            rec_max_length = caps["recommended"]["max_length"]

            # Dynamic validation based on actual system RAM
            max_safe_length = min(2048, rec_max_length * 2)  # Allow 2x recommended as maximum

            if self.data.max_length > max_safe_length:
                raise ValueError(
                    f"max_length ({self.data.max_length}) exceeds safe limit ({max_safe_length}) "
                    f"for your system ({total_ram_gb}GB RAM). Reduce to ≤{max_safe_length} tokens."
                )

            if self.data.max_length > rec_max_length:
                print(
                    f"⚠️  max_length ({self.data.max_length}) exceeds recommended value "
                    f"({rec_max_length}) for {total_ram_gb}GB RAM. Training may be unstable."
                )

            if self.model.max_length > max_safe_length:
                raise ValueError(
                    f"model.max_length ({self.model.max_length}) exceeds safe limit "
                    f"({max_safe_length}) for {total_ram_gb}GB RAM."
                )

        except ImportError:
            # Fallback to hardcoded limits if utils not available
            if self.data.max_length > 2048 or self.model.max_length > 2048:
                raise ValueError("max_length must be ≤2048 tokens")

        print("✓ Configuration validated successfully")


def create_default_config(output_path: str = "configs/default.yaml") -> Config:
    """
    Create and save default configuration file with auto-detected settings.

    The configuration will be optimized for the current system's RAM.
    """
    config = Config.get_default()
    config.to_yaml(output_path)

    # Print what was detected
    try:
        from llm_training.utils import get_system_capabilities

        caps = get_system_capabilities()
        print(f"Auto-detected settings for {caps['total_ram_gb']}GB RAM system:")
        print(f"  batch_size: {config.training.batch_size}")
        print(f"  max_length: {config.data.max_length}")
        print(f"  gradient_accumulation_steps: {config.training.gradient_accumulation_steps}")
    except ImportError:
        pass

    print(f"\nDefault configuration saved to {output_path}")
    return config


if __name__ == "__main__":
    # Example usage
    config = create_default_config()
    config.validate()
