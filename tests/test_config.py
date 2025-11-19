"""Tests for configuration module."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_training.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    FineTuningConfig,
    create_default_config,
)


class TestDataConfig:
    """Test DataConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()
        assert config.raw_data_path == "data/raw"
        assert config.max_length == 512
        assert config.train_test_split == 0.9
        assert config.seed == 42
        assert ".md" in config.file_extensions
        assert ".mdx" in config.file_extensions

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DataConfig(
            raw_data_path="custom/path",
            max_length=1024,
            train_test_split=0.8,
        )
        assert config.raw_data_path == "custom/path"
        assert config.max_length == 1024
        assert config.train_test_split == 0.8


class TestModelConfig:
    """Test ModelConfig class."""

    def test_default_model(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.model_name == "gpt2"
        assert config.max_length == 512

    def test_custom_model(self):
        """Test custom model configuration."""
        config = ModelConfig(model_name="distilgpt2", max_length=256)
        assert config.model_name == "distilgpt2"
        assert config.max_length == 256


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 2
        assert config.gradient_accumulation_steps == 8
        assert config.bf16 is True
        assert config.use_mps is True
        assert config.gradient_checkpointing is True

    def test_memory_optimization_settings(self):
        """Test memory optimization settings."""
        config = TrainingConfig()
        assert config.gradient_checkpointing is True
        assert config.bf16 is True
        assert config.dataloader_num_workers == 0  # Important for MPS


class TestFineTuningConfig:
    """Test FineTuningConfig class."""

    def test_lora_defaults(self):
        """Test LoRA default configuration."""
        config = FineTuningConfig()
        assert config.lora_r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestConfig:
    """Test main Config class."""

    def test_default_config(self):
        """Test creating default configuration."""
        config = Config.get_default()
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.finetuning, FineTuningConfig)

    def test_yaml_save_load(self):
        """Test saving and loading configuration to/from YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "test_config.yaml")

            # Create and save config
            original_config = Config.get_default()
            original_config.model.model_name = "distilgpt2"
            original_config.training.num_train_epochs = 5
            original_config.to_yaml(yaml_path)

            # Load config
            loaded_config = Config.from_yaml(yaml_path)

            # Verify
            assert loaded_config.model.model_name == "distilgpt2"
            assert loaded_config.training.num_train_epochs == 5

    def test_validation(self):
        """Test configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config.get_default()
            # Update paths to temp directory
            config.data.raw_data_path = os.path.join(tmpdir, "data/raw")
            config.data.processed_data_path = os.path.join(tmpdir, "data/processed")
            config.model.cache_dir = os.path.join(tmpdir, "models/cache")
            config.model.output_dir = os.path.join(tmpdir, "models/output")
            config.training.output_dir = os.path.join(tmpdir, "checkpoints")
            config.training.logging_dir = os.path.join(tmpdir, "logs")

            # Should not raise any errors
            config.validate()

            # Check that directories were created
            assert os.path.exists(config.data.raw_data_path)
            assert os.path.exists(config.data.processed_data_path)

    def test_invalid_validation(self):
        """Test validation with invalid values."""
        config = Config.get_default()

        # Invalid batch size
        config.training.per_device_train_batch_size = 0
        with pytest.raises(AssertionError):
            config.validate()

        # Reset and test invalid split
        config = Config.get_default()
        config.data.train_test_split = 1.5
        with pytest.raises(AssertionError):
            config.validate()


class TestCreateDefaultConfig:
    """Test create_default_config function."""

    def test_create_default_config(self):
        """Test creating and saving default config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = os.path.join(tmpdir, "config.yaml")
            config = create_default_config(yaml_path)

            # Check file was created
            assert os.path.exists(yaml_path)

            # Check config is valid
            assert isinstance(config, Config)
            assert config.model.model_name == "gpt2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
