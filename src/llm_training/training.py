"""
Training pipeline optimized for M3 MacBook Pro with 16GB RAM.
Uses PyTorch MPS backend for GPU acceleration on Apple Silicon.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import psutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer as HFTrainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset

from llm_training.config import Config, TrainingConfig
from llm_training.utils import setup_logging, get_device, estimate_memory

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor memory usage during training."""

    def __init__(self):
        self.process = psutil.Process()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        mem_info = self.process.memory_info()
        return {
            "ram_used_gb": mem_info.rss / (1024**3),
            "ram_percent": self.process.memory_percent(),
        }

    def log_memory(self):
        """Log current memory usage."""
        usage = self.get_memory_usage()
        logger.info(f"Memory usage: {usage['ram_used_gb']:.2f} GB ({usage['ram_percent']:.1f}%)")


class Trainer:
    """
    LLM Trainer optimized for M3 MacBook Pro.

    Features:
    - MPS (Metal Performance Shaders) acceleration for M3
    - Memory-efficient training with gradient accumulation
    - Mixed precision training (bfloat16)
    - Gradient checkpointing to reduce memory
    - Automatic checkpoint management
    """

    def __init__(
        self,
        config: Config,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            test_dataset: Test dataset
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.memory_monitor = MemoryMonitor()

        # Setup logging
        setup_logging(config.training.logging_dir)

        # Get device
        self.device = get_device(use_mps=config.training.use_mps)
        logger.info(f"Using device: {self.device}")

        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # Load model
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float32,
        )

        # Enable gradient checkpointing for memory efficiency
        if self.config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Move to device
        self.model = self.model.to(self.device)

        # Log memory usage
        self.memory_monitor.log_memory()

        # Estimate memory requirements
        estimate_memory(
            self.model,
            self.config.training.per_device_train_batch_size,
            self.config.model.max_length,
        )

    def _create_training_args(self) -> TrainingArguments:
        """Create Hugging Face training arguments."""
        return TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
            logging_dir=self.config.training.logging_dir,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            evaluation_strategy=self.config.training.evaluation_strategy,
            save_strategy=self.config.training.save_strategy,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            report_to=self.config.training.report_to,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            seed=self.config.training.seed,
            # MPS-specific settings
            use_cpu=False if self.device.type == "mps" else True,
            # Memory optimization
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        # Log initial memory
        self.memory_monitor.log_memory()

        # Create training arguments
        training_args = self._create_training_args()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked
        )

        # Create trainer
        self.trainer = HFTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Train
        try:
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.training.resume_from_checkpoint
            )

            # Save final model
            self.trainer.save_model(self.config.model.output_dir)
            self.tokenizer.save_pretrained(self.config.model.output_dir)

            # Log metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            logger.info("Training completed successfully!")
            self.memory_monitor.log_memory()

            return train_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            dataset: Dataset to evaluate on (uses eval_dataset if None)

        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")

        eval_dataset = dataset if dataset is not None else self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate(eval_dataset=eval_dataset)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint."""
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float32,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info("Checkpoint loaded successfully")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of generated texts
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        self.model.eval()

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

        return generated_texts
