"""
Training pipeline using MLX for Apple Silicon optimization.
Optimized for M3 MacBook Pro with 16GB RAM.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate

from llm_training.config import Config
from llm_training.dataset import MarkdownDataset
from llm_training.utils import setup_logging, estimate_memory_mlx

logger = logging.getLogger(__name__)


class Trainer:
    """
    LLM Trainer using MLX for Apple Silicon.

    Features:
    - Native M3 optimization with MLX
    - Efficient unified memory usage
    - Simple training loop
    - Automatic checkpoint management
    """

    def __init__(
        self,
        config: Config,
        train_dataset: MarkdownDataset,
        eval_dataset: Optional[MarkdownDataset] = None,
        test_dataset: Optional[MarkdownDataset] = None,
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

        # Setup logging
        setup_logging(config.training.logging_dir)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None

        # Load model
        self._load_model()

        # Setup optimizer
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        logger.info("MLX Trainer initialized successfully")

    def _load_model(self):
        """Load model and tokenizer using mlx-lm."""
        logger.info(f"Loading model: {self.config.model.model_name}")

        try:
            # Load model and tokenizer from mlx-community
            self.model, self.tokenizer = load(self.config.model.model_name)
            logger.info("Model loaded successfully")

            # Estimate memory
            estimate_memory_mlx(self.model, self.config.training.batch_size)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _setup_optimizer(self):
        """Setup optimizer."""
        # Use AdamW optimizer (standard for LLM training)
        self.optimizer = optim.AdamW(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Learning rate schedule (warmup + cosine decay)
        def lr_schedule(step):
            warmup = self.config.training.warmup_steps
            if step < warmup:
                return step / warmup
            else:
                # Cosine decay after warmup
                progress = (step - warmup) / (
                    self.config.training.num_train_epochs
                    * len(self.train_dataset)
                    // self.config.training.batch_size
                    - warmup
                )
                return 0.5 * (1 + mx.cos(mx.pi * progress))

        self.lr_schedule = lr_schedule

    def loss_fn(self, model, inputs, targets, attention_mask):
        """
        Compute loss for language modeling.

        Args:
            model: The model
            inputs: Input token IDs
            targets: Target token IDs
            attention_mask: Attention mask

        Returns:
            Loss value
        """
        # Forward pass
        logits = model(inputs)

        # Compute cross-entropy loss
        # Shift logits and targets for next-token prediction
        shift_logits = logits[..., :-1, :]
        shift_targets = targets[..., 1:]
        shift_mask = attention_mask[..., 1:]

        # Flatten for loss computation
        vocab_size = shift_logits.shape[-1]
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, vocab_size), shift_targets.reshape(-1), reduction="none"
        )

        # Apply mask and average
        loss = loss.reshape(shift_targets.shape)
        loss = (loss * shift_mask).sum() / shift_mask.sum()

        return loss

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: Batch of data

        Returns:
            Loss value
        """
        # Convert to MLX arrays
        inputs = mx.array(batch["input_ids"])
        targets = mx.array(batch["labels"])
        attention_mask = mx.array(batch["attention_mask"])

        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad_fn(self.model, inputs, targets, attention_mask)

        # Update model
        self.optimizer.update(self.model, grads)

        # Update learning rate
        lr = self.config.training.learning_rate * self.lr_schedule(self.global_step)
        self.optimizer.learning_rate = lr

        mx.eval(self.model.parameters(), self.optimizer.state)

        return loss.item()

    def train(self):
        """Train the model."""
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        best_eval_loss = float("inf")
        steps_per_epoch = len(self.train_dataset) // self.config.training.batch_size

        for epoch in range(self.config.training.num_train_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_train_epochs}")

            # Training loop
            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_dataset.batch_iterate(self.config.training.batch_size, shuffle=True),
                total=steps_per_epoch,
                desc=f"Training Epoch {epoch + 1}",
            )

            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                self.global_step += 1

                # Logging
                if self.global_step % self.config.training.logging_steps == 0:
                    avg_loss = epoch_loss / (self.global_step % steps_per_epoch + 1)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    logger.info(f"Step {self.global_step}, Loss: {avg_loss:.4f}")

                # Evaluation
                if self.global_step % self.config.training.eval_steps == 0 and self.eval_dataset:
                    eval_loss = self.evaluate()
                    logger.info(f"Evaluation loss: {eval_loss:.4f}")

                    # Save best model
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        self.save_checkpoint(os.path.join(self.config.training.output_dir, "best"))

                # Save checkpoint
                if self.global_step % self.config.training.save_steps == 0:
                    checkpoint_path = os.path.join(
                        self.config.training.output_dir, f"checkpoint-{self.global_step}"
                    )
                    self.save_checkpoint(checkpoint_path)

            # End of epoch
            avg_epoch_loss = epoch_loss / steps_per_epoch
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save final model
        self.save_checkpoint(self.config.model.output_dir)
        logger.info("Training completed successfully!")

        return {"final_loss": avg_epoch_loss}

    def evaluate(self, dataset: Optional[MarkdownDataset] = None) -> float:
        """
        Evaluate the model.

        Args:
            dataset: Dataset to evaluate on (uses eval_dataset if None)

        Returns:
            Average evaluation loss
        """
        eval_dataset = dataset if dataset is not None else self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        logger.info("Evaluating model...")

        total_loss = 0.0
        num_batches = 0

        for batch in eval_dataset.batch_iterate(self.config.training.batch_size, shuffle=False):
            inputs = mx.array(batch["input_ids"])
            targets = mx.array(batch["labels"])
            attention_mask = mx.array(batch["attention_mask"])

            loss = self.loss_fn(self.model, inputs, targets, attention_mask)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, checkpoint_path: str):
        """Save model checkpoint in safetensors format compatible with mlx_lm."""
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save model weights in safetensors format
        # Use MLX's native safetensors saving
        weights_path = os.path.join(checkpoint_path, "model.safetensors")
        mx.save_safetensors(weights_path, dict(mx.utils.tree_flatten(self.model.parameters())))

        # Save model config
        model_config_path = os.path.join(checkpoint_path, "config.json")
        if hasattr(self.model, "config"):
            import json

            with open(model_config_path, "w") as f:
                json.dump(self.model.config.__dict__, f, indent=2)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save additional training state
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        with open(training_state_path, "w") as f:
            json.dump(
                {
                    "model_name": self.config.model.model_name,
                    "max_length": self.config.model.max_length,
                    "global_step": self.global_step,
                    "epoch": self.current_epoch,
                },
                f,
            )

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint from safetensors format."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load model and tokenizer from safetensors
        self.model, self.tokenizer = load(checkpoint_path)

        # Load training state if available
        state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                checkpoint_config = json.load(f)
                self.global_step = checkpoint_config.get("global_step", 0)
                self.current_epoch = checkpoint_config.get("epoch", 0)

        logger.info("Checkpoint loaded successfully")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        # Use mlx-lm generate function
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_length,
            temp=temperature,
            top_p=top_p,
        )

        return response
