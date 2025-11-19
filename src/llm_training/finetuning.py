"""
Fine-tuning module using LoRA with MLX.
Memory-efficient fine-tuning optimized for Apple Silicon.
"""

import logging
from typing import Optional
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.lora import LoRALinear
from tqdm import tqdm

from llm_training.config import Config, FineTuningConfig
from llm_training.dataset import MarkdownDataset
from llm_training.utils import setup_logging

logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """
    LoRA Fine-Tuner using MLX.

    LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method:
    - Reduces memory usage by only training adapter layers
    - Faster training than full fine-tuning
    - Can be merged back into base model

    Perfect for M3 MacBook Pro with 16GB RAM!
    """

    def __init__(
        self,
        base_model_path: str,
        config: Config,
        train_dataset: MarkdownDataset,
        eval_dataset: Optional[MarkdownDataset] = None,
    ):
        """
        Initialize LoRA fine-tuner.

        Args:
            base_model_path: Path to base model
            config: Configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        self.base_model_path = base_model_path
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Setup logging
        setup_logging(config.finetuning.logging_dir)

        logger.info(f"Using base model: {base_model_path}")

        # Load model and tokenizer
        self._load_model()

        # Apply LoRA
        self._apply_lora()

        # Setup optimizer
        self._setup_optimizer()

        logger.info("LoRA fine-tuner initialized successfully")

    def _load_model(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading base model from {self.base_model_path}")
        self.model, self.tokenizer = load(self.base_model_path)

    def _apply_lora(self):
        """Apply LoRA to model layers."""
        logger.info("Applying LoRA adapters...")

        # Freeze base model parameters
        self.model.freeze()

        # Apply LoRA to specified layers
        # MLX-LM typically applies LoRA to attention layers
        for layer in self.model.layers[-self.config.finetuning.lora_layers :]:
            # Apply LoRA to attention layers
            if hasattr(layer, "self_attn"):
                self_attn = layer.self_attn
                if hasattr(self_attn, "q_proj"):
                    self_attn.q_proj = LoRALinear.from_linear(
                        self_attn.q_proj,
                        r=self.config.finetuning.lora_rank,
                        scale=self.config.finetuning.lora_alpha / self.config.finetuning.lora_rank,
                        dropout=self.config.finetuning.lora_dropout,
                    )
                if hasattr(self_attn, "v_proj"):
                    self_attn.v_proj = LoRALinear.from_linear(
                        self_attn.v_proj,
                        r=self.config.finetuning.lora_rank,
                        scale=self.config.finetuning.lora_alpha / self.config.finetuning.lora_rank,
                        dropout=self.config.finetuning.lora_dropout,
                    )

        # Count parameters
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print information about trainable parameters."""
        total_params = sum(p.size for p in self.model.parameters().values())
        trainable_params = sum(p.size for p in self.model.trainable_parameters().values())

        percentage = 100 * trainable_params / total_params if total_params > 0 else 0

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {total_params:,} || "
            f"Trainable%: {percentage:.2f}%"
        )

    def _setup_optimizer(self):
        """Setup optimizer for fine-tuning."""
        self.optimizer = optim.AdamW(
            learning_rate=self.config.finetuning.learning_rate,
        )

    def loss_fn(self, model, inputs, targets, attention_mask):
        """Compute loss."""
        logits = model(inputs)

        shift_logits = logits[..., :-1, :]
        shift_targets = targets[..., 1:]
        shift_mask = attention_mask[..., 1:]

        vocab_size = shift_logits.shape[-1]
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, vocab_size), shift_targets.reshape(-1), reduction="none"
        )

        loss = loss.reshape(shift_targets.shape)
        loss = (loss * shift_mask).sum() / shift_mask.sum()

        return loss

    def finetune(self):
        """Fine-tune the model using LoRA."""
        logger.info("Starting LoRA fine-tuning...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        steps_per_epoch = len(self.train_dataset) // self.config.finetuning.batch_size
        total_steps = steps_per_epoch * self.config.finetuning.num_train_epochs

        global_step = 0

        for epoch in range(self.config.finetuning.num_train_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.finetuning.num_train_epochs}")

            epoch_loss = 0.0
            progress_bar = tqdm(
                self.train_dataset.batch_iterate(self.config.finetuning.batch_size, shuffle=True),
                total=steps_per_epoch,
                desc=f"Fine-tuning Epoch {epoch + 1}",
            )

            for batch in progress_bar:
                # Convert to MLX arrays
                inputs = mx.array(batch["input_ids"])
                targets = mx.array(batch["labels"])
                attention_mask = mx.array(batch["attention_mask"])

                # Compute loss and gradients
                loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
                loss, grads = loss_and_grad_fn(self.model, inputs, targets, attention_mask)

                # Update only LoRA parameters
                self.optimizer.update(self.model, grads)
                mx.eval(self.model.parameters(), self.optimizer.state)

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_epoch_loss = epoch_loss / steps_per_epoch
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save fine-tuned model
        self.save_model(self.config.finetuning.output_dir)

        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Model saved to {self.config.finetuning.output_dir}")

        return {"final_loss": avg_epoch_loss}

    def save_model(self, output_path: str):
        """
        Save fine-tuned model.

        Args:
            output_path: Path to save model
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        weights_path = Path(output_path) / "lora_weights.npz"
        lora_weights = {k: v for k, v in self.model.parameters().items() if "lora" in k}
        mx.savez(str(weights_path), **lora_weights)

        # Save config
        config_path = Path(output_path) / "lora_config.json"
        with open(config_path, "w") as f:
            json.dump(
                {
                    "base_model": self.base_model_path,
                    "lora_rank": self.config.finetuning.lora_rank,
                    "lora_alpha": self.config.finetuning.lora_alpha,
                    "lora_layers": self.config.finetuning.lora_layers,
                },
                f,
                indent=2,
            )

        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"LoRA model saved to {output_path}")

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using fine-tuned model.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated text
        """
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_length,
            temp=temperature,
            top_p=top_p,
        )

        return response


if __name__ == "__main__":
    print("LoRA Fine-Tuning Module for MLX")
    print("See examples for usage")
