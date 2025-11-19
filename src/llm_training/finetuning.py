"""
Fine-tuning module using LoRA (Low-Rank Adaptation).
Memory-efficient fine-tuning for resource-constrained environments.
"""

import logging
from typing import Optional, Dict
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from torch.utils.data import Dataset

from llm_training.config import Config, FineTuningConfig
from llm_training.utils import setup_logging, get_device

logger = logging.getLogger(__name__)


class LoRAFineTuner:
    """
    LoRA (Low-Rank Adaptation) Fine-Tuner.

    LoRA is a parameter-efficient fine-tuning method that:
    - Reduces memory usage by only training a small subset of parameters
    - Allows fine-tuning large models on limited hardware
    - Maintains model quality while being much faster

    Perfect for M3 MacBook Pro with 16GB RAM!
    """

    def __init__(
        self,
        base_model_path: str,
        config: Config,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Initialize LoRA fine-tuner.

        Args:
            base_model_path: Path to base model (can be pretrained or your trained model)
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

        # Get device
        self.device = get_device(use_mps=config.training.use_mps)
        logger.info(f"Using device: {self.device}")

        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.trainer = None

        # Load and prepare model
        self._load_and_prepare_model()

    def _load_and_prepare_model(self):
        """Load base model and apply LoRA configuration."""
        logger.info(f"Loading base model from {self.base_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.finetuning.lora_r,
            lora_alpha=self.config.finetuning.lora_alpha,
            target_modules=self.config.finetuning.target_modules,
            lora_dropout=self.config.finetuning.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print information about trainable parameters."""
        trainable_params = 0
        all_param = 0

        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        percentage = 100 * trainable_params / all_param

        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {percentage:.2f}%"
        )

    def _create_training_args(self) -> TrainingArguments:
        """Create training arguments for fine-tuning."""
        return TrainingArguments(
            output_dir=self.config.finetuning.output_dir,
            num_train_epochs=self.config.finetuning.num_train_epochs,
            per_device_train_batch_size=self.config.finetuning.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.finetuning.gradient_accumulation_steps,
            learning_rate=self.config.finetuning.learning_rate,
            bf16=True,
            logging_dir=self.config.finetuning.logging_dir,
            logging_steps=100,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            eval_steps=500 if self.eval_dataset else None,
            save_strategy="steps",
            load_best_model_at_end=True if self.eval_dataset else False,
            report_to="tensorboard",
            warmup_steps=100,
            weight_decay=0.01,
            # MPS optimization
            dataloader_num_workers=0,
        )

    def finetune(self):
        """Fine-tune the model using LoRA."""
        logger.info("Starting LoRA fine-tuning...")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Evaluation samples: {len(self.eval_dataset)}")

        # Create training arguments
        training_args = self._create_training_args()

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )

        # Train
        try:
            train_result = self.trainer.train()

            # Save fine-tuned model
            self.model.save_pretrained(self.config.finetuning.output_dir)
            self.tokenizer.save_pretrained(self.config.finetuning.output_dir)

            # Log metrics
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            logger.info("Fine-tuning completed successfully!")
            logger.info(f"Model saved to {self.config.finetuning.output_dir}")

            return train_result

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise

    def merge_and_save(self, output_path: str):
        """
        Merge LoRA weights with base model and save.

        This creates a standalone model that doesn't require LoRA at inference.

        Args:
            output_path: Path to save merged model
        """
        logger.info("Merging LoRA weights with base model...")

        # Merge weights
        merged_model = self.model.merge_and_unload()

        # Save merged model
        Path(output_path).mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Merged model saved to {output_path}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate fine-tuned model."""
        if self.trainer is None:
            raise ValueError("Model not fine-tuned yet. Call finetune() first.")

        if self.eval_dataset is None:
            raise ValueError("No evaluation dataset provided")

        logger.info("Evaluating fine-tuned model...")
        metrics = self.trainer.evaluate()

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

        return metrics

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
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def load_finetuned_model(
    model_path: str,
    device: Optional[str] = None,
) -> tuple[PeftModel, AutoTokenizer]:
    """
    Load a fine-tuned LoRA model.

    Args:
        model_path: Path to fine-tuned model
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading fine-tuned model from {model_path}")

    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    logger.info("Model loaded successfully")
    return model, tokenizer


if __name__ == "__main__":
    # Example usage
    print("LoRA Fine-Tuning Module")
    print("See examples/finetune_example.py for usage examples")
