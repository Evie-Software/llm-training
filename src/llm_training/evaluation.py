"""
Evaluation and testing module for trained LLMs.
Provides comprehensive metrics and testing capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llm_training.config import Config
from llm_training.utils import get_device

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for LLM models.

    Provides:
    - Perplexity calculation
    - Loss evaluation
    - Text generation quality assessment
    - Custom metric computation
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            config: Configuration object
            device: Device to use (auto-detected if None)
        """
        self.model_path = model_path
        self.config = config or Config.get_default()
        self.device = device or get_device(use_mps=self.config.training.use_mps)

        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float32,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded and ready for evaluation")

    def calculate_perplexity(
        self,
        dataset: Dataset,
        batch_size: int = 8,
    ) -> float:
        """
        Calculate perplexity on a dataset.

        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for evaluation

        Returns:
            Perplexity score (lower is better)
        """
        logger.info("Calculating perplexity...")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                # Calculate loss only on non-padding tokens
                loss = outputs.loss
                n_tokens = attention_mask.sum().item()

                total_loss += loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def evaluate_loss(
        self,
        dataset: Dataset,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate average loss on dataset.

        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size

        Returns:
            Dictionary with loss metrics
        """
        logger.info("Calculating loss metrics...")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating loss"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches

        return {
            "avg_loss": avg_loss,
            "perplexity": np.exp(avg_loss),
        }

    def generate_samples(
        self,
        prompts: List[str],
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
    ) -> List[Dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences per prompt

        Returns:
            List of dictionaries with prompt and generated text
        """
        logger.info(f"Generating samples for {len(prompts)} prompts...")

        results = []

        for prompt in tqdm(prompts, desc="Generating"):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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

            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]

            results.append(
                {
                    "prompt": prompt,
                    "generated": generated_texts,
                }
            )

        return results

    def comprehensive_evaluation(
        self,
        dataset: Dataset,
        test_prompts: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Run comprehensive evaluation.

        Args:
            dataset: Test dataset
            test_prompts: Optional prompts for generation testing
            output_path: Path to save results

        Returns:
            Dictionary of all evaluation metrics
        """
        logger.info("Running comprehensive evaluation...")

        results = {}

        # Calculate perplexity
        results["perplexity"] = self.calculate_perplexity(dataset)

        # Calculate loss
        loss_metrics = self.evaluate_loss(dataset)
        results.update(loss_metrics)

        # Generate samples if prompts provided
        if test_prompts:
            samples = self.generate_samples(test_prompts)
            results["generated_samples"] = samples

        # Save results
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results

    def compare_models(
        self,
        other_model_path: str,
        dataset: Dataset,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare current model with another model.

        Args:
            other_model_path: Path to comparison model
            dataset: Test dataset

        Returns:
            Dictionary comparing metrics
        """
        logger.info(f"Comparing with model at {other_model_path}")

        # Evaluate current model
        current_metrics = self.evaluate_loss(dataset)
        current_metrics["model"] = self.model_path

        # Evaluate other model
        other_evaluator = Evaluator(other_model_path, self.config, self.device)
        other_metrics = other_evaluator.evaluate_loss(dataset)
        other_metrics["model"] = other_model_path

        return {
            "current_model": current_metrics,
            "comparison_model": other_metrics,
            "improvement": {
                "loss": current_metrics["avg_loss"] - other_metrics["avg_loss"],
                "perplexity": current_metrics["perplexity"] - other_metrics["perplexity"],
            },
        }


def run_quick_test(
    model_path: str,
    test_prompts: List[str] = None,
) -> None:
    """
    Quick test function for model inference.

    Args:
        model_path: Path to trained model
        test_prompts: Prompts to test (uses defaults if None)
    """
    if test_prompts is None:
        test_prompts = [
            "The main purpose of this documentation is",
            "To get started with this project, you need to",
            "The key features include",
        ]

    evaluator = Evaluator(model_path)
    results = evaluator.generate_samples(
        test_prompts,
        max_length=50,
        temperature=0.7,
    )

    print("\n" + "=" * 80)
    print("MODEL GENERATION TEST")
    print("=" * 80 + "\n")

    for result in results:
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated'][0]}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        run_quick_test(model_path)
    else:
        print("Usage: python evaluation.py <model_path>")
