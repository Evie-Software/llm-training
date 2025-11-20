"""
Evaluation and testing module for MLX-trained LLMs.
Provides comprehensive metrics and testing capabilities.
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load, generate
from tqdm import tqdm

from llm_training.config import Config
from llm_training.dataset import MarkdownDataset

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluator for MLX LLM models.

    Provides:
    - Perplexity calculation
    - Loss evaluation
    - Text generation quality assessment
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Config] = None,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            config: Configuration object
        """
        self.model_path = model_path
        self.config = config or Config.get_default()

        logger.info(f"Loading model from {model_path}")
        self.model, self.tokenizer = load(model_path)

        logger.info("Model loaded and ready for evaluation")

    def loss_fn(self, model, inputs, targets, attention_mask):
        """
        Compute loss for a batch.

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

    def calculate_perplexity(
        self,
        dataset: MarkdownDataset,
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

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(
            dataset.batch_iterate(batch_size, shuffle=False),
            desc="Calculating perplexity",
            total=len(dataset) // batch_size,
        ):
            inputs = mx.array(batch["input_ids"])
            targets = mx.array(batch["labels"])
            attention_mask = mx.array(batch["attention_mask"])

            loss = self.loss_fn(self.model, inputs, targets, attention_mask)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)

        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity

    def evaluate_loss(
        self,
        dataset: MarkdownDataset,
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

        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(
            dataset.batch_iterate(batch_size, shuffle=False),
            desc="Evaluating loss",
            total=len(dataset) // batch_size,
        ):
            inputs = mx.array(batch["input_ids"])
            targets = mx.array(batch["labels"])
            attention_mask = mx.array(batch["attention_mask"])

            loss = self.loss_fn(self.model, inputs, targets, attention_mask)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

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
    ) -> List[Dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            List of dictionaries with prompt and generated text
        """
        logger.info(f"Generating samples for {len(prompts)} prompts...")

        results = []

        for prompt in tqdm(prompts, desc="Generating"):
            generated_text = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
            )

            results.append(
                {
                    "prompt": prompt,
                    "generated": generated_text,
                }
            )

        return results

    def comprehensive_evaluation(
        self,
        dataset: MarkdownDataset,
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
        print(f"Generated: {result['generated']}")
        print("-" * 80 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        run_quick_test(model_path)
    else:
        print("Usage: python evaluation.py <model_path>")
