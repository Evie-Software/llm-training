#!/usr/bin/env python3
"""
Quick training script for testing.
Uses default settings optimized for M3 MacBook Pro.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_training.config import Config
from llm_training.dataset import prepare_dataset
from llm_training.training import Trainer
from llm_training.utils import print_system_info
from transformers import AutoTokenizer


def main():
    """Quick training with defaults."""
    print("=" * 70)
    print("QUICK TRAINING SCRIPT")
    print("=" * 70)

    # Print system info
    print_system_info()

    # Get data directory from command line
    if len(sys.argv) < 2:
        print("Usage: python quick_train.py <data_directory>")
        print("\nExample: python quick_train.py data/raw")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not Path(data_dir).exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create config
    print("\nUsing default configuration optimized for M3 MacBook Pro...")
    config = Config.get_default()

    # Override some settings for quick testing
    config.training.num_train_epochs = 1
    config.training.save_steps = 100
    config.training.eval_steps = 100

    config.validate()

    # Load tokenizer
    print(f"\nLoading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=config.model.cache_dir,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    print(f"\nPreparing datasets from {data_dir}...")
    train_dataset, val_dataset, test_dataset = prepare_dataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        train_split=config.data.train_test_split,
        validation_split=config.data.validation_split,
        seed=config.data.seed,
        extensions=config.data.file_extensions,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Train
    print("\nStarting training...")
    print("This may take a while depending on your dataset size...")
    trainer.train()

    # Evaluate
    if test_dataset and len(test_dataset) > 0:
        print("\nEvaluating on test set...")
        metrics = trainer.evaluate(test_dataset)
        print(f"\nTest metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    # Test generation
    print("\n" + "=" * 70)
    print("TESTING TEXT GENERATION")
    print("=" * 70)

    test_prompts = [
        "The main purpose of",
        "To get started,",
        "The key features include",
    ]

    for prompt in test_prompts:
        generated = trainer.generate_text(prompt, max_length=50)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[0]}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"\nModel saved to: {config.model.output_dir}")
    print("\nTo use your trained model:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{config.model.output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{config.model.output_dir}')")


if __name__ == "__main__":
    main()
