"""
Command-line interface for MLX-based LLM training.
"""

import argparse
import sys
from pathlib import Path

from llm_training.config import Config, create_default_config
from llm_training.dataset import prepare_dataset
from llm_training.training import Trainer
from llm_training.evaluation import Evaluator, run_quick_test
from llm_training.finetuning import LoRAFineTuner
from llm_training.utils import print_system_info
from transformers import AutoTokenizer


def create_config_command(args):
    """Create a default configuration file."""
    output_path = args.output or "configs/default.yaml"
    config = create_default_config(output_path)
    print(f"✓ Configuration file created at {output_path}")
    print("\nEdit this file to customize your training settings.")


def train_command(args):
    """Train a model."""
    print("Starting MLX-based training...")

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.get_default()

    # Validate config
    config.validate()

    # Load tokenizer
    print(f"Loading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    data_dir = args.data_dir or config.data.raw_data_path
    print(f"Loading data from {data_dir}")

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
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        test_dataset=test_dataset,
    )

    # Train
    trainer.train()

    # Evaluate on test set
    if test_dataset and len(test_dataset) > 0:
        print("\nEvaluating on test set...")
        test_loss = trainer.evaluate(test_dataset)
        print(f"Test loss: {test_loss:.4f}")

    print("\n✓ Training completed!")
    print(f"Model saved to: {config.model.output_dir}")


def evaluate_command(args):
    """Evaluate a trained model."""
    print("Evaluating MLX model...")

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.get_default()

    # Create evaluator
    evaluator = Evaluator(args.model_path, config)

    # Quick test if prompts provided
    if args.prompts:
        results = evaluator.generate_samples(
            prompts=args.prompts,
            max_length=args.max_length,
        )

        print("\nGeneration Results:")
        print("=" * 80)
        for result in results:
            print(f"\nPrompt: {result['prompt']}")
            print(f"Generated: {result['generated']}")
            print("-" * 80)

    # If test data provided, compute metrics
    if args.test_data:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, _, test_dataset = prepare_dataset(
            data_dir=args.test_data,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
        )

        results = evaluator.comprehensive_evaluation(
            dataset=test_dataset,
            output_path=args.output,
        )

        print("\nEvaluation Results:")
        print("=" * 80)
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Average Loss: {results['avg_loss']:.4f}")


def finetune_command(args):
    """Fine-tune a model using LoRA."""
    print("Starting MLX LoRA fine-tuning...")

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.get_default()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    print(f"Loading data from {args.data_dir}")
    train_dataset, val_dataset, _ = prepare_dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        train_split=config.data.train_test_split,
        validation_split=config.data.validation_split,
        seed=config.data.seed,
        extensions=config.data.file_extensions,
    )

    # Create fine-tuner
    finetuner = LoRAFineTuner(
        base_model_path=args.base_model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Fine-tune
    finetuner.finetune()

    print("\n✓ Fine-tuning completed!")
    print(f"Model saved to: {config.finetuning.output_dir}")


def info_command(args):
    """Print system information."""
    print_system_info()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Training Framework using MLX for M3 MacBook Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser("config", help="Create default configuration file")
    config_parser.add_argument("-o", "--output", help="Output path for config file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--data-dir", help="Directory containing markdown files")
    train_parser.add_argument("--config", help="Path to configuration file")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("model_path", help="Path to trained model")
    eval_parser.add_argument("--test-data", help="Path to test data directory")
    eval_parser.add_argument("--prompts", nargs="+", help="Test prompts for generation")
    eval_parser.add_argument("--max-length", type=int, default=100, help="Max generation length")
    eval_parser.add_argument("--output", help="Output path for results")
    eval_parser.add_argument("--config", help="Path to configuration file")

    # Finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune model with LoRA")
    finetune_parser.add_argument("base_model", help="Path to base model")
    finetune_parser.add_argument("data_dir", help="Directory containing training data")
    finetune_parser.add_argument("--config", help="Path to configuration file")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Execute command
    commands = {
        "config": create_config_command,
        "train": train_command,
        "evaluate": evaluate_command,
        "finetune": finetune_command,
        "info": info_command,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
