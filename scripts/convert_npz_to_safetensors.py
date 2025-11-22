#!/usr/bin/env python3
"""
Convert old NPZ format model to safetensors format.
Use this to convert models trained before the safetensors update.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx
import mlx.utils
from mlx_lm import load
from transformers import AutoTokenizer


def convert_npz_to_safetensors(model_dir: str):
    """
    Convert NPZ format model to safetensors.

    Args:
        model_dir: Directory containing weights.npz and tokenizer files
    """
    model_dir = Path(model_dir)

    # Check if weights.npz exists
    npz_path = model_dir / "weights.npz"
    if not npz_path.exists():
        print(f"Error: {npz_path} not found")
        return False

    # Check if already converted
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        print(f"Model already converted to safetensors format")
        return True

    print(f"Converting {model_dir} to safetensors format...")

    try:
        # Load the original model name from config
        config_path = model_dir / "config.json"
        if config_path.exists():
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
                model_name = config.get("model_name", "mlx-community/gpt2-base-mlx")
        else:
            model_name = "mlx-community/gpt2-base-mlx"

        print(f"Loading base model: {model_name}")

        # Load base model and tokenizer
        model, tokenizer = load(model_name)

        # Load trained weights from NPZ
        print(f"Loading weights from {npz_path}")
        model.load_weights(str(npz_path))

        # Save in safetensors format using MLX's native function
        print(f"Saving to safetensors format...")
        safetensors_path = model_dir / "model.safetensors"
        mx.save_safetensors(str(safetensors_path), dict(mlx.utils.tree_flatten(model.parameters())))

        # Save model config - mlx_lm models have args attribute
        import json

        config_path = model_dir / "config.json"
        if hasattr(model, "args"):
            # Convert model args to dict for saving
            model_config = vars(model.args) if hasattr(model.args, "__dict__") else model.args
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"  - Saved model config from model.args")
        elif hasattr(model, "config"):
            # Fallback to config attribute
            model_config = vars(model.config) if hasattr(model.config, "__dict__") else model.config
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"  - Saved model config from model.config")
        else:
            print(f"  - Warning: No model config found to save")

        # Save tokenizer
        tokenizer.save_pretrained(str(model_dir))

        print(f"✓ Conversion successful!")
        print(f"  - Created: {safetensors_path}")
        print(f"  - Saved tokenizer files")
        print(f"  - Old NPZ file preserved: {npz_path}")

        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/convert_npz_to_safetensors.py <model_dir>")
        print("Example: python scripts/convert_npz_to_safetensors.py models/output")
        sys.exit(1)

    model_dir = sys.argv[1]
    success = convert_npz_to_safetensors(model_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
