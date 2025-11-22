"""
Utility functions for MLX-based LLM training.
"""

import logging
import sys
from pathlib import Path

import mlx.core as mx
import psutil


def setup_logging(log_dir: str = "logs", log_file: str = "training.log"):
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        log_file: Log file name
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / log_file

    # Create formatters
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def estimate_memory_mlx(
    model,
    batch_size: int,
    print_info: bool = True,
) -> dict:
    """
    Estimate memory requirements for MLX training.

    Args:
        model: MLX model
        batch_size: Batch size
        print_info: Whether to print information

    Returns:
        Dictionary with memory estimates
    """

    def count_params(params_dict):
        """Recursively count parameters in nested MLX model dict."""
        import mlx.core as mx

        total = 0
        for value in params_dict.values():
            if isinstance(value, dict):
                # Recursively count nested dicts
                total += count_params(value)
            elif isinstance(value, mx.array):
                # MLX array - count elements
                total += value.size
        return total

    # Count parameters
    total_params = count_params(model.parameters())
    trainable_params = count_params(model.trainable_parameters())

    # Estimate model size (MLX uses float32 by default = 4 bytes per parameter)
    model_size_gb = (total_params * 4) / (1024**3)

    # Get current available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    estimates = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_gb": model_size_gb,
        "available_memory_gb": available_memory_gb,
        "fits_in_memory": model_size_gb < available_memory_gb * 0.6,  # 60% safety margin
    }

    if print_info:
        print("\n" + "=" * 60)
        print("MEMORY ESTIMATION (MLX)")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_gb:.2f} GB")
        print(f"Available memory: {available_memory_gb:.2f} GB")

        if estimates["fits_in_memory"]:
            print("✓ Should fit in available memory")
        else:
            print("⚠ WARNING: May exceed available memory!")
            print("  Consider:")
            print("  - Reducing batch size")
            print("  - Using a smaller model")
            print("  - Using LoRA fine-tuning instead of full training")

        print("=" * 60 + "\n")

    return estimates


def count_parameters(model) -> dict:
    """
    Count MLX model parameters.

    Args:
        model: MLX model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.size for p in model.parameters().values())
    trainable = sum(p.size for p in model.trainable_parameters().values())
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": (trainable / total * 100) if total > 0 else 0,
    }


def format_number(num: int) -> str:
    """Format large numbers with suffixes (K, M, B)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def check_mlx_availability() -> dict:
    """
    Check MLX availability and configuration.

    Returns:
        Dictionary with MLX information
    """
    info = {
        "mlx_available": True,  # If we can import mlx, it's available
        "mlx_version": mx.__version__ if hasattr(mx, "__version__") else "unknown",
        "device": "Apple Silicon",
    }

    try:
        # Test basic MLX operation
        x = mx.ones((2, 2))
        y = x * 2
        mx.eval(y)
        info["mlx_working"] = True
    except Exception as e:
        info["mlx_working"] = False
        info["error"] = str(e)

    return info


def get_system_capabilities() -> dict:
    """
    Detect system capabilities and recommend optimal training settings.

    Returns:
        Dictionary with system info and recommended settings
    """
    import platform

    mem = psutil.virtual_memory()
    total_ram_gb = mem.total / (1024**3)
    available_ram_gb = mem.available / (1024**3)

    # Detect chip type
    chip_info = platform.processor()
    chip_type = "Apple Silicon" if "arm" in chip_info.lower() else "Unknown"

    # Try to detect specific M-series chip
    try:
        import subprocess  # nosec B404  # Safe: used only for chip detection

        # Safe: hardcoded sysctl command, no user input
        result = subprocess.run(  # nosec B603, B607
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            brand = result.stdout.strip()
            if "Apple" in brand:
                chip_type = brand.split()[1] if len(brand.split()) > 1 else "Apple Silicon"
    except Exception:  # nosec B110  # Intentional: fallback to generic chip detection
        pass

    # Calculate recommended settings based on total RAM
    # Conservative approach: use ~30% of total RAM for model + activations
    if total_ram_gb < 10:
        # 8GB system
        recommendations = {
            "batch_size": 1,
            "max_length": 256,
            "gradient_accumulation_steps": 16,
            "max_model_size": "small (distilgpt2)",
            "warning": "Low RAM detected. Training larger models may be unstable.",
        }
    elif total_ram_gb < 20:
        # 16GB system
        recommendations = {
            "batch_size": 2,
            "max_length": 512,
            "gradient_accumulation_steps": 8,
            "max_model_size": "medium (gpt2, gpt2-medium with care)",
            "warning": None,
        }
    elif total_ram_gb < 40:
        # 32GB system
        recommendations = {
            "batch_size": 4,
            "max_length": 1024,
            "gradient_accumulation_steps": 4,
            "max_model_size": "large (gpt2-medium, gpt2-large)",
            "warning": None,
        }
    else:
        # 64GB+ system
        recommendations = {
            "batch_size": 8,
            "max_length": 2048,
            "gradient_accumulation_steps": 2,
            "max_model_size": "very large (llama-2-7b, mistral-7b)",
            "warning": None,
        }

    return {
        "total_ram_gb": round(total_ram_gb, 1),
        "available_ram_gb": round(available_ram_gb, 1),
        "chip_type": chip_type,
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "recommended": recommendations,
    }


def print_system_info():
    """Print system information with recommendations."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # Python version
    print(f"Python version: {sys.version.split()[0]}")

    # Get system capabilities
    caps = get_system_capabilities()

    # Chip and RAM
    print(f"Chip: {caps['chip_type']}")
    print(f"Total RAM: {caps['total_ram_gb']} GB")
    print(f"Available RAM: {caps['available_ram_gb']} GB")
    print(
        f"CPU cores: {caps['cpu_cores_physical']} physical, " f"{caps['cpu_cores_logical']} logical"
    )

    # MLX information
    mlx_info = check_mlx_availability()
    print(f"MLX version: {mlx_info.get('mlx_version', 'unknown')}")
    print(f"MLX working: {mlx_info.get('mlx_working', False)}")

    # Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDED SETTINGS FOR YOUR SYSTEM")
    print("-" * 60)
    rec = caps["recommended"]
    print(f"Batch size: {rec['batch_size']}")
    print(f"Max sequence length: {rec['max_length']} tokens")
    print(f"Gradient accumulation: {rec['gradient_accumulation_steps']} steps")
    print(f"Recommended models: {rec['max_model_size']}")

    if rec["warning"]:
        print(f"\n⚠️  {rec['warning']}")

    print("=" * 60 + "\n")


def cleanup_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 2,
    dry_run: bool = False,
):
    """
    Clean up old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of checkpoints to keep
        dry_run: If True, only print what would be deleted
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        logging.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return

    # Find checkpoint directories (usually named checkpoint-XXXX)
    checkpoints = sorted(
        [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: (int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0),
    )

    if len(checkpoints) <= keep_last_n:
        logging.info(f"Only {len(checkpoints)} checkpoints found, no cleanup needed")
        return

    # Determine which to delete
    to_delete = checkpoints[:-keep_last_n]

    logging.info(f"Found {len(checkpoints)} checkpoints, will delete {len(to_delete)}")

    for checkpoint in to_delete:
        if dry_run:
            logging.info(f"Would delete: {checkpoint}")
        else:
            import shutil

            shutil.rmtree(checkpoint)
            logging.info(f"Deleted: {checkpoint}")


if __name__ == "__main__":
    # Test utilities
    print_system_info()

    # Test MLX
    mlx_info = check_mlx_availability()
    if mlx_info["mlx_working"]:
        print("✓ MLX is working correctly!")
    else:
        print("✗ MLX is not available or not working")
        print(f"Error: {mlx_info.get('error', 'Unknown')}")
