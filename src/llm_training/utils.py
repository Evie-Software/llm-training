"""
Utility functions for LLM training.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
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


def get_device(use_mps: bool = True) -> torch.device:
    """
    Get appropriate device for training/inference.

    Args:
        use_mps: Whether to use MPS (Metal Performance Shaders) for M3

    Returns:
        torch.device
    """
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Metal Performance Shaders) on Apple Silicon")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    return device


def estimate_memory(
    model: torch.nn.Module,
    batch_size: int,
    seq_length: int,
    print_info: bool = True,
) -> dict:
    """
    Estimate memory requirements for training.

    Args:
        model: PyTorch model
        batch_size: Batch size
        seq_length: Sequence length
        print_info: Whether to print information

    Returns:
        Dictionary with memory estimates
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate model size (in GB)
    # Assuming bfloat16 (2 bytes per parameter)
    model_size_gb = (total_params * 2) / (1024**3)

    # Estimate activation memory (rough estimate)
    # For transformers: batch_size * seq_length * hidden_size * num_layers * 4 (activations) * 2 (bytes)
    # This is a simplified estimate
    activation_estimate_gb = (batch_size * seq_length * 768 * 12 * 4 * 2) / (1024**3)

    # Optimizer state (Adam: 2x model parameters for momentum and variance)
    optimizer_gb = model_size_gb * 2

    # Total estimate
    total_estimate_gb = model_size_gb + activation_estimate_gb + optimizer_gb

    # Get current available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    estimates = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_gb": model_size_gb,
        "activation_estimate_gb": activation_estimate_gb,
        "optimizer_state_gb": optimizer_gb,
        "total_estimate_gb": total_estimate_gb,
        "available_memory_gb": available_memory_gb,
        "fits_in_memory": total_estimate_gb < available_memory_gb * 0.8,  # Use 80% as safety margin
    }

    if print_info:
        print("\n" + "=" * 60)
        print("MEMORY ESTIMATION")
        print("=" * 60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {model_size_gb:.2f} GB")
        print(f"Estimated activation memory: {activation_estimate_gb:.2f} GB")
        print(f"Optimizer state: {optimizer_gb:.2f} GB")
        print(f"Total estimated: {total_estimate_gb:.2f} GB")
        print(f"Available memory: {available_memory_gb:.2f} GB")

        if estimates["fits_in_memory"]:
            print("✓ Should fit in available memory")
        else:
            print("⚠ WARNING: May exceed available memory!")
            print("  Consider:")
            print("  - Reducing batch size")
            print("  - Using gradient checkpointing")
            print("  - Using a smaller model")

        print("=" * 60 + "\n")

    return estimates


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
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


def check_mps_availability() -> dict:
    """
    Check MPS (Metal Performance Shaders) availability and configuration.

    Returns:
        Dictionary with MPS information
    """
    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "torch_version": torch.__version__,
    }

    if info["mps_available"]:
        # Test MPS with a simple operation
        try:
            x = torch.ones(1, device="mps")
            y = x * 2
            info["mps_working"] = True
        except Exception as e:
            info["mps_working"] = False
            info["mps_error"] = str(e)
    else:
        info["mps_working"] = False

    return info


def print_system_info():
    """Print system information."""
    print("\n" + "=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)

    # Python version
    print(f"Python version: {sys.version.split()[0]}")

    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # MPS information
    mps_info = check_mps_availability()
    print(f"MPS available: {mps_info['mps_available']}")
    print(f"MPS working: {mps_info.get('mps_working', False)}")

    # Memory information
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"RAM usage: {mem.percent}%")

    # CPU information
    print(
        f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )

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
        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
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

    # Test MPS
    mps_info = check_mps_availability()
    if mps_info["mps_working"]:
        print("✓ MPS is working correctly!")
    else:
        print("✗ MPS is not available or not working")
