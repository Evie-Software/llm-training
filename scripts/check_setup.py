#!/usr/bin/env python3
"""
Setup verification script.
Checks that all dependencies are installed and working correctly.
"""

import sys
import importlib


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info

    if version.major == 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro}")
        print("    Required: Python 3.9 or higher")
        return False


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"  ✓ {package_name} ({version})")
        return True
    except ImportError:
        print(f"  ✗ {package_name} - NOT INSTALLED")
        return False


def check_torch_mps():
    """Check PyTorch MPS availability."""
    print("\nChecking PyTorch MPS (Metal Performance Shaders)...")

    try:
        import torch

        print(f"  PyTorch version: {torch.__version__}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")

        if torch.backends.mps.is_available():
            try:
                # Test MPS with simple operation
                x = torch.ones(1, device="mps")
                y = x * 2
                print("  ✓ MPS is working correctly!")
                return True
            except Exception as e:
                print(f"  ✗ MPS test failed: {e}")
                return False
        else:
            print("  ⚠ MPS not available (not on Apple Silicon?)")
            return False

    except ImportError:
        print("  ✗ PyTorch not installed")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\nChecking required packages...")

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("tokenizers", "tokenizers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil"),
        ("markdown", "markdown"),
        ("beautifulsoup4", "bs4"),
    ]

    results = []
    for package_name, import_name in packages:
        results.append(check_package(package_name, import_name))

    return all(results)


def check_llm_training_package():
    """Check if llm_training package is installed."""
    print("\nChecking llm_training package...")

    try:
        import llm_training

        print(f"  ✓ llm_training package installed (v{llm_training.__version__})")
        return True
    except ImportError:
        print("  ✗ llm_training package not installed")
        print("    Run: pip install -e .")
        return False


def check_directories():
    """Check if required directories exist."""
    print("\nChecking project directories...")

    from pathlib import Path

    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "checkpoints",
        "logs",
        "configs",
    ]

    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ⚠ {dir_path} - creating...")
            path.mkdir(parents=True, exist_ok=True)

    return True


def print_system_info():
    """Print system information."""
    print("\n" + "=" * 70)
    print("SYSTEM INFORMATION")
    print("=" * 70)

    import platform
    import psutil

    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"Available RAM: {mem.available / (1024**3):.2f} GB")

    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")


def main():
    """Run all checks."""
    print("=" * 70)
    print("LLM TRAINING SETUP VERIFICATION")
    print("=" * 70)

    all_ok = True

    # Python version
    all_ok &= check_python_version()

    # Dependencies
    all_ok &= check_dependencies()

    # PyTorch MPS
    mps_ok = check_torch_mps()

    # LLM training package
    all_ok &= check_llm_training_package()

    # Directories
    all_ok &= check_directories()

    # System info
    print_system_info()

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ ALL CHECKS PASSED!")
        print("\nYour environment is ready for LLM training.")
        print("\nNext steps:")
        print("1. Add your markdown files to data/raw/")
        print("2. Create a config: llm-train config")
        print("3. Start training: llm-train train --data-dir data/raw")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Run './setup.sh' to install dependencies.")

    if not mps_ok:
        print("\n⚠ WARNING: MPS not available")
        print("Training will use CPU, which will be significantly slower.")
        print("For M3 Mac, ensure you have PyTorch 2.0+ installed.")

    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
