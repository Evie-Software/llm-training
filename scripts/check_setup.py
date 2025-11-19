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


def check_mlx():
    """Check MLX availability."""
    print("\nChecking MLX (Apple's ML framework)...")

    try:
        import mlx.core as mx

        print(f"  MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")

        try:
            # Test MLX with simple operation
            x = mx.ones((2, 2))
            y = x * 2
            mx.eval(y)
            print("  ✓ MLX is working correctly!")
            return True
        except Exception as e:
            print(f"  ✗ MLX test failed: {e}")
            return False

    except ImportError:
        print("  ✗ MLX not installed")
        print("    Install with: pip install mlx mlx-lm")
        return False


def check_dependencies():
    """Check all required dependencies."""
    print("\nChecking required packages...")

    packages = [
        ("mlx", "mlx.core"),
        ("mlx-lm", "mlx_lm"),
        ("transformers", "transformers"),
        ("tokenizers", "tokenizers"),
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

    print(
        f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical"
    )


def main():
    """Run all checks."""
    print("=" * 70)
    print("LLM TRAINING SETUP VERIFICATION (MLX)")
    print("=" * 70)

    all_ok = True

    # Python version
    all_ok &= check_python_version()

    # Dependencies
    all_ok &= check_dependencies()

    # MLX
    mlx_ok = check_mlx()
    all_ok &= mlx_ok

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
        print("\nYour environment is ready for MLX-based LLM training.")
        print("\nNext steps:")
        print("1. Add your markdown files to data/raw/")
        print("2. Create a config: llm-train config")
        print("3. Start training: llm-train train --data-dir data/raw")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        print("Run './setup.sh' to install dependencies.")

    if not mlx_ok:
        print("\n⚠ WARNING: MLX not available")
        print("Training will not work without MLX.")
        print("Ensure you're on Apple Silicon (M1/M2/M3) and run:")
        print("  pip install mlx mlx-lm")

    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
