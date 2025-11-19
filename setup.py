"""Setup script for LLM Training package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-training",
    version="0.2.0",
    author="Your Name",
    description="LLM training framework using MLX, optimized for M3 MacBook Pro with 16GB RAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-training",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        # MLX - Apple's ML framework for Apple Silicon
        "mlx>=0.4.0",
        "mlx-lm>=0.4.0",
        # Hugging Face ecosystem (for tokenizers and model configs)
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "huggingface-hub>=0.20.0",
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        # Utilities
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
        # Markdown processing
        "markdown>=3.5.0",
        "beautifulsoup4>=4.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "bandit[toml]>=1.7.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-train=llm_training.cli:main",
        ],
    },
)
