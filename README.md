# MLX LLM Training Framework for Apple Silicon

[![Tests](https://github.com/Evie-Software/llm-training/actions/workflows/tests.yml/badge.svg)](https://github.com/Evie-Software/llm-training/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready framework for training small Language Models (LLMs) on Markdown documentation, optimized for Apple Silicon (M1/M2/M3/M4) with auto-detection using **MLX** - Apple's ML framework for Apple Silicon.

## 🎯 Key Features

- **Apple Silicon Native**: Uses MLX framework for all M-series chips (M1/M2/M3/M4+)
- **Memory Efficient**: Carefully tuned for 16GB RAM with gradient accumulation and checkpointing
- **Unified Memory**: Leverages Apple Silicon's unified memory architecture
- **Markdown Focused**: Purpose-built for training on `.md` and `.mdx` documentation files
- **Complete Pipeline**: Dataset preparation → Training → Evaluation → Fine-tuning
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using MLX's native LoRA support
- **Self-Serve**: Comprehensive documentation and automated setup
- **Production Ready**: Includes tests, monitoring, and artifact management

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9 or higher
- 16GB RAM (minimum)
- 10GB+ free disk space

### Installation

```bash
# Clone or download this repository
cd llm-training

# Run automated setup
chmod +x setup.sh
./setup.sh

# Verify installation
python scripts/check_setup.py
```

### Your First Training Run

Follow these steps for a complete training and evaluation workflow:

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Add your markdown files to data/raw/
#    You can organize them in subdirectories
#    Example: data/raw/laravel-docs/*.md

# 3. Create a configuration file (optional, uses defaults if skipped)
llm-train config -o configs/my_config.yaml
# Edit configs/my_config.yaml to customize settings

# 4. Start training
llm-train train --data-dir data/raw --config configs/my_config.yaml

# 5. Monitor training progress
tail -f logs/training.log  # In another terminal

# 6. Evaluate your trained model
llm-train evaluate models/output \
    --prompts "Your test prompt" "Another prompt" \
    --max-length 100 \
    --config configs/my_config.yaml

# 7. Check system info anytime
llm-train info
```

**What happens during training:**
- Markdown files are automatically cleaned and parsed
- Data is split into train/validation/test sets (90/5/5 by default)
- Model checkpoints are saved every 1000 steps to `checkpoints/`
- Final trained model is saved to `models/output/`
- Training logs are saved to `logs/training.log`

**Expected timeline:**
- Small dataset (<100 files): 15-30 minutes
- Medium dataset (100-500 files): 30-90 minutes
- Large dataset (500+ files): 1-3 hours

(Times are for M1/M2/M3 with 16GB RAM using default settings)

## 📥 Installation

### Automated Setup (Recommended)

```bash
./setup.sh
```

This script will:
- Create a Python virtual environment
- Install all dependencies (including MLX)
- Create necessary directories
- Install the package in editable mode

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Create directories
mkdir -p data/raw data/processed models checkpoints logs configs
```

### Verify Installation

```bash
python scripts/check_setup.py
```

This will check:
- Python version
- All dependencies
- MLX availability and functionality
- Package installation
- Directory structure

## 📖 Usage Guide

### 1. Prepare Your Data

Place your `.md` or `.mdx` files in `data/raw/`. The framework will automatically:
- Recursively find all markdown files
- Clean and parse MDX components
- Convert to plain text while preserving structure
- Split into train/validation/test sets

Data preparation happens automatically when you run the `train` command.

### 2. Configure Training

Create a configuration file:

```bash
llm-train config -o configs/my_config.yaml
```

Edit `configs/my_config.yaml` to customize:
- Model selection (default: mlx-community/gpt2-base-mlx)
- Batch size and memory settings
- Learning rate and optimization
- LoRA parameters for fine-tuning

### 3. Train a Model

```bash
# Using default config
llm-train train --data-dir data/raw

# Using custom config
llm-train train --data-dir data/raw --config configs/my_config.yaml

# Quick test training
python scripts/quick_train.py data/raw
```

**Training Options:**
- `--data-dir`: Directory containing markdown files
- `--config`: Path to configuration file
- Logs are saved to `logs/`
- Checkpoints saved to `checkpoints/`
- Final model saved to `models/output/`

### 4. Monitor Training

Training logs are automatically saved to `logs/training.log`. Watch progress in real-time:

```bash
tail -f logs/training.log
```

For programmatic monitoring, check the logs directory for structured output.

### 5. Evaluate Your Model

After training, evaluate your model's performance and test text generation:

#### Option A: Evaluate with Test Data
```bash
# Full evaluation with perplexity and loss metrics
llm-train evaluate models/output \
    --test-data data/raw \
    --output results.json \
    --config configs/gpt2-medium-backup.yaml
```

This will:
- Calculate perplexity on your test dataset
- Compute average loss metrics
- Save detailed results to `results.json`

#### Option B: Quick Generation Test (Recommended)
```bash
# Test text generation with custom prompts
llm-train evaluate models/output \
    --prompts "Laravel routing" "How to deploy" "Database migrations" \
    --max-length 100 \
    --config configs/gpt2-medium-backup.yaml
```

This will:
- Load your trained model
- Generate text completions for each prompt
- Display results in the terminal

**Note:** The framework automatically handles both quantized and non-quantized models. If you trained with a quantized model (e.g., `viktor2698/gpt2-medium-mlx-8Bit`), the evaluation will filter quantization parameters as needed.

#### Troubleshooting Evaluation

**Error: "Received X parameters not in model"**
- This occurs with quantized models and is now automatically handled
- The evaluator will filter quantization parameters and retry
- No action needed from you

**Slow evaluation:**
- Reduce `--max-length` for faster generation
- Use a smaller batch of test prompts
- Ensure other applications aren't consuming memory

### 6. Fine-tune for Specific Tasks

Use LoRA for memory-efficient fine-tuning:

```bash
llm-train finetune mlx-community/gpt2-base-mlx data/specialized \
    --config configs/my_config.yaml
```

**LoRA Benefits:**
- Trains only 0.1-1% of parameters
- Much faster than full fine-tuning
- Can fine-tune on specific documentation sets
- Native MLX LoRA implementation for optimal performance

## ⚙️ Configuration

### Default Configuration

The default configuration is optimized for all Apple Silicon Macs:

```yaml
model:
  model_name: mlx-community/gpt2-base-mlx  # 124M parameters
  max_length: 512

training:
  num_train_epochs: 3
  batch_size: 4  # MLX handles batches efficiently
  gradient_accumulation_steps: 4  # Effective batch size: 16
  learning_rate: 5e-5
  grad_checkpoint: true  # Saves memory
```

### Model Selection

**Recommended models for 16GB RAM:**

| Model | Parameters | RAM Usage | Speed | Quality |
|-------|-----------|-----------|-------|---------|
| `mlx-community/gpt2-base-mlx` | 124M | ~4GB | Medium | Good |
| `mlx-community/deepseek-coder-1.3b-base-mlx` | 1.3B | ~5GB | Medium | Better (code) |
| `viktor2698/gpt2-medium-mlx-8Bit` | 355M | ~3GB | Medium | Better (quantized) |

**Models for 32GB+ RAM:**
- `MCES10/gpt2-large-mlx-fp16` (774M params) - Better quality
- `MCES10/gpt2-xl-mlx-fp16` (1.5B params) - Best quality

> **Note:** MLX models are available on Hugging Face Hub. Search for models with `library:mlx` filter or visit the `mlx-community` organization. Over 3,000 MLX-optimized models are available.

### Memory Optimization Tips

If you run out of memory:

1. **Reduce batch size:**
   ```yaml
   training:
     batch_size: 2
     gradient_accumulation_steps: 8
   ```

2. **Reduce sequence length:**
   ```yaml
   data:
     max_length: 256  # Instead of 512
   ```

3. **Use a smaller model:**
   ```yaml
   model:
     model_name: mlx-community/gpt2-base-mlx
   ```

4. **Enable gradient checkpointing:**
   ```yaml
   training:
     grad_checkpoint: true
   ```

## 🎓 Advanced Topics

### Training with Multiple Documentation Sources

The framework supports training on documentation from multiple projects/products simultaneously. There are two strategies depending on whether you want the model to distinguish between sources:

#### Strategy 1: Source Prefix Tagging (Recommended for Multi-Product)

**Use this when:** Training on multiple related products (e.g., Laravel Framework, Laravel Forge, Laravel Cloud) and you want the model to understand which product each content belongs to.

**Directory Structure:**
```
data/raw/
├── laravel-framework/
│   ├── getting-started.md
│   ├── routing.md
│   └── controllers.md
├── laravel-forge/
│   ├── servers.md
│   ├── deployment.md
│   └── sites.md
└── laravel-cloud/
    ├── introduction.md
    ├── teams.md
    └── projects.md
```

**Configuration:**
```yaml
# config.yaml
data:
  raw_data_path: "data/raw"
  add_source_prefix: true  # Enable source tagging
  max_length: 512
```

**What Happens:**
The model learns content with source prefixes:
- `"[laravel-framework] Routes are defined in the routes directory..."`
- `"[laravel-forge] Deploying your application to a server..."`
- `"[laravel-cloud] Teams allow you to collaborate..."`

**Benefits:**
- ✅ Single model understands all products
- ✅ Model knows which product each content belongs to
- ✅ Can generate product-specific responses
- ✅ Easier deployment (one model vs multiple)

**Usage:**
```bash
# Train with source prefixing enabled
llm-train config -o configs/multi-source.yaml
# Edit config and set add_source_prefix: true
llm-train train --config configs/multi-source.yaml
```

#### Strategy 2: Mixed Training (No Source Distinction)

**Use this when:** Training on documentation that's all related to one ecosystem and you don't need the model to distinguish sources.

**Configuration:**
```yaml
# config.yaml
data:
  raw_data_path: "data/raw"
  add_source_prefix: false  # Default - no tagging
  max_length: 512
```

**What Happens:**
All documentation is mixed together without source tags. The model learns general patterns across all content.

**Benefits:**
- ✅ Simpler - just organize files in subdirectories
- ✅ Model learns unified knowledge base
- ✅ Good for supplementary docs, guides, READMEs

**Usage:**
```bash
# Standard training (no source prefixes)
llm-train train --data-dir data/raw
```

#### Strategy 3: Separate Models per Source

**Use this when:** You need completely isolated models for different products or have very different documentation styles.

**Directory Structure:**
```
data/
├── laravel-framework/
│   └── *.md files
├── laravel-forge/
│   └── *.md files
└── laravel-cloud/
    └── *.md files
```

**Train Separate Models:**
```bash
# Framework model
llm-train train --data-dir data/laravel-framework \
  --config configs/framework.yaml

# Forge model
llm-train train --data-dir data/laravel-forge \
  --config configs/forge.yaml

# Cloud model
llm-train train --data-dir data/laravel-cloud \
  --config configs/cloud.yaml
```

**Benefits:**
- ✅ Complete isolation between products
- ✅ Can optimize each model separately
- ✅ Different model sizes per product complexity

**Drawbacks:**
- ❌ More models to maintain and deploy
- ❌ More storage space required
- ❌ Harder to cross-reference between products

#### Recommendation for Laravel Use Case

For **Laravel Framework + Forge + Cloud** specifically:

**Recommended:** Strategy 1 (Source Prefix Tagging)

```yaml
# configs/laravel-multi.yaml
data:
  raw_data_path: "data/raw"
  add_source_prefix: true  # Enable this!
  max_length: 512  # Or 1024 if you have 32GB+ RAM

model:
  model_name: "mlx-community/gpt2-base-mlx"  # 124M params, good for 16GB+
  # For code-focused: "mlx-community/deepseek-coder-1.3b-base-mlx"
```

This gives you:
1. **Single model** that knows all Laravel products
2. **Context awareness** - knows when discussing Framework vs Forge vs Cloud
3. **Better responses** - can say "In Laravel Framework..." vs "In Forge..."
4. **Easier deployment** - one model, simpler infrastructure

#### How Source Prefixes Work

When you enable `add_source_prefix: true`, the framework automatically:

1. **Extracts source name** from directory structure:
   - `data/raw/laravel-framework/routing.md` → source name: `laravel-framework`
   - `data/raw/laravel-forge/servers.md` → source name: `laravel-forge`

2. **Adds prefix to content**:
   ```
   Original: "Routes are defined in the routes directory..."
   Prefixed: "[laravel-framework] Routes are defined in the routes directory..."
   ```

3. **Model learns the pattern**:
   - Understands that `[source-name]` indicates the product context
   - Can generate responses with appropriate context
   - Differentiates between similar concepts across products

#### Example Queries After Training

With source prefixes enabled:

**Query:** "How do I deploy my application?"

**Model Response:**
- Understands this could be Framework (local dev) OR Forge (server deployment)
- Can provide context-specific answers based on training data
- Learned pattern: `[laravel-forge]` content discusses deployment

**Query:** "Tell me about routing"

**Model Response:**
- Knows this is primarily a `[laravel-framework]` topic
- Provides Framework-specific routing information

### Custom Training Loop

```python
from llm_training import Config, Trainer, prepare_dataset
from transformers import AutoTokenizer

# Load config
config = Config.from_yaml("configs/my_config.yaml")

# Prepare data
tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
train_ds, val_ds, test_ds = prepare_dataset(
    "data/raw",
    tokenizer,
    max_length=512,
)

# Train
trainer = Trainer(config, train_ds, val_ds, test_ds)
trainer.train()

# Generate text
text = trainer.generate_text("Once upon a time", max_length=100)
print(text)
```

### Custom Dataset Processing

```python
from llm_training.dataset import MarkdownDataset, MarkdownParser

# Custom parsing
parser = MarkdownParser()
cleaned_text = parser.clean_markdown(raw_content, is_mdx=True)

# Custom dataset
dataset = MarkdownDataset(
    file_paths=["doc1.md", "doc2.md"],
    tokenizer=tokenizer,
    max_length=512,
    stride=256,  # Overlapping chunks
)

# Iterate over batches
for batch in dataset.batch_iterate(batch_size=4, shuffle=True):
    # batch contains numpy arrays ready for MLX
    pass
```

### Programmatic Evaluation

```python
from llm_training.evaluation import Evaluator

evaluator = Evaluator("models/output")

# Calculate perplexity
perplexity = evaluator.calculate_perplexity(test_dataset)

# Generate samples
samples = evaluator.generate_samples(
    prompts=["Hello", "The main features"],
    temperature=0.7,
)

# Comprehensive evaluation
results = evaluator.comprehensive_evaluation(
    dataset=test_dataset,
    test_prompts=["Test prompt 1", "Test prompt 2"],
    output_path="eval_results.json",
)
```

### LoRA Fine-tuning Programmatically

```python
from llm_training.finetuning import LoRAFineTuner

finetuner = LoRAFineTuner(
    base_model_path="mlx-community/gpt2-base-mlx",
    config=config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Fine-tune
finetuner.finetune()

# Model is automatically saved with LoRA weights
```

## 🛠️ Troubleshooting

### MLX Not Available

**Symptom:** Import errors or MLX not working

**Solutions:**
1. Ensure you're on Apple Silicon (M1/M2/M3)
2. Update MLX: `pip install --upgrade mlx mlx-lm`
3. Check: `python -c "import mlx.core as mx; print(mx.ones((2,2)))"`
4. Run: `llm-train info` to see system information

### Out of Memory Errors

**Solutions:**
1. Reduce batch size in config
2. Reduce sequence length
3. Use smaller model (mlx-community/distilgpt2)
4. Close other applications
5. Enable gradient checkpointing

### Training Very Slow

**Possible causes:**
1. Large model for hardware
2. Dataset too large
3. Other applications consuming memory
4. Disk I/O bottleneck

**Solutions:**
- Use Activity Monitor to check CPU/Memory/Disk usage
- Reduce model size
- Reduce `num_train_epochs`
- Ensure data is on fast SSD

### Import Errors

**Solution:**
```bash
# Reinstall package
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Generation Quality Issues

If generated text is poor:
1. Train for more epochs
2. Use more training data
3. Use a larger model
4. Adjust temperature and top_p during generation
5. Check training loss - should be decreasing

## 📁 Project Structure

```
llm-training/
├── README.md                 # This file
├── setup.sh                  # Automated setup script
├── cleanup.sh               # Artifact cleanup utility
├── requirements.txt         # Python dependencies (MLX-based)
├── setup.py                # Package configuration
├── .gitignore              # Git ignore rules
│
├── src/llm_training/       # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── dataset.py          # Dataset preparation
│   ├── training.py         # MLX training pipeline
│   ├── evaluation.py       # Evaluation and testing
│   ├── finetuning.py       # MLX LoRA fine-tuning
│   ├── utils.py            # Utility functions
│   └── cli.py              # Command-line interface
│
├── scripts/                # Executable scripts
│   ├── check_setup.py      # Setup verification
│   └── quick_train.py      # Quick training script
│
├── tests/                  # Test files
│   └── ...
│
├── docs/                   # Additional documentation
│   ├── QUICKSTART.md
│   ├── CONFIGURATION.md
│   ├── MLX_MIGRATION.md    # MLX migration guide
│   └── API.md
│
├── examples/               # Example configs and code
│   ├── configs/
│   └── notebooks/
│
├── data/                   # Data directory (gitignored)
│   ├── raw/               # Your markdown files
│   └── processed/         # Processed datasets
│
├── models/                 # Models (gitignored)
│   ├── cache/             # Downloaded models
│   ├── output/            # Trained models
│   └── finetuned/         # Fine-tuned models
│
├── checkpoints/            # Training checkpoints (gitignored)
├── logs/                   # Training logs (gitignored)
└── configs/                # Configuration files
```

## 🧪 Testing

Run tests:

```bash
# All tests
pytest

# With coverage
pytest --cov=llm_training --cov-report=html

# Specific test file
pytest tests/test_dataset.py
```

## 🧹 Cleanup

Manage disk space:

```bash
# Interactive cleanup
./cleanup.sh

# This will help you remove:
# - Old checkpoints
# - Training logs
# - Processed datasets
# - Cache files
```

## 📊 Performance Expectations

On Apple Silicon (performance varies by RAM) with MLX:

- **Small model (distilgpt2, 82M):** ~800-1000 samples/sec
- **Medium model (gpt2, 124M):** ~500-700 samples/sec
- **Larger model (gpt2-medium, 355M):** ~200-300 samples/sec

Training time for 10,000 samples:
- Small: ~15-20 minutes
- Medium: ~25-35 minutes
- Large: ~1-1.5 hours

> **Note:** MLX is generally faster than PyTorch on Apple Silicon due to native optimizations.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Format and lint your code: `black src/ tests/ && bandit -c pyproject.toml -r src/`
5. Add tests if applicable
6. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines and [docs/LINTING.md](docs/LINTING.md) for code formatting instructions.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [MLX-LM](https://github.com/ml-explore/mlx-examples/tree/main/llms) - Language model utilities for MLX
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Tokenizers and model configs

---

## 💡 Tips for Success

1. **Start small:** Test with a small subset of data first
2. **Monitor memory:** Use Activity Monitor to watch RAM usage
3. **Check logs:** Watch `logs/training.log` for progress
4. **Version configs:** Track which config produced which model
5. **Backup models:** Your trained models are valuable!
6. **Use LoRA:** For fine-tuning, LoRA is much more memory-efficient

## 🆘 Getting Help

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/training.log`
3. Run `python scripts/check_setup.py`
4. Check system info: `llm-train info`
5. See [MLX_MIGRATION.md](docs/MLX_MIGRATION.md) for migration details

Happy training with MLX! 🚀
