# LLM Training Framework for M3 MacBook Pro

[![Tests](https://github.com/Evie-Software/llm-training/actions/workflows/tests.yml/badge.svg)](https://github.com/Evie-Software/llm-training/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-ready framework for training small Language Models (LLMs) on Markdown documentation, specifically optimized for Apple M3 MacBook Pro with 16GB RAM.

## 🎯 Key Features

- **M3 Optimized**: Uses Metal Performance Shaders (MPS) for GPU acceleration on Apple Silicon
- **Memory Efficient**: Carefully tuned for 16GB RAM with gradient accumulation and checkpointing
- **Markdown Focused**: Purpose-built for training on `.md` and `.mdx` documentation files
- **Complete Pipeline**: Dataset preparation → Training → Evaluation → Fine-tuning
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning for specialized tasks
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

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Add your markdown files to data/raw/
#    You can organize them in subdirectories

# 3. Create a default configuration
llm-train config

# 4. Start training
llm-train train --data-dir data/raw

# 5. Monitor training
tensorboard --logdir logs
```

That's it! Your model will be saved to `models/output/` when training completes.

## 📥 Installation

### Automated Setup (Recommended)

```bash
./setup.sh
```

This script will:
- Create a Python virtual environment
- Install all dependencies
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
- MPS (GPU) availability
- Package installation
- Directory structure

## 📖 Usage Guide

### 1. Prepare Your Data

Place your `.md` or `.mdx` files in `data/raw/`. The framework will:
- Recursively find all markdown files
- Clean and parse MDX components
- Convert to plain text while preserving structure
- Split into train/validation/test sets

```bash
# Check your data
llm-train prepare --data-dir data/raw
```

### 2. Configure Training

Create a configuration file:

```bash
llm-train config -o configs/my_config.yaml
```

Edit `configs/my_config.yaml` to customize:
- Model selection (default: GPT-2)
- Batch size and memory settings
- Learning rate and optimization
- Evaluation frequency

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

#### Using TensorBoard

```bash
tensorboard --logdir logs
# Open http://localhost:6006 in your browser
```

#### Using Weights & Biases (Optional)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Update config to use wandb
# training.report_to: "wandb"
```

### 5. Evaluate Your Model

```bash
# Evaluate with test data
llm-train evaluate models/output \
    --test-data data/raw \
    --output results.json

# Quick generation test
llm-train evaluate models/output \
    --prompts "The main purpose of" "To get started," \
    --max-length 100
```

### 6. Fine-tune for Specific Tasks

Use LoRA for memory-efficient fine-tuning:

```bash
llm-train finetune models/output data/specialized \
    --config configs/my_config.yaml \
    --merge
```

**LoRA Benefits:**
- Trains only 0.1-1% of parameters
- Much faster than full fine-tuning
- Can fine-tune on specific documentation sets
- Optionally merge weights for standalone model

## ⚙️ Configuration

### Default Configuration

The default configuration is optimized for M3 MacBook Pro with 16GB RAM:

```yaml
model:
  model_name: gpt2  # 124M parameters
  max_length: 512

training:
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8  # Effective batch size: 16
  learning_rate: 5e-5
  bf16: true  # Use bfloat16 for M3
  gradient_checkpointing: true  # Saves memory
  use_mps: true  # Use Metal GPU
```

### Model Selection

**Recommended models for 16GB RAM:**

| Model | Parameters | RAM Usage | Speed | Quality |
|-------|-----------|-----------|-------|---------|
| `distilgpt2` | 82M | ~4GB | Fast | Good |
| `gpt2` | 124M | ~5GB | Medium | Better |
| `gpt2-medium` | 355M | ~8GB | Slower | Best* |

*May be tight on 16GB RAM

**Alternative models:**
- `facebook/opt-125m` - Similar to GPT-2
- `EleutherAI/pythia-160m` - Well-trained small model
- `microsoft/phi-1_5` - 1.3B params, good quality but may need adjustments

### Memory Optimization Tips

If you run out of memory:

1. **Reduce batch size:**
   ```yaml
   training:
     per_device_train_batch_size: 1
     gradient_accumulation_steps: 16
   ```

2. **Reduce sequence length:**
   ```yaml
   data:
     max_length: 256  # Instead of 512
   ```

3. **Use a smaller model:**
   ```yaml
   model:
     model_name: distilgpt2
   ```

4. **Enable gradient checkpointing:**
   ```yaml
   training:
     gradient_checkpointing: true
   ```

## 🎓 Advanced Topics

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
    base_model_path="models/output",
    config=config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

# Fine-tune
finetuner.finetune()

# Merge weights (optional)
finetuner.merge_and_save("models/finetuned_merged")
```

## 🛠️ Troubleshooting

### MPS (GPU) Not Available

**Symptom:** Training uses CPU instead of GPU

**Solutions:**
1. Ensure you're on Apple Silicon (M1/M2/M3)
2. Update PyTorch: `pip install --upgrade torch`
3. Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

### Out of Memory Errors

**Solutions:**
1. Reduce batch size in config
2. Reduce sequence length
3. Use smaller model (distilgpt2)
4. Close other applications
5. Enable gradient checkpointing

### Training Very Slow

**Possible causes:**
1. MPS not enabled (check config: `use_mps: true`)
2. Too many data loader workers (set to 0 for MPS)
3. Large model for hardware
4. Dataset too large

**Solutions:**
- Verify GPU usage with Activity Monitor
- Reduce model size
- Reduce `num_train_epochs`

### Import Errors

**Solution:**
```bash
# Reinstall package
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Checkpoint Issues

If training crashes and you want to resume:

```yaml
training:
  resume_from_checkpoint: "checkpoints/checkpoint-1000"
```

## 📁 Project Structure

```
llm-training/
├── README.md                 # This file
├── setup.sh                  # Automated setup script
├── cleanup.sh               # Artifact cleanup utility
├── requirements.txt         # Python dependencies
├── setup.py                # Package configuration
├── .gitignore              # Git ignore rules
│
├── src/llm_training/       # Main package
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── dataset.py          # Dataset preparation
│   ├── training.py         # Training pipeline
│   ├── evaluation.py       # Evaluation and testing
│   ├── finetuning.py       # LoRA fine-tuning
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

On M3 MacBook Pro (16GB RAM):

- **Small model (distilgpt2, 82M):** ~500 samples/sec
- **Medium model (gpt2, 124M):** ~300 samples/sec
- **Larger model (gpt2-medium, 355M):** ~100 samples/sec

Training time for 10,000 samples:
- Small: ~30 minutes
- Medium: ~1 hour
- Large: ~3 hours

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model library
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning

---

## 💡 Tips for Success

1. **Start small:** Test with a small subset of data first
2. **Monitor memory:** Use Activity Monitor to watch RAM usage
3. **Use TensorBoard:** Visualize training progress in real-time
4. **Save often:** Keep `save_total_limit` at 2-3 to conserve space
5. **Version configs:** Track which config produced which model
6. **Backup models:** Your trained models are valuable!

## 🆘 Getting Help

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/training.log`
3. Run `python scripts/check_setup.py`
4. Check system info: `llm-train info`

Happy training! 🚀