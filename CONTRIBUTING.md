# Contributing to LLM Training Framework

Thank you for your interest in contributing to the LLM Training Framework! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project aims to be welcoming and inclusive. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/llm-training.git`
3. Add upstream remote: `git remote add upstream https://github.com/Evie-Software/llm-training.git`

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) or Linux
- Python 3.9 or higher
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies including dev tools:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. Verify installation:
   ```bash
   python scripts/check_setup.py
   ```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-model-support`
- `fix/memory-leak-in-training`
- `docs/update-quickstart-guide`
- `test/add-finetuning-tests`

### Commit Messages

Write clear, concise commit messages:

```
Add support for Llama-2 model

- Add Llama-2 to supported models list
- Update configuration defaults for Llama-2
- Add example configuration file
- Update documentation

Fixes #123
```

**Format:**
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description if needed
- Reference issues/PRs

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_training --cov-report=html

# Run specific test file
pytest tests/test_dataset.py

# Run specific test
pytest tests/test_config.py::TestConfig::test_default_config
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test classes `Test*`
- Name test functions `test_*`
- Use descriptive test names

Example:

```python
def test_markdown_parser_removes_html_comments():
    """Test that HTML comments are properly removed from markdown."""
    parser = MarkdownParser()
    content = "# Title\n<!-- comment -->\nContent"
    cleaned = parser.clean_markdown(content)
    assert "comment" not in cleaned
    assert "Title" in cleaned
```

### Test Coverage

- Aim for >80% code coverage
- All new features should include tests
- Bug fixes should include regression tests

## Code Style

We use automated tools to maintain code quality:

### Black (Code Formatting)

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Flake8 (Linting)

```bash
# Run linter
flake8 src/ tests/
```

### Type Hints

While not strictly required, type hints are encouraged:

```python
def prepare_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> Tuple[MarkdownDataset, MarkdownDataset, MarkdownDataset]:
    """Prepare train, validation, and test datasets."""
    ...
```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples for complex functions

Example:

```python
def calculate_perplexity(self, dataset: Dataset) -> float:
    """
    Calculate perplexity on a dataset.

    Args:
        dataset: Dataset to evaluate

    Returns:
        Perplexity score (lower is better)

    Example:
        >>> evaluator = Evaluator("models/output")
        >>> perplexity = evaluator.calculate_perplexity(test_dataset)
        >>> print(f"Perplexity: {perplexity:.2f}")
        Perplexity: 42.17
    """
    ...
```

## Submitting Changes

### Pull Request Process

1. **Update your fork:**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes:**
   - Write code
   - Add tests
   - Update documentation

4. **Verify everything works:**
   ```bash
   # Run tests
   pytest

   # Check formatting
   black --check src/ tests/

   # Run linter
   flake8 src/ tests/
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add my new feature"
   ```

6. **Push to your fork:**
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Create a Pull Request:**
   - Go to the repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Pull Request Template

Your PR should include:

**Description:**
- What does this PR do?
- Why is this change needed?
- What approach did you take?

**Testing:**
- What tests did you add?
- How did you verify the changes work?

**Documentation:**
- Did you update the README?
- Did you add/update docstrings?
- Did you update examples?

**Checklist:**
- [ ] Tests pass locally
- [ ] Code follows style guidelines (black, flake8)
- [ ] Documentation updated
- [ ] Tests added for new features
- [ ] No breaking changes (or documented)

## Types of Contributions

### Bug Reports

When reporting bugs, include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal reproducible example

### Feature Requests

When requesting features, explain:
- What problem does it solve?
- How should it work?
- Alternative solutions considered
- Willingness to implement

### Code Contributions

Areas where contributions are welcome:
- **Model support**: Add support for new models
- **Optimizations**: Improve training speed/memory usage
- **Documentation**: Improve guides and examples
- **Tests**: Increase test coverage
- **Bug fixes**: Fix reported issues
- **Examples**: Add example notebooks/scripts

## Development Tips

### Memory Testing

When making changes to memory usage:

```python
from llm_training.utils import estimate_memory

# Test memory estimation
estimate_memory(model, batch_size=2, seq_length=512)
```

### Local Testing on M3

Test M3-specific features:

```python
from llm_training.utils import check_mps_availability

# Verify MPS works
mps_info = check_mps_availability()
assert mps_info["mps_working"], "MPS should work on M3"
```

### Documentation Preview

Preview documentation changes:

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Serve documentation locally
mkdocs serve
```

## Questions?

- Open an issue for general questions
- Use discussions for broader topics
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🚀
