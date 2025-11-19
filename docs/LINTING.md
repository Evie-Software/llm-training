# Code Formatting and Linting Guide

This document explains how to format and lint your code before committing.

## Quick Reference

```bash
# Format all code with Black
black src/ tests/ scripts/ --line-length 100

# Check formatting (without making changes)
black --check src/ tests/ scripts/

# Run linter (flake8)
flake8 src/ tests/ scripts/

# Run security checks
bandit -r src/ -f screen

# Run all checks at once
./scripts/lint_all.sh
```

## Installation

The linting tools are installed automatically when you run `./setup.sh` or manually install dev dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- **Black** - Code formatter
- **flake8** - Code linter
- **mypy** - Type checker
- **bandit** - Security scanner
- **pytest** - Testing framework

## Black (Code Formatting)

Black is an opinionated code formatter that automatically formats your Python code.

### Format code

```bash
# Format all Python files
black src/ tests/ scripts/

# Format specific file
black src/llm_training/config.py

# Custom line length (default is 100 for this project)
black src/ --line-length 100
```

### Check formatting without changes

```bash
# Check if files need formatting
black --check src/ tests/ scripts/

# Show diff of what would change
black --diff src/ tests/ scripts/
```

### Configuration

Black is configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']
```

## Flake8 (Linting)

Flake8 checks your code for style issues and potential errors.

### Run linter

```bash
# Check all files
flake8 src/ tests/ scripts/

# Check specific file
flake8 src/llm_training/config.py

# Show statistics
flake8 src/ --statistics

# Show source code for each error
flake8 src/ --show-source
```

### Configuration

Flake8 is configured in `.flake8`:

```ini
[flake8]
max-line-length = 127
max-complexity = 10
ignore = E203, W503, E501  # Conflicts with Black
```

### Common flake8 errors

- **E/W**: PEP 8 style violations
- **F**: PyFlakes errors (undefined variables, imports, etc.)
- **C**: McCabe complexity warnings

## Bandit (Security Scanning)

Bandit checks for common security issues in Python code.

### Run security checks

```bash
# Scan all source code
bandit -r src/

# Output as JSON
bandit -r src/ -f json -o bandit-report.json

# Show only medium/high severity issues
bandit -r src/ -ll

# Scan specific file
bandit src/llm_training/config.py
```

### Configuration

Bandit is configured in `pyproject.toml`:

```toml
[tool.bandit]
exclude_dirs = ["tests", "venv", ".venv", "build", "dist"]
skips = ["B101", "B601", "B615"]
```

### Skipped checks

- **B101**: `assert_used` - We use asserts in tests
- **B601**: `paramiko` - Not used in this project
- **B615**: `huggingface_unsafe_download` - Expected behavior for training framework

### Ignoring false positives

Add `# nosec` comment to ignore specific lines:

```python
# nosec B614 - Loading user's own processed data
samples = torch.load(input_path, weights_only=False)
```

## MyPy (Type Checking)

MyPy performs static type checking (optional but recommended).

### Run type checker

```bash
# Check all files
mypy src/

# Check specific file
mypy src/llm_training/config.py

# Ignore missing imports
mypy src/ --ignore-missing-imports
```

### Configuration

MyPy is configured in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.9"
ignore_missing_imports = true
warn_redundant_casts = true
```

## Pre-commit Workflow

Before committing code, run:

```bash
# 1. Format code
black src/ tests/ scripts/

# 2. Check linting
flake8 src/ tests/ scripts/

# 3. Run tests
pytest

# 4. Check security (optional)
bandit -r src/
```

Or use the provided script:

```bash
./scripts/lint_all.sh
```

## IDE Integration

### VS Code

Install extensions:
- **Python** (ms-python.python)
- **Black Formatter** (ms-python.black-formatter)
- **Flake8** (ms-python.flake8)

Add to `.vscode/settings.json`:

```json
{
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true
}
```

### PyCharm

1. **Black**: Settings → Tools → Black → Enable on save
2. **Flake8**: Settings → Tools → External Tools → Add flake8
3. Set line length to 100 in both

## Continuous Integration

Our GitHub Actions workflow automatically runs these checks:

- Black formatting check
- Flake8 linting
- Bandit security scan
- pytest tests

See `.github/workflows/tests.yml` for details.

## Fixing Common Issues

### Black and flake8 conflicts

If Black and flake8 disagree, Black wins. Our flake8 config ignores conflicts:

```ini
ignore = E203, W503, E501
```

### Line too long

Black will automatically wrap lines at 100 characters. If it can't, you may need to:

```python
# Bad
result = some_function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# Good
result = some_function(
    arg1, arg2, arg3, arg4, arg5,
    arg6, arg7, arg8, arg9, arg10
)
```

### Import errors (flake8)

Organize imports with isort (installed with dev dependencies):

```bash
isort src/ tests/ scripts/
```

### Security warnings

Read the Bandit warning carefully. If it's a false positive:

1. Add explanation comment
2. Add `# nosec BXXX` with the check number

```python
# This is safe because we validate input
# nosec B602
subprocess.call(command)
```

## Troubleshooting

### Black not found

```bash
pip install black
# or
pip install -e ".[dev]"
```

### Flake8 giving too many errors

Focus on critical errors first:

```bash
# Only show errors (not warnings)
flake8 src/ --select=E9,F63,F7,F82
```

### Bandit finding too many issues

Focus on high severity:

```bash
bandit -r src/ -ll  # Low confidence, low severity threshold
```

## Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [MyPy Documentation](https://mypy.readthedocs.io/)
- [PEP 8 Style Guide](https://pep8.org/)

## Summary

Always run before committing:

1. `black src/ tests/ scripts/` - Format code
2. `flake8 src/ tests/ scripts/` - Check style
3. `pytest` - Run tests
4. Optional: `bandit -r src/` - Security check

Or simply: `./scripts/lint_all.sh`
