#!/bin/bash

# LLM Training - Code Quality Check Script
# Runs all linting and formatting checks before committing

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$PROJECT_DIR"

echo "=================================="
echo "LLM Training - Code Quality Checks"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall success
OVERALL_SUCCESS=true

# Function to run a check
run_check() {
    local name=$1
    local command=$2

    echo "Running $name..."
    if eval "$command"; then
        echo -e "${GREEN}✓${NC} $name passed"
        echo ""
        return 0
    else
        echo -e "${RED}✗${NC} $name failed"
        echo ""
        OVERALL_SUCCESS=false
        return 1
    fi
}

# 1. Black formatting check
run_check "Black (code formatting)" \
    "black --check src/ tests/ scripts/ --line-length 100" \
    || echo -e "${YELLOW}Run: black src/ tests/ scripts/${NC}"

# 2. Flake8 linting
run_check "Flake8 (linting)" \
    "flake8 src/ tests/ scripts/" \
    || echo -e "${YELLOW}Fix linting errors manually${NC}"

# 3. Bandit security check (optional, can fail)
echo "Running Bandit (security checks)..."
if bandit -r src/ -c pyproject.toml -f screen -ll 2>&1; then
    echo -e "${GREEN}✓${NC} Bandit passed"
else
    echo -e "${YELLOW}⚠${NC} Bandit found potential issues (review manually)"
fi
echo ""

# 4. Tests
run_check "Pytest (tests)" \
    "pytest --tb=short -q" \
    || echo -e "${YELLOW}Fix failing tests${NC}"

# Summary
echo "=================================="
if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "Code is ready to commit."
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo "Please fix the issues above before committing."
    exit 1
fi
