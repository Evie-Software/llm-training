#!/bin/bash

# LLM Training Cleanup Script
# This script helps manage disk space by cleaning up training artifacts

set -e  # Exit on error

echo "=================================="
echo "LLM Training Cleanup Utility"
echo "=================================="
echo ""

# Function to get directory size
get_dir_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0B"
    fi
}

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Show current disk usage
echo "Current disk usage by directory:"
echo "  Checkpoints: $(get_dir_size 'checkpoints')"
echo "  Logs: $(get_dir_size 'logs')"
echo "  Models: $(get_dir_size 'models')"
echo "  Processed data: $(get_dir_size 'data/processed')"
echo "  Cache (__pycache__): $(find . -type d -name '__pycache__' -exec du -sh {} + 2>/dev/null | awk '{sum+=$1} END {print sum "B"}' || echo "0B")"
echo ""

# Cleanup options
echo "What would you like to clean up?"
echo ""

# Clean checkpoints
if [ -d "checkpoints" ] && [ "$(ls -A checkpoints)" ]; then
    if confirm "1. Remove training checkpoints ($(get_dir_size 'checkpoints'))?"; then
        rm -rf checkpoints/*
        echo "   ✓ Checkpoints cleaned"
    fi
fi

# Clean logs
if [ -d "logs" ] && [ "$(ls -A logs)" ]; then
    if confirm "2. Remove training logs ($(get_dir_size 'logs'))?"; then
        rm -rf logs/*
        echo "   ✓ Logs cleaned"
    fi
fi

# Clean processed data
if [ -d "data/processed" ] && [ "$(ls -A data/processed)" ]; then
    if confirm "3. Remove processed datasets ($(get_dir_size 'data/processed'))?"; then
        rm -rf data/processed/*
        echo "   ✓ Processed data cleaned"
    fi
fi

# Clean Python cache
if confirm "4. Remove Python cache files (__pycache__, .pyc)?"; then
    find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name '*.pyc' -delete 2>/dev/null || true
    find . -type f -name '*.pyo' -delete 2>/dev/null || true
    echo "   ✓ Python cache cleaned"
fi

# Clean pytest cache
if [ -d ".pytest_cache" ]; then
    if confirm "5. Remove pytest cache?"; then
        rm -rf .pytest_cache
        echo "   ✓ Pytest cache cleaned"
    fi
fi

# Clean wandb
if [ -d "wandb" ]; then
    if confirm "6. Remove wandb logs ($(get_dir_size 'wandb'))?"; then
        rm -rf wandb
        echo "   ✓ Wandb logs cleaned"
    fi
fi

# Clean tensorboard
if [ -d "runs" ]; then
    if confirm "7. Remove tensorboard runs ($(get_dir_size 'runs'))?"; then
        rm -rf runs
        echo "   ✓ Tensorboard runs cleaned"
    fi
fi

# Deep clean option
echo ""
if confirm "8. DEEP CLEAN: Remove ALL artifacts including models (DESTRUCTIVE)?"; then
    echo "   This will remove: checkpoints, logs, models, processed data, and all caches"
    if confirm "   Are you absolutely sure?"; then
        rm -rf checkpoints/* logs/* models/* data/processed/* wandb runs .pytest_cache
        find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name '*.pyc' -delete 2>/dev/null || true
        echo "   ✓ Deep clean completed"
    fi
fi

echo ""
echo "=================================="
echo "Cleanup completed!"
echo "=================================="
echo ""
echo "New disk usage:"
echo "  Checkpoints: $(get_dir_size 'checkpoints')"
echo "  Logs: $(get_dir_size 'logs')"
echo "  Models: $(get_dir_size 'models')"
echo "  Processed data: $(get_dir_size 'data/processed')"
echo ""
