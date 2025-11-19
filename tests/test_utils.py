"""Tests for utils module."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_training.utils import (
    format_number,
    check_mlx_availability,
    cleanup_checkpoints,
    count_parameters,
)


class TestFormatNumber:
    """Test format_number function."""

    def test_small_numbers(self):
        """Test formatting small numbers."""
        assert format_number(1) == "1"
        assert format_number(10) == "10"
        assert format_number(999) == "999"

    def test_thousands(self):
        """Test formatting thousands."""
        assert format_number(1000) == "1.0K"
        assert format_number(1500) == "1.5K"
        assert format_number(999000) == "999.0K"

    def test_millions(self):
        """Test formatting millions."""
        assert format_number(1_000_000) == "1.0M"
        assert format_number(1_500_000) == "1.5M"
        assert format_number(124_000_000) == "124.0M"

    def test_billions(self):
        """Test formatting billions."""
        assert format_number(1_000_000_000) == "1.0B"
        assert format_number(1_500_000_000) == "1.5B"
        assert format_number(7_000_000_000) == "7.0B"


class TestCheckMLXAvailability:
    """Test check_mlx_availability function."""

    def test_mlx_info_structure(self):
        """Test that MLX info has correct structure."""
        info = check_mlx_availability()

        # Check required keys
        assert "mlx_available" in info
        assert "mlx_version" in info
        assert "device" in info

        # Check types
        assert isinstance(info["mlx_available"], bool)
        assert isinstance(info["mlx_version"], str)
        assert isinstance(info["device"], str)

    def test_mlx_working_field(self):
        """Test mlx_working field."""
        info = check_mlx_availability()

        # mlx_working should be present
        assert "mlx_working" in info
        assert isinstance(info["mlx_working"], bool)

        # If MLX is not working, there might be an error message
        if not info["mlx_working"]:
            # This is expected on non-Apple Silicon machines
            assert "error" in info


class TestCleanupCheckpoints:
    """Test cleanup_checkpoints function."""

    def test_cleanup_with_no_checkpoints(self):
        """Test cleanup when no checkpoints exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise any errors
            cleanup_checkpoints(tmpdir, keep_last_n=2, dry_run=True)

    def test_cleanup_dry_run(self):
        """Test cleanup in dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            for i in range(5):
                checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{i*100}")
                os.makedirs(checkpoint_dir)

            # Dry run should not delete anything
            cleanup_checkpoints(tmpdir, keep_last_n=2, dry_run=True)

            # Check all checkpoints still exist
            checkpoints = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")]
            assert len(checkpoints) == 5

    def test_cleanup_actual(self):
        """Test actual cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            for i in range(5):
                checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{i*100}")
                os.makedirs(checkpoint_dir)

            # Cleanup, keeping only last 2
            cleanup_checkpoints(tmpdir, keep_last_n=2, dry_run=False)

            # Check only 2 checkpoints remain
            checkpoints = sorted([d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")])
            assert len(checkpoints) == 2

            # Check that the last 2 are kept
            assert "checkpoint-300" in checkpoints
            assert "checkpoint-400" in checkpoints

    def test_cleanup_keep_all(self):
        """Test cleanup when keeping more than exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 2 checkpoint directories
            for i in range(2):
                checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{i*100}")
                os.makedirs(checkpoint_dir)

            # Try to keep 5 (more than exist)
            cleanup_checkpoints(tmpdir, keep_last_n=5, dry_run=False)

            # Check all checkpoints still exist
            checkpoints = [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")]
            assert len(checkpoints) == 2


class TestCountParameters:
    """Test count_parameters function with MLX models."""

    def test_count_parameters_mock(self):
        """Test parameter counting with mock MLX model."""
        # Create a mock model object that mimics MLX model structure
        class MockModel:
            def __init__(self):
                self._params = {
                    "layer1.weight": type("Param", (), {"size": 200})(),
                    "layer1.bias": type("Param", (), {"size": 20})(),
                    "layer2.weight": type("Param", (), {"size": 100})(),
                    "layer2.bias": type("Param", (), {"size": 5})(),
                }

            def parameters(self):
                return self._params

            def trainable_parameters(self):
                return self._params

        model = MockModel()
        counts = count_parameters(model)

        # Check structure
        assert "total" in counts
        assert "trainable" in counts
        assert "frozen" in counts
        assert "trainable_percent" in counts

        # Check values
        assert counts["total"] == 325  # 200 + 20 + 100 + 5
        assert counts["trainable"] == 325
        assert counts["frozen"] == 0
        assert counts["trainable_percent"] == 100.0

    def test_count_parameters_with_frozen_mock(self):
        """Test parameter counting with frozen parameters using mock."""

        class MockModelWithFrozen:
            def __init__(self):
                self._all_params = {
                    "layer1.weight": type("Param", (), {"size": 200})(),
                    "layer1.bias": type("Param", (), {"size": 20})(),
                    "layer2.weight": type("Param", (), {"size": 100})(),
                    "layer2.bias": type("Param", (), {"size": 5})(),
                }
                # Only layer2 is trainable
                self._trainable_params = {
                    "layer2.weight": type("Param", (), {"size": 100})(),
                    "layer2.bias": type("Param", (), {"size": 5})(),
                }

            def parameters(self):
                return self._all_params

            def trainable_parameters(self):
                return self._trainable_params

        model = MockModelWithFrozen()
        counts = count_parameters(model)

        # Check that frozen parameters are counted correctly
        assert counts["total"] == 325
        assert counts["trainable"] == 105  # Only layer2
        assert counts["frozen"] == 220  # layer1
        assert counts["trainable_percent"] < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
