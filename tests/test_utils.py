"""Tests for utils module."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_training.utils import (
    format_number,
    check_mps_availability,
    cleanup_checkpoints,
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


class TestCheckMPSAvailability:
    """Test check_mps_availability function."""

    def test_mps_info_structure(self):
        """Test that MPS info has correct structure."""
        info = check_mps_availability()

        # Check required keys
        assert "mps_available" in info
        assert "mps_built" in info
        assert "torch_version" in info

        # Check types
        assert isinstance(info["mps_available"], bool)
        assert isinstance(info["mps_built"], bool)
        assert isinstance(info["torch_version"], str)

    def test_mps_working_field(self):
        """Test mps_working field."""
        info = check_mps_availability()

        # mps_working should be present
        assert "mps_working" in info
        assert isinstance(info["mps_working"], bool)

        # If MPS is not working, there might be an error message
        if not info["mps_working"] and not info["mps_available"]:
            # This is expected on non-Apple Silicon machines
            pass


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


class TestGetDevice:
    """Test get_device function."""

    def test_get_device(self):
        """Test getting device."""
        from llm_training.utils import get_device
        import torch

        # Get device
        device = get_device()

        # Should be one of the valid types
        assert device.type in ["mps", "cuda", "cpu"]

        # Should be a torch device
        assert isinstance(device, torch.device)


class TestCountParameters:
    """Test count_parameters function."""

    def test_count_parameters(self):
        """Test parameter counting."""
        from llm_training.utils import count_parameters
        import torch.nn as nn

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.Linear(20, 5),  # 20*5 + 5 = 105 params
        )

        counts = count_parameters(model)

        # Check structure
        assert "total" in counts
        assert "trainable" in counts
        assert "frozen" in counts
        assert "trainable_percent" in counts

        # Check values
        assert counts["total"] == 325  # 220 + 105
        assert counts["trainable"] == 325
        assert counts["frozen"] == 0
        assert counts["trainable_percent"] == 100.0

    def test_count_parameters_with_frozen(self):
        """Test parameter counting with frozen parameters."""
        from llm_training.utils import count_parameters
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        counts = count_parameters(model)

        # Check that frozen parameters are counted correctly
        assert counts["total"] == 325
        assert counts["trainable"] == 105  # Only second layer
        assert counts["frozen"] == 220  # First layer
        assert counts["trainable_percent"] < 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
