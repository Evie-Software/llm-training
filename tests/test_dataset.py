"""Tests for dataset module."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_training.dataset import (
    MarkdownParser,
    MarkdownDataset,
    collect_markdown_files,
)


class TestMarkdownParser:
    """Test MarkdownParser class."""

    def test_clean_markdown_basic(self):
        """Test basic markdown cleaning."""
        parser = MarkdownParser()
        content = "# Hello\n\nThis is  a   test\n\n\n\nWith extra spaces"
        cleaned = parser.clean_markdown(content)

        assert "# Hello" in cleaned
        assert "This is a test" in cleaned
        assert "\n\n\n" not in cleaned

    def test_parse_mdx_imports(self):
        """Test MDX import removal."""
        parser = MarkdownParser()
        content = """
import { Component } from '@/components'
import React from 'react'

# Hello World

Some content
"""
        cleaned = parser.parse_mdx(content)

        assert "import" not in cleaned
        assert "# Hello World" in cleaned
        assert "Some content" in cleaned

    def test_parse_mdx_components(self):
        """Test MDX component removal."""
        parser = MarkdownParser()
        content = """
# Title

<Component prop="value" />

Some text

<AnotherComponent>
  Content
</AnotherComponent>

More text
"""
        cleaned = parser.parse_mdx(content)

        assert "Component" not in cleaned
        assert "# Title" in cleaned
        assert "Some text" in cleaned
        assert "More text" in cleaned

    def test_parse_mdx_expressions(self):
        """Test MDX expression removal."""
        parser = MarkdownParser()
        content = "Text {variable} more text {expression}"
        cleaned = parser.parse_mdx(content)

        assert "variable" not in cleaned
        assert "expression" not in cleaned
        assert "Text" in cleaned
        assert "more text" in cleaned

    def test_remove_html_comments(self):
        """Test HTML comment removal."""
        parser = MarkdownParser()
        content = """
# Title

<!-- This is a comment -->

Content

<!-- Multi-line
comment
-->

More content
"""
        cleaned = parser.clean_markdown(content)

        assert "comment" not in cleaned.lower()
        assert "# Title" in cleaned
        assert "Content" in cleaned
        assert "More content" in cleaned

    def test_markdown_to_text(self):
        """Test markdown to text conversion."""
        parser = MarkdownParser()
        markdown = """
# Heading

**Bold text** and *italic text*

- List item 1
- List item 2

[Link](http://example.com)
"""
        text = parser.markdown_to_text(markdown)

        assert "Heading" in text
        assert "Bold text" in text
        assert "italic text" in text
        assert "List item 1" in text


class TestMarkdownDataset:
    """Test MarkdownDataset class."""

    @pytest.fixture
    def sample_files(self):
        """Create sample markdown files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample files
            file1 = os.path.join(tmpdir, "doc1.md")
            file2 = os.path.join(tmpdir, "doc2.md")

            with open(file1, "w") as f:
                f.write("# Document 1\n\nThis is the first document with some content.")

            with open(file2, "w") as f:
                f.write("# Document 2\n\nThis is the second document with more content.")

            yield [file1, file2]

    def test_dataset_creation(self, sample_files):
        """Test creating a dataset from markdown files."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = MarkdownDataset(
            file_paths=sample_files,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Check dataset was created
        assert len(dataset) > 0

        # Check sample structure
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        # Check tensor shapes
        assert sample["input_ids"].shape[0] == 128
        assert sample["attention_mask"].shape[0] == 128

    def test_dataset_getitem(self, sample_files):
        """Test getting items from dataset."""
        from transformers import AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = MarkdownDataset(
            file_paths=sample_files,
            tokenizer=tokenizer,
            max_length=128,
        )

        # Get first item
        item = dataset[0]

        # Check types
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

        # Check dtypes
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["labels"].dtype == torch.long


class TestCollectMarkdownFiles:
    """Test collect_markdown_files function."""

    def test_collect_files(self):
        """Test collecting markdown files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            Path(tmpdir, "subdir").mkdir()

            # Create files
            (Path(tmpdir) / "file1.md").write_text("# File 1")
            (Path(tmpdir) / "file2.mdx").write_text("# File 2")
            (Path(tmpdir) / "subdir" / "file3.md").write_text("# File 3")
            (Path(tmpdir) / "ignore.txt").write_text("Should be ignored")

            # Collect files
            files = collect_markdown_files(tmpdir)

            # Check results
            assert len(files) == 3
            assert all(f.endswith((".md", ".mdx")) for f in files)

    def test_exclude_patterns(self):
        """Test excluding patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            Path(tmpdir, "node_modules").mkdir()
            Path(tmpdir, "valid").mkdir()

            # Create files
            (Path(tmpdir) / "valid" / "file1.md").write_text("# File 1")
            (Path(tmpdir) / "node_modules" / "file2.md").write_text("# File 2")

            # Collect files
            files = collect_markdown_files(tmpdir, exclude_patterns=["node_modules"])

            # Check that node_modules was excluded
            assert len(files) == 1
            assert "node_modules" not in files[0]

    def test_custom_extensions(self):
        """Test custom file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "file1.md").write_text("# File 1")
            (Path(tmpdir) / "file2.mdx").write_text("# File 2")
            (Path(tmpdir) / "file3.txt").write_text("# File 3")

            # Collect only .md files
            files = collect_markdown_files(tmpdir, extensions=[".md"])

            # Check results
            assert len(files) == 1
            assert files[0].endswith(".md")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
