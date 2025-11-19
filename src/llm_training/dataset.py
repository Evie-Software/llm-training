"""
Dataset preparation for markdown (.md and .mdx) files using MLX.
Handles parsing, cleaning, and tokenization of documentation files.
"""

import os
import re
import pickle  # nosec B403  # Used for saving/loading user's own processed datasets
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
from transformers import PreTrainedTokenizer
import markdown
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarkdownParser:
    """Parse and clean markdown files."""

    @staticmethod
    def parse_mdx(content: str) -> str:
        """
        Parse MDX content by removing JSX/React components and extracting text.

        Args:
            content: Raw MDX file content

        Returns:
            Cleaned markdown content
        """
        # Remove JSX imports
        content = re.sub(
            r"^import\s+.*?from\s+['\"].*?['\"];?\s*$", "", content, flags=re.MULTILINE
        )

        # Remove JSX export statements
        content = re.sub(r"^export\s+.*?;?\s*$", "", content, flags=re.MULTILINE)

        # Remove self-closing JSX tags
        content = re.sub(r"<\w+\s+[^>]*?/>", "", content)

        # Remove JSX component tags (opening and closing)
        content = re.sub(r"</?[A-Z]\w*[^>]*>", "", content)

        # Remove inline JSX expressions
        content = re.sub(r"\{[^}]*\}", "", content)

        return content

    @staticmethod
    def markdown_to_text(markdown_content: str) -> str:
        """
        Convert markdown to plain text while preserving structure.

        Args:
            markdown_content: Markdown formatted text

        Returns:
            Plain text with preserved structure
        """
        # Convert markdown to HTML
        html = markdown.markdown(markdown_content)

        # Parse HTML and extract text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = text.strip()

        return text

    @staticmethod
    def clean_markdown(content: str, is_mdx: bool = False) -> str:
        """
        Clean and normalize markdown content.

        Args:
            content: Raw file content
            is_mdx: Whether the content is MDX format

        Returns:
            Cleaned markdown text
        """
        if is_mdx:
            content = MarkdownParser.parse_mdx(content)

        # Remove HTML comments
        content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

        # Remove code block language specifiers but keep code
        content = re.sub(r"```(\w+)\n", "```\n", content)

        # Normalize whitespace
        content = re.sub(r" +", " ", content)
        content = re.sub(r"\n\n+", "\n\n", content)

        return content.strip()


class MarkdownDataset:
    """Dataset for markdown files compatible with MLX."""

    def __init__(
        self,
        file_paths: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Initialize the dataset.

        Args:
            file_paths: List of paths to markdown files
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            stride: Stride for overlapping chunks (for long documents)
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.parser = MarkdownParser()
        self.samples = []

        logger.info(f"Loading {len(file_paths)} files...")
        self._process_files()
        logger.info(f"Created {len(self.samples)} training samples")

    def _process_files(self):
        """Process all files and create training samples."""
        for file_path in self.file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Determine if MDX
                is_mdx = file_path.endswith(".mdx")

                # Clean content
                cleaned = self.parser.clean_markdown(content, is_mdx=is_mdx)

                # Tokenize and create chunks
                tokens = self.tokenizer.encode(cleaned, add_special_tokens=True)

                # Create overlapping chunks for long documents
                for i in range(0, len(tokens), self.stride):
                    chunk = tokens[i : i + self.max_length]

                    if len(chunk) > 10:  # Skip very short chunks
                        # Pad if necessary
                        if len(chunk) < self.max_length:
                            chunk = chunk + [self.tokenizer.pad_token_id] * (
                                self.max_length - len(chunk)
                            )

                        # Create attention mask
                        attention_mask = [
                            1 if token != self.tokenizer.pad_token_id else 0 for token in chunk
                        ]

                        self.samples.append(
                            {
                                "input_ids": np.array(chunk, dtype=np.int32),
                                "attention_mask": np.array(attention_mask, dtype=np.int32),
                                "source_file": file_path,
                            }
                        )

            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get a single sample."""
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["input_ids"],  # For language modeling
        }

    def batch_iterate(self, batch_size: int, shuffle: bool = False):
        """
        Iterate over batches of data.

        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data

        Yields:
            Dictionary containing batched arrays
        """
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = {
                "input_ids": np.stack([self.samples[idx]["input_ids"] for idx in batch_indices]),
                "attention_mask": np.stack(
                    [self.samples[idx]["attention_mask"] for idx in batch_indices]
                ),
                "labels": np.stack([self.samples[idx]["input_ids"] for idx in batch_indices]),
            }
            yield batch


def collect_markdown_files(
    data_dir: str,
    extensions: List[str] = [".md", ".mdx"],
    exclude_patterns: List[str] = ["node_modules", ".git", "venv"],
) -> List[str]:
    """
    Recursively collect all markdown files from a directory.

    Args:
        data_dir: Directory to search
        extensions: File extensions to include
        exclude_patterns: Directory patterns to exclude

    Returns:
        List of file paths
    """
    file_paths = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    for ext in extensions:
        for file_path in data_path.rglob(f"*{ext}"):
            # Check if file should be excluded
            should_exclude = any(pattern in str(file_path) for pattern in exclude_patterns)

            if not should_exclude:
                file_paths.append(str(file_path))

    logger.info(f"Found {len(file_paths)} markdown files in {data_dir}")
    return file_paths


def prepare_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    train_split: float = 0.9,
    validation_split: float = 0.05,
    seed: int = 42,
    extensions: List[str] = [".md", ".mdx"],
) -> Tuple[MarkdownDataset, MarkdownDataset, MarkdownDataset]:
    """
    Prepare train, validation, and test datasets from markdown files.

    Args:
        data_dir: Directory containing markdown files
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        train_split: Proportion of data for training
        validation_split: Proportion of data for validation
        seed: Random seed for reproducibility
        extensions: File extensions to include

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Collect files
    file_paths = collect_markdown_files(data_dir, extensions=extensions)

    if len(file_paths) == 0:
        raise ValueError(f"No markdown files found in {data_dir}")

    # Shuffle files
    np.random.seed(seed)
    np.random.shuffle(file_paths)

    # Split files
    n_train = int(len(file_paths) * train_split)
    n_val = int(len(file_paths) * validation_split)

    train_files = file_paths[:n_train]
    val_files = file_paths[n_train : n_train + n_val]
    test_files = file_paths[n_train + n_val :]

    logger.info(
        f"Dataset split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test"
    )

    # Create datasets
    train_dataset = MarkdownDataset(train_files, tokenizer, max_length=max_length)
    val_dataset = MarkdownDataset(val_files, tokenizer, max_length=max_length)
    test_dataset = MarkdownDataset(test_files, tokenizer, max_length=max_length)

    return train_dataset, val_dataset, test_dataset


def save_processed_dataset(dataset: MarkdownDataset, output_path: str):
    """Save processed dataset to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset.samples, f)
    logger.info(f"Dataset saved to {output_path}")


def load_processed_dataset(
    input_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
) -> MarkdownDataset:
    """Load processed dataset from disk."""
    with open(input_path, "rb") as f:
        samples = pickle.load(f)  # nosec B301  # Loading user's own processed data
    dataset = MarkdownDataset([], tokenizer, max_length=max_length)
    dataset.samples = samples
    logger.info(f"Dataset loaded from {input_path}")
    return dataset
