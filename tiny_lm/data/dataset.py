"""
Dataset utilities for language model training.
"""
import torch
from torch.utils.data import Dataset
from typing import List


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Loads text data and tokenizes it for autoregressive language modeling.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance (CharTokenizer or BPETokenizer)
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
        self.encoded_texts = []
        for text in texts:
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            self.encoded_texts.append(token_ids)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        token_ids = self.encoded_texts[idx]
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # For language modeling, input is shifted by 1 from labels
        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StreamingTextDataset(Dataset):
    """
    Streaming text dataset that loads data from a file.
    
    Useful for large datasets that don't fit in memory.
    """
    
    def __init__(
        self,
        filepath: str,
        tokenizer,
        max_length: int = 512,
        encoding: str = "utf-8",
    ):
        """
        Args:
            filepath: Path to text file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            encoding: File encoding
        """
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoding = encoding
        
        # Read file and split into lines
        with open(filepath, 'r', encoding=encoding) as f:
            self.lines = [line.strip() for line in f if line.strip()]
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.lines)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        text = self.lines[idx]
        token_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # For language modeling, input is shifted by 1 from labels
        input_ids = token_ids[:-1]
        labels = token_ids[1:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[dict]) -> dict:
    """
    Collate function for batching samples with padding.
    
    Args:
        batch: List of samples from dataset
    
    Returns:
        Dictionary with batched and padded tensors
    """
    # Get max length in batch
    max_input_len = max(item["input_ids"].size(0) for item in batch)
    max_label_len = max(item["labels"].size(0) for item in batch)
    max_len = max(max_input_len, max_label_len)
    
    # Pad sequences
    input_ids = []
    labels = []
    
    for item in batch:
        input_id = item["input_ids"]
        label = item["labels"]
        
        # Pad input_ids with 0 (assuming 0 is PAD token)
        input_padding = torch.zeros(max_len - input_id.size(0), dtype=torch.long)
        input_ids.append(torch.cat([input_id, input_padding]))
        
        # Pad labels with -100 (ignore_index in cross_entropy)
        label_padding = torch.full((max_len - label.size(0),), -100, dtype=torch.long)
        labels.append(torch.cat([label, label_padding]))
    
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }
