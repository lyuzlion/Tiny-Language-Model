"""
Utilities for working with common text datasets.
"""
import os
from typing import List


def load_text_file(filepath: str, encoding: str = "utf-8") -> List[str]:
    """
    Load a text file and return lines as a list.
    
    Args:
        filepath: Path to text file
        encoding: File encoding
    
    Returns:
        List of text lines
    """
    with open(filepath, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_directory_text_files(dirpath: str, extension: str = ".txt", encoding: str = "utf-8") -> List[str]:
    """
    Load all text files from a directory.
    
    Args:
        dirpath: Path to directory
        extension: File extension to filter
        encoding: File encoding
    
    Returns:
        List of text lines from all files
    """
    all_lines = []
    
    for filename in os.listdir(dirpath):
        if filename.endswith(extension):
            filepath = os.path.join(dirpath, filename)
            lines = load_text_file(filepath, encoding)
            all_lines.extend(lines)
    
    return all_lines


def split_train_val(texts: List[str], val_ratio: float = 0.1, shuffle: bool = True) -> tuple:
    """
    Split texts into training and validation sets.
    
    Args:
        texts: List of text strings
        val_ratio: Ratio of validation data
        shuffle: Whether to shuffle before splitting
    
    Returns:
        Tuple of (train_texts, val_texts)
    """
    import random
    
    if shuffle:
        texts = texts.copy()
        random.shuffle(texts)
    
    val_size = int(len(texts) * val_ratio)
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]
    
    return train_texts, val_texts
