"""
Tiny Language Model - Train small language models from scratch using native PyTorch.

This package provides a complete implementation of language model training
without relying on abstract interfaces from third-party libraries.
"""

__version__ = "0.1.0"

from tiny_lm.model.transformer import TransformerLM
from tiny_lm.tokenizer.char_tokenizer import CharTokenizer
from tiny_lm.tokenizer.bpe_tokenizer import BPETokenizer

__all__ = [
    "TransformerLM",
    "CharTokenizer",
    "BPETokenizer",
]
