"""
Model configuration utilities.
"""
from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """
    Configuration for TransformerLM model.
    
    This provides a convenient way to define and save model configurations.
    """
    vocab_size: int
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    positional_encoding: str = "sinusoidal"
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert to dictionary."""
        return self.__dict__


# Predefined configurations for common model sizes
TINY_CONFIG = ModelConfig(
    vocab_size=1000,
    d_model=128,
    num_layers=2,
    num_heads=4,
    d_ff=512,
    max_seq_len=256,
    dropout=0.1,
)

SMALL_CONFIG = ModelConfig(
    vocab_size=5000,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    max_seq_len=512,
    dropout=0.1,
)

MEDIUM_CONFIG = ModelConfig(
    vocab_size=10000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    max_seq_len=1024,
    dropout=0.1,
)

LARGE_CONFIG = ModelConfig(
    vocab_size=50000,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    max_seq_len=2048,
    dropout=0.1,
)
