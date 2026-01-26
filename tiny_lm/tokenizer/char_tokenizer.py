"""
Character-level tokenizer implementation from scratch.
"""
from typing import List, Dict


class CharTokenizer:
    """
    Simple character-level tokenizer built from scratch.
    
    Each character is mapped to a unique integer ID. This is the simplest
    form of tokenization and works well for small datasets or educational purposes.
    """
    
    def __init__(self):
        """Initialize an empty tokenizer."""
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        # Start with special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Collect all unique characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Sort for deterministic ordering
        chars = sorted(chars)
        
        # Build mappings
        all_tokens = special_tokens + chars
        self.char_to_id = {char: idx for idx, char in enumerate(all_tokens)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text string to encode
            add_special_tokens: Whether to add BOS and EOS tokens
        
        Returns:
            List of token IDs
        """
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.char_to_id[self.bos_token])
        
        for char in text:
            token_ids.append(self.char_to_id.get(char, self.char_to_id[self.unk_token]))
        
        if add_special_tokens:
            token_ids.append(self.char_to_id[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            Decoded text string
        """
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        chars = []
        
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, self.unk_token)
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size
    
    def save(self, filepath: str):
        """
        Save tokenizer vocabulary to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        vocab_data = {
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """
        Load tokenizer vocabulary from file.
        
        Args:
            filepath: Path to load file
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_id = vocab_data["char_to_id"]
        self.vocab_size = vocab_data["vocab_size"]
        self.id_to_char = {int(idx): char for char, idx in self.char_to_id.items()}
