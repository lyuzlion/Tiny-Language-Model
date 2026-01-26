"""
Byte Pair Encoding (BPE) tokenizer implementation from scratch.
"""
from typing import List, Dict, Tuple
from collections import Counter
import re


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer built entirely from scratch.
    
    BPE is a subword tokenization algorithm that iteratively merges the most
    frequent pairs of bytes/characters in the training data.
    """
    
    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def _get_pairs(self, word: List[str]) -> Dict[Tuple[str, str], int]:
        """
        Get all adjacent pairs in a word.
        
        Args:
            word: List of characters/tokens
        
        Returns:
            Dictionary of pairs and their counts
        """
        pairs = {}
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs
    
    def build_vocab(self, texts: List[str]):
        """
        Build BPE vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        # Initialize vocabulary with special tokens and characters
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Get all characters
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Initialize vocab with special tokens and characters
        vocab = special_tokens + sorted(chars)
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        
        # Split texts into words (simple whitespace tokenization)
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            word_freqs.update(words)
        
        # Convert words to character sequences with end token
        word_splits = {}
        for word, freq in word_freqs.items():
            word_splits[word] = (list(word) + ['</w>'], freq)
        
        # Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            # Count all pairs
            pair_counts = Counter()
            for word, (split, freq) in word_splits.items():
                pairs = self._get_pairs(split)
                for pair, count in pairs.items():
                    pair_counts[pair] += count * freq
            
            if not pair_counts:
                break
            
            # Get most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            # Merge the pair
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            
            # Update word splits
            for word in word_splits:
                split, freq = word_splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = (new_split, freq)
    
    def _apply_merges(self, word: List[str]) -> List[str]:
        """
        Apply learned merges to a word.
        
        Args:
            word: List of characters
        
        Returns:
            List of tokens after applying merges
        """
        for merge_pair in self.merges:
            new_token = ''.join(merge_pair)
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == merge_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word
    
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
            token_ids.append(self.vocab[self.bos_token])
        
        # Split into words
        words = text.split()
        for word in words:
            # Convert to character list with end token
            chars = list(word) + ['</w>']
            
            # Apply merges
            tokens = self._apply_merges(chars)
            
            # Convert to IDs
            for token in tokens:
                token_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text string
        """
        # Create reverse mapping
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        special_tokens = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        tokens = []
        
        for token_id in token_ids:
            token = id_to_token.get(token_id, self.unk_token)
            if skip_special_tokens and token in special_tokens:
                continue
            tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath: str):
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save file
        """
        import json
        
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to load file
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data["vocab"]
        self.merges = [tuple(merge) for merge in data["merges"]]
        self.vocab_size = data["vocab_size"]
