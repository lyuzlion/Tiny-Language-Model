"""
Complete Transformer language model implementation from scratch.
"""
import torch
import torch.nn as nn
from tiny_lm.model.attention import MultiHeadAttention
from tiny_lm.model.feedforward import FeedForward
from tiny_lm.model.positional import PositionalEncoding, LearnedPositionalEncoding


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feedforward layers.
    
    Implements: LayerNorm(x + SelfAttention(x)) -> LayerNorm(x + FFN(x))
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feedforward network
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class TransformerLM(nn.Module):
    """
    Transformer-based language model built entirely from scratch.
    
    This model implements a decoder-only transformer architecture suitable
    for autoregressive language modeling tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model embeddings
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            d_ff: Dimension of feedforward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            positional_encoding: Type of positional encoding ("sinusoidal" or "learned")
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        elif positional_encoding == "learned":
            self.positional_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal (autoregressive) attention mask.
        
        Args:
            seq_len: Sequence length
            device: Device to create mask on
        
        Returns:
            Causal mask of shape (seq_len, seq_len) where mask[i, j] = 1 if j > i
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            labels: Optional labels for computing loss, shape (batch_size, seq_len)
        
        Returns:
            Dictionary containing:
                - logits: Output logits of shape (batch_size, seq_len, vocab_size)
                - loss: Cross-entropy loss if labels provided, otherwise None
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_embedding(input_ids) * (self.d_model ** 0.5)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, input_ids.device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Get logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Initial token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from tokens with cumulative probability >= top_p
        
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs["logits"]
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
