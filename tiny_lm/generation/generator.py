"""
Text generation utilities from scratch.
"""
import torch
import torch.nn as nn


class Generator:
    """
    Text generator with various decoding strategies.
    
    Implements greedy decoding, top-k sampling, top-p (nucleus) sampling,
    and beam search, all built from scratch.
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        """
        Args:
            model: The language model
            tokenizer: Tokenizer instance
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generate text using greedy decoding.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                # Get most likely next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        generated_ids = input_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def generate_sampling(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> str:
        """
        Generate text using sampling with temperature, top-k, and top-p.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, only sample from tokens with cumulative prob >= top_p
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt, add_special_tokens=True),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                # Get logits for last token and apply temperature
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep at least one token
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Decode
        generated_ids = input_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def generate_beam_search(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        length_penalty: float = 1.0,
    ) -> str:
        """
        Generate text using beam search.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            num_beams: Number of beams
            length_penalty: Length penalty (> 1.0 favors longer sequences)
        
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_ids)
        
        # Initialize beams: (sequence, score)
        beams = [(prompt_ids, 0.0)]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                all_candidates = []
                
                for seq, score in beams:
                    # Convert to tensor
                    input_ids = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids)
                    logits = outputs["logits"]
                    
                    # Get log probabilities for last token
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    
                    # Get top k candidates
                    topk_log_probs, topk_indices = torch.topk(log_probs, num_beams)
                    
                    # Add candidates
                    for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                        candidate_seq = seq + [idx.item()]
                        candidate_score = score + log_prob.item()
                        
                        # Apply length penalty
                        normalized_score = candidate_score / (len(candidate_seq) ** length_penalty)
                        
                        all_candidates.append((candidate_seq, candidate_score, normalized_score))
                
                # Select top num_beams candidates
                beams = sorted(all_candidates, key=lambda x: x[2], reverse=True)[:num_beams]
                beams = [(seq, score) for seq, score, _ in beams]
        
        # Return best sequence
        best_seq = beams[0][0]
        generated_text = self.tokenizer.decode(best_seq, skip_special_tokens=True)
        
        return generated_text
