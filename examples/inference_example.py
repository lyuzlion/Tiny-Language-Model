"""
Example inference script for Tiny Language Model.

This script demonstrates how to load a trained model and generate text
using different sampling strategies.
"""
import torch
from tiny_lm.model.transformer import TransformerLM
from tiny_lm.tokenizer.char_tokenizer import CharTokenizer
from tiny_lm.generation.generator import Generator


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./outputs/best_model.pt"
    tokenizer_path = "./outputs/tokenizer.json"
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Initialize model
    print("Loading model...")
    model = TransformerLM(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1,
        positional_encoding="sinusoidal",
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Initialize generator
    generator = Generator(model, tokenizer, device)
    
    # Example prompts
    prompts = [
        "The quick",
        "Machine learning",
        "Deep learning models",
    ]
    
    print("\n" + "=" * 80)
    print("Generating text with different strategies:")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        print("-" * 80)
        
        # Greedy decoding
        print("\n1. Greedy decoding:")
        generated = generator.generate_greedy(prompt, max_new_tokens=30)
        print(f"   {generated}")
        
        # Sampling with temperature
        print("\n2. Sampling (temperature=0.8):")
        generated = generator.generate_sampling(
            prompt, max_new_tokens=30, temperature=0.8
        )
        print(f"   {generated}")
        
        # Top-k sampling
        print("\n3. Top-k sampling (k=10):")
        generated = generator.generate_sampling(
            prompt, max_new_tokens=30, temperature=1.0, top_k=10
        )
        print(f"   {generated}")
        
        # Top-p sampling
        print("\n4. Top-p sampling (p=0.9):")
        generated = generator.generate_sampling(
            prompt, max_new_tokens=30, temperature=1.0, top_p=0.9
        )
        print(f"   {generated}")
        
        # Beam search
        print("\n5. Beam search (beams=3):")
        generated = generator.generate_beam_search(
            prompt, max_new_tokens=30, num_beams=3
        )
        print(f"   {generated}")
        
        print()


if __name__ == "__main__":
    main()
