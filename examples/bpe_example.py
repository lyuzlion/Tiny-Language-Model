"""
Example script demonstrating BPE tokenizer training and usage.
"""
from tiny_lm.tokenizer.bpe_tokenizer import BPETokenizer


def main():
    # Sample texts for training BPE tokenizer
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of training data.",
        "Natural language processing enables computers to understand text.",
        "Transformers have revolutionized the field of NLP.",
    ]
    
    # Initialize and train BPE tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=200)
    tokenizer.build_vocab(texts)
    
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test encoding and decoding
    test_text = "Machine learning enables computers to learn from data."
    print(f"\nOriginal text: {test_text}")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    
    # Save tokenizer
    tokenizer.save("bpe_tokenizer.json")
    print("\nTokenizer saved to bpe_tokenizer.json")
    
    # Load tokenizer
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("bpe_tokenizer.json")
    print("Tokenizer loaded successfully!")
    
    # Test loaded tokenizer
    decoded_text2 = new_tokenizer.decode(token_ids)
    print(f"Decoded with loaded tokenizer: {decoded_text2}")


if __name__ == "__main__":
    main()
