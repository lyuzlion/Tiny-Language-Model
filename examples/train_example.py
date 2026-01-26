"""
Example training script for Tiny Language Model.

This script demonstrates how to train a small language model from scratch
using the tiny_lm library with native PyTorch implementations.
"""
import torch
from torch.utils.data import DataLoader
from tiny_lm.model.transformer import TransformerLM
from tiny_lm.tokenizer.char_tokenizer import CharTokenizer
from tiny_lm.data.dataset import TextDataset, collate_fn
from tiny_lm.training.trainer import Trainer, get_cosine_schedule_with_warmup


def main():
    # Sample training data
    train_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning models require large amounts of training data.",
        "Natural language processing enables computers to understand text.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Large language models can generate human-like text.",
        "Training neural networks requires careful hyperparameter tuning.",
        "GPUs accelerate the training process for deep learning models.",
    ] * 10  # Repeat for more data
    
    eval_texts = [
        "Neural networks learn patterns from data.",
        "Text generation is an important NLP task.",
    ]
    
    # Initialize tokenizer
    print("Building tokenizer vocabulary...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(train_texts + eval_texts)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
    eval_dataset = TextDataset(eval_texts, tokenizer, max_length=128)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Initialize model
    print("Initializing model...")
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    num_epochs = 10
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_grad_norm=1.0,
        log_interval=10,
        eval_interval=50,
        save_interval=100,
        output_dir="./outputs",
    )
    
    # Train the model
    print("Starting training...")
    trainer.train(num_epochs=num_epochs)
    
    print("Training completed!")
    
    # Save tokenizer
    tokenizer.save("./outputs/tokenizer.json")
    print("Tokenizer saved to ./outputs/tokenizer.json")


if __name__ == "__main__":
    main()
