# Tutorial: Building Language Models from Scratch

This tutorial will guide you through training a small language model entirely from scratch using the Tiny Language Model library.

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Preparing Your Data](#preparing-your-data)
4. [Training Your First Model](#training-your-first-model)
5. [Generating Text](#generating-text)
6. [Advanced Topics](#advanced-topics)

## Introduction

Language models predict the next word (or token) in a sequence. The Tiny Language Model library implements a transformer-based architecture, similar to GPT, built entirely from scratch using native PyTorch.

### What You'll Learn

- How transformers work at a code level
- How to implement attention mechanisms from scratch
- How to train a language model on your own data
- Different text generation strategies

## Understanding the Architecture

### Transformer Components

Our language model consists of several key components:

#### 1. Token Embeddings

Convert discrete tokens (words/characters) into continuous vectors:

```python
# Each token ID is mapped to a d_model dimensional vector
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

#### 2. Positional Encodings

Since transformers have no inherent notion of position, we add positional information:

```python
# Sinusoidal positional encoding
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 3. Multi-Head Attention

Allows the model to focus on different parts of the input:

```python
# Self-attention mechanism
scores = QK^T / sqrt(d_k)
attention = softmax(scores)
output = attention * V
```

#### 4. Feedforward Networks

Position-wise transformations:

```python
FFN(x) = W2 * GELU(W1 * x + b1) + b2
```

### How It All Fits Together

```
Input Tokens
    ↓
Token Embedding + Positional Encoding
    ↓
[Transformer Block] × N layers
    ├── Multi-Head Attention
    ├── Layer Norm + Residual
    ├── Feedforward Network
    └── Layer Norm + Residual
    ↓
Language Model Head (Linear)
    ↓
Output Logits
```

## Preparing Your Data

### Step 1: Collect Your Text Data

Gather text files for training. For example:

```python
train_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a branch of artificial intelligence.",
    "Python is a popular programming language.",
    # ... more texts
]
```

### Step 2: Choose a Tokenizer

**Character-Level Tokenizer** (Simple, good for small datasets):

```python
from tiny_lm.tokenizer.char_tokenizer import CharTokenizer

tokenizer = CharTokenizer()
tokenizer.build_vocab(train_texts)
```

**BPE Tokenizer** (More efficient, better for larger datasets):

```python
from tiny_lm.tokenizer.bpe_tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=5000)
tokenizer.build_vocab(train_texts)
```

### Step 3: Create Dataset

```python
from tiny_lm.data.dataset import TextDataset
from torch.utils.data import DataLoader

dataset = TextDataset(
    texts=train_texts,
    tokenizer=tokenizer,
    max_length=128,
)

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
)
```

## Training Your First Model

### Step 1: Define Model Configuration

```python
from tiny_lm.model.transformer import TransformerLM

model = TransformerLM(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=256,        # Embedding dimension
    num_layers=4,       # Number of transformer blocks
    num_heads=4,        # Number of attention heads
    d_ff=1024,          # Feedforward dimension
    max_seq_len=128,    # Maximum sequence length
    dropout=0.1,        # Dropout rate
)
```

### Step 2: Setup Training

```python
from tiny_lm.training.trainer import Trainer
import torch

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    output_dir="./outputs",
)
```

### Step 3: Train!

```python
trainer.train(num_epochs=10)
```

### Monitoring Training

The trainer will print logs like:

```
Epoch 1/10
Step 100: loss = 3.2145, avg_loss = 3.4567, lr = 0.000300
Step 200: loss = 2.9876, avg_loss = 3.1234, lr = 0.000295
...
```

Lower loss = better model!

## Generating Text

### Basic Generation

```python
from tiny_lm.generation.generator import Generator

generator = Generator(model, tokenizer, device="cuda")

# Greedy decoding (deterministic)
text = generator.generate_greedy(
    prompt="The quick brown",
    max_new_tokens=50,
)
```

### Advanced Sampling Strategies

#### Temperature Sampling

Controls randomness (higher = more random):

```python
text = generator.generate_sampling(
    prompt="Machine learning",
    max_new_tokens=50,
    temperature=0.8,  # Try 0.5 for more focused, 1.5 for more random
)
```

#### Top-K Sampling

Only sample from the K most likely tokens:

```python
text = generator.generate_sampling(
    prompt="Deep learning",
    max_new_tokens=50,
    top_k=10,
)
```

#### Top-P (Nucleus) Sampling

Sample from tokens with cumulative probability >= P:

```python
text = generator.generate_sampling(
    prompt="Neural networks",
    max_new_tokens=50,
    top_p=0.9,
)
```

#### Beam Search

Maintains multiple hypotheses:

```python
text = generator.generate_beam_search(
    prompt="Artificial intelligence",
    max_new_tokens=50,
    num_beams=5,
)
```

## Advanced Topics

### Learning Rate Scheduling

Improve training with warmup and cosine decay:

```python
from tiny_lm.training.trainer import get_cosine_schedule_with_warmup

num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_training_steps // 10,
    num_training_steps=num_training_steps,
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    ...
)
```

### Using Predefined Configurations

```python
from tiny_lm.model.config import SMALL_CONFIG, MEDIUM_CONFIG

config = SMALL_CONFIG
config.vocab_size = tokenizer.get_vocab_size()

model = TransformerLM(**config.to_dict())
```

### Checkpointing

The trainer automatically saves checkpoints:

- `checkpoint_epoch_N.pt`: After each epoch
- `checkpoint_step_N.pt`: At specified intervals
- `best_model.pt`: Best model based on validation loss

To resume training:

```python
trainer.train(
    num_epochs=10,
    resume_from_checkpoint="./outputs/checkpoint_epoch_5.pt"
)
```

### Loading a Trained Model

```python
# Load checkpoint
checkpoint = torch.load("./outputs/best_model.pt")

# Create model with same configuration
model = TransformerLM(...)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Evaluation

```python
# Create evaluation dataloader
eval_dataloader = DataLoader(eval_dataset, ...)

# Evaluate
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    eval_interval=500,  # Evaluate every 500 steps
    ...
)

# Or evaluate manually
eval_loss = trainer.evaluate()
print(f"Evaluation loss: {eval_loss:.4f}")
```

### Working with Large Datasets

For datasets that don't fit in memory:

```python
from tiny_lm.data.dataset import StreamingTextDataset

dataset = StreamingTextDataset(
    filepath="large_text_file.txt",
    tokenizer=tokenizer,
    max_length=128,
)
```

### Tips for Better Models

1. **More Data**: Language models benefit greatly from more training data
2. **Longer Training**: Don't stop too early, monitor validation loss
3. **Larger Models**: More parameters can learn more complex patterns
4. **Learning Rate**: Try different learning rates (1e-4 to 5e-4)
5. **Regularization**: Use dropout to prevent overfitting
6. **Preprocessing**: Clean and normalize your text data

### Common Issues

**High Loss**: 
- Train longer
- Reduce learning rate
- Check your data quality

**Overfitting**:
- Increase dropout
- Get more training data
- Use a smaller model

**Out of Memory**:
- Reduce batch size
- Reduce model size (d_model, num_layers)
- Reduce max_seq_len

**Generated Text is Repetitive**:
- Use temperature > 1.0
- Try top-k or top-p sampling
- Train on more diverse data

## Next Steps

1. Try training on your own text dataset
2. Experiment with different model sizes
3. Compare different generation strategies
4. Fine-tune on specific tasks (e.g., poetry, code)
5. Implement additional features (e.g., different attention patterns)

## Resources

- Original Transformer Paper: "Attention is All You Need" (Vaswani et al., 2017)
- PyTorch Documentation: https://pytorch.org/docs/
- Understanding LLMs: Check out various blogs and papers on language models

Happy Learning! 🚀
