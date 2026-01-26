# Tiny Language Model 🚀

Train small language models entirely from scratch using native PyTorch!

## Overview

This open-source project provides a complete implementation for training small language models from the ground up. **All core algorithmic code has been rebuilt from scratch using native PyTorch**, with **no reliance on abstract interfaces provided by third-party libraries**. This represents not only a full-stage open-source recreation of large language models but also serves as an **introductory tutorial to LLM development**.

### Key Features

✨ **Pure PyTorch Implementation**: Every component is built from scratch using native PyTorch
- Multi-head attention mechanism
- Transformer blocks with layer normalization
- Positional encodings (sinusoidal and learned)
- Custom training loop and optimization

🔤 **Built-in Tokenizers**: Two tokenization methods implemented from scratch
- Character-level tokenizer
- Byte Pair Encoding (BPE) tokenizer

📚 **Complete Training Pipeline**:
- Data loading and preprocessing
- Training with gradient clipping and learning rate scheduling
- Checkpointing and model saving
- Evaluation loop

🎯 **Advanced Text Generation**:
- Greedy decoding
- Temperature-based sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Beam search

📖 **Educational**: Extensively documented code suitable for learning

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy

### Install from source

```bash
git clone https://github.com/lyuzlion/Tiny-Language-Model.git
cd Tiny-Language-Model
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training a Model

```python
import torch
from torch.utils.data import DataLoader
from tiny_lm.model.transformer import TransformerLM
from tiny_lm.tokenizer.char_tokenizer import CharTokenizer
from tiny_lm.data.dataset import TextDataset, collate_fn
from tiny_lm.training.trainer import Trainer

# Prepare your training data
train_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
    # Add more texts...
]

# Build tokenizer vocabulary
tokenizer = CharTokenizer()
tokenizer.build_vocab(train_texts)

# Create dataset and dataloader
train_dataset = TextDataset(train_texts, tokenizer, max_length=128)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# Initialize model
model = TransformerLM(
    vocab_size=tokenizer.get_vocab_size(),
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    max_seq_len=128,
)

# Train the model
trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    output_dir="./outputs",
)
trainer.train(num_epochs=10)
```

### Generating Text

```python
from tiny_lm.generation.generator import Generator

# Initialize generator
generator = Generator(model, tokenizer, device="cuda")

# Generate text
generated_text = generator.generate_sampling(
    prompt="Machine learning",
    max_new_tokens=50,
    temperature=0.8,
    top_k=10,
)
print(generated_text)
```

## Architecture

### Transformer Model

The model implements a decoder-only transformer architecture:

- **Token Embeddings**: Maps input tokens to dense vectors
- **Positional Encodings**: Adds position information (sinusoidal or learned)
- **Transformer Blocks**: Multiple layers of:
  - Multi-head self-attention
  - Position-wise feedforward networks
  - Layer normalization
  - Residual connections
- **Language Model Head**: Projects to vocabulary logits

### Model Components

```python
TransformerLM(
    vocab_size=1000,        # Vocabulary size
    d_model=512,            # Model dimension
    num_layers=6,           # Number of transformer layers
    num_heads=8,            # Number of attention heads
    d_ff=2048,              # Feedforward dimension
    max_seq_len=512,        # Maximum sequence length
    dropout=0.1,            # Dropout rate
    positional_encoding="sinusoidal"  # or "learned"
)
```

## Examples

The `examples/` directory contains complete scripts demonstrating various use cases:

- **`train_example.py`**: Complete training pipeline
- **`inference_example.py`**: Text generation with different strategies
- **`bpe_example.py`**: BPE tokenizer training and usage

Run examples:

```bash
# Train a model
python examples/train_example.py

# Generate text (after training)
python examples/inference_example.py

# Test BPE tokenizer
python examples/bpe_example.py
```

## Project Structure

```
Tiny-Language-Model/
├── tiny_lm/                    # Main package
│   ├── model/                  # Model components
│   │   ├── attention.py        # Multi-head attention
│   │   ├── feedforward.py      # Feedforward networks
│   │   ├── positional.py       # Positional encodings
│   │   └── transformer.py      # Complete transformer model
│   ├── tokenizer/              # Tokenization
│   │   ├── char_tokenizer.py   # Character-level tokenizer
│   │   └── bpe_tokenizer.py    # BPE tokenizer
│   ├── data/                   # Data utilities
│   │   └── dataset.py          # Dataset classes
│   ├── training/               # Training infrastructure
│   │   └── trainer.py          # Trainer class
│   └── generation/             # Text generation
│       └── generator.py        # Generation strategies
├── examples/                   # Example scripts
│   ├── train_example.py
│   ├── inference_example.py
│   └── bpe_example.py
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Key Concepts

### 1. Multi-Head Attention

The attention mechanism allows the model to focus on different parts of the input sequence. Multi-head attention runs multiple attention operations in parallel, enabling the model to capture various types of relationships.

### 2. Positional Encoding

Since transformers don't inherently understand sequence order, positional encodings add position information to the input embeddings.

### 3. Causal Masking

For language modeling, we use causal (autoregressive) masking to ensure the model can only attend to previous tokens, not future ones.

### 4. Autoregressive Generation

The model generates text one token at a time, using its own predictions as input for the next step.

## Training Tips

1. **Start Small**: Begin with a small model to verify your pipeline works
2. **Monitor Loss**: Watch training and evaluation loss to detect overfitting
3. **Learning Rate**: Use learning rate warmup and cosine decay for stable training
4. **Gradient Clipping**: Essential for preventing exploding gradients
5. **Checkpointing**: Save checkpoints regularly in case training is interrupted

## Hyperparameters

Common hyperparameter ranges:

- **Learning rate**: 1e-4 to 5e-4
- **Batch size**: 8 to 64 (depending on GPU memory)
- **Model dimension**: 128 to 1024
- **Number of layers**: 2 to 12
- **Number of heads**: 4 to 16
- **Dropout**: 0.1 to 0.3

## Performance

The model size and training speed depend on your hardware:

- **Small model** (256 dim, 4 layers): ~5M parameters, trains on CPU
- **Medium model** (512 dim, 6 layers): ~25M parameters, GPU recommended
- **Large model** (768 dim, 12 layers): ~85M parameters, GPU required

## Contributing

Contributions are welcome! This project aims to be educational and accessible. Please:

1. Keep implementations clear and well-documented
2. Avoid adding unnecessary dependencies
3. Include examples for new features
4. Follow the existing code style

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Educational Resources

This implementation is designed for learning. Here are some concepts covered:

- Transformer architecture
- Self-attention mechanisms
- Tokenization algorithms (BPE)
- Training deep neural networks
- Text generation strategies
- Optimization techniques

## Citation

If you use this code for research or educational purposes, please cite:

```bibtex
@software{tiny_language_model,
  author = {Zilong Liu},
  title = {Tiny Language Model: Train Small Language Models from Scratch},
  year = {2026},
  url = {https://github.com/lyuzlion/Tiny-Language-Model}
}
```

## Acknowledgments

This project is inspired by:
- "Attention is All You Need" (Vaswani et al., 2017)
- Various open-source implementations of transformers
- The PyTorch community

---

**Built with ❤️ for learning and education**