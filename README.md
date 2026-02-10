# Tiny Language Model 🚀

Train tiny language models entirely from scratch using native PyTorch!

## Overview

This open-source project provides a complete implementation for training tiny language models from the ground up. **Most core algorithmic code has been rebuilt from scratch using native PyTorch**. This represents not only a full-stage open-source recreation of large language models but also serves as an **introductory tutorial to LLM development**.

### Key Features

✨ **Pure PyTorch Implementation**: Every component is built from scratch using native PyTorch
- Multi-head attention mechanism
- Transformer blocks with RMSNorm
- Positional encodings (RoPE)
- Custom training loop and optimization

📚 **Complete Training Pipeline**:
- Data loading and preprocessing
- Training with gradient clipping and learning rate scheduling
- Checkpointing and model saving
- Evaluation loop

📖 **Educational**: Extensively documented code suitable for learning

## Installation

### Requirements

- Python 3.10
- PyTorch 2.6.0
- transformers 4.57.1
- NumPy 1.26.4

### Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Pretraining a Model

```bash
cd ./trainer
torchrun --nproc_per_node=4 train_pretrain.py
```


### Full SFT
```bash
cd ./trainer
torchrun --nproc_per_node=4 train_full_sft.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/pretrain/pretrain_768_epoch2.pth
```

### LoRA Fine-Tuning
```bash
torchrun --nproc_per_node=4 train_lora.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/full_sft/full_sft_768_epoch1.pth
```

### DPO
```bash
torchrun --nproc_per_node=4 train_dpo.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/full_sft/full_sft_768_epoch1.pth
```


### Inference

```bash
python inference.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/pretrain/pretrain_768_epoch2.pth --weight pretrain

python inference.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/full_sft/full_sft_768_epoch1.pth --weight full_sft

python inference.py --model_path /home/liuzilong/data/models/Tiny-Language-Model/dpo/dpo_768_epoch0.pth --weight dpo

```

## Architecture

### Transformer Model

The model implements a decoder-only transformer architecture:

- **Token Embeddings**: Maps input tokens to dense vectors
- **Positional Encodings**: Adds position information (sinusoidal or learned)
- **Transformer Blocks**: Multiple layers of:
  - Grouped query attention
  - SwiGLU feedforward networks
  - RMS normalization
  - Residual connections
- **Language Model Head**: Projects to vocabulary logits


## Project Structure

```
Tiny-Language-Model/
├── model/                      # Model architecture implementation
│   └── model_tinylm.py         # Tiny Language Model neural network
├── tokenizer/                  # Tokenization utilities
│   ├── tokenizer_config.json   # Tokenizer configuration file
│   └── tokenizer.json          # Tokenizer vocabulary and rules
├── dataset/                    # Data handling utilities
│   └── dataset.py              # Dataset loading and preprocessing
├── trainer/                    # Training utilities
│   ├── trainer_pretrain.py     # Pre-training script
│   ├── trainer_full_sft.py     # Supervised Fine-Tuning script
│   ├── trainer_lora.py         # LoRA script
│   ├── trainer_dpo.py          # DPO script
│   └── trainer_utils.py        # Training helper functions
├── inference.py                # Inference script
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Key Concepts

### 1. Grouped Query Attention

GQA is an attention mechanism that bridges the gap between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). It provides a balance between model quality and inference efficiency by grouping multiple query heads to share the same key and value heads.

### 2. Rotary Positional Encoding

RoPE is a positional encoding method that encodes absolute positional information using rotation matrices, allowing the model to naturally capture relative positional relationships through the attention mechanism.

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
- **Model dimension**: 512 or 768
- **Number of layers**: 8 or 16
- **Number of heads**: 8
- **Dropout**: 0

## Performance

The model size and training speed depend on your hardware:

- **Small model** (512 dim, 8 layers): ~25M parameters, GPU required
- **Large model** (768 dim, 16 layers): ~104M parameters, GPU required

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