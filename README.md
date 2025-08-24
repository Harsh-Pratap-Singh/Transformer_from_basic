# Transformer from Scratch 🚀

A comprehensive, step-by-step implementation of the Transformer architecture built from the ground up using PyTorch. This project demonstrates how to construct each component of the Transformer model, from basic attention mechanisms to the complete encoder-decoder architecture.

## 🎯 Project Overview

This implementation follows the seminal "Attention Is All You Need" paper, building each component progressively to create a full understanding of how Transformers work. The code is educational, well-documented, and includes detailed print statements to visualize tensor shapes throughout the forward pass.

## 📚 What You'll Learn

- **Self-Attention Mechanism**: From basic dot-product attention to scaled attention
- **Multi-Head Attention**: Parallel attention heads for richer representations
- **Positional Encoding**: Sinusoidal embeddings to inject position information
- **Layer Normalization**: Stabilizing training with proper normalization techniques
- **Masking**: Preventing future information leakage in decoder layers
- **Encoder-Decoder Architecture**: Complete transformer structure with cross-attention
- **Feed-Forward Networks**: Position-wise fully connected layers

## 🏗️ Implementation Journey

### Phase 1: Foundation - Self-Attention Mechanism
```python
# Basic attention calculation
scaled = np.matmul(q, k.T) / math.sqrt(d_k)
attention = softmax(scaled + mask)
output = np.matmul(attention, v)
```

**Key Insights:**
- Implemented scaled dot-product attention from scratch
- Understood why we divide by √d_k (variance control)
- Applied triangular masking for causal attention

### Phase 2: Multi-Head Attention
```python
# Reshape for multiple heads
qkv = qkv.reshape(batch_size, sequence_length, num_heads, 3 * head_dim)
qkv = qkv.permute(0, 2, 1, 3)  # [batch, heads, seq, dim]
q, k, v = qkv.chunk(3, dim=-1)
```

**Architecture Details:**
- **Input Dimension**: 512 (d_model)
- **Number of Heads**: 8
- **Head Dimension**: 64 (d_model // num_heads)
- **QKV Linear Layer**: 512 → 1536 (3 × d_model)

### Phase 3: Positional Encoding
```python
# Sinusoidal position embeddings
even_PE = torch.sin(position / denominator)
odd_PE = torch.cos(position / denominator)
PE = torch.flatten(torch.stack([even_PE, odd_PE], dim=2), start_dim=1)
```

**Mathematical Foundation:**
- Even positions: sin(pos/10000^(2i/d_model))
- Odd positions: cos(pos/10000^(2i/d_model))
- Creates unique position signatures for each sequence position

### Phase 4: Layer Normalization
```python
# Normalize across feature dimension
mean = input.mean(dim=dims, keepdim=True)
var = ((input - mean)**2).mean(dim=dims, keepdim=True)
normalized = (input - mean) / sqrt(var + eps)
output = gamma * normalized + beta
```

**Benefits:**
- Stabilizes training by normalizing layer inputs
- Learnable parameters (γ, β) for scaling and shifting
- Applied after each sub-layer in the transformer

### Phase 5: Encoder Architecture
```python
class EncoderLayer(nn.Module):
    def forward(self, x):
        # Multi-head self-attention
        residual = x
        x = self.attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual)  # Add & Norm
        
        # Feed-forward network
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)  # Add & Norm
        return x
```

**Encoder Stack:**
- **Layers**: 5 encoder layers
- **Self-Attention**: No masking (bidirectional)
- **Feed-Forward**: 512 → 2048 → 512
- **Residual Connections**: Around each sub-layer

### Phase 6: Decoder Architecture
```python
class DecoderLayer(nn.Module):
    def forward(self, x, y, decoder_mask):
        # Masked self-attention
        residual = y
        y = self.self_attention(y, mask=decoder_mask)
        y = self.norm1(y + residual)
        
        # Cross-attention (encoder-decoder)
        residual = y
        y = self.cross_attention(x, y)  # K,V from encoder
        y = self.norm2(y + residual)
        
        # Feed-forward
        residual = y
        y = self.ffn(y)
        y = self.norm3(y + residual)
        return y
```

**Decoder Features:**
- **Masked Self-Attention**: Prevents looking at future tokens
- **Cross-Attention**: Attends to encoder outputs
- **Three Sub-layers**: Self-attention, cross-attention, FFN

## 🔧 Architecture Configuration

```python
# Model hyperparameters
config = {
    'd_model': 512,              # Model dimension
    'num_heads': 8,              # Multi-head attention heads
    'num_layers': 5,             # Number of encoder/decoder layers
    'ffn_hidden': 2048,          # Feed-forward hidden dimension
    'max_sequence_length': 200,  # Maximum input length
    'batch_size': 30,            # Training batch size
    'dropout': 0.1,              # Dropout probability
}
```

## 📊 Tensor Flow Visualization

The implementation includes detailed tensor shape tracking:

```
# Example output from encoder forward pass
x.size(): torch.Size([30, 200, 512])
qkv.size(): torch.Size([30, 200, 1536])
q, k, v sizes: torch.Size([30, 8, 200, 64]) each
attention.size(): torch.Size([30, 8, 200, 200])
output.size(): torch.Size([30, 200, 512])
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch numpy matplotlib
```

### Running the Code
```python
# Import required libraries
import torch
import torch.nn as nn
import numpy as np
import math

# Initialize model components
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 200

# Create sample input
x = torch.randn(batch_size, sequence_length, d_model)

# Build and test encoder
encoder = Encoder(d_model, ffn_hidden=2048, num_heads=8, 
                  drop_prob=0.1, num_layers=5)
encoder_output = encoder(x)

# Build and test decoder
y = torch.randn(batch_size, sequence_length, d_model)
mask = torch.triu(torch.full([sequence_length, sequence_length], 
                             float('-inf')), diagonal=1)

decoder = Decoder(d_model, ffn_hidden=2048, num_heads=8, 
                  drop_prob=0.1, num_layers=5)
decoder_output = decoder(encoder_output, y, mask)
```

## 📝 Code Organization

### Part 1: Basic Attention
- NumPy implementation of scaled dot-product attention
- Variance analysis and scaling factor derivation
- Causal masking for decoder attention

### Part 2: Multi-Head Attention & Positional Encoding
- PyTorch implementation with multiple attention heads
- Sinusoidal positional encoding generation
- Layer normalization from scratch

### Part 3: Transformer Encoder
- Complete encoder layer with residual connections
- Multi-layer encoder stack
- Feed-forward position-wise networks

### Part 4: Transformer Decoder
- Masked self-attention implementation
- Cross-attention mechanism (encoder-decoder attention)
- Complete decoder stack with proper masking

## 🎯 Key Features

- **Educational Focus**: Extensive comments and print statements for learning
- **From Scratch**: Every component built without pre-existing transformer layers
- **Modular Design**: Each component can be understood and tested independently
- **Shape Debugging**: Detailed tensor shape tracking throughout forward passes
- **Mathematical Clarity**: Clear implementation of all mathematical concepts

## 🔍 Understanding the Flow

### Encoder Process:
1. **Input Embeddings** + **Positional Encoding**
2. **Multi-Head Self-Attention** (no masking)
3. **Add & Normalize**
4. **Feed-Forward Network**
5. **Add & Normalize**
6. **Repeat** for N layers

### Decoder Process:
1. **Output Embeddings** + **Positional Encoding**
2. **Masked Multi-Head Self-Attention**
3. **Add & Normalize**
4. **Multi-Head Cross-Attention** (with encoder output)
5. **Add & Normalize**
6. **Feed-Forward Network**
7. **Add & Normalize**
8. **Repeat** for N layers

## 💡 Key Insights from Implementation

### Attention Mechanism
- **Scaled Attention**: Division by √d_k prevents softmax saturation
- **Multi-Head**: Parallel attention allows focusing on different aspects
- **Masking**: Essential for maintaining causality in generation tasks

### Positional Encoding
- **Sinusoidal Functions**: Allow extrapolation to longer sequences
- **Fixed vs Learned**: This implementation uses fixed sinusoidal embeddings
- **Additive**: Added to input embeddings, not concatenated

### Normalization & Residuals
- **Layer Norm**: Applied after each sub-layer
- **Residual Connections**: Enable deep network training
- **Add & Norm**: Critical for gradient flow

## 🎓 Educational Value

This implementation serves as:
- **Learning Resource**: Understand transformer internals step-by-step
- **Research Foundation**: Modify components for custom architectures
- **Debugging Tool**: Extensive logging helps identify issues
- **Teaching Aid**: Clear, commented code for educational purposes

## 🔧 Customization Options

### Modify Architecture:
```python
# Different model sizes
d_model = 256  # Smaller model
num_heads = 4  # Fewer attention heads
num_layers = 3 # Shallower network

# Custom feed-forward size
ffn_hidden = d_model * 4  # Standard scaling
```

### Experiment with Masking:
```python
# Create custom attention masks
lookahead_mask = create_lookahead_mask(seq_len)
padding_mask = create_padding_mask(input_ids, pad_token_id)
```

## 📚 Mathematical Foundations

### Self-Attention Formula:
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Multi-Head Attention:
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

### Positional Encoding:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## 🤝 Contributing

Feel free to contribute by:
- Adding visualization tools for attention weights
- Implementing different positional encoding schemes
- Adding training loops and optimization
- Creating example applications (translation, text generation)

## 📖 References

- **Original Paper**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **The Illustrated Transformer**: [Jay Alammar's Blog](http://jalammar.github.io/illustrated-transformer/)
- **PyTorch Documentation**: [Official PyTorch Docs](https://pytorch.org/docs/)

## 👨‍💻 Author

**Harsh Pratap Singh**
- GitHub: [@Harsh-Pratap-Singh](https://github.com/Harsh-Pratap-Singh)
- Project: [Transformer_from_basic](https://github.com/Harsh-Pratap-Singh/Transformer_from_basic)

---

⭐ **If this implementation helped you understand Transformers, please give it a star!** ⭐

This project demonstrates the beauty and elegance of the Transformer architecture through clean, educational code. Perfect for students, researchers, and practitioners who want to truly understand how these revolutionary models work under the hood.
