import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores + mask
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (B, H, L, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        values, _ = scaled_dot_product_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return self.linear(values)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.q = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, enc_output, dec_input, mask=None):
        batch_size, seq_len, _ = enc_output.shape
        kv = self.kv(enc_output).reshape(batch_size, seq_len, self.num_heads, 2 * self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)

        q = self.q(dec_input).reshape(batch_size, dec_input.size(1), self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        values, _ = scaled_dot_product_attention(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, dec_input.size(1), self.d_model)
        return self.linear(values)