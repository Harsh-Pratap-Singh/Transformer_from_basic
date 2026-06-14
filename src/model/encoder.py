import torch.nn as nn
from .layers import LayerNormalization, PositionwiseFeedForward
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        # Self‑attention
        residual = x
        x = self.self_attn(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual)
        # Feed‑forward
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float,
                 num_layers: int, max_seq_len: int, tokenizer):
        super().__init__()
        self.embedding = SentenceEmbedding(tokenizer, d_model, max_seq_len, drop_prob)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, sentences, mask=None, add_start=False, add_end=False):
        x = self.embedding(sentences, add_start, add_end)
        for layer in self.layers:
            x = layer(x, mask)
        return x