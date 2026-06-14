import torch.nn as nn
from .layers import LayerNormalization, PositionwiseFeedForward
from .attention import MultiHeadAttention, MultiHeadCrossAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.cross_attn = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalization(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, enc_output, dec_input, self_mask=None, cross_mask=None):
        # Masked self‑attention
        residual = dec_input
        dec_input = self.self_attn(dec_input, self_mask)
        dec_input = self.dropout1(dec_input)
        dec_input = self.norm1(dec_input + residual)
        # Cross‑attention
        residual = dec_input
        dec_input = self.cross_attn(enc_output, dec_input, cross_mask)
        dec_input = self.dropout2(dec_input)
        dec_input = self.norm2(dec_input + residual)
        # Feed‑forward
        residual = dec_input
        dec_input = self.ffn(dec_input)
        dec_input = self.dropout3(dec_input)
        dec_input = self.norm3(dec_input + residual)
        return dec_input

class Decoder(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float,
                 num_layers: int, max_seq_len: int, tokenizer):
        super().__init__()
        self.embedding = SentenceEmbedding(tokenizer, d_model, max_seq_len, drop_prob)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
            for _ in range(num_layers)
        ])

    def forward(self, enc_output, target_sentences, self_mask=None, cross_mask=None,
                add_start=True, add_end=True):
        x = self.embedding(target_sentences, add_start, add_end)
        for layer in self.layers:
            x = layer(enc_output, x, self_mask, cross_mask)
        return x