import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SentenceEmbedding(nn.Module):
    def __init__(self, tokenizer, d_model: int, max_seq_len: int, drop_prob: float = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(len(tokenizer), d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, sentences, add_start=True, add_end=True):
        # sentences: list of strings or pre‑tokenized tensors
        if isinstance(sentences, list):
            device = next(self.parameters()).device
            token_ids = [self.tokenizer.encode(s, add_start, add_end).to(device) for s in sentences]
            token_ids = torch.stack(token_ids)
        else:
            token_ids = sentences
        x = self.embedding(token_ids)
        x = self.pos_encoding(x)
        return self.dropout(x)