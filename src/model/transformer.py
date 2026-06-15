import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, config, src_tokenizer, tgt_tokenizer):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            d_model=config.model.d_model,
            ffn_hidden=config.model.ffn_hidden,
            num_heads=config.model.num_heads,
            drop_prob=config.model.dropout,
            num_layers=config.model.num_layers,
            max_seq_len=config.model.max_seq_len,
            tokenizer=src_tokenizer
        )
        self.decoder = Decoder(
            d_model=config.model.d_model,
            ffn_hidden=config.model.ffn_hidden,
            num_heads=config.model.num_heads,
            drop_prob=config.model.dropout,
            num_layers=config.model.num_layers,
            max_seq_len=config.model.max_seq_len,
            tokenizer=tgt_tokenizer
        )
        self.output_proj = nn.Linear(config.model.d_model, len(tgt_tokenizer))

    def forward(self, src_sentences, tgt_sentences,
                encoder_mask=None, decoder_self_mask=None, decoder_cross_mask=None,
                src_add_start=False, src_add_end=False,
                tgt_add_start=True, tgt_add_end=True):
        enc_out = self.encoder(src_sentences, encoder_mask, src_add_start, src_add_end)
        dec_out = self.decoder(enc_out, tgt_sentences, decoder_self_mask, decoder_cross_mask,
                               tgt_add_start, tgt_add_end)
        logits = self.output_proj(dec_out)
        return logits