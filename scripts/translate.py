#!/usr/bin/env python
import argparse
import torch
from src.utils.config_utils import load_config
from src.data.tokenizer import CharTokenizer
from src.model.transformer import Transformer
from src.inference.translator import Translator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # Build tokenizers from config
    src_tokenizer = CharTokenizer(config.vocab.english_chars, max_len=config.model.max_seq_len)
    tgt_tokenizer = CharTokenizer(config.vocab.hindi_chars, max_len=config.model.max_seq_len)

    model = Transformer(config, src_tokenizer, tgt_tokenizer)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    translator = Translator(model, src_tokenizer, tgt_tokenizer, device,
                            max_gen_len=config.inference.max_gen_len,
                            beam_size=config.inference.beam_size)
    translation = translator.translate(args.sentence)
    print(f"Input: {args.sentence}")
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main()