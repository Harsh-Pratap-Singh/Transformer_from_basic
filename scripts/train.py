#!/usr/bin/env python
import argparse
import sys
import torch
import pandas as pd
from src.utils.config_utils import load_config
from src.data.tokenizer import CharTokenizer
from src.data.preprocess import filter_valid_pairs, clean_sentences
from src.data.dataset import create_dataloaders
from src.model.transformer import Transformer
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with 'english','hindi' columns")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)

    # Windows compatibility for num_workers
    if sys.platform == "win32":
        config.data.num_workers = 0

    device = torch.device(args.device)

    # Load data
    df = pd.read_csv(args.data)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['english'] = df['english'].str.lower()

    src_sentences = df['english'].tolist()
    tgt_sentences = df['hindi'].tolist()

    # Create tokenizers from config vocab
    src_tokenizer = CharTokenizer(config.vocab.english_chars, max_len=config.model.max_seq_len)
    tgt_tokenizer = CharTokenizer(config.vocab.hindi_chars, max_len=config.model.max_seq_len)

    # Clean and filter
    src_vocab_set = set(src_tokenizer.vocab)
    tgt_vocab_set = set(tgt_tokenizer.vocab)

    # Optional: clean sentences first (replace unknown chars)
    src_sentences = clean_sentences(src_sentences, src_vocab_set, unk_char="�")
    tgt_sentences = clean_sentences(tgt_sentences, tgt_vocab_set, unk_char="�")

    src_sentences, tgt_sentences = filter_valid_pairs(
        src_sentences, tgt_sentences, src_vocab_set, tgt_vocab_set, config.model.max_seq_len
    )

    print(f"Total valid sentence pairs: {len(src_sentences)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer,
        batch_size=config.data.batch_size,
        train_ratio=config.data.train_ratio,
        num_workers=config.data.num_workers
    )

    # Model
    model = Transformer(config, src_tokenizer, tgt_tokenizer)

    # Trainer
    trainer = Trainer(model, train_loader, val_loader, config, device, src_tokenizer, tgt_tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()