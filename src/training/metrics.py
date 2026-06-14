import torch
import numpy as np
from sacrebleu import corpus_bleu

def compute_bleu(preds: list, targets: list) -> float:
    """Compute corpus‑level BLEU score."""
    # preds and targets are lists of strings
    return corpus_bleu(preds, [targets]).score

def compute_accuracy(logits, labels, pad_idx):
    """Compute token‑level accuracy ignoring padding."""
    preds = logits.argmax(dim=-1)
    mask = labels != pad_idx
    correct = (preds == labels) & mask
    return correct.sum().item() / mask.sum().item()