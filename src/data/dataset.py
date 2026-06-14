import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class TranslationDataset(Dataset):
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str],
                 src_tokenizer, tgt_tokenizer):
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx) -> Tuple[torch.LongTensor, torch.LongTensor]:
        src_ids = self.src_tokenizer.encode(self.src[idx], add_start=False, add_end=False)
        tgt_ids = self.tgt_tokenizer.encode(self.tgt[idx], add_start=True, add_end=True)
        return src_ids, tgt_ids

def create_dataloaders(
    src_sentences: List[str],
    tgt_sentences: List[str],
    src_tokenizer,
    tgt_tokenizer,
    batch_size: int,
    train_ratio: float = 0.95,
    shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Split data and return train and val dataloaders."""
    split_idx = int(len(src_sentences) * train_ratio)
    train_dataset = TranslationDataset(
        src_sentences[:split_idx], tgt_sentences[:split_idx],
        src_tokenizer, tgt_tokenizer
    )
    val_dataset = TranslationDataset(
        src_sentences[split_idx:], tgt_sentences[split_idx:],
        src_tokenizer, tgt_tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader