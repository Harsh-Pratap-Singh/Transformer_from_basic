import torch
from typing import List, Dict, Optional

# Use a single character for UNK to avoid splitting across multiple characters
SPECIAL_TOKENS = ["<START>", "<PAD>", "<END>", "�"]

class CharTokenizer:
    def __init__(self, chars: List[str], max_len: int = 300):
        # Build vocabulary with special tokens first
        self.vocab = SPECIAL_TOKENS + chars
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.vocab)}
        self.max_len = max_len
        self.start_id = self.stoi["<START>"]
        self.unk_id = self.stoi["�"]
        self.pad_id = self.stoi["<PAD>"]
        self.end_id = self.stoi["<END>"]

    def encode(self, sentence: str, add_start: bool = True, add_end: bool = True) -> torch.LongTensor:
        ids = [self.stoi.get(ch, self.unk_id) for ch in sentence]
        if add_start:
            ids.insert(0, self.start_id)
        if add_end:
            ids.append(self.end_id)
        # truncate if necessary (keep at least start/end)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len - 1] + [self.end_id]
        # pad
        ids += [self.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.LongTensor, skip_special: bool = True) -> str:
        tokens = []
        for idx in ids:
            ch = self.itos[int(idx)]
            if skip_special and ch in SPECIAL_TOKENS:
                continue
            tokens.append(ch)
        return "".join(tokens)

    def __len__(self):
        return len(self.vocab)