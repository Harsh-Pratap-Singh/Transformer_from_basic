from typing import List, Set, Tuple

def is_valid_tokens(sentence: str, vocab_set: Set[str]) -> bool:
    """Check all characters in sentence exist in vocab_set."""
    # vocab_set includes all characters, but also special tokens? We only check characters.
    # We'll assume vocab_set contains the normal character set without special tokens.
    return all(ch in vocab_set for ch in sentence)

def is_valid_length(sentence: str, max_seq_len: int) -> bool:
    """Check if sentence length (including start/end) fits max_seq_len."""
    return len(sentence) < (max_seq_len - 1)  # -1 for end token

def clean_sentences(sentences: List[str], vocab_set: Set[str], unk_char: str = "�") -> List[str]:
    """Replace unknown characters with unk_char."""
    cleaned = []
    for sent in sentences:
        new_sent = "".join(ch if ch in vocab_set else unk_char for ch in sent)
        cleaned.append(new_sent)
    return cleaned

def filter_valid_pairs(
    src_sentences: List[str],
    tgt_sentences: List[str],
    src_vocab_set: Set[str],
    tgt_vocab_set: Set[str],
    max_seq_len: int
) -> Tuple[List[str], List[str]]:
    """Keep only pairs where both are within length and have valid tokens."""
    valid_indices = []
    for i, (src, tgt) in enumerate(zip(src_sentences, tgt_sentences)):
        if (is_valid_length(src, max_seq_len) and
            is_valid_length(tgt, max_seq_len) and
            is_valid_tokens(src, src_vocab_set) and
            is_valid_tokens(tgt, tgt_vocab_set)):
            valid_indices.append(i)
    return [src_sentences[i] for i in valid_indices], [tgt_sentences[i] for i in valid_indices]