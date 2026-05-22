import os
from collections import Counter
import regex
from pathlib import Path

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    if not text:
        return {}

    segs = [text]
    for st in special_tokens:
        new_segs = []
        for seg in segs:
            new_segs.extend(seg.split(st))
        segs = new_segs

    counts = Counter()
    for seg in segs:
        if not seg:
            continue
        for match in regex.finditer(GPT2_PATTERN, seg):
            s = match.group()
            atom = tuple(bytes([b]) for b in s.encode("utf-8"))
            counts[atom] += 1

    return counts

def get_pair_count_from_word(word: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    """
    Count adjacent pair count from a single word
    """
    counts: dict[tuple[bytes, bytes], int] = {}
    if len(word < 2):
        return pair_counts
    prev = word[0]
    for w in word[1:]:
        pair = (prev, w)
        counts[p] = counts.get(p, 0) + 1
        prev = w
    return counts


def build_pair_counts(word_freq: Counter[tuple[bytes, ...]]) -> dict[tuple[bytes, bytes], int]:
    """
    Returns:
    pair_counts: adjacent bytes pair counting
    """

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for word, freq in word_freq.items():
        if len(word) < 2:
            continue
        local_pair_counts: dict[tuple[bytes, bytes], int]  = get_pair_count_from_word(word)
        for pair, count in local_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count * freq

    return pair_counts

def train_bpe(
    input_path: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    """
    Train BPE
    
    Returns:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    """

    # vocab init: 256 single byte token + special tokens 
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1


    # pretokenize
    text: str = Path(input_path).read_text(encoding="utf-8")
    word_seq_freq: Counter[tuple[bytes, ...]] = pretokenize(text)

    # BPE merge
    pair_counts: dict[tuple[bytes, bytes], int] = build_pair_counts(word_seq_freq)
    merges: List[tuple[bytes, bytes]] = []

    while next_id < vocab_size:
        if not pair_counts:
            break
        # choose best pair
        (a, b), best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count <= 0:
            break

        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        ## 

        





def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    raise NotImplementedError
