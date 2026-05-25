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
    if len(word) < 2:
        return counts
    prev = word[0]
    for w in word[1:]:
        pair = (prev, w)
        counts[pair] = counts.get(pair, 0) + 1
        prev = w
    return counts


def build_pair_stats(seq_freq: Counter[tuple[bytes, ...]]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Returns:
    pair_counts: adjacent bytes pair counting
    pair_to_seqs: map between tuple to a set of sequence
    """

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for seq, freq in seq_freq.items():
        if len(seq) < 2:
            continue

        local_pair_counts: dict[tuple[bytes, bytes], int]  = get_pair_count_from_word(seq)
        for pair, count in local_pair_counts.items():
            pair_counts[pair] = pair_counts.get(pair, 0) + count * freq
            pair_to_seqs.setdefault(pair, set()).add(seq)

    return pair_counts, pair_to_seqs

def apply_merge(seq: tuple[bytes, ...], a: bytes, b: bytes, new_token: bytes) -> tuple[bytes, ...]:
    merged_seq = []
    i = 0
    while i < len(seq):
        if i < len(seq)-1 and seq[i] == a and seq[i+1] == b:
            merged_seq.append(new_token)
            i += 2
        else:
            merged_seq.append(seq[i])
            i += 1

    return tuple(merged_seq)


# def update_seq_freq(
#     word_seq_freq: Counter[tuple[bytes, ...]], 
#     affected_seqs: set[tuple[bytes, ...]],
#     a: bytes, 
#     b: bytes, 
#     new_token: bytes
# ) -> Counter[tuple[bytes, ...]]:
#     updated_seq_freq = word_seq_freq.copy()

#     for old_seq in affected_seqs:
#         freq = word_seq_freq[old_seq]
#         if freq <= 0:
#             updated_seq_freq.pop(old_seq, None)
#             continue

#         merged_seq = apply_merge(old_seq, a, b, new_token)

#         updated_seq_freq.pop(old_seq, None)
#         updated_seq_freq[merged_seq] += freq
        
#     return updated_seq_freq

def remove_seq_contribution(
    old_seq: tuple[bytes, ...], 
    freq: int, 
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]) -> None:
    """
    Remove old sequence contribution for pair_counts and pair_to_seqs dict
    """

    for pair, count in get_pair_count_from_word(old_seq).items():
        seqs = pair_to_seqs.get(pair)
        if seqs is not None:
            seqs.discard(old_seq)
            if not seqs:
                del pair_to_seqs[pair]

        new_count = pair_counts.get(pair, 0) - count * freq
        if new_count <= 0:
            pair_counts.pop(pair, None)
        else:
            pair_counts[pair] = new_count

def add_seq_contribution(
    new_seq: tuple[bytes, ...],
    freq: int, 
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]) -> None:
    """
    Add new sequence contribution for pair_counts and pair_to_seqs dict
    """

    if len(new_seq) < 2:
        return
    for pair, count in get_pair_count_from_word(new_seq).items():
        pair_counts[pair] = pair_counts.get(pair, 0) + count * freq
        seqs = pair_to_seqs.get(pair)
        if seqs is None:
            pair_to_seqs[pair] = {new_seq}
        else:
            seqs.add(new_seq)


def update_bpe_stats(
    word_seq_freq: Counter[tuple[bytes, ...]], 
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
    affected_seqs: list[[tuple[bytes, ...]]], 
    a: bytes, 
    b: bytes, 
    new_token: bytes) -> None:
    
    for old_seq in affected_seqs:
        freq = word_seq_freq.pop(old_seq, 0)
        if freq <= 0:
            continue

        remove_seq_contribution(old_seq, freq, pair_counts, pair_to_seqs)
        merged_seq = apply_merge(old_seq, a, b, new_token)
        word_seq_freq[merged_seq] += freq 
        add_seq_contribution(merged_seq, freq, pair_counts, pair_to_seqs)

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
    word_seq_freq: Counter[tuple[bytes, ...]] = pretokenize(text, special_tokens)

    # BPE merge
    pair_counts, pair_to_seqs = build_pair_stats(word_seq_freq)
    merges: list[tuple[bytes, bytes]] = []

    while next_id < vocab_size:
        if not pair_counts:
            break
        # choose best pair
        (a, b), best_count = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
        if best_count <= 0:
            break

        affected_seqs = list(pair_to_seqs[(a, b)])
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        update_bpe_stats(word_seq_freq, pair_counts, pair_to_seqs, affected_seqs, a, b, new_token)

    return vocab, merges