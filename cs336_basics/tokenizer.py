from collections.abc import Iterable, Iterator
import regex

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_RE = regex.compile(GPT2_PATTERN)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[bytes] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.merge_rank: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(self.merges)}
        self.byte_to_id: dict[bytes, int] = {token_byte: token_id for token_id, token_byte in self.vocab.items()}

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return None

    def encode(self, text: str) -> list[int]:
        return self._encode_normal(text)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return None

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        token_bytes = [self.vocab[idx] for idx in ids]
        b = b"".join(token_bytes)
        return b.decode("utf-8", errors="replace")

    # -----------------------
    # Helper functions
    # -----------------------
    def _encode_normal(self, text: str) -> list[int]:
        """
        encode text without special tokens
        """
        res: list[int] = []
        if not text:
            return res

        for match in GPT2_RE.finditer(text):
            s = match.group()
            if not s:
                continue
            raw_bytes = s.encode("utf-8")
            encoded_bytes: list[bytes] = self._apply_bpe(raw_bytes)
            for b in encoded_bytes:
                res.append(self.byte_to_id[b])

        return res

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply the right merge based on the learned merges from BPE
        token_bytes example: " cat".encode()
        """

        seq: list[bytes] = [bytes([b]) for b in token_bytes]
        if len(seq) <= 1:
            return seq

        while True:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank: int | None = None

            # For each adjacent pair, find best from merge_rank derived from merges
            prev = seq[0]
            for curr in seq[1:]:
                pair = (prev, curr)
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
                prev = curr

            if best_pair is None:
                break

            # Merge all occurance of best pair
            a, b = best_pair
            new_token = a + b

            merged_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    merged_seq.append(new_token)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1

            seq = merged_seq
            if len(seq) <= 1:
                return seq

        return seq
