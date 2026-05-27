from collections.abc import Iterable, Iterator
import regex

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_RE = regex.compile(GPT2_PATTERN)


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.merge_rank: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(self.merges)}
        self.byte_to_id: dict[bytes, int] = {token_byte: token_id for token_id, token_byte in self.vocab.items()}

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        self.special_pattern = "|".join(regex.escape(token) for token in self.special_tokens)
        self.special_re = regex.compile(f"({self.special_pattern})") if self.special_pattern else None

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return None

    def encode(self, text: str) -> list[int]:
        if self.special_re is None:
            return self._encode_normal(text)

        if not text:
            return []

        ids: list[int] = []
        # the first charater index that has not been processed
        last = 0

        for match in self.special_re.finditer(text):
            start, end = match.span()
            if last < start:
                ids.extend(self._encode_normal(text[last:start]))

            special = match.group()
            ids.append(self.byte_to_id[special.encode("utf-8")])

            last = end

        # final part
        if last < len(text):
            ids.extend(self._encode_normal(text[last:]))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_re is None:
            yield from self.encode("".join(iterable))
            return

        buf = ""

        for chunk in iterable:
            if not chunk:
                continue

            buf += chunk

            # while buffer contains a complete special token
            while True:
                # find earliest special token.
                match = self.special_re.search(buf)
                if match is None:
                    break

                # encode normal text before it and yield ids
                start, end = match.span()
                special = match.group()

                if start > 0:
                    yield from self._encode_normal(buf[:start])

                # yield special token id
                yield self.byte_to_id[special.encode("utf-8")]
                # remove processed prefix
                buf = buf[end:]

        if buf:
            yield from self._encode_normal(buf)

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
