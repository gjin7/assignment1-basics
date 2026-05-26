import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return None

    def encode(self, text: str) -> list[int]:
        return None

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        return None

    def decode(self, ids: list[int]) -> str:
        return None
