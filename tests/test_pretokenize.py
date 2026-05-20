from collections import Counter

from cs336_basics.train_bpe import pretokenize


def byte_tuple(text: str) -> tuple[bytes, ...]:
    return tuple(bytes([b]) for b in text.encode("utf-8"))


def test_pretokenize_empty_input_has_no_counts():
    assert pretokenize("", special_tokens=[]) == Counter()


def test_pretokenize_counts_repeated_pretokens():
    counts = pretokenize("hello\nhello", special_tokens=[])

    assert counts == Counter(
        {
            byte_tuple("hello"): 2,
            byte_tuple("\n"): 1,
        }
    )


def test_pretokenize_preserves_gpt2_leading_space_behavior():
    counts = pretokenize("hello hello", special_tokens=[])

    assert counts == Counter(
        {
            byte_tuple("hello"): 1,
            byte_tuple(" hello"): 1,
        }
    )


def test_pretokenize_encodes_unicode_as_utf8_byte_tokens():
    counts = pretokenize("é", special_tokens=[])

    assert counts == Counter({(b"\xc3", b"\xa9"): 1})
    for pretoken in counts:
        assert isinstance(pretoken, tuple)
        assert all(isinstance(part, bytes) for part in pretoken)
        assert all(len(part) == 1 for part in pretoken)


def test_pretokenize_splits_out_special_tokens():
    counts = pretokenize("hello <|endoftext|> world", special_tokens=["<|endoftext|>"])

    assert byte_tuple("<|endoftext|>") not in counts
    assert all(b"<|endoftext|>" not in b"".join(pretoken) for pretoken in counts)


def test_pretokenize_does_not_join_across_special_token_boundary():
    counts = pretokenize("abc<|endoftext|>def", special_tokens=["<|endoftext|>"])

    assert byte_tuple("abcdef") not in counts
    assert counts == Counter(
        {
            byte_tuple("abc"): 1,
            byte_tuple("def"): 1,
        }
    )
