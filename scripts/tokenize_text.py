import argparse
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def tokenize_file_to_bin(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    dtype: np.dtype = np.uint16,
    flush_size: int = 1_000_000,
) -> int:
    """
    Stream-encode a text file into token IDs and write them as a raw binary file.
    """
    dtype = np.dtype(dtype)

    max_token_id = max(tokenizer.vocab.keys())
    if max_token_id > np.iinfo(dtype).max:
        raise ValueError(f"token id {max_token_id} does not fit in {dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer: list[int] = []
    total_tokens = 0

    with input_path.open("r", encoding="utf-8") as in_f, output_path.open("wb") as out_f:
        for token_id in tokenizer.encode_iterable(in_f):
            buffer.append(token_id)

            if len(buffer) >= flush_size:
                arr = np.asarray(buffer, dtype=dtype)
                arr.tofile(out_f)
                total_tokens += arr.size
                buffer.clear()

        if buffer:
            arr = np.asarray(buffer, dtype=dtype)
            arr.tofile(out_f)
            total_tokens += arr.size

    return total_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    parser.add_argument("--flush-size", type=int, default=1_000_000)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.special_token],
    )

    total_tokens = tokenize_file_to_bin(
        tokenizer=tokenizer,
        input_path=args.input_path,
        output_path=args.output_path,
        dtype=np.dtype(args.dtype),
        flush_size=args.flush_size,
    )

    print(f"wrote {total_tokens:,} tokens to {args.output_path}")
    print(f"dtype: {args.dtype}")


if __name__ == "__main__":
    main()
