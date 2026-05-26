import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

from cs336_basics.train_bpe import run_train_bpe


def peak_rss_gb() -> float:
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    max_rss = max(self_rss, child_rss)
    if sys.platform == "darwin":
        return max_rss / (1024**3)
    return max_rss / (1024**2)


def token_display(token: bytes) -> str:
    return token.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tinystories_bpe_10k"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--num-processes", type=int, default=1)
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input_path}. "
            "Download TinyStories first, or pass --input-path tests/fixtures/tinystories_sample_5M.txt."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    vocab, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
        num_processes=args.num_processes,
    )
    elapsed_seconds = time.perf_counter() - start
    peak_gb = peak_rss_gb()

    longest_id, longest_token = max(vocab.items(), key=lambda item: len(item[1]))

    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.txt"
    summary_path = args.output_dir / "summary.json"

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({str(idx): list(token) for idx, token in vocab.items()}, f, indent=2)

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left.hex()} {right.hex()}\n")

    summary = {
        "input_path": os.fspath(args.input_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_token": args.special_token,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_gb": peak_gb,
        "longest_token": {
            "id": longest_id,
            "num_bytes": len(longest_token),
            "bytes": list(longest_token),
            "hex": longest_token.hex(),
            "utf8": token_display(longest_token),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote vocab: {vocab_path}")
    print(f"wrote merges: {merges_path}")
    print(f"wrote summary: {summary_path}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print(f"peak_rss_gb: {peak_gb:.3f}")
    print(
        "longest_token: "
        f"id={longest_id}, bytes={len(longest_token)}, utf8={token_display(longest_token)!r}, hex={longest_token.hex()}"
    )


if __name__ == "__main__":
    main()
