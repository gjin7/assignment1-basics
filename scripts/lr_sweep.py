import argparse
import gc
import json
from pathlib import Path

import torch

from cs336_basics.config import Config, get_default_config, get_mini_config
from cs336_basics.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Use the mini model config.")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--run-name-prefix", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--ckpt-interval", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def build_config(args: argparse.Namespace, lr: float) -> Config:
    cfg = get_mini_config() if args.mini else get_default_config()

    cfg.optimizer.lr_max = lr
    cfg.optimizer.lr_min = lr * args.lr_min_ratio

    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.warmup_steps is not None:
        cfg.optimizer.warmup_steps = args.warmup_steps
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.eval_interval is not None:
        cfg.train.eval_interval = args.eval_interval
    if args.ckpt_interval is not None:
        cfg.train.ckpt_interval = args.ckpt_interval
    if args.device is not None:
        cfg.train.device = args.device

    cfg.optimizer.cosine_cycle_steps = cfg.train.max_steps

    base_prefix = args.run_name_prefix
    if base_prefix is None:
        base_prefix = "ts_mini_sweep" if args.mini else "ts_sweep"
    cfg.run.run_name_prefix = f"{base_prefix}_lr{format_lr(lr)}"

    return cfg


def last_metric(run_dir: Path, metric_name: str) -> float | None:
    metrics_path = run_dir / "metrics.jsonl"
    value = None
    with metrics_path.open() as f:
        for line in f:
            row = json.loads(line)
            if metric_name in row:
                value = float(row[metric_name])
    return value


def main() -> None:
    args = parse_args()
    results: list[tuple[float, Path, float | None]] = []

    for lr in args.lrs:
        cfg = build_config(args, lr)
        print(f"starting lr_max={cfg.optimizer.lr_max:.2e}, lr_min={cfg.optimizer.lr_min:.2e}")
        run_dir = train(cfg)
        val_loss = last_metric(run_dir, "val/loss")
        results.append((lr, run_dir, val_loss))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("learning-rate sweep summary:")
    for lr, run_dir, val_loss in results:
        val_loss_text = "none" if val_loss is None else f"{val_loss:.4f}"
        print(f"lr_max={lr:.2e} final_val_loss={val_loss_text} run_dir={run_dir}")


if __name__ == "__main__":
    main()
