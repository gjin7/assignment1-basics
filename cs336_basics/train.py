import argparse

import numpy as np
import torch

from pathlib import Path
from datetime import datetime

from cs336_basics.config import Config, get_default_config, get_mini_config
from cs336_basics.data import get_batch, open_memmap
from cs336_basics.optimizer import AdamW, learning_rate_schedule
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import clip_gradient, cross_entropy, load_checkpoint, save_checkpoint
from cs336_basics.experiment_tracker import ExperimentTracker

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Run a tiny training job to smoke-test the loop.")
    return parser.parse_args()


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda:0")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def resolve_run_dir(base_run_dir: str, run_name_prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m%d_%H%M")
    run_dir = Path(base_run_dir) / f"{run_name_prefix}_{timestamp}"
    if not run_dir.exists():
        return run_dir

    suffix = 1
    while True:
        candidate = Path(base_run_dir) / f"{run_name_prefix}_{timestamp}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
) -> float:
    if eval_batches <= 0:
        raise ValueError(f"eval_batches must be positive, got {eval_batches}")

    was_training = model.training
    model.eval()

    losses: list[float] = []
    for _ in range(eval_batches):
        input_ids, target_ids = get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)
        losses.append(loss.item())

    if was_training:
        model.train()

    return sum(losses) / len(losses)


def train(cfg: Config) -> Path:
    # 1. Resolve runtime config.
    device = get_device(cfg.train.device)

    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    run_dir = resolve_run_dir(cfg.run.run_dir, cfg.run.run_name_prefix)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load tokenized datasets.
    train_mm = open_memmap(cfg.data.train_data_path, cfg.data.dtype)
    val_mm = open_memmap(cfg.data.val_data_path, cfg.data.dtype)

    # 3. Create model.
    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        num_layers=cfg.model.num_layers,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
        device=device,
    ).to(device)

    # 4. Create optimizer.
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # 5. Resume from checkpoint if requested.
    start_step = 0
    wall_time_offset = 0.0
    if cfg.run.resume_from is not None:
        start_step, checkpoint_metadata = load_checkpoint(
            src=cfg.run.resume_from,
            model=model,
            optimizer=optimizer,
            return_metadata=True,
        )
        wall_time_offset = float(checkpoint_metadata["wall_time"])

    tracker = ExperimentTracker(run_dir=run_dir, config=cfg, wall_time_offset=wall_time_offset)

    # 6. Training loop.
    model.train()
    recent_losses: list[float] = []
    for step in range(start_step, cfg.train.max_steps):
        global_step = step + 1
        ckpt_path = ckpt_dir / f"step_{global_step}.pt"

        # 6.1. Update learning rate based on schedule.
        lr = learning_rate_schedule(
            t=step,
            alpha_max=cfg.optimizer.lr_max,
            alpha_min=cfg.optimizer.lr_min,
            T_w=cfg.optimizer.warmup_steps,
            T_c=cfg.optimizer.cosine_cycle_steps or cfg.train.max_steps,
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        # 6.2. Sample a training batch.
        # input_ids, target_ids: (batch_size, context_length)
        input_ids, target_ids = get_batch(
            dataset=train_mm,
            batch_size=cfg.train.batch_size,
            context_length=cfg.model.context_length,
            device=device,
        )

        # 6.3. Clear gradients from the previous optimization step.
        optimizer.zero_grad()

        # 6.4. Forward pass.
        # logits: (batch_size, context_length, vocab_size)
        logits = model(input_ids)

        # 6.5. Compute next-token prediction loss.
        loss = cross_entropy(logits, target_ids)

        # 6.6. Backward pass.
        loss.backward()

        # 6.7. Clip gradients before the optimizer update.
        if cfg.optimizer.max_grad_norm is not None:
            clip_gradient(model.parameters(), cfg.optimizer.max_grad_norm)

        # 6.8. Update model parameters.
        optimizer.step()

        # 6.9. Log training progress.
        recent_losses.append(loss.item())
        if cfg.train.log_interval > 0 and global_step % cfg.train.log_interval == 0:
            avg_loss = sum(recent_losses[-cfg.train.log_interval :]) / len(recent_losses[-cfg.train.log_interval :])
            print(f"step {global_step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={lr:.2e}")
            tracker.log(
                step=global_step,
                metrics={
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/lr": lr,
                },
            )

        # 6.10. Periodically estimate validation loss.
        if cfg.train.eval_interval > 0 and global_step % cfg.train.eval_interval == 0:
            val_loss = estimate_loss(
                model=model,
                dataset=val_mm,
                batch_size=cfg.train.batch_size,
                context_length=cfg.model.context_length,
                eval_batches=cfg.train.eval_batches,
                device=device,
            )
            print(f"step {global_step}: val_loss={val_loss:.4f}")
            tracker.log(
                step=global_step,
                metrics={
                    "val/loss": val_loss,
                },
            )

        # 6.11. Periodically save checkpoint.
        if cfg.train.ckpt_interval > 0 and global_step % cfg.train.ckpt_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=global_step,
                out=ckpt_path,
                wall_time=tracker.elapsed_time(),
            )
    # 7. Save final checkpoint.
    final_ckpt_path = ckpt_dir / "final.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=cfg.train.max_steps,
        out=final_ckpt_path,
        wall_time=tracker.elapsed_time(),
    )

    return run_dir


def main() -> None:
    args = parse_args()
    cfg = get_mini_config() if args.mini else get_default_config()
    train(cfg)


if __name__ == "__main__":
    main()
