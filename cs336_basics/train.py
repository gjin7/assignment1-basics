import argparse

import numpy as np
import torch

from cs336_basics.config import get_default_config, get_mini_config
from cs336_basics.data import get_batch, open_memmap
from cs336_basics.optimizer import AdamW, learning_rate_schedule
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.utils import clip_gradient, cross_entropy, load_checkpoint


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


def main() -> None:
    # 1. Load config.
    args = parse_args()
    cfg = get_mini_config() if args.mini else get_default_config()
    device = get_device(cfg.train.device)

    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    # 2. Load tokenized datasets.
    train_mm = open_memmap(cfg.data.train_data_path, cfg.data.dtype)
    _val_mm = open_memmap(cfg.data.val_data_path, cfg.data.dtype)

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
    if cfg.train.resume_from is not None:
        start_step = load_checkpoint(src=cfg.train.resume_from, model=model, optimizer=optimizer)

    # 6. Training loop.
    model.train()
    recent_losses: list[float] = []
    for step in range(start_step, cfg.train.max_steps):
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
        if step % cfg.train.log_interval == 0:
            avg_loss = sum(recent_losses[-cfg.train.log_interval:]) / len(recent_losses[-cfg.train.log_interval:])
            print(f"step {step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}, lr={lr:.2e}")


if __name__ == "__main__":
    main()
