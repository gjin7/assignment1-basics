import torch

def top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to a probability vector.

    Args:
        probs: 1D tensor of probabilities.
        top_p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered and renormalized probability vector with the same shape as probs.
    """
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1D, got shape {tuple(probs.shape)}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    # Keep the first token that makes cumulative probability cross top_p.
    keep = cumsum - sorted_probs < top_p

    kept_probs = sorted_probs[keep]
    kept_idx = sorted_idx[keep]
    kept_probs = kept_probs / kept_probs.sum()

    filtered_probs = torch.zeros_like(probs)
    filtered_probs[kept_idx] = kept_probs

    return filtered_probs


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    context_length: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if prompt_ids.ndim != 1:
        raise ValueError(f"prompt_ids must be 1D, got shape {tuple(prompt_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if context_length is not None and context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    model_was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    output_ids = prompt_ids.to(device=device, dtype=torch.long)

    for _ in range(max_new_tokens):
        # input_ids: (1, min(current_seq_len, context_length))
        if context_length is not None:
            input_ids = output_ids[-context_length:]
        else:
            input_ids = output_ids

        # logits: (1, seq_len, vocab_size)
        logits = model(input_ids.unsqueeze(0))
        next_logits = logits[0, -1, :]  # (vocab_size,)

        if temperature == 0.0:
            next_id = torch.argmax(next_logits, dim=-1).reshape(1)
        else:
            probs = torch.softmax(next_logits / temperature, dim=-1)

            if top_p is not None:
                probs = top_p_sampling(probs, top_p)

            next_id = torch.multinomial(probs, num_samples=1)

        output_ids = torch.cat([output_ids, next_id], dim=-1)

        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    if model_was_training:
        model.train()

    return output_ids

def main() -> None:
    import argparse
    from pathlib import Path

    from cs336_basics.config import get_default_config, get_mini_config
    from cs336_basics.optimizer import AdamW
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.train import get_device
    from cs336_basics.transformer_lm import TransformerLM
    from cs336_basics.utils import load_checkpoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Use the mini model config.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, default=Path("outputs/tinystories_bpe_10k/vocab.pkl"))
    parser.add_argument("--merges-path", type=Path, default=Path("outputs/tinystories_bpe_10k/merges.pkl"))
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    cfg = get_mini_config() if args.mini else get_default_config()
    device = get_device(cfg.train.device)

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

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr_max,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    load_checkpoint(args.checkpoint_path, model, optimizer)

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long)
    eos_token_id = tokenizer.byte_to_id[b"<|endoftext|>"]

    output_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_length=cfg.model.context_length,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    print(tokenizer.decode(output_ids.tolist()))


if __name__ == "__main__":
    main()
