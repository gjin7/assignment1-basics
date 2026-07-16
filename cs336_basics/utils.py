import torch
from typing import Iterable
import math

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        logits: tensor of shape [..., vocab_size]
        targets: tensor of shape [...]
    Returns:
        cross entropy: a scalar tensor of negative log likelihood
    """

    largest = logits.max(dim=-1, keepdim=True).values
    logits = logits - largest

    logsum = logits.exp().sum(dim=-1).log() # shape [...]

    idx = targets.unsqueeze(-1) # shape [..., 1]
    targets_logits = logits.gather(dim=-1, index=idx).squeeze(-1) # shape [...]

    nll = logsum - targets_logits

    return nll.mean()


def clip_gradient(
    params: Iterable[torch.nn.Parameter],
    max_norm: float,
    epsilon: float = 1e-6,
) -> None:
    """
    in-place graident clipping to avoid graident execeeding max value. 

    Args:
        params: torch.nn.Parameter iterables, where gradient be updated in place 
        max_norm: maximum L2 norm allowed 
        epsilon: small constant used to scale down gradient
    """

    if max_norm < 0:
        raise ValueError(f"max_norm must be non-negative, got {max_norm}")

    grads = []
    for p in params:
        if p is None:
            continue
        g = p.grad
        if g is None:
            continue

        grads.append(g)

    if len(grads) == 0:
        return
    
    total_sq = 0.0
    for g in grads:
        total_sq += g.square().sum()
    total_norm = math.sqrt(total_sq)

    coeffcient = max_norm / (total_norm + epsilon)
    if coeffcient < 1:
        for g in grads:
            g.mul_(coeffcient)

    return
