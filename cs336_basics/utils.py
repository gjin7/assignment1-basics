import torch
import os
from collections.abc import Iterable
import typing
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

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    wall_time: float | None = None,
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    obj = {
        "iteration": iteration, 
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    } 
    if wall_time is not None:
        obj["wall_time"] = wall_time
    
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    return_metadata: bool = False,
) -> int | tuple[int, dict[str, object]]:
    ckpt = torch.load(src)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    iteration = int(ckpt["iteration"])
    if return_metadata:
        return iteration, {
            "wall_time": float(ckpt.get("wall_time", 0.0)),
        }

    return iteration
