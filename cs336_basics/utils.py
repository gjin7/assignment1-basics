import torch

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
