import torch
from typing import Optional, Callable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                adjusted_learning_rate = lr * math.sqrt(1.0-beta2**step) / (1.0-beta1**step)

                # apply weight decay
                if wd != 0.0:
                   p.add_(p, alpha=-lr*wd)

                # update first/second moment estimate
                m, v = state["m"], state["v"]
                m.mul_(beta1).add_(grad, alpha=1-beta1) # m = beta1 * m + (1-beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2) # v = beta2 * m + (1-beta2) * grad^2

                # update parameters
                denom = torch.sqrt(v).add_(eps)
                p.addcdiv_(m, denom, value=-adjusted_learning_rate)

        return loss


def learning_rate_schedule(
        t: int, 
        alpha_max: float, 
        alpha_min: float, 
        T_w: int, 
        T_c: int
    ) -> float:
    """
    Cosine learning rate schedule with warm up. 

    Args:
        t: current iteration
        alpha_max: max learning rate
        alpha_min: min learning rate
        T_w: warm up iteration
        T_c: final iteration of cosine annealing 
    """
    if t < T_w:
        return 1.0 * t / T_w * alpha_max
    elif t > T_c:
        return alpha_min
    else:
        num = t - T_w
        denom = T_c - T_w
        
        return alpha_min + 0.5 * (1 + math.cos(1.0 * num / denom * math.pi)) * (alpha_max - alpha_min)
