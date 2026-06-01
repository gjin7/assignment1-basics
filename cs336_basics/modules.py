from torch import nn
import torch
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # store W (NOT W^T)
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype))
        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, d_in) -> (b, d_out)
        return torch.einsum("... i, o i -> ... o", x, self.weight)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(torch.ones((self.d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model) and return of same shape
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Normalize and apply gain
        normed = (x_fp32 / rms) * self.weight.to(torch.float32)

        return normed.to(in_dtype)


def round_up_to_multiple_of(x: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        raise ValueError("multiple_of must be a positive integer")

    return ((x + multiple_of - 1) // multiple_of) * multiple_of


def default_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """
    default 8/3 and round up to multiple of 64 by default for hardware efficiency
    """
    val = int(math.ceil(8.0 * d_model / 3.0))
    return round_up_to_multiple_of(val, multiple_of)


class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network, composed of SiLU activation and a GLU

    FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1 x) ⊙ W3 x)
    d(ff) = 8/3 d(model)

    input: (..., d_model)
    W1, W3: (dff, d_model)
    W2: (d_model, dff)
    dff: 8/3 * d_model
    """

    def __init__(self, d_model: int, d_ff: int, multiple_of: int = 64, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else default_d_ff(d_model, multiple_of)

        self.w1 = Linear(d_model, d_ff, dtype=dtype, device=device)
        self.w2 = Linear(d_ff, d_model, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff, dtype=dtype, device=device)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)

        gated = self.silu(a) * b
        return self.w2(gated)
