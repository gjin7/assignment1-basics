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
