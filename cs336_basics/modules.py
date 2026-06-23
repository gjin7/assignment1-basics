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


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("RoPE only supports even embedding dimensions, d_k = {d_k}")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer, max_seq_len = {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_idx = torch.arange(
            0, self.d_k // 2, device=device, dtype=torch.float32
        )  # use torch.arrange to create pair_idx as tensor
        freq_factor = self.theta ** (-2 * pair_idx / self.d_k)  # (d_k/2, )

        # angle = position * freq_factor
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32).unsqueeze(-1)  # (max_seq_len, 1)
        angles = positions * freq_factor  # (max_seq_len, d_k/2)

        sin = torch.sin(angles)
        cos = torch.cos(angles)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate x with an arbitrary number of batch dimensions. Assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        x along the sequence dimension.

        input: (..., seq_len, d_k)
        token_positions: (..., seq_len)

        output: (..., seq_len, d_k)
        """

        if x.size(-1) != self.d_k:
            raise ValueError(f"Expected x.size(-1) == d_k, but got {x.size(-1)}")

        positions = token_positions.to(device=x.device)
        cos_selected = self.cos[positions]  # (..., seq_len, d_k/2)
        sin_selected = self.sin[positions]  # (..., seq_len, d_k/2)

        x_fp32 = x.to(torch.float32)
        cos = cos_selected.to(torch.float32)
        sin = sin_selected.to(torch.float32)

        x_even = x_fp32[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x_fp32[..., 1::2]  # (..., seq_len, d_k/2)

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd

        return out


def softmax(x: torch.Tensor, dim: int):
    """
    apply softmax to ith dimension of the input tensor

    subtract the largest entry of v from all elements of v, making
    the new largest entry 0 for numberical stability
    """

    x_max = torch.max(x, dim=dim, keepdim=True).values
    z = x - x_max

    exp_z = torch.exp(z)
    sum_exp = torch.sum(exp_z, dim=dim, keepdim=True)

    return exp_z / sum_exp


def scaled_dot_product_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
):
    """
    Dot product attention function with optional user provided mask

    Args:
    query: tensor of shape (batch_size, ..., seq_len, d_k) -- n x d_k
    key: tensor of shape (batch_size, ..., seq_len, d_k) -- m x d_k
    value: tensor of shape (batch_size, ..., seq_len, d_v) -- m x d_v
    mask: Optional bool tensor of shape (seq_len, seq_len). True means attend and False means not attend

    Return:
    tensor of shape (batch_size, ..., seq_len, d_v)
    """
    if query.dim() < 2 or key.dim() < 2 or value.dim() < 2:
        raise ValueError("k, q, v must have shape of (..., seq_len, d_*)")

    if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise ValueError("batch-like dimension of k, q, v must be same")

    d_k = key.shape[-1]
    if d_k != value.shape[-1]:
        raise ValueError("d_k for query and key must be the same")

    scale = 1.0 / math.sqrt(d_k)
    qk_scaled = torch.einsum("...qd, ...kd -> ...qk", query, key) * scale

    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError("mask dtype must be bool")

        qk_scaled_masked = qk_scaled.masked_fill(~mask, -torch.inf)  # (..., q_seq_len, k_seq_len)

    # q over each key dimension
    logits = softmax(qk_scaled_masked, dim=-1)
    out = torch.einsum("... nm, ... mv -> ...nv", logits, value)

    return out


class CasualMultiHeadSelfAttention(nn.Module):
    """
    Casual Multi-Head Self Attention

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
    for head_i = Attention(Q_i, K_i, V_i)

    MultiHeadSelfAttention(x) = W_oMultiHead(W_q x, W_k x, W_v x)

    Shapes:
    x: (..., seq_len, d_model)
    QKV: (..., seq_len, d_model)
    output: (..., seq_len, d_model)
    """

    def __init__(
        self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.head_dim = self.d_model // self.num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.o_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project

        # 2. Split heads

        # 3. Build casual mask

        # 4. Apply attention

        # 5. Merge heads back

        return None
