import torch
from torch import nn
from cs336_basics.modules import Embedding, Linear, RMSNorm, TransformerBlock

class TransformerLM(torch.nn.Module):
    """
    Transformer language model: token embedding -> N transformer block -> Norm -> Linear -> Softmax -> Output probabilities
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        max_seq_len: int | None = None,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len if max_seq_len is not None else context_length

        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model = self.d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    max_seq_len = self.max_seq_len,
                    theta = rope_theta,
                    eps = eps,
                    device = device,
                    dtype = dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model=self.d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor):
        """
        Args:
            in_indices: tensor of shape (batch_size, seq_len)

        Output:
            logits: tensor of shape (batch, seq_len, vocab_size)
        """

        batch_size, seq_len = in_indices.shape
        positions = torch.arange(seq_len, device=in_indices.device) # (seq_len, )
        token_positions = positions.unsqueeze(0).expand(batch_size, seq_len) # (batch, seq_len)

        x = self.token_embeddings(in_indices)

        for block in self.layers:
            x = block(x, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
